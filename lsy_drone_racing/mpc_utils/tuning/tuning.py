"""A demo of parameter optimization of MPC controller using TuRBO (TrUst Region Bayesian Optimization)[1].

Uses uber-research's implementation of TuRBO[2] to optimize the time it takes for mpc_acados.py controller to reach the goal.
Since we need to change the weight parameters after initialization, some fiddling of internal variables of the
controller class is done. This demo can easily be extended to MPCC++ controller's case.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium
import numpy as np
from turbo import TurboM
import contextlib
from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.controller import BaseController
    from lsy_drone_racing.control.mpc import MPC
    from lsy_drone_racing.envs.drone_racing_env import DroneRacingEnv


class Levy:
    def __init__(self, x0):
        # Define the bounds of the hyperparameters around the initial guess
        self.dim = len(x0)
        self.lb = 0.1 * x0
        self.ub = 10 * x0

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        cost = 0.0
        hyperparams = {
            "Qs_pos": x[0],
            "Qs_vel": x[1],
            "Qs_ang": x[2],
            "Qs_quat": x[3],
            "Qs_dang": x[4],
            "Qt_pos": x[5],
            "Qt_vel": x[6],
            "Qt_ang": x[7],
            "Qt_quat": x[8],
            "Qt_dang": x[9],
            "Rs": x[10],
        }
        try:
            avg_error, avg_collided, avg_gate_times, avg_num_gates_passed = do_simulation(
                hyperparams, n_runs=1
            )
            cost = self.cost_function(avg_error, avg_collided, avg_gate_times, avg_num_gates_passed)
        except BaseException as e:  # Solver error - penalize harshly
            cost += 1000.0
            print(f"run failed:{e}")
        return cost

    def cost_function(self, avg_error, avg_collided, avg_gate_times, avg_num_gates_passed) -> float:
        """Cost function for the TurBO hyperparameter search.

        Args:
            times_per_gate: The time taken to pass each gate.
            gates_passed: The number of gates passed.
            avg_pos_error: The average position error.
            crashed: Whether the drone crashed.
        """
        print("avg gate times", avg_gate_times)
        print("avg num gates passed", avg_num_gates_passed)
        print("avg collided", avg_collided)
        print("avg error", avg_error)
        cost = 0.0
        cost += 700 * avg_collided
        cost += 100 * avg_error
        for i in range(len(avg_gate_times)):
            if avg_gate_times[i] > 0:
                cost -= 100 * avg_num_gates_passed / avg_gate_times[i]
            else:
                cost += 100
        print(f"cost: {cost}")
        return cost


def do_simulation(
    hyperparams: dict,
    config: str = "level3.toml",
    controller: str = "mpc.py",
    n_runs: int = 1,
    gui: bool | None = None,
    env_id: str | None = None,
):
    """Evaluate the drone controller over one episode.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.
        env_id: The id of the environment to use. If None, the environment specified in the config
            file is used.

    Returns:
        An episode time, and the gate the controller managed to reach in this episode.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(os.path.abspath(__file__)).parents[3] / "config" / config)
    if gui is None:
        gui = config.sim.gui
    else:
        config.sim.gui = gui
    # Load the controller module
    control_path = Path(os.path.abspath(__file__)).parents[2] / "control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRacingEnv = gymnasium.make(env_id or config.env.id, config=config)
    # print("Environment created")

    avg_gate_times = np.zeros(4)
    avg_error = 0
    avg_collided = 0
    avg_num_gates_passed = 0
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        for _ in range(n_runs):  # Run n_runs episodes with the controller
            done = False
            obs, info = env.reset()

            controller: MPC = controller_cls(obs, info, hyperparams=hyperparams)
            # raise Exception("This is a test exception")

            # controller.__init__(obs, info, hyperparams=hyperparams)

            i = 0
            while not done:
                curr_time = i / config.env.freq
                action = controller.compute_control(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # Update the controller internal state and models.
                controller.step_callback(action, obs, reward, terminated, truncated, info)
                i += 1

            error, collided, gate_times, num_gates_passed = (
                controller.episode_callback()
            )  # Update the controller internal state and models.

            # Debug prints to check dimensions

            avg_error += error / n_runs
            avg_collided += int(collided) / n_runs
            avg_gate_times += gate_times / n_runs
            avg_num_gates_passed += num_gates_passed / n_runs
            controller.episode_reset()

    env.close()
    return avg_error, avg_collided, avg_gate_times, avg_num_gates_passed


def do_optimization():
    """Conducts parameter optimization.

    TuRBO optimizer is set up and executed in this function.
    Various optimization parameters including upper and lower bounds of the parameters are set here.
    """
    # create optimizer instance
    # Hyperparameters for TuRBO 6 parameters, only 5 active (either Qs_ang or Qs_quat)
    # [Qs_pos, Qs_vel, Qs_ang, Qs_quat, Qs_dang, Qt_pos, Qt_vel, Qt_ang, Qt_quat, Qt_dang,Rs, softPenalty]
    x0 = np.array([2, 0.1, 0.5, 0.05, 0.1, 10, 0.005, 0.1, 2, 2, 0.2]).T
    f = Levy(x0)

    n_init = 2 * len(x0)
    turbo = TurboM(
        f=f,  # Handle to objective function
        lb=f.lb,  # Numpy array specifying lower bounds
        ub=f.ub,  # Numpy array specifying upper bounds
        n_init=n_init,  # Number of initial bounds from an Latin hypercube design
        max_evals=300,  # Maximum number of evaluations
        batch_size=10,  # How large batch size TuRBO uses
        n_trust_regions=8,  # Sets m
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float32",  # float64 or float32
    )

    turbo.optimize()
    X = turbo.X  # Evaluated points
    fX = turbo.fX  # Observed values
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]
    print(
        "Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3))
    )


if __name__ == "__main__":
    do_optimization()
    # print(target_function(np.zeros(32)))
