"""A demo of parameter optimization of MPC controller using TuRBO (TrUst Region Bayesian Optimization)[1].

Uses uber-research's implementation of TuRBO[2] to optimize the time it takes for mpc_acados.py controller to reach the goal.
Since we need to change the weight parameters after initialization, some fiddling of internal variables of the
controller class is done. This demo can easily be extended to MPCC++ controller's case.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
import time
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
    def __init__(self, x0, x_lb=None, x_ub=None):
        # Define the bounds of the hyperparameters around the initial guess
        self.dim = len(x0)
        if x_lb is None:
            self.lb = 0.4 * x0
        else:
            self.lb = x_lb
        if x_ub is None:
            self.ub = 2.5 * x0
        else:
            self.ub = x_ub

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        x = np.around(x, 4)
        cost = 0.0
        hyperparams = {
            "cost_info": {
                "linear": {
                    "Qs_pos": x[0],
                    "Qs_vel": x[1],
                    "Qs_rpy": x[2],
                    "Qs_drpy": x[3],
                    "Qt_pos": x[4],
                    "Qt_vel": x[5],
                    "Qt_rpy": x[6],
                    "Qt_drpy": x[7],
                    "R_f": x[8],
                    "R_df": x[9],
                }
            },
            "constraints_info": {
                "obstacle_diameter": x[10],
                "Wn": x[11],  # Updated Wn to use x[11]
                "Wgate": x[12],  # Updated Wgate to use x[12]
                "tunnelTransitionMargin": x[13],
                "tunnelTransitionSteepness": x[14],
            },
            "optimizer_info": {"softPenalty": x[15]},
            "solver_options": {"acados": {"build": True, "generate": True}},
        }
        try:
            avg_error, avg_collided, avg_gate_times, avg_num_gates_passed = do_simulation(
                hyperparams, n_runs=5
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
        cost += 100 * avg_collided
        cost += 10 * avg_error
        cost -= 2000 / avg_gate_times if avg_gate_times != 0 else 1

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

    avg_gate_time = 0
    avg_error = 0
    avg_collided = 0
    avg_num_gates_passed = 0
    random_seed = int(time.time()) % (2**32 - 1)
    print(f"Generated random seed: {random_seed}")
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        for run_number in range(n_runs):  # Run n_runs episodes with the controller
            done = False
            random_seed = int(time.time()) % (2**32 - 1)
            obs, info = env.reset(seed=random_seed)
            if run_number != 0:
                hyperparams["solver_options"]["acados"]["build"] = False
                hyperparams["solver_options"]["acados"]["generate"] = False

            controller: MPC = controller_cls(obs, info, hyperparams=hyperparams)

            while not done:
                action = controller.compute_control(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # Update the controller internal state and models.
                controller.step_callback(action, obs, reward, terminated, truncated, info)

            error, collided, gate_times, num_gates_passed = (
                controller.episode_callback()
            )  # Update the controller internal state and models.

            avg_error += error / n_runs
            avg_collided += int(collided) / n_runs
            avg_gate_time += gate_times / n_runs
            avg_num_gates_passed += num_gates_passed / n_runs
            controller.episode_reset()

    env.close()
    return avg_error, avg_collided, avg_gate_time, avg_num_gates_passed


def do_optimization():
    """Conducts parameter optimization."""
    # Hyperparameters for TuRBO 6 parameters, only 5 active (either Qs_ang or Qs_quat)
    # [Qs_pos, Qs_vel, Qs_ang, Qs_quat, Qs_dang, Qt_pos, Qt_vel, Qt_ang, Qt_quat, Qt_dang,Rs, softPenalty]
    x0 = np.array([4, 0.05, 0.5, 0.1, 10, 0.01, 2, 2, 0.2, 0.01, 0.15, 0.9, 0.15, 0.05, 1, 1000]).T
    x_lb = np.array(
        [1, 0.05, 0.03, 0.05, 1, 0.01, 0.5, 0.5, 0.1, 0.01, 0.14, 0.7, 0.13, 0.05, 0.8, 1000]
    ).T
    x_ub = np.array([10, 1, 2, 1, 20, 0.1, 5, 5, 1, 0.05, 0.25, 1, 0.25, 0.2, 5, 5000]).T
    f = Levy(x0, x_lb, x_ub)

    hyperparams = {
        "cost_info": {
            "linear": {
                "Qs_pos": 4,
                "Qs_vel": 0.05,
                "Qs_rpy": 0.5,
                "Qs_drpy": 0.1,
                "Qt_pos": 10,
                "Qt_vel": 0.01,
                "Qt_rpy": 2,
                "Qt_drpy": 2,
                "R_f": 0.2,
                "R_df": 0.01,
            }
        },
        "constraints_info": {
            "obstacle_diameter": 0.15,
            "Wn": 0.9,
            "Wgate": 0.15,
            "tunnelTransitionMargin": 0.05,
            "tunnelTransitionSteepness": 1,
        },
        "optimizer_info": {"softPenalty": 1000},
    }

    n_init = 2 * len(x0)
    turbo = TurboM(
        f=f,  # Handle to objective function
        lb=f.lb,  # Numpy array specifying lower bounds
        ub=f.ub,  # Numpy array specifying upper bounds
        n_init=n_init,  # Number of initial bounds from an Latin hypercube design
        max_evals=300,  # Maximum number of evaluations
        batch_size=30,  # How large batch size TuRBO uses
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
    # Extract the best 10 observed values and corresponding evaluated points

    indices_best_10 = np.argsort(fX)[:10]
    f_best_10 = fX[indices_best_10]
    x_best_10 = X[indices_best_10, :]

    print("Best 10 values found:")
    for i in range(10):
        print(f"{i + 1}: f(x) = {f_best_10[i]:.3f}, x = {np.around(x_best_10[i], 3)}")

    # Extract the best overall value
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]
    print(
        "Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3))
    )


if __name__ == "__main__":
    do_optimization()
    # print(target_function(np.zeros(32)))
