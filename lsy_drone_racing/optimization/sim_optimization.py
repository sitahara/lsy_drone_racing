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

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:

    from numpy.typing import NDArray

    from lsy_drone_racing.control.controller import BaseController
    from lsy_drone_racing.envs.drone_racing_env import DroneRacingEnv

def do_simulation(
    config: str = "level1.toml",
    controller: str | None = None,
    n_runs: int = 1,
    gui: bool | None = None,
    env_id: str | None = None,
    Qs: NDArray[np.floating] | None = None, # (12,) NDArray
    Qt: NDArray[np.floating] | None = None, # (12,) NDArray
    R: NDArray[np.floating] | None = None, # (4, ) NDArray
    dR: NDArray[np.floating] | None = None, # (4, ) NDArray
) -> list[float]:
    """Evaluate the drone controller over one episode.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        gui: Enable/disable the simulation GUI.
        env_id: The id of the environment to use. If None, the environment specified in the config
            file is used.
        Qs: State error weight matrix's diagonal elements to be used in this run.
        Qt: Terminal state error weight matrix's diagonal elements to be used in this run.
        R:  Control input error weight matrix's diagonal elements to be used in this run.
        dR: A parameter of the cost function. Not sure what it's used for but it's there.

    Returns:
        An episode time, and the gate the controller managed to reach in this episode.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(os.path.abspath(__file__)).parents[2] / "config" / config)
    if gui is None:
        gui = config.sim.gui
    else:
        config.sim.gui = gui
    # Load the controller module
    control_path = Path(os.path.abspath(__file__)).parents[1] / "control"
    controller = "mpc_acados.py"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRacingEnv = gymnasium.make(env_id or config.env.id, config=config)

    ep_times = []
    final_gate = 1
    for _ in range(n_runs):  # Run n_runs episodes with the controller
        done = False
        obs, info = env.reset()
        controller: BaseController = controller_cls(obs, info)
        if Qs is not None: 
            assert Qs.shape == (12,)
            controller.Qs = np.diag(Qs) / controller.x_scal
        
        if Qt is not None:
            assert Qt.shape == (12,)
            controller.Qt = np.diag(Qt) / controller.x_scal
        
        if R is not None:
            assert R.shape == (4,)
            controller.Rs = np.diag(R) / controller.u_scal
        
        if dR is not None:
            assert dR.shape == (4,)
            controller.Rsdelta = np.diag(dR) / controller.u_rate_scal
        controller.model = None
        controller.ocp_solver = None
        controller.setupAcadosModel()
        controller.setupIPOPTOptimizer()
        controller.setupAcadosOptimizer()
        
        i = 0
        while not done:
            curr_time = i / config.env.freq
            action = controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # Update the controller internal state and models.
            controller.step_callback(action, obs, reward, terminated, truncated, info)
            i += 1

        controller.episode_callback()  # Update the controller internal state and models.
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)
        final_gate = obs["target_gate"] if obs["target_gate"] != -1 else 4

    # Close the environment
    env.close()
    return ep_times[0], final_gate

# Create a "function", where you enter the 12x2+4x2=32 parameters and receive the time
def target_function(x: NDArray[np.floating]) -> float:
    """A functionalized wrapper of the simulation.

    Since TuRBO optimizer expects a scalar-valued function that takes one vector as the argument as the optimization target,
    wrapping of the whole simulation function was necessary.
    The cost of the simulation was set as 
    -(Number of completed gates) - (Episode time (if successful)) + (Penalty for non-completion)


    Args:
        x: 32-dimension vector representing flattened weight vectors. This value is the deviation from the baseline weights.
    
    Returns:
        Cost of the simulation.
    """
    # Sanity check for input
    assert x.ndim==1
    assert len(x)==32

    # Baseline parameter, the optimizer looks around this point
    x0=np.array([1, 1, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1,
                 1, 1, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1,
                 1e-2, 1e-2, 1e-2, 1e-2,
                 1e-2, 1e-2, 1e-2, 1e-2])

    # Separate input into arries and create episode weights
    sizes = [12, 12, 4, 4]
    indices = np.cumsum(sizes)[:-1]
    Qs, Qt, Rs, deltaR = np.split(x+x0, indices)

    # Run the simulation
    try:
        time, final_gate = do_simulation(n_runs=1, Qs=Qs, Qt=Qt, R=Rs, dR=deltaR)
        print(time, final_gate)
        if time is not None: # Episode completed
            return time -final_gate*30.0
        else: # Crashed
            return 100.0 - final_gate*30.0
    except BaseException as e: # Solver error - penalize harshly
        print(f"run failed:{e}")
        return 1000.0

def do_optimization():
    """Conducts parameter optimization.

    TuRBO optimizer is set up and executed in this function.
    Various optimization parameters including upper and lower bounds of the parameters are set here.
    """
    # create optimizer instance
    x0=np.array([1, 1, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1,
                 1, 1, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1,
                 1e-2, 1e-2, 1e-2, 1e-2,
                 1e-2, 1e-2, 1e-2, 1e-2])
    lb = x0 * -0.5
    ub = x0 * 0.5
    turbo = TurboM(
        f=target_function,  # Handle to objective function
        lb=lb,  # Numpy array specifying lower bounds
        ub=ub,  # Numpy array specifying upper bounds
        n_init=5,  # Number of initial bounds from an Latin hypercube design
        max_evals = 1000,  # Maximum number of evaluations
        batch_size=10,  # How large batch size TuRBO uses
        n_trust_regions=8, # Sets m
        verbose=True,  # Print information from each batch
        use_ard=True,  # Set to true if you want to use ARD for the GP kernel
        max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
        n_training_steps=50,  # Number of steps of ADAM to learn the hypers
        min_cuda=1024,  # Run on the CPU for small datasets
        device="cpu",  # "cpu" or "cuda"
        dtype="float64",  # float64 or float32
    )
    turbo.optimize()
    X = turbo.X  # Evaluated points
    fX = turbo.fX  # Observed values
    ind_best = np.argmin(fX)
    f_best, x_best = fX[ind_best], X[ind_best, :]
    print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))


if __name__=="__main__":
    do_optimization()
    # print(target_function(np.zeros(32)))