import logging
import numpy as np
import os
import re
import rlgym_sim
import warnings

from rewards import EventReward, FaceBallReward, InAirReward, SpeedTowardBallReward
from rlgym_ppo import Learner
from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils import common_values, RewardFunction, math
from rlgym_sim.utils.action_parsers import DiscreteAction
from rlgym_sim.utils.common_values import *
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.reward_functions import CombinedReward, RewardFunction
from rlgym_sim.utils.reward_functions.common_rewards import *
from rlgym_sim.utils.state_setters import RandomState
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition

# Suppress specific FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set up a logger
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        num_days_played = cumulative_timesteps / (120/8) /60 /60 / 24
        report = {"x_vel": avg_linvel[0],
                  "y_vel": avg_linvel[1],
                  "z_vel": avg_linvel[2],
                  "Cumulative Timesteps": cumulative_timesteps,
                  "Days Played": num_days_played,
                  "Years Played": num_days_played / 365}
        wandb_run.log(report)

#def get_most_recent_checkpoint() -> str:
#    checkpoint_load_dir = "data/checkpoints/"
#    # Get the most recent checkpoint directory based on the numeric suffix
#    recent_checkpoint_dir = max(os.listdir(checkpoint_load_dir), key=lambda d: int(d.split("-")[-1]))
#    checkpoint_load_dir += recent_checkpoint_dir + "/"
#    # Get the most recent PPO policy file directory
#    recent_policy_dir = max(os.listdir(checkpoint_load_dir), key=lambda d: int(d))
#    return os.path.join(checkpoint_load_dir, recent_policy_dir)  # Return the full path of the latest policy

def get_most_recent_checkpoint() -> str:
    checkpoint_load_dir = "data/checkpoints/"
    checkpoint_load_dir += str(
        max(os.listdir(checkpoint_load_dir), key=lambda d: int(d.split("-")[-1])))
    checkpoint_load_dir += "/"
    checkpoint_load_dir += str(
        max(os.listdir(checkpoint_load_dir), key=lambda d: int(d)))
    return checkpoint_load_dir

def build_rocketsim_env():
    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = DiscreteAction()  # Use default DiscreteAction settings
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    rewards_to_combine = [
    EventReward(touch=1),  # Reward for event actions
    FaceBallReward(),                               # Reward for facing the ball
#    InAirReward(),                                  # Reward for being in the air
    SpeedTowardBallReward(),                        # Reward for moving toward the ball
    ]

    # Adjust the weights to match the number of reward functions
    reward_weights = [
        50,       # Weight for EventReward
        1,       # Weight for FaceBallReward
#        0.01,     # Weight for InAirReward
        5,     # Weight for SpeedTowardBallReward
    ]

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL
    )

    state_setter = RandomState(ball_rand_speed=True, cars_rand_speed=True, cars_on_ground=False)

    env = rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=team_size,
        spawn_opponents=spawn_opponents,
        terminal_conditions=terminal_conditions,
        reward_fn=CombinedReward(reward_functions=rewards_to_combine, reward_weights=reward_weights),
        obs_builder=obs_builder,
        action_parser=action_parser,
        state_setter=state_setter
    )

    return env

if __name__ == "__main__":
    logger.info("Starting main execution.")
    metrics_logger = ExampleLogger()
    logger.debug("Initialized metrics logger.")

    # Get the most recent checkpoint directory
    try:
        checkpoint_load_dir = get_most_recent_checkpoint()
        print(f"Loading checkpoint: {checkpoint_load_dir}")
    except:
        print("checkpoint load dir not found.")
        checkpoint_load_dir = None

    # Number of processes and inference size setup
    n_proc = 32
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    logger.debug(f"Number of processes: {n_proc}")
    logger.debug(f"Minimum inference size: {min_inference_size}")

    # Ask user for rendering option
    render_input = input("Do you want to enable rendering? (y/n): ").strip().lower()
    render = render_input == 'y'
    logger.info(f"Rendering enabled: {render}")

    logger.debug("Initializing learner with configuration.")
    learner = Learner(
        build_rocketsim_env,
        checkpoint_load_folder=checkpoint_load_dir,  # Pass the full path to the policy file
        critic_lr=1e-4,
        exp_buffer_size=150000,
        log_to_wandb=True,
        metrics_logger=metrics_logger,
        min_inference_size=min_inference_size,
        n_proc=n_proc,
        policy_lr=1e-4,
        ppo_batch_size=50000,
        ppo_ent_coef=0.01,
        ppo_epochs=3,
        ppo_minibatch_size=50000,
        render=render,
        save_every_ts=100000,
        standardize_obs=False,
        standardize_returns=True,
        timestep_limit=20_000_000,
        ts_per_iteration=100000, 
    )
    logger.info("Learner initialized successfully.")

    # Start the learning process
    try:
        logger.info("Starting learning process.")
        learner.learn()
        logger.info("Learning process completed.")
    except Exception as e:
        logger.error("An error occurred during the learning process.", exc_info=True)
