import logging
import numpy as np
import os
import re
import rlgym_sim

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

# Set up a logger
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class AlignBallGoal(RewardFunction):
    def __init__(self, defense=1., offense=1.):
        super().__init__()
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc

        # Align player->ball and net->player vectors
        defensive_reward = self.defense * math.cosine_similarity(ball - pos, pos - protecc)

        # Align player->ball and player->net vectors
        offensive_reward = self.offense * math.cosine_similarity(ball - pos, attacc - pos)

        return defensive_reward + offensive_reward

class BallYCoordinateReward(RewardFunction):
    def __init__(self, exponent=1):
        # Exponent should be odd so that negative y -> negative reward
        self.exponent = exponent

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            return (state.ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)) ** self.exponent
        else:
            return (state.inverted_ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)) ** self.exponent

class ConditionalRewardFunction(RewardFunction):
    def __init__(self, reward_func: RewardFunction):
        super().__init__()
        self.reward_func = reward_func

    @abstractmethod
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        raise NotImplementedError

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_reward(player, state, previous_action)
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_final_reward(player, state, previous_action)
        return 0

class ConstantReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 1

class EventReward(RewardFunction):
    def __init__(self, goal=0.5, team_goal=0.5, concede=-5, touch=5.0, shot=1.0, save=1.5, demo=0.1, boost_pickup=0.05):
        """
        :param goal: reward for goal scored by player.
        :param team_goal: reward for goal scored by player's team.
        :param concede: reward for goal scored by opponents. Should be negative if used as punishment.
        :param touch: reward for touching the ball.
        :param shot: reward for shooting the ball (as detected by Rocket League).
        :param save: reward for saving the ball (as detected by Rocket League).
        :param demo: reward for demolishing a player.
        :param boost_pickup: reward for picking up boost. big pad = +1.0 boost, small pad = +0.12 boost.
        """
        super().__init__()
        self.weights = np.array([goal, team_goal, concede, touch, shot, save, demo, boost_pickup])

        # Need to keep track of last registered value to detect changes
        self.last_registered_values = {}

    @staticmethod
    def _extract_values(player: PlayerData, state: GameState):
        if player.team_num == BLUE_TEAM:
            team, opponent = state.blue_score, state.orange_score
        else:
            team, opponent = state.orange_score, state.blue_score

        return np.array([player.match_goals, team, opponent, player.ball_touched, player.match_shots,
                         player.match_saves, player.match_demolishes, player.boost_amount])

    def reset(self, initial_state: GameState, optional_data=None):
        # Update every reset since rocket league may crash and be restarted with clean values
        self.last_registered_values = {}
        for player in initial_state.players:
            self.last_registered_values[player.car_id] = self._extract_values(player, initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        old_values = self.last_registered_values[player.car_id]
        new_values = self._extract_values(player, state)

        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # We only care about increasing values

        reward = np.dot(self.weights, diff_values)

        self.last_registered_values[player.car_id] = new_values
        return reward

class FaceBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        return float(np.dot(player.car_data.forward(), norm_pos_diff))

class InAirReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        return 1 if not player.on_ground else 0

class LiuDistanceBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False):
        super().__init__()
        self.own_goal = own_goal

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        # Compensate for moving objective to back of net
        dist = np.linalg.norm(state.ball.position - objective) - (BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)
        return np.exp(-0.5 * dist / BALL_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

class LiuDistancePlayerToBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Compensate for inside of ball being unreachable (keep max reward at 1)
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        return np.exp(-0.5 * dist / CAR_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

class RewardIfBehindBall(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return player.team_num == BLUE_TEAM and player.car_data.position[1] < state.ball.position[1] \
               or player.team_num == ORANGE_TEAM and player.car_data.position[1] > state.ball.position[1]

class RewardIfClosestToBall(ConditionalRewardFunction):
    def __init__(self, reward_func: RewardFunction, team_only=True):
        super().__init__(reward_func)
        self.team_only = team_only

    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        dist = np.linalg.norm(player.car_data.position - state.ball.position)
        for player2 in state.players:
            if not self.team_only or player2.team_num == player.team_num:
                dist2 = np.linalg.norm(player2.car_data.position - state.ball.position)
                if dist2 < dist:
                    return False
        return True

#class RewardIfTouchedLast(ConditionalRewardFunction):
#    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
#        return state.last_touch == player.car_id

class SaveBoostReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # 1 reward for each frame with 100 boost, sqrt because 0->20 makes bigger difference than 80->100
        return np.sqrt(player.boost_amount)

class SpeedTowardBallReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        player_vel = player.car_data.linear_velocity
        pos_diff = (state.ball.position - player.car_data.position)
        dist_to_ball = np.linalg.norm(pos_diff)

        if dist_to_ball == 0:  # Avoid division by zero
            return 0

        dir_to_ball = pos_diff / dist_to_ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)

        if speed_toward_ball > 0:
            return speed_toward_ball / CAR_MAX_SPEED
        else:
            return 0

class TouchBallReward(RewardFunction):
    def __init__(self, aerial_weight=0.):
        self.aerial_weight = aerial_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            # Default just rewards 1, set aerial weight to reward more depending on ball height
            return ((state.ball.position[2] + BALL_RADIUS) / (2 * BALL_RADIUS)) ** self.aerial_weight
        return 0

class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = objective - state.ball.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / BALL_MAX_SPEED
            return float(np.dot(norm_pos_diff, norm_vel))

class VelocityPlayerToBallReward(RewardFunction):
    def __init__(self, use_scalar_projection=False):
        super().__init__()
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 92.75 = 24.8
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t
        else:
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / CAR_MAX_SPEED
            return float(np.dot(norm_pos_diff, norm_vel))

class VelocityReward(RewardFunction):
    # Simple reward function to ensure the model is training.
    def __init__(self, negative=False):
        super().__init__()
        self.negative = negative

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return np.linalg.norm(player.car_data.linear_velocity) / CAR_MAX_SPEED * (1 - 2 * self.negative)

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
        report = {"x_vel": avg_linvel[0],
                  "y_vel": avg_linvel[1],
                  "z_vel": avg_linvel[2],
                  "Cumulative Timesteps": cumulative_timesteps}
        wandb_run.log(report)

def build_rocketsim_env():
#    import rlgym_sim
#    from rlgym_sim.utils.reward_functions import CombinedReward
#    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
#        EventReward
#    from rlgym_sim.utils.obs_builders import DefaultObs
#    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
#    from rlgym_sim.utils import common_values
#    from rlgym_sim.utils.action_parsers import ContinuousAction

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    # Create action parser without jump action parameter
    action_parser = DiscreteAction()  # Use default DiscreteAction settings

    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    # Early stage rewards
    rewards_to_combine = [
        EventReward(goal=0.5, team_goal=0.5, concede=-5, touch=5.0, shot=1.0, save=1.5, demo=0.1, boost_pickup=0.05),  # Event-based rewards
        SpeedTowardBallReward(),                 # Speed of car toward the ball
        FaceBallReward(),                        # Align the car to face the ball
        InAirReward(),                           # Reward for being in the air
        VelocityPlayerToBallReward(),            # Speed towards the ball
        VelocityBallToGoalReward(),              # Speed of the ball toward the goal
        AlignBallGoal(defense=0.3, offense=0.7), # Alignment with ball towards goal for offense and defense
        LiuDistancePlayerToBallReward(),         # Distance of player to ball
        LiuDistanceBallToGoalReward(),           # Distance of ball to goal
        BallYCoordinateReward(exponent=1),       # Ball's y-coordinate position
        SaveBoostReward(),                       # Reward for boost saving
        ConstantReward(),                        # Constant reward (e.g., baseline reward)
        TouchBallReward(aerial_weight=0.5),      # Ball touch reward with aerial factor
        VelocityReward(negative=False),          # Car's velocity reward
        RewardIfClosestToBall(reward_func=TouchBallReward(), team_only=True), # Reward if player is closest to the ball
#        RewardIfTouchedLast(reward_func=TouchBallReward()), # Reward for the last player to touch the ball
        RewardIfBehindBall(reward_func=FaceBallReward())    # Reward for staying behind the ball based on team
    ]
    # Assign example weights based on reward importance (adjust as needed)
    reward_weights = [
        5,   # EventReward - Strong reward for touch and save events, etc.
        0.5,    # SpeedTowardBallReward - Moderate weight to encourage moving toward the ball
        1,    # FaceBallReward - Lower weight for facing the ball alignment
        -50, # InAirReward - Low weight for airborne presence
        0.25,  # VelocityPlayerToBallReward - Slight boost for velocity toward ball
        1,  # VelocityBallToGoalReward - Encourages directing ball toward goal
        2.5,  # AlignBallGoal - Medium importance for offensive/defensive alignment
        2,  # LiuDistancePlayerToBallReward - Encourages player to be near the ball
        1,  # LiuDistanceBallToGoalReward - Encourages ball proximity to goal
        0.2,  # BallYCoordinateReward - Medium weight for ball y-position
        20,  # SaveBoostReward - Reward for conserving boost
        0.001,  # ConstantReward - Baseline reward, low weight
        20,   # TouchBallReward - Significant reward for touching the ball
        0.25,  # VelocityReward - Minor boost for car speed
        2,  # RewardIfClosestToBall - Higher reward if player is closest to the ball
#        2.0,  # RewardIfTouchedLast - High reward for last touch on the ball
        2   # RewardIfBehindBall - Reward for positioning behind the ball
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
    # Start logging the main execution
    logger.info("Starting main execution.")

    # Initialize logger for metrics
    metrics_logger = ExampleLogger()
    logger.debug("Initialized metrics logger.")

    # Checkpoint directory setup
    checkpoint_dir = os.path.join("data", "checkpoints")
    logger.debug(f"Checkpoint directory set to: {checkpoint_dir}")

    if not os.path.exists(checkpoint_dir):
        logger.info(f"Checkpoint directory {checkpoint_dir} does not exist. Creating it.")
        os.makedirs(checkpoint_dir)

    # Default the policy file path to None
    policy_file_path = None

    # Check for existing checkpoints with additional logging
    checkpoint_files = os.listdir(checkpoint_dir)
    logger.debug(f"Files found in checkpoint directory: {checkpoint_files}")

    if checkpoint_files:
        try:
            # Filtering directories matching 'rlgym-ppo-run-<number>'
            numeric_checkpoints = [
                d for d in checkpoint_files if re.match(r"rlgym-ppo-run-\d+$", d)
            ]
            logger.debug(f"Numeric checkpoint directories found: {numeric_checkpoints}")

            if numeric_checkpoints:
                # Sort based on the numeric suffix
                latest_checkpoint = max(
                    numeric_checkpoints, key=lambda d: int(d.split("-")[-1])
                )
                checkpoint_load_folder = os.path.join(checkpoint_dir, latest_checkpoint)
                logger.info(f"Found latest checkpoint: {latest_checkpoint}")

                # Now, search for the PPO_POLICY.pt file in subdirectories
                for root, dirs, files in os.walk(checkpoint_load_folder):
                    if "PPO_POLICY.pt" in files:
                        policy_file_path = os.path.join(root, "PPO_POLICY.pt")
                        logger.info(f"Loaded policy file from: {policy_file_path}")
                        break

                if policy_file_path is None:
                    logger.error(f"Checkpoint file PPO_POLICY.pt does not exist in {checkpoint_load_folder}. Cannot load checkpoint.")
                    checkpoint_load_folder = None
            else:
                logger.info("No valid numeric checkpoint directories found.")
                checkpoint_load_folder = None
        except ValueError as e:
            logger.error("Error parsing checkpoint directories.", exc_info=True)
            checkpoint_load_folder = None
    else:
        checkpoint_load_folder = None
        logger.info("No checkpoints found, starting from scratch.")

    # Number of processes and inference size setup
    n_proc = 32
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    logger.debug(f"Number of processes: {n_proc}")
    logger.debug(f"Minimum inference size: {min_inference_size}")

    # Ask user for rendering option
    render_input = input("Do you want to enable rendering? (y/n): ").strip().lower()
    render = render_input == 'y'  # Set render to True if user inputs 'y'
    logger.info(f"Rendering enabled: {render}")

    # Initialize the learner with detailed configuration logging
    logger.debug("Initializing learner with configuration.")
    learner = Learner(
        build_rocketsim_env,
        checkpoint_load_folder=policy_file_path,  # Pass the correct path to the policy file
        critic_lr=2e-4,
        exp_buffer_size=300000,
        log_to_wandb=True,
        metrics_logger=metrics_logger,
        min_inference_size=min_inference_size,
        n_proc=n_proc,
        policy_lr=2e-4,
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
