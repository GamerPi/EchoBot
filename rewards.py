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
        print(f"Resetting {self.__class__.__name__}")
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
        print(f"Resetting {self.__class__.__name__}")
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
        print(f"Resetting {self.__class__.__name__}")
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
