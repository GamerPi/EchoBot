## Player vars from eventreward bullshit
#car_id: Unique identifier for each car/player.
#team_num: The team number (0.0 for one team, 1.0 for the other).
#match_goals, match_saves, match_shots, match_demolishes: Statistics for that car during the match.
#boost_pickups: Number of boost pads picked up.
#is_demoed: Whether the car is demolished (True/False).
#on_ground: Whether the car is on the ground.
#ball_touched: Whether the car has touched the ball in this frame.
#has_jump and has_flip: Whether the car has jump and flip abilities available.
#boost_amount: Boost level of the car (range: 0 to 1.0).
#car_data and inverted_car_data: These are PhysicsObject instances representing the car's physical state and its state in the opponent's perspective.

import logging
import math
import numpy as np
import os
import re
import rlgym_sim

from abc import ABC, abstractmethod
from bubo_misc_utils import clamp, distance, distance2D, normalize, relative_velocity_mag, sign # *
from rlgym_ppo import Learner
from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils import common_values, RewardFunction
from rlgym_sim.utils import math as rl_math
from rlgym_sim.utils.action_parsers import DiscreteAction
from rlgym_sim.utils.common_values import *
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.reward_functions import CombinedReward, RewardFunction
from rlgym_sim.utils.reward_functions.common_rewards import *
from rlgym_sim.utils.reward_functions.common_rewards.player_ball_rewards import FaceBallReward
from rlgym_sim.utils.state_setters import RandomState
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, NoTouchTimeoutCondition

SIDE_WALL_X = 4096  # +/-
BACK_WALL_Y = 5120  # +/-
CEILING_Z = 2044
BACK_NET_Y = 6000  # +/-

GOAL_HEIGHT = 642.775

ORANGE_GOAL_CENTER = (0, BACK_WALL_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_CENTER = (0, -BACK_WALL_Y, GOAL_HEIGHT / 2)

# Often more useful than center
ORANGE_GOAL_BACK = (0, BACK_NET_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_BACK = (0, -BACK_NET_Y, GOAL_HEIGHT / 2)

# ORANGE_GOAL_WAY_BACK = (0, 8000, GOAL_HEIGHT / 2)
# BLUE_GOAL_WAY_BACK = (0, -8000, GOAL_HEIGHT / 2)
# [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
BALL_RADIUS = 92.75

BALL_MAX_SPEED = 6000
CAR_MAX_SPEED = 2300
SUPERSONIC_THRESHOLD = 2200
CAR_MAX_ANG_VEL = 5.5

BLUE_TEAM = 0
ORANGE_TEAM = 1
NUM_ACTIONS = 8

BOOST_LOCATIONS = (
    (0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0),
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (-940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0),
    (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0),
    (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0),
    (-940.0, 3310.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0),
)

#sort
class AerialDistanceReward:
    def __init__(self, reward_fn, min_height_val=150, max_height_val=800, 
                 min_val_ratio=0.1, max_val_ratio=4.0):
        self.reward_fn = reward_fn
        self.min_height_ratio = min_val_ratio
        self.max_height_ratio = max_val_ratio
        self.target_height = min_height_val
        self.max_height = max_height_val
        self.min_height = min_height_val
        self.total_height = 0.0
        self.num_ticks_touched = 0
        self.curr_tick = 0

    def reset(self, initial_state):
        avg_height = self.total_height / self.num_ticks_touched if self.num_ticks_touched > 0 else self.min_height
        self.target_height = min(max(avg_height, self.min_height), self.max_height)
        self.num_ticks_touched = 0
        self.curr_tick = initial_state['tick_num']
        self.total_height = 0.0

    def get_reward(self, player, state, previous_action):
        if self.curr_tick != state['tick_num'] and player['ball_touched']:
            self.total_height += state['ball']['position']['z']
            self.num_ticks_touched += 1
            self.curr_tick = state['tick_num']

        height_diff = state['ball']['position']['z'] - self.target_height + (self.target_height / 4.0)
        height_ratio = height_diff / (self.target_height / 4.0)**1.25

        height_ratio = max(self.min_height_ratio, min(self.max_height_ratio, height_ratio))
        return self.reward_fn.get_reward(player, state, previous_action) * height_ratio

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
class AerialNavigation(RewardFunction):
    def __init__(
        self, ball_height_min=400, player_height_min=200, beginner=True
    ) -> None:
        super().__init__()
        self.ball_height_min = ball_height_min
        self.player_height_min = player_height_min
        self.face_reward = FaceBallReward()
        self.beginner = beginner

    def reset(self, initial_state: GameState) -> None:
        self.face_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if (
            not player.on_ground
            and state.ball.position[2]
            > self.ball_height_min
            > player.car_data.position[2]
            and player.car_data.linear_velocity[2] > 0
            and distance2D(player.car_data.position, state.ball.position)
            < state.ball.position[2] * 3
        ):
            # vel check
            ball_direction = normalize(state.ball.position - player.car_data.position)
            alignment = ball_direction.dot(normalize(player.car_data.linear_velocity))
            if self.beginner:
                reward += max(
                    0, alignment * 0.5
                )  # * (np.linalg.norm(player.car_data.linear_velocity)/2300)
                # old
                # #face check
                # face_reward = self.face_reward.get_reward(player, state, previous_action)
                # if face_reward > 0:
                #     reward += face_reward * 0.25
                # #boost check
                #     if previous_action[6] == 1 and player.boost_amount > 0:
                #         reward += face_reward

            reward += alignment * (
                np.linalg.norm(player.car_data.linear_velocity) / 2300.0
            )

        return max(reward, 0)

class AerialTraining(RewardFunction):
    def __init__(self, ball_height_min=400, player_min_height=300) -> None:
        super().__init__()
        self.vel_reward = VelocityPlayerToBallReward()
        self.ball_height_min = ball_height_min
        self.player_min_height = player_min_height

    def reset(self, initial_state: GameState) -> None:
        self.vel_reward.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
                not player.on_ground
                and state.ball.position[2] > self.ball_height_min
                and self.player_min_height < player.car_data.position[2] < state.ball.position[2]
        ):
            divisor = max(1, distance(player.car_data.position, state.ball.position)/1000)
            return self.vel_reward.get_reward(player, state, previous_action)/divisor

        return 0

class AirReward(RewardFunction):
    def __init__(
        self,
    ):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if not player.on_ground:
            if player.has_flip:
                return 0.5
            else:
                return 1
        return 0

class AirTouchReward:
    def __init__(self, weight=2.0, min_height=150):
        """
        :param weight: The weight of the reward.
        :param min_height: Minimum height for the ball to count as an aerial touch.
        """
        self.weight = weight
        self.min_height = min_height
    
    def reset(self, initial_state):
        pass

    def pre_step(self, state: GameState):
        # Cache the ball's current position and velocity for the next step
        self.previous_ball_position = state.ball.position.copy()
        self.previous_ball_velocity = state.ball.linear_velocity.copy()

        # Cache all player positions and velocities
        self.previous_player_states = []
        for player in state.players:
            self.previous_player_states.append({
                'position': player.car_data.position.copy(),
                'velocity': player.car_data.linear_velocity.copy(),
                'has_wheel_contact': player.on_ground,
            })

    def get_reward(self, player, state, previous_action):
        ball = state.ball
        try:
            is_air_touch = ball.position[2] > self.min_height and state.ball.last_touch == player.id
        except AttributeError:
            is_air_touch = False  # Fallback if `last_touch` isn't available
        return self.weight if is_air_touch else 0

    def get_final_reward(self, player, state, previous_action):
        """Pass all required arguments to get_reward"""
        return self.get_reward(player, state, previous_action)

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

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            return (
                state.ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)
            ) ** self.exponent
        else:
            return (
                state.inverted_ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)
            ) ** self.exponent

class BoostAcquisitions(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.boost_reserves = 1

    def reset(self, initial_state: GameState) -> None:
        self.boost_reserves = 1

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        boost_gain = player.boost_amount - self.boost_reserves
        self.boost_reserves = player.boost_amount
        return 0 if boost_gain <= 0 else boost_gain

class BoostDiscipline(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return float(-previous_action[6])

class BoostPickupReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.previous_boost = {}

    def reset(self, initial_state: GameState):
        # Initialize previous_boost as a dictionary with car_id as the key
        self.previous_boost = {player.car_id: player.boost_amount for player in initial_state.players}

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        # Calculate boost difference
        boost_diff = player.boost_amount - self.previous_boost.get(player.car_id, player.boost_amount)
        
        # Ensure boost_diff is not negative
        if boost_diff < 0:
            boost_diff = 0  # Can also set it to a minimum value if needed, e.g., 0 or another value

        # Update the previous boost value
        self.previous_boost[player.car_id] = player.boost_amount

        # Return the square root of boost_diff
        return np.sqrt(boost_diff)
    
class BoostTrainer(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return previous_action[6] == 0

class CenterReward(RewardFunction):
    def __init__(self, centered_distance=1200, punish_area_exit=False, non_participation_reward=0.0):
        self.centered_distance = centered_distance
        self.punish_area_exit = punish_area_exit
        self.non_participation_reward = non_participation_reward
        self.centered = False
        self.goal_spot = np.array([0, 5120, 0])

    def reset(self, initial_state: GameState):
        self.centered = False

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball_loc = state.ball.position
        if player.team_num != 0:
            ball_loc = state.inverted_ball.position

        coord_diff = self.goal_spot - ball_loc
        ball_dist_2d = np.linalg.norm(coord_diff[:2])
        #ball_dist_2d = math.sqrt(coord_diff[0] * coord_diff[0] + coord_diff[1] * coord_diff[1])
        reward = 0

        if self.centered:
            if ball_dist_2d > self.centered_distance:
                self.centered = False
                if self.punish_area_exit:
                    reward -= 1
        else:
            if ball_dist_2d < self.centered_distance:
                self.centered = True
                if state.last_touch == player.car_id:
                    reward += 1
                else:
                    reward += self.non_participation_reward
        return reward

class ChallengeReward(RewardFunction):
    def __init__(self, challenge_distance=300):
        super().__init__()
        self.challenge_distance = challenge_distance

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if (
            not player.on_ground
            and distance(player.car_data.position, state.ball.position)
            < self.challenge_distance
        ):
            for _player in state.players:
                if (
                    _player.team_num != player.team_num
                    and distance(_player.car_data.position, state.ball.position)
                    < self.challenge_distance
                ):
                    reward += 0.1
                    if not player.has_flip:
                        # ball_dir_norm = normalize(state.ball.position-player.car_data.position)
                        # direction = ball_dir_norm.dot(normalize(player.car_data.linear_velocity))
                        # return direction + reward
                        reward += 0.9
                    break

        return reward

class ClearReward(RewardFunction):
    def __init__(self, protected_distance=1200, punish_area_entry=False, non_participation_reward=0.0):
        self.protected_distance = protected_distance
        self.punish_area_entry=punish_area_entry
        self.non_participation_reward = non_participation_reward
        self.needs_clear = False
        self.goal_spot = np.array([0, -5120, 0])

    def reset(self, initial_state: GameState):
        self.needs_clear = False

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball_loc = state.ball.position
        if player.team_num != 0:
            ball_loc = state.inverted_ball.position

        coord_diff = self.goal_spot - ball_loc
        ball_dist_2d = np.linalg.norm(coord_diff[:2])
        #ball_dist_2d = math.sqrt(coord_diff[0]*coord_diff[0] + coord_diff[1]*coord_diff[1])
        reward = 0

        if self.needs_clear:
            if ball_dist_2d > self.protected_distance:
                self.needs_clear = False
                if state.last_touch == player.car_id:
                    reward += 1
                else:
                    reward += self.non_participation_reward
        else:
            if ball_dist_2d < self.protected_distance:
                self.needs_clear = True
                if self.punish_area_entry:
                    reward -= 1
        return reward

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

class ControllerModerator(RewardFunction):
    def __init__(self, index: int = 0, val: int = 0, reward: float = -1):
        super().__init__()
        self.index = index
        self.val = val
        self.reward = reward

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if previous_action[self.index] == self.val:
            return self.reward
        return 0

class CradleFlickReward(RewardFunction):
    def __init__(
        self,
        minimum_barrier: float = 400,
        max_vel_diff: float = 400,
        training: bool = True,
    ):
        super().__init__()
        self.min_distance = minimum_barrier
        self.max_vel_diff = max_vel_diff
        self.training = training
        self.cradle_reward = CradleReward(minimum_barrier=0)

    def reset(self, initial_state: GameState):
        self.cradle_reward.reset(initial_state)

    def stable_carry(self, player: PlayerData, state: GameState) -> bool:
        if BALL_RADIUS + 20 < state.ball.position[2] < BALL_RADIUS + 80:
            if (
                abs(
                    np.linalg.norm(
                        player.car_data.linear_velocity - state.ball.linear_velocity
                    )
                )
                <= self.max_vel_diff
            ):
                return True
        return False

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = self.cradle_reward.get_reward(player, state, previous_action) * 0.5
        if reward > 0:
            if not self.training:
                reward = 0
            stable = self.stable_carry(player, state)
            challenged = False
            for _player in state.players:
                if (
                    _player.team_num != player.team_num
                    and distance(_player.car_data.position, state.ball.position)
                    < self.min_distance
                ):
                    challenged = True
                    break
            if challenged:
                if stable:
                    if player.on_ground:
                        return reward - 0.5
                    else:
                        if player.has_flip:
                            # small reward for jumping
                            return reward + 2
                        else:
                            # print("PLAYER FLICKED!!!")
                            # big reward for flicking
                            return reward + 5
            else:
                if stable:
                    return reward + 1

        return reward

class CradleReward(RewardFunction):
    def __init__(self, minimum_barrier: float = 200):
        super().__init__()
        self.min_distance = minimum_barrier

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            player.car_data.position[2] < state.ball.position[2]
            and (BALL_RADIUS + 20 < state.ball.position[2] < BALL_RADIUS + 200)
            and distance2D(player.car_data.position, state.ball.position) <= 170
        ):
            if (
                abs(state.ball.position[0]) < 3946
                and abs(state.ball.position[1]) < 4970
            ):  # side and back wall values - 150
                if self.min_distance > 0:
                    for _player in state.players:
                        if (
                            _player.team_num != player.team_num
                            and distance(_player.car_data.position, state.ball.position)
                            < self.min_distance
                        ):
                            return 0

                return 1

        return 0

class DefenderReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.enemy_goals = 0


    def reset(self, initial_state: GameState):
        pass

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.team_num == BLUE_TEAM:
            e_score = state.orange_score
            defend_loc = BLUE_GOAL_CENTER
        else:
            e_score = state.blue_score
            defend_loc = ORANGE_GOAL_CENTER

        if e_score > self.enemy_goals:
            self.enemy_goals = e_score
            dist = distance2D(np.asarray(defend_loc), player.car_data.position)
            if dist > 900:
                reward -= clamp(1, 0, dist/10000)
        return reward

class DefenseTrainer(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            defense_objective = np.array(BLUE_GOAL_BACK)
        else:
            defense_objective = np.array(ORANGE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = defense_objective - state.ball.position
        norm_pos_diff = normalize(pos_diff)
        vel = vel/BALL_MAX_SPEED
        scale = clamp(1, 0, 1 - (distance2D(state.ball.position, defense_objective)/10000))
        return -clamp(1, 0, float(norm_pos_diff.dot(vel)*scale))

class DemoPunish(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.demo_statuses = [True for _ in range(64)]

    def reset(self, initial_state: GameState) -> None:
        self.demo_statuses = [True for _ in range(64)]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        if player.is_demoed and not self.demo_statuses[player.car_id]:
            reward = -1

        self.demo_statuses[player.car_id] = player.is_demoed
        return reward

class DistanceReward(RewardFunction):
    def __init__(self, dist_max=1000, max_reward=2):
        super().__init__()
        self.dist_max = dist_max

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        difference = state.ball.position - player.car_data.position
        distance = (
            math.sqrt(
                difference[0] * difference[0]
                + difference[1] * difference[1]
                + difference[2] * difference[2]
            )
            - 110
        )

        if distance > self.dist_max:
            return 0

        return 1 - (distance / self.dist_max)

class EventReward(RewardFunction):
    def __init__(self, goal=0., team_goal=0., concede=-0., touch=0., shot=0., save=0., demo=0., boost_pickup=0.):
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

class FlatSpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return abs(np.linalg.norm(player.car_data.linear_velocity[:2])) / 2300

class FlipCorrecter(RewardFunction):
    def __init__(self) -> None:
        self.last_velocity = np.zeros(3)
        self.armed = False

    def reset(self, initial_state: GameState) -> None:
        self.last_velocity = np.zeros(3)
        self.armed = False

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if self.armed and player.on_ground:
            self.armed = False

        elif self.armed and not player.has_flip:
            self.armed = False
            if distance(player.car_data.position, state.ball.position) <= 500:
                vel_diff = player.car_data.linear_velocity - self.last_velocity
                if np.linalg.norm(vel_diff) > 100 and previous_action[5] == 1:
                    ball_dir = normalize(state.ball.position - player.car_data.position)
                    reward = ball_dir.dot(normalize(vel_diff))
                # if distance(player.car_data.position, state.ball.position) >= 1200:
                #     rew2 = normalize(self.last_velocity).dot(normalize(vel_diff))
                #     if rew2 > reward:
                #         reward = rew2

        elif not self.armed and not player.on_ground and player.has_flip:
            self.armed = True

        self.last_velocity = player.car_data.linear_velocity
        return reward

class ForwardBiasReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return player.car_data.forward().dot(normalize(player.car_data.linear_velocity))

class GoalboxPenalty(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if abs(player.car_data.position[1]) >= 5120:
            return -1
        return 0

class GoalSpeedAndPlacementReward(RewardFunction):
    def __init__(self):
        self.prev_score_blue = 0
        self.prev_score_orange = 0
        self.prev_state_blue = None
        self.prev_state_orange = None
        self.min_height = BALL_RADIUS + 10
        self.height_reward = 1.75

    def reset(self, initial_state: GameState):
        self.prev_score_blue = initial_state.blue_score
        self.prev_score_orange = initial_state.orange_score
        self.prev_state_blue = initial_state
        self.prev_state_orange = initial_state
    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            score = state.blue_score
            # check to see if goal scored
            if score > self.prev_score_blue:
                reward = np.linalg.norm(self.prev_state_blue.ball.linear_velocity) / BALL_MAX_SPEED
                if self.prev_state_blue.ball.position[2] > self.min_height:
                    reward = self.height_reward * reward
                self.prev_state_blue = state
                self.prev_score_blue = score
                return reward
            else:
                self.prev_state_blue = state
                return 0.0
        else:
            score = state.orange_score
            # check to see if goal scored
            if score > self.prev_score_orange:
                reward = np.linalg.norm(self.prev_state_orange.ball.linear_velocity) / BALL_MAX_SPEED
                if self.prev_state_orange.ball.position[2] > self.min_height:
                    reward = self.height_reward * reward
                self.prev_state_orange = state
                self.prev_score_orange = score
                return reward
            else:
                self.prev_state_orange = state
                return 0.0
    
class GoodVelocityPlayerToBallReward:
    def __init__(self, weight=1.0, max_speed=2300):
        self.weight = weight
        self.max_speed = max_speed

    def reset(self, initial_state):
        pass

    def get_reward(self, player, ball):
        player_to_ball = ball.position - player.position
        player_to_ball_direction = player_to_ball.normalized()

        player_velocity = player.velocity
        player_speed = player_velocity.magnitude()
        velocity_direction = player_velocity.normalized() if player_speed > 0 else Vec3(0, 0, 0)

        alignment = max(0, velocity_direction.dot(player_to_ball_direction))
        scaled_reward = alignment * min(player_speed / self.max_speed, 1.0)

        return self.weight * scaled_reward

class GroundDribbleReward:
    def __init__(self, weight=1.0, max_distance=300):
        """
        :param weight: The weight of the reward.
        :param max_distance: Maximum distance between the player and ball for dribbling.
        """
        self.weight = weight
        self.max_distance = max_distance

    def reset(self, initial_state):
        pass

    def get_reward(self, player, ball):
        """
        Calculates the reward for ground dribbling.

        :param player: The player object.
        :param ball: The ball object.
        :return: Reward value.
        """
        is_grounded = ball.position.z < 120  # Check if ball is on the ground.
        player_to_ball_distance = (player.position - ball.position).magnitude()

        if is_grounded and player_to_ball_distance <= self.max_distance:
            control_factor = max(0, 1 - (player_to_ball_distance / self.max_distance))
            return self.weight * control_factor
        return 0

class GroundedReward(RewardFunction):
    def __init__(
        self,
    ):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return player.on_ground is True

class HeightTouchReward(RewardFunction):
    def __init__(self, min_height=92, exp=0.2, coop_dist=0):
        super().__init__()
        self.min_height = min_height
        self.exp = exp
        self.cooperation_dist = coop_dist

    def reset(self, initial_state: GameState):
        pass

    def cooperation_detector(self, player: PlayerData, state: GameState):
        for p in state.players:
            if p.car_id != player.car_id and \
                    distance(player.car_data.position, p.car_data.position) < self.cooperation_dist:
                return True

        return False


    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            if state.ball.position[2] >= self.min_height:
                if not player.on_ground or self.cooperation_dist < 90 or not self.cooperation_detector(player, state):
                    if player.on_ground:
                        reward += clamp(5000, 0.0001, (state.ball.position[2]-92)) ** self.exp
                    else:
                        reward += clamp(500, 1, (state.ball.position[2] ** (self.exp*2)))

            elif not player.on_ground:
                reward += 1

        return reward

class InAirReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        return 1 if not player.on_ground else 0

class JumpTouchReward(RewardFunction):
    def __init__(self, min_height=92.75):
        self.min_height = min_height
        self.max_height = 2044-92.75
        self.range = self.max_height - self.min_height

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return (state.ball.position[2] - self.min_height) / self.range

        return 0

class KickoffProximityReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[0] == 0 and state.ball.position[1] == 0:
            player_pos = np.array(player.car_data.position)
            ball_pos = np.array(state.ball.position)
            player_dist_to_ball = np.linalg.norm(player_pos - ball_pos)

            opponent_distances = []
            for p in state.players:
                if p.team_num != player.team_num:
                    opponent_pos = np.array(p.car_data.position)
                    opponent_dist_to_ball = np.linalg.norm(opponent_pos - ball_pos)
                    opponent_distances.append(opponent_dist_to_ball)

            if opponent_distances and player_dist_to_ball < min(opponent_distances):
                return 1
            else:
                return -1
        return 0

class KickoffReward(RewardFunction):
    def __init__(self, boost_punish: bool = True):
        super().__init__()
        self.vel_dir_reward = VelocityPlayerToBallReward()
        self.vel_reward = NaiveSpeedReward()
        self.boost_punish = boost_punish
        self.primed = False
        self.ticks = 0

    def reset(self, initial_state: GameState):
        self.primed = False
        self.ticks = 0
        self.vel_dir_reward.reset(initial_state)

    def closest_to_ball(self, player: PlayerData, state: GameState) -> bool:
        player_dist = distance(player.car_data.position, state.ball.position)
        for p in state.players:
            if p.team_num == player.team_num and p.car_id != p.car_id:
                dist = distance(p.car_data.position, state.ball.position)
                if dist < player_dist:
                    return False

        return True


    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if state.ball.position[0] == 0 and state.ball.position[1] == 0 and self.closest_to_ball(player, state):
            if self.ticks > 0 and self.boost_punish:

                if (
                    previous_action[6] < 1
                    and np.linalg.norm(player.car_data.linear_velocity) < 2200
                ):
                    reward -= (1 - previous_action[6]) * 0.334

                if previous_action[0] < 1:
                    reward -= (1 - previous_action[0]) * 0.334

                if previous_action[7] > 0:
                    reward -= previous_action[7] * 0.334

            reward += self.vel_reward.get_reward(player, state, previous_action)
            reward += self.vel_dir_reward.get_reward(player, state, previous_action)
        self.ticks += 1
        return reward

class LandingRecoveryReward(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self.up = np.array([0, 0, 1])

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if (
            not player.on_ground
            and player.car_data.linear_velocity[2] < 0
            and player.car_data.position[2] > 250
        ):
            flattened_vel = normalize(
                np.array(
                    [
                        player.car_data.linear_velocity[0],
                        player.car_data.linear_velocity[1],
                        0,
                    ]
                )
            )
            forward = player.car_data.forward()
            flattened_forward = normalize(np.array([forward[0], forward[1], 0]))
            reward += flattened_vel.dot(flattened_forward)
            reward += self.up.dot(player.car_data.up())
            reward /= 2

        return reward

class LavaFloorReward(RewardFunction):
    @staticmethod
    def get_reward(player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        if player.on_ground and player.car_data.position[2] < 50:
            return -1
        return 0

    @staticmethod
    def reset(initial_state: GameState):
        pass

class LemTouchBallReward(RewardFunction):
    def init(self, aerial_weight=0):
        self.aerial_weight = aerial_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            if not player.on_ground and player.car_data.position[2] >= 256:
                height_reward = np.log1p(player.car_data.position[2] - 256)
                return height_reward
        return 0

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

class MillennialKickoffReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def closest_to_ball(self, player: PlayerData, state: GameState) -> bool:
        player_dist = np.linalg.norm(player.car_data.position - state.ball.position)
        for p in state.players:
            if p.team_num == player.team_num and p.car_id != p.car_id:
                dist = np.linalg.norm(p.car_data.position - state.ball.position)
                if dist < player_dist:
                    return False

        return True

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[0] == 0 and state.ball.position[1] == 0 and self.closest_to_ball(player, state):
            return -1
        return 0

class ModifiedTouchReward(RewardFunction):
    def __init__(self, min_change: float = 300, min_height: float = 200, vel_scale: float = 1, touch_scale: float = 1, jump_reward: bool = False, jump_scale: float = 0.1, tick_min: int = 0):
        super().__init__()
        self.psr = PowerShotReward(min_change)
        self.min_height = min_height
        self.height_cap = 2044-92.75
        self.vel_scale = vel_scale
        self.touch_scale = touch_scale
        self.jump_reward = jump_reward
        self.jump_scale = jump_scale
        self.tick_count = 0
        self.tick_min = tick_min

    def reset(self, initial_state: GameState):
        self.psr.reset(initial_state)
        self.tick_count = 0

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        psr = self.psr.get_reward(player, state, previous_action)
        if player.ball_touched:
            if self.tick_count <= 0:
                self.tick_count = self.tick_min
                reward += abs(psr * self.vel_scale)

                if not player.on_ground:
                    if self.jump_reward:
                        reward += self.jump_scale
                        if not player.has_flip:
                            reward += self.jump_scale
                    if state.ball.position[2] > self.min_height:
                        reward += abs((state.ball.position[2]/self.height_cap) * self.touch_scale)
            else:
                self.tick_count -= 1
        else:
            self.tick_count -= 1

        return reward

class NaiveSpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return abs(np.linalg.norm(player.car_data.linear_velocity)) / 2300

class OmniBoostDiscipline(RewardFunction):
    def __init__(self, aerial_forgiveness=False):
        super().__init__()
        self.values = [0 for _ in range(64)]
        self.af = aerial_forgiveness

    def reset(self, initial_state: GameState):
        self.values = [0 for _ in range(64)]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        old, self.values[player.car_id] = self.values[player.car_id], player.boost_amount
        if player.on_ground or not self.af:
            return -int(self.values[player.car_id] < old)
        return 0

class OncePerStepRewardWrapper(RewardFunction):
    def __init__(self, reward):
        super().__init__()
        self.reward = reward
        self.gs = None
        self.rv = 0

    def reset(self, initial_state: GameState):
        self.reward.reset(initial_state)
        self.gs = None
        self.rv = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state == self.gs:
            return self.rv

        self.gs = state
        reward = self.reward.get_reward(player, state, previous_action)
        self.rv = reward
        return self.rv

class PlayerAlignment(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        defending = ball[1] < 0
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc
            defending = ball[1] > 0

        if defending:
            reward = rl_math.cosine_similarity(ball - pos, pos - protecc)
        else:
            reward = rl_math.cosine_similarity(ball - pos, attacc - pos)

        return reward

class PositioningReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        if player.team_num != BLUE_TEAM:
            ball = state.inverted_ball.position
            pos = player.inverted_car_data.position

        reward = 0.0
        if ball[1] < pos[1]:
            diff = ball[1] - pos[1]
            reward = -clamp(1, 0, abs(diff) / 5000)
        return reward

class PositionReward(RewardFunction):
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        optimal_position = np.array([0, 0])
        distance_to_optimal = np.linalg.norm(player.car_data.position[:2] - optimal_position)
        return max(0, self.weight - distance_to_optimal / 1000)
     
class PositiveBallVelToGoalReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.rew = VelocityBallToGoalReward()

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return clamp(1, 0, self.rew.get_reward(player, state, previous_action))

class PositivePlayerVelToBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.rew = VelocityPlayerToBallReward()

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return clamp(1, 0, self.rew.get_reward(player, state, previous_action))

class PositiveWrapperReward(RewardFunction):
    """A simple wrapper to ensure a reward only returns positive values"""
    def __init__(self, base_reward):
        super().__init__()
        #pass in instantiated reward object
        self.rew = base_reward

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        rew = self.rew.get_reward(player, state, previous_action)
        return 0 if rew < 0 else rew

class PowerShotReward(RewardFunction):
    def __init__(self, min_change: float = 300):
        super().__init__()
        self.min_change = min_change
        self.last_velocity = np.array([0, 0])

    def reset(self, initial_state: GameState):
        self.last_velocity = np.array([0, 0])

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        cur_vel = np.array(
            [state.ball.linear_velocity[0], state.ball.linear_velocity[1]]
        )
        if player.ball_touched:
            vel_change = rl_math.vecmag(self.last_velocity - cur_vel)
            if vel_change > self.min_change:
                reward = vel_change / (2300*2)

        self.last_velocity = cur_vel
        return reward

class ProximityToBallReward(RewardFunction):
    def __init__(self, max_distance=700.0, weight=10.0):
        super().__init__()
        self.max_distance = max_distance
        self.weight = weight

    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        ball_position = state.ball.position
        player_position = player.car_data.position

        distance_to_ball = np.linalg.norm(ball_position - player_position)

        if distance_to_ball < self.max_distance:
            normalized_distance = distance_to_ball / self.max_distance
            reward = (1 - normalized_distance) * self.weight
            return reward
        else:
            return 0

class PushReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        pos = state.ball.position
        if player.team_num != BLUE_TEAM:
            pos = state.inverted_ball.position

        if pos[1] > 0:
            y_scale = pos[1] / 5213
            if abs(pos[0]) > 800:
                x_scale = (abs(pos[0]) / 4096) * y_scale
                scale = y_scale - x_scale
                return scale
            return y_scale

        elif pos[1] < 0:
            y_scale = pos[1] / 5213
            if abs(pos[0]) > 800:
                x_scale = (abs(pos[0]) / 4096) * abs(y_scale)
                scale = y_scale + x_scale
                return scale
            return y_scale

        return 0

class QuickestTouchReward(RewardFunction):
    def __init__(self, timeout=0, tick_skip=8):
        self.timeout = timeout * 120 # convert to ticks
        self.tick_skip = tick_skip

    def reset(self, initial_state: GameState):
        self.timer = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.timer += self.tick_skip
                
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            reward = max(((self.timeout - self.timer) / self.timeout) * 100, 30) 
        else:
            reward = -100
        print("QuickestTouchReward.FinalReward(): ", reward)
        return reward

class RegularTouchVelChange(RewardFunction): #this is regular touchvel change without my threshold added
    def __init__(self):
        self.last_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.last_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            vel_difference = abs(np.linalg.norm(self.last_vel - state.ball.linear_velocity))
            reward = vel_difference / 4600.0

        self.last_vel = state.ball.linear_velocity

        return reward

class RetreatReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.defense_target = np.array(BLUE_GOAL_BACK)

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            ball = state.ball.position
            pos = player.car_data.position
            vel = player.car_data.linear_velocity
        else:
            ball = state.inverted_ball.position
            pos = player.inverted_car_data.position
            vel = player.inverted_car_data.linear_velocity

        reward = 0.0
        if ball[1]+200 < pos[1]:
            pos_diff = self.defense_target - pos
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / CAR_MAX_SPEED
            reward = np.dot(norm_pos_diff, norm_vel)
        return reward

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

class RewardIfTouchedLast(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return state.last_touch == player.car_id

class RuleOnePunishment(RewardFunction):
    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.on_ground and np.linalg.norm(player.car_data.linear_velocity) < 300:
            for p in state.players:
                if (
                    p.car_id != player.car_id
                    and p.on_ground
                    and distance(player.car_data.position, p.car_data.position) < 300
                    and relative_velocity_mag(
                        player.car_data.linear_velocity, p.car_data.linear_velocity
                    )
                    < 200
                ):
                    return -1

        return 0

class SaveBoostReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return player.boost_amount * 0.01

class SelectiveChaseReward(RewardFunction):
    def __init__(self, distance_req: float = 500):
        super().__init__()
        self.vel_dir_reward = VelocityPlayerToBallReward()
        self.distance_requirement = distance_req

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            distance2D(player.car_data.position, state.ball.position)
            > self.distance_requirement
        ):
            return self.vel_dir_reward.get_reward(player, state, previous_action)
        return 0

class SequentialRewards(RewardFunction):
    def __init__(self, rewards: list, steps: list):
        super().__init__()
        self.rewards_list = rewards
        self.step_counts = steps
        self.step_count = 0
        self.step_index = 0
        assert len(self.rewards_list) == len(self.step_counts)

    def reset(self, initial_state: GameState):
        for rew in self.rewards_list:
            rew.reset(initial_state)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            self.step_index < len(self.step_counts)
            and self.step_count > self.step_counts[self.step_index]
        ):
            self.step_index += 1
            print(f"Switching to Reward index {self.step_index}")

        self.step_count += 1
        return self.rewards_list[self.step_index].get_reward(
            player, state, previous_action
        )

class SpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        car_speed = np.linalg.norm(player.car_data.linear_velocity)
        car_dir = sign(player.car_data.forward().dot(player.car_data.linear_velocity))
        if car_dir < 0:
            car_speed /= -2300

        else:
            car_speed /= 2300
        return min(car_speed, 1)

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

class StarterReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.goal_reward = 10
        self.boost_weight = 0.1
        self.rew = CombinedReward(
            (
                EventReward(team_goal=self.goal_reward, concede=-self.goal_reward, demo=self.goal_reward/3),
                TouchVelChange(),
                VelocityBallToGoalReward(),
                VelocityPlayerToBallReward(),
                JumpTouchReward(min_height=120),
                KickoffReward(boost_punish=False)
            ),
            (1.0, 1.5, 0.075, 0.075, 2.0, 0.1))
            #(1.0, 1.0, 0.1, 2.0, 0.3334, 0.05))

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.rew.get_reward(player, state, previous_action)

class SwiftGroundDribbleReward:
    def __init__(self, weight=1.5, min_dribble_speed=500, max_dribble_speed=2300):
        self.weight = weight
        self.min_dribble_speed = min_dribble_speed
        self.max_dribble_speed = max_dribble_speed

    def reset(self, initial_state):
        pass

    def get_reward(self, player, ball):
        is_grounded = ball.position.z < 120
        player_to_ball_distance = (player.position - ball.position).magnitude()
        player_speed = player.velocity.magnitude()

        speed_factor = max(0, min((player_speed - self.min_dribble_speed) / 
                                  (self.max_dribble_speed - self.min_dribble_speed), 1.0))
        proximity_factor = max(0, 1 - player_to_ball_distance / 300)

        return self.weight * is_grounded * speed_factor * proximity_factor

class TeamSpacingReward(RewardFunction):
    def __init__(self, min_spacing: float = 1000) -> None:
        super().__init__()
        self.min_spacing = clamp(math.inf, 0.0000001, min_spacing)

    def reset(self, initial_state: GameState):
        pass

    def spacing_reward(self, player: PlayerData, state: GameState) -> float:
        reward = 0
        for p in state.players:
            if p.team_num == player.team_num and p.car_id != player.car_id and not player.is_demoed and not p.is_demoed:
                separation = distance(player.car_data.position, p.car_data.position)
                if separation < self.min_spacing:
                    reward -= 1-(separation / self.min_spacing)

        return reward

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return self.spacing_reward(player, state)

class ThreeManRewards(RewardFunction):
    def __init__(self, min_spacing: float = 1500):
        super().__init__()
        self.min_spacing = min_spacing
        self.vel_reward = VelocityBallToGoalReward()
        self.KOR = KickoffReward()

    def spacing_reward(self, player: PlayerData, state: GameState, role: int):
        reward = 0
        if role != 0:
            for p in state.players:
                if p.team_num == player.team_num and p.car_id != player.car_id:
                    separation = distance(player.car_data.position, p.car_data.position)
                    if separation < self.min_spacing:
                        reward -= 1 - (separation / self.min_spacing)
        return reward

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        player_distances = []
        for p in state.players:
            if p.team_num == player.team_num:
                player_distances.append(
                    (distance(p.car_data.position, state.ball.position), p.car_id)
                )

        role = 0
        player_distances.sort(key=lambda x: x[0])
        for count, pd in enumerate(player_distances):
            if pd[1] == player.car_id:
                role = count
                break

        reward = self.spacing_reward(player, state, role)
        if role == 1:
            reward += self.vel_reward.get_reward(player, state, previous_action)
            reward += self.KOR.get_reward(player, state, previous_action)

        return reward

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

class TouchBallTweakedReward(RewardFunction):
    def __init__(
        self,
        min_touch: float = 0.05,
        min_height: float = 170,
        min_distance: float = 300,
        aerial_weight: float = 0.15,
        air_reward: bool = True,
        first_touch: bool = False,
    ):
        self.min_touch = min_touch
        self.min_height = min_height
        self.aerial_weight = aerial_weight
        self.air_reward = air_reward
        self.first_touch = first_touch
        self.min_distance = min_distance
        self.min_change = 500
        self.last_velocity = np.array([0, 0, 0])

    def reset(self, initial_state: GameState):
        self.last_velocity = np.array([0, 0, 0])

    def get_closest_enemy_distance(self, player: PlayerData, state: GameState) -> float:
        closest_dist = 50000
        for car in state.players:
            if car.team_num != player.team_num:
                dist = distance2D(state.ball.position, car.car_data.position)
                if dist < closest_dist:
                    closest_dist = dist
        return closest_dist

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        current_vel = state.ball.linear_velocity
        if player.ball_touched:
            if state.ball.position[2] >= self.min_height or (
                state.ball.position[2] >= BALL_RADIUS + 20
                and (
                    self.min_distance == 0
                    or self.get_closest_enemy_distance(player, state)
                    > self.min_distance
                )
            ):
                reward += max(
                    [
                        self.min_touch,
                        (
                            abs(state.ball.position[2] - BALL_RADIUS)
                            ** self.aerial_weight
                        )
                        - 1,
                    ]
                )
                reward += np.linalg.norm(self.last_velocity - current_vel) / 2300

            if self.air_reward and not player.on_ground:
                reward += 0.5
                if not player.has_flip:
                    reward += 0.5

        self.last_velocity = current_vel
        # if abs(state.ball.position[0]) > 3896 or abs(state.ball.position[1]) > 4920:
        #     reward *= 0.75
        return reward

class TouchVelChange(RewardFunction):
    def __init__(self):
        self.last_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.last_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            vel_difference = abs(np.linalg.norm(self.last_vel - state.ball.linear_velocity))
            reward = vel_difference / 4600.0

        self.last_vel = state.ball.linear_velocity

        return reward

class TweakedVelocityPlayerToGoalReward(RewardFunction):
    def __init__(self, max_leeway=100, default_power=0.0) -> None:
        super().__init__()
        self.max_leeway = max_leeway
        self.default_power = default_power

    def reset(self, initial_state: GameState) -> None:
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball = state.ball
        player_pos = player.car_data.position
        player_goal = BLUE_GOAL_BACK
        if player.team_num == ORANGE_TEAM:
            ball = state.inverted_ball
            player_pos = player.inverted_car_data.position
            player_goal = ORANGE_GOAL_BACK

        diff = player_pos - ball.position
        if diff[1] < self.max_leeway:
            return 0

        direction = normalize(np.array(player_goal) - player_pos)
        vel = player.car_data.linear_velocity
        norm_pos_diff = direction / np.linalg.norm(direction)
        vel = vel/CAR_MAX_SPEED
        return float(np.dot(norm_pos_diff, vel))

class VelocityBallDefense(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            defense_objective = np.array(BLUE_GOAL_BACK)
        else:
            defense_objective = np.array(ORANGE_GOAL_BACK)

        vel = state.ball.linear_velocity
        pos_diff = state.ball.position - defense_objective
        norm_pos_diff = normalize(pos_diff)
        vel = vel/BALL_MAX_SPEED
        return float(norm_pos_diff.dot(vel))

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

class VersatileBallVelocityReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.offensive_reward = VelocityBallToGoalReward()
        self.defensive_reward = VelocityBallDefense()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (player.team_num == BLUE_TEAM and state.ball.position[1] < 0) or (
            player.team_num == ORANGE_TEAM and state.ball.position[1] > 0
        ):
            return self.defensive_reward.get_reward(player, state, previous_action)
        else:
            return self.offensive_reward.get_reward(player, state, previous_action)

class WallTouchReward(RewardFunction):
    def __init__(self, min_height=92, exp=0.2):
        self.min_height = min_height
        self.exp = exp
        self.max = math.inf

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and player.on_ground and state.ball.position[2] >= self.min_height:
            return (clamp(self.max, 0.0001, state.ball.position[2] - 92) ** self.exp)-1

        return 0

class ZeroSumReward(RewardFunction):
    '''
    child_reward: The underlying reward function
    team_spirit: How much to share this reward with teammates (0-1)
    opp_scale: How to scale the penalty we get for the opponents getting this reward (usually 1)
    '''
    def __init__(self, child_reward: RewardFunction, team_spirit, opp_scale = 1.0):
        self.child_reward = child_reward # type: RewardFunction
        self.team_spirit = team_spirit
        self.opp_scale = opp_scale

        self._update_next = True
        self._rewards_cache = {}

    def reset(self, initial_state: GameState):
        self.child_reward.reset(initial_state)

    def pre_step(self, state: GameState):
        self.child_reward.pre_step(state)

        # Mark the next get_reward call as being the first reward call of the step
        self._update_next = True

    def update(self, state: GameState, is_final):
        self._rewards_cache = {}

        '''
        Each player's reward is calculated using this equation:
        reward = individual_rewards * (1-team_spirit) + avg_team_reward * team_spirit - avg_opp_reward * opp_scale
        '''

        # Get the individual rewards from each player while also adding them to that team's reward list
        individual_rewards = {}
        team_reward_lists = [ [], [] ]
        for player in state.players:
            if is_final:
                reward = self.child_reward.get_final_reward(player, state, None)
            else:
                reward = self.child_reward.get_reward(player, state, None)
            individual_rewards[player.car_id] = reward
            team_reward_lists[int(player.team_num)].append(reward)

        # If a team has no players, add a single 0 to their team rewards so the average doesn't break
        for i in range(2):
            if len(team_reward_lists[i]) == 0:
                team_reward_lists[i].append(0)

        # Turn the team-sorted reward lists into averages for each time
        # Example:
        #    Before: team_rewards = [ [1, 3], [4, 8] ]
        #    After:  team_rewards = [ 2, 6 ]
        team_rewards = np.average(team_reward_lists, 1)

        # Compute and cache:
        # reward = individual_rewards * (1-team_spirit)
        #          + avg_team_reward * team_spirit
        #          - avg_opp_reward * opp_scale
        for player in state.players:
            self._rewards_cache[player.car_id] = (
                    individual_rewards[player.car_id] * (1 - self.team_spirit)
                    + team_rewards[int(player.team_num)] * self.team_spirit
                    - team_rewards[1 - int(player.team_num)] * self.opp_scale
            )

    '''
    I made get_reward and get_final_reward both call get_reward_multi, using the "is_final" argument to distinguish
    Otherwise I would have to rewrite this function for final rewards, which is lame
    '''
    def get_reward_multi(self, player: PlayerData, state: GameState, previous_action: np.ndarray, is_final) -> float:
        # If this is the first get_reward call this step, we need to update the rewards for all players
        if self._update_next:
            self.update(state, is_final)
            self._update_next = False
        return self._rewards_cache[player.car_id]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward_multi(player, state, previous_action, False)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward_multi(player, state, previous_action, True)
    
