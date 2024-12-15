## Player vars from eventreward bullshit
#car_id: Unique identifier for each car/player.
#team_num: The team number (0.0 for one team, 1.0 for the other).
# match_goals, match_saves, match_shots, match_demolishes:
# Statistics for that car during the match.
#boost_pickups: Number of boost pads picked up.
#is_demoed: Whether the car is demolished (True/False).
#on_ground: Whether the car is on the ground.
#ball_touched: Whether the car has touched the ball in this frame.
#has_jump and has_flip: Whether the car has jump and flip abilities available.
#boost_amount: Boost level of the car (range: 0 to 1.0).
# car_data and inverted_car_data: These are PhysicsObject instances representing
# the car's physical state and its state in the opponent's perspective.

import math
import numpy as np

from abc import abstractmethod
from bubo_misc_utils import (
    clamp, distance, distance2D, normalize,
    relative_velocity_mag, sign
)
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils import math as rl_math
from rlgym_sim.utils.common_values import (
    BLUE_TEAM, ORANGE_TEAM, BALL_MAX_SPEED, BALL_RADIUS,
    CAR_MAX_SPEED, BLUE_GOAL_BACK, ORANGE_GOAL_BACK,
    BACK_WALL_Y, BLUE_GOAL_CENTER, ORANGE_GOAL_CENTER,
    BACK_NET_Y
)
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.reward_functions import CombinedReward, RewardFunction

class Vec3:
    """
    A class to represent a 3D vector.

    Attributes:
        x (float): The x-coordinate of the vector.
        y (float): The y-coordinate of the vector.
        z (float): The z-coordinate of the vector.

    Methods:
        __sub__(self, other): Subtracts another Vec3 from the current Vec3.
        normalized(self): Returns a new normalized (unit) vector.
        magnitude(self): Returns the magnitude (length) of the vector.
        dot(self, other): Returns the dot product of the current Vec3 with another Vec3.
    """
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def normalized(self):
        magnitude = self.magnitude()
        return Vec3(
            self.x / magnitude,
            self.y / magnitude,
            self.z / magnitude
        ) if magnitude != 0 else Vec3()

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

class AerialDistanceReward:
    """
    A reward function based on the distance the ball travels in the air.

    Attributes:
        reward_fn (RewardFunction): The reward function to apply.
        min_height_ratio (float): Minimum ratio for height in the reward calculation.
        max_height_ratio (float): Maximum ratio for height in the reward calculation.
        target_height (float): The target height for the ball to reach.
        max_height (float): The maximum height allowed.
        min_height (float): The minimum height allowed.
        total_height (float): Accumulated height of the ball in aerial distance.
        num_ticks_touched (int): Number of times the ball has been touched.
        curr_tick (int): The current tick number of the game.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        get_reward(self, player, state, previous_action):
        Calculates and returns the reward.
        get_final_reward(self, player, state, previous_action):
        Returns the final reward.
    """
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
        avg_height = (
            self.total_height / self.num_ticks_touched
            if self.num_ticks_touched > 0
            else self.min_height
        )
        self.target_height = min(max(avg_height, self.min_height), self.max_height)
        self.num_ticks_touched = 0
        self.curr_tick = initial_state['tick_num']
        self.total_height = 0.0

    def get_reward(self, player, state, previous_action):
        if self.curr_tick != state['tick_num'] and player['ball_touched']:
            self.total_height += state['ball']['position']['z']
            self.num_ticks_touched += 1
            self.curr_tick = state['tick_num']

        height_diff = (
            state['ball']['position']['z'] - self.target_height +
            (self.target_height / 4.0)
        )
        height_ratio = height_diff / (self.target_height / 4.0)**1.25

        height_ratio = max(self.min_height_ratio, min(self.max_height_ratio, height_ratio))
        return self.reward_fn.get_reward(player, state, previous_action) * height_ratio

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class AerialNavigation(RewardFunction):
    """
    A reward function based on the aerial navigation of the player in the game.

    Attributes:
        ball_height_min (float): Minimum height for the ball to be
        relevant for the reward.
        player_height_min (float): Minimum height for the player to be
        relevant for the reward.
        face_reward (FaceBallReward): A reward function for the player's
        alignment with the ball.
        beginner (bool): A flag indicating whether the player is a beginner.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        get_reward(self, player, state, previous_action):
        Calculates and returns the reward.
    """
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
    """
    A reward function for training aerial movements with specific player height
    and ball height criteria.

    Attributes:
        vel_reward (VelocityPlayerToBallReward): A reward function based on the
        player's velocity toward the ball.
        ball_height_min (float): Minimum height for the ball to be considered
        for the reward.
        player_min_height (float): Minimum height for the player to be considered
        for the reward.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        get_reward(self, player, state, previous_action):
        Calculates and returns the reward.
    """
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
    """
    A reward function that rewards the player when they are in the air.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        get_reward(self, player, state, previous_action):
        Calculates and returns the reward.
    """
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
    """
    A reward function for rewarding players when they make an aerial touch with the ball.

    Attributes:
        weight (float): The weight of the reward for an aerial touch.
        min_height (float): Minimum height for the ball to count as an aerial touch.
        previous_ball_position (Vec3): Cached position of the ball from the previous tick.
        previous_ball_velocity (Vec3): Cached velocity of the ball from the previous tick.
        previous_player_states (list): Cached states of players from the previous tick.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        pre_step(self, state): Caches the ball and player data for the next step.
        get_reward(self, player, state, previous_action):
        Calculates and returns the reward for an aerial touch.
        get_final_reward(self, player, state, previous_action):
        Returns the final reward for the aerial touch.
    """
    def __init__(self, weight=2.0, min_height=150):
        """
        :param weight: The weight of the reward.
        :param min_height: Minimum height for the ball to count as an aerial touch.
        """
        self.weight = weight
        self.min_height = min_height
        self.previous_ball_position = None
        self.previous_ball_velocity = None
        self.previous_player_states = None

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
    """
    A reward function for aligning the ball with the player's goal,
    considering offensive and defensive alignments.

    Attributes:
        defense (float): Weight for the defensive alignment.
        offense (float): Weight for the offensive alignment.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        get_reward(self, player, state, previous_action):
        Calculates and returns the reward based on alignment.
    """
    def __init__(self, defense=1., offense=1.):
        super().__init__()
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
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
    """
    A reward function based on the Y-coordinate of the ball.

    Attributes:
        exponent (int): Exponent to adjust the reward calculation.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        get_reward(self, player, state, previous_action):
        Calculates and returns the reward based on the ball's Y-coordinate.
    """
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
    """
    A reward function based on the acquisition of boost by the player.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        get_reward(self, player, state, previous_action):
        Calculates and returns the reward for boost acquisition.
    """
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
    """
    A reward function that penalizes players for using boost unnecessarily.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        get_reward(self, player, state, previous_action):
        Returns the reward based on boost usage discipline.
    """
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return float(-previous_action[6])

class BoostPickupReward(RewardFunction):
    """
    A reward function for rewarding players when they pick up boost.

    Attributes:
        previous_boost (dict): Stores the previous boost amount for each player.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        get_reward(self, player, state, previous_action):
        Calculates and returns the reward for boost pickup.
    """
    def __init__(self):
        super().__init__()
        self.previous_boost = {}

    def reset(self, initial_state: GameState):
        # Initialize previous_boost as a dictionary with car_id as the key
        self.previous_boost = {
            player.car_id: player.boost_amount
            for player in initial_state.players
        }

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        # Calculate boost difference
        boost_diff = player.boost_amount - self.previous_boost.get(
            player.car_id, player.boost_amount
        )

        # Ensure boost_diff is not negative
        if boost_diff < 0:
            boost_diff = 0  # Can also set it to a minimum value if needed, e.g., 0 or another value

        # Update the previous boost value
        self.previous_boost[player.car_id] = player.boost_amount

        # Return the square root of boost_diff
        return np.sqrt(boost_diff)

class BoostTrainer(RewardFunction):
    """
    A reward function for training boost-related actions.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        get_reward(self, player, state, previous_action):
        Returns a reward based on boost training actions.
    """
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return previous_action[6] == 0

class CenterReward(RewardFunction):
    """
    A reward function based on the player's proximity to the center of the field.

    Attributes:
        centered_distance (float): The distance from the center
        where the player is considered centered.
        punish_area_exit (bool): Whether to penalize players who leave the centered area.
        non_participation_reward (float): Reward for non-participation.
        centered (bool): Whether the player is centered on the field.
        goal_spot (np.ndarray): The coordinates of the goal spot.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        get_reward(self, player, state, previous_action):
        Calculates and returns the reward for being centered on the field.
    """
    def __init__(self, centered_distance=1200, punish_area_exit=False,
                 non_participation_reward=0.0):
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
    """
    A reward function for rewarding players based on their
    proximity to the ball during a challenge.

    Attributes:
        challenge_distance (float): The distance from the
        ball at which a challenge occurs.

    Methods:
        reset(self, initial_state): Resets the reward function state.
        get_reward(self, player, state, previous_action):
        Calculates and returns the reward for participating in a challenge.
    """
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
    """
    Reward function for clearing the ball from a protected area. 
    The reward depends on the distance to the goal and whether
    the player is involved in the clearance. 
    If the ball is within the protected area, the player is expected to clear it. 
    Non-participation in the clearance yields a penalty.
    """
    def __init__(self, protected_distance=1200, punish_area_entry=False,
                 non_participation_reward=0.0):
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
    """
    A wrapper for applying a specific reward function conditionally based on custom logic. 
    The reward is only awarded if a condition is met.
    """
    def __init__(self, reward_func: RewardFunction):
        super().__init__()
        self.reward_func = reward_func

    @abstractmethod
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        raise NotImplementedError

    def reset(self, initial_state: GameState):
        print(f"Resetting {self.__class__.__name__}")
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_reward(player, state, previous_action)
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if self.condition(player, state, previous_action):
            return self.reward_func.get_final_reward(player, state, previous_action)
        return 0

class ConstantReward(RewardFunction):
    """
    A constant reward function that always returns the same fixed reward (1).
    """
    def reset(self, initial_state: GameState):
        print(f"Resetting {self.__class__.__name__}")
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return 1

class ControllerModerator(RewardFunction):
    """
    Reward function that applies a reward when a specific controller action
    (e.g., button press) is detected.
    """
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
    """
    Reward function for promoting stable carries and rewarding flicking
    or jump-based ball movements.
    It applies different rewards based on the player's actions and stability
    in carrying the ball.
    """
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
    """
    Reward function for encouraging stable cradling of the
    ball within a specific proximity to the player.
    """
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
    """
    Reward function for defensive players.
    It encourages players to stay near their goal and prevent scoring opportunities.
    """
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
    """
    Reward function that incentivizes defending actions,
    such as positioning the player between the ball and the goal.
    """
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
    """
    Reward function that penalizes players for being demoed.
    A penalty is given when a player is demoed.
    """
    def __init__(self) -> None:
        super().__init__()
        self.demo_statuses = [True for _ in range(64)]

    def reset(self, initial_state: GameState) -> None:
        self.demo_statuses = [True for _ in range(64)]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.is_demoed and not self.demo_statuses[player.car_id]:
            reward = -1

        self.demo_statuses[player.car_id] = player.is_demoed
        return reward

class DistanceReward(RewardFunction):
    """
    Reward function that provides rewards based on the player's proximity to the ball.
    The reward decreases as the distance to the ball increases.
    """
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
    """
    Reward function that provides rewards for various
    in-game events such as goals, shots, saves, demos, and boost pickups.
    """
    def __init__(self, goal=0., team_goal=0., concede=-0., touch=0.,
                 shot=0., save=0., demo=0., boost_pickup=0.):
        """
        :param goal: reward for goal scored by player.
        :param team_goal: reward for goal scored by player's team.
        :param concede: reward for goal scored by opponents.
        # Should be negative if used as punishment.
        :param touch: reward for touching the ball.
        :param shot: reward for shooting the ball (as detected by Rocket League).
        :param save: reward for saving the ball (as detected by Rocket League).
        :param demo: reward for demolishing a player.
        :param boost_pickup: reward for picking up boost. 
        # Big pad = +1.0 boost, 
        # small pad = +0.12 boost.
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

        return np.array([
            player.match_goals,
            team, opponent,
            player.ball_touched,
            player.match_shots,
            player.match_saves,
            player.match_demolishes,
            player.boost_amount
        ])

    def reset(self, initial_state: GameState, optional_data=None):
        # Update every reset since rocket league may crash and be restarted with clean values
        self.last_registered_values = {}
        for player in initial_state.players:
            self.last_registered_values[player.car_id] = self._extract_values(player, initial_state)

    def get_reward(
        self,
        player: PlayerData,
        state: GameState,
        previous_action: np.ndarray,
        optional_data=None
    ):
        old_values = self.last_registered_values[player.car_id]
        new_values = self._extract_values(player, state)

        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # We only care about increasing values

        reward = np.dot(self.weights, diff_values)

        self.last_registered_values[player.car_id] = new_values
        return reward

class FaceBallReward(RewardFunction):
    """
    Reward function that encourages the player to face the ball,
    with rewards based on the alignment between the player's car and the ball.
    """
    def reset(self, initial_state: GameState):
        print(f"Resetting {self.__class__.__name__}")
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        pos_diff = state.ball.position - player.car_data.position
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        return float(np.dot(player.car_data.forward(), norm_pos_diff))

class FlatSpeedReward(RewardFunction):
    """
    Reward function that provides a reward based on the player's flat speed,
    normalized to a maximum speed of 2300.
    """
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return abs(np.linalg.norm(player.car_data.linear_velocity[:2])) / 2300

class FlipCorrecter(RewardFunction):
    """
    Reward function that encourages correct flip mechanics,
    rewarding players for proper flip timing and direction.
    """
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
    """
    Reward function that biases the player's forward movement by
    rewarding the player for moving forward.
    """
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return player.car_data.forward().dot(normalize(player.car_data.linear_velocity))

class GoalboxPenalty(RewardFunction):
    """
    Penalizes players who enter the goalbox,
    discouraging undesirable positions near the goal area.
    """
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if abs(player.car_data.position[1]) >= 5120:
            return -1
        return 0

class GoalSpeedAndPlacementReward(RewardFunction):
    """
    Rewards players based on scoring, ball velocity, and height,
    with different reward scales for each condition.
    """
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

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
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
            self.prev_state_blue = state
            return 0.0

        score = state.orange_score
        # check to see if goal scored
        if score > self.prev_score_orange:
            reward = np.linalg.norm(self.prev_state_orange.ball.linear_velocity) / BALL_MAX_SPEED
            if self.prev_state_orange.ball.position[2] > self.min_height:
                reward = self.height_reward * reward
            self.prev_state_orange = state
            self.prev_score_orange = score
            return reward
        self.prev_state_orange = state
        return 0.0

class GoodVelocityPlayerToBallReward:
    """
    Encourages players to approach the ball with
    Appropriate speed and alignment, rewarding good positioning.
    """
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
    """
    Rewards players for maintaining control of the ball while
    dribbling on the ground within a specified distance.
    """
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
    """
    Rewards players for being grounded, providing a reward when
    the player is not in the air.
    """
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
    """
    Rewards players for touching the ball at a height greater than a
    specified minimum, with additional rewards for ground behavior and cooperation.
    """
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
                if (
                    not player.on_ground
                    or self.cooperation_dist < 90
                    or not self.cooperation_detector(player, state)
                ):
                    if player.on_ground:
                        reward += clamp(5000, 0.0001, (state.ball.position[2] - 92)) ** self.exp
                    else:
                        reward += clamp(500, 1, (state.ball.position[2] ** (self.exp * 2)))
            elif not player.on_ground:
                reward += 1

        return reward

class InAirReward(RewardFunction):
    """
    Rewards players for being in the air, providing a reward when the player
    is not on the ground.
    """
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        return 1 if not player.on_ground else 0

class JumpTouchReward(RewardFunction):
    """
    Rewards players for touching the ball while airborne and at a
    specific height range, with scaled rewards based on the height.
    """
    def __init__(self, min_height=92.75):
        self.min_height = min_height
        self.max_height = 2044 - 92.75
        self.range = self.max_height - self.min_height

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            player.ball_touched
            and not player.on_ground
            and state.ball.position[2] >= self.min_height
        ):
            return (state.ball.position[2] - self.min_height) / self.range

        return 0

class KickoffProximityReward(RewardFunction):
    """
    Rewards players for being the closest to the ball during a kickoff,
    penalizing players who are further from the ball.
    """
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
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
            return -1
        return 0

class KickoffReward(RewardFunction):
    """
    Rewards players based on their positioning during kickoffs,
    penalizing poor positioning or improper use of boost.
    """
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
        ball_position = state.ball.position
        if (
            ball_position[0] == 0
            and ball_position[1] == 0
            and self.closest_to_ball(player, state)
        ):
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
    """
    Rewards players for recovering from a mid-air position and landing in
    alignment with their forward movement direction.
    """
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
    """
    Penalizes players who are on the ground below a certain height,
    discouraging movement near the "lava" area.
    """
    def reset(self, initial_state: GameState):
        pass

    # @staticmethod
    def get_reward(player: PlayerData, state: GameState, previous_action: np.ndarray):
        if player.on_ground and player.car_data.position[2] < 50:
            return -1
        return 0

class LemTouchBallReward(RewardFunction):
    """
    Rewards players for touching the ball at higher altitudes,
    with more reward for higher jumps.
    """
    def init(self):
        self.aerial_weight = 0

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched:
            if not player.on_ground and player.car_data.position[2] >= 256:
                height_reward = np.log1p(player.car_data.position[2] - 256)
                return height_reward
        return 0

class LiuDistanceBallToGoalReward(RewardFunction):
    """
    Rewards players based on their distance to the goal,
    with a higher reward for positioning the ball closer to the opponent's goal.
    """
    def __init__(self, own_goal=False):
        super().__init__()
        self.own_goal = own_goal

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM and not self.own_goal \
                or player.team_num == ORANGE_TEAM and self.own_goal:
            objective = np.array(ORANGE_GOAL_BACK)
        else:
            objective = np.array(BLUE_GOAL_BACK)

        # Compensate for moving objective to back of net
        dist = np.linalg.norm(state.ball.position - objective) - \
               (BACK_NET_Y - BACK_WALL_Y + BALL_RADIUS)
        return np.exp(-0.5 * dist / BALL_MAX_SPEED)
    # Inspired by https://arxiv.org/abs/2105.12196

class LiuDistancePlayerToBallReward(RewardFunction):
    """
    Rewards players based on their proximity to the ball,
    with a higher reward for being closer to the ball.
    """
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # Compensate for inside of ball being unreachable (keep max reward at 1)
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        return np.exp(-0.5 * dist / CAR_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

class MillennialKickoffReward(RewardFunction):
    """
    Rewards players for being the closest to the ball during kickoffs,
    penalizing players who are further away.
    """
    def __init__(self):
        pass  # super().__init__()

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

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            state.ball.position[0] == 0
            and state.ball.position[1] == 0
            and self.closest_to_ball(player, state)
        ):
            return -1
        return 0

class ModifiedTouchReward(RewardFunction):
    """
    A reward function that modifies rewards based on power shot rewards, height,
    and player jump behavior.
    It also includes a timer mechanism to ensure rewards are distributed over time.
    """
    def __init__(self):
        super().__init__()
        self.psr = PowerShotReward(min_change=300)
        self.min_height = 200
        self.height_cap = 2044 - 92.75
        self.vel_scale = 1
        self.touch_scale = 1
        self.jump_reward = False
        self.jump_scale = 0.1
        self.tick_count = 0
        self.tick_min = 0

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
    """
    A simple reward function that provides rewards based on the player's speed,
    normalized by a constant.
    """
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return abs(np.linalg.norm(player.car_data.linear_velocity)) / 2300

class OmniBoostDiscipline(RewardFunction):
    """
    Penalizes players for excessive boost usage, with an optional
    aerial forgiveness feature.
    """
    def __init__(self, aerial_forgiveness=False):
        super().__init__()
        self.values = [0 for _ in range(64)]
        self.af = aerial_forgiveness

    def reset(self, initial_state: GameState):
        self.values = [0 for _ in range(64)]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        old, self.values[player.car_id] = self.values[player.car_id], player.boost_amount
        if player.on_ground or not self.af:
            return -int(self.values[player.car_id] < old)
        return 0

class OncePerStepRewardWrapper(RewardFunction):
    """
    A wrapper to ensure the wrapped reward function only provides a reward once per step.
    """
    def __init__(self, reward):
        super().__init__()
        self.reward = reward
        self.gs = None
        self.rv = 0

    def reset(self, initial_state: GameState):
        self.reward.reset(initial_state)
        self.gs = None
        self.rv = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if state == self.gs:
            return self.rv

        self.gs = state
        reward = self.reward.get_reward(player, state, previous_action)
        self.rv = reward
        return self.rv

class PlayerAlignment(RewardFunction):
    """
    Rewards players for maintaining proper alignment with the ball
    and their respective goal direction.
    """
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
    """
    Rewards players for positioning themselves closer to the ball,
    penalizing those who are too far.
    """
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
    """
    Rewards players for positioning themselves near an optimal point on the field.
    """
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        optimal_position = np.array([0, 0])
        distance_to_optimal = np.linalg.norm(player.car_data.position[:2] - optimal_position)
        return max(0, self.weight - distance_to_optimal / 1000)

class PositiveBallVelToGoalReward(RewardFunction):
    """
    Rewards players based on the velocity of the ball toward the goal,
    encouraging goal-scoring positions.
    """
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
    """
    Rewards players based on their velocity toward the ball,
    promoting quicker ball possession.
    """
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
    """
    A wrapper that ensures the wrapped reward function always returns a positive value.
    """
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
    """
    Rewards players for powerful shots that significantly change the ball's
    velocity, based on a minimum threshold.
    """
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
    """
    Rewards players for being closer to the ball, with a maximum distance and
    a scaling reward.
    """
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
        return 0

class PushReward(RewardFunction):
    """
    Rewards players for pushing the ball in the right direction,
    based on its position relative to the field.
    """
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

        if pos[1] < 0:
            y_scale = pos[1] / 5213
            if abs(pos[0]) > 800:
                x_scale = (abs(pos[0]) / 4096) * abs(y_scale)
                scale = y_scale + x_scale
                return scale
            return y_scale

        return 0

class QuickestTouchReward(RewardFunction):
    """
    Rewards players for making the quickest touch on the ball after a timeout period,
    with a final reward based on the time taken.
    """
    def __init__(self, timeout=0, tick_skip=8):
        self.timeout = timeout * 120 # convert to ticks
        self.tick_skip = tick_skip
        self.timer = 0

    def reset(self, initial_state: GameState):
        self.timer = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        self.timer += self.tick_skip
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched:
            reward = max(((self.timeout - self.timer) / self.timeout) * 100, 30)
        else:
            reward = -100
        print("QuickestTouchReward.FinalReward(): ", reward)
        return reward

class RegularTouchVelChange(RewardFunction):
    """
    A reward function that calculates the difference in the ball's
    velocity before and after a player touches the ball.
    The reward is based on the magnitude of this change,
    normalized by a constant.
    """
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
    """
    A reward function for players on defense, encouraging them to move toward a
    defensive target position when the ball is behind them.
    The reward depends on the player's speed and position
    relative to the defensive target.
    """
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
    """
    A conditional reward function that rewards players for positioning themselves
    behind the ball, based on their team.
    """
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        is_blue_team_behind = (
            player.team_num == BLUE_TEAM
            and player.car_data.position[1] < state.ball.position[1]
        )
        is_orange_team_behind = (
            player.team_num == ORANGE_TEAM
            and player.car_data.position[1] > state.ball.position[1]
        )
        return is_blue_team_behind or is_orange_team_behind

class RewardIfClosestToBall(ConditionalRewardFunction):
    """
    A conditional reward function that rewards a player if they are the closest player
    to the ball, with an option to reward only team members.
    """
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
    """
    A conditional reward function that rewards a player if they were the last to touch
    the ball.
    """
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return state.last_touch == player.car_id

class RuleOnePunishment(RewardFunction):
    """
    A penalty for players who are on the ground and moving too slowly, especially when
    in close proximity to another player with similar velocity.
    """
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
    """
    A reward function that rewards players based on their boost amount,
    scaled by a constant.
    """
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return player.boost_amount * 0.01

class SelectiveChaseReward(RewardFunction):
    """
    A reward function that applies a velocity-based reward for players if they are not
    within a specified distance from the ball.
    """
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
    """
    A reward function that sequentially applies a list of reward functions over time,
    switching based on step counts.
    """
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
    """
    A reward function that calculates the player's speed, adjusting based on the direction
    of their movement relative to their car's forward direction.
    """
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
    """
    A reward function that rewards players for moving toward the ball,
    based on their speed and distance to the ball.
    """
    def __init__(self):
        pass # super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        player_vel = player.car_data.linear_velocity
        pos_diff = (state.ball.position - player.car_data.position)
        dist_to_ball = np.linalg.norm(pos_diff)

        if dist_to_ball == 0:  # Avoid division by zero
            return 0

        dir_to_ball = pos_diff / dist_to_ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)

        if speed_toward_ball > 0:
            return speed_toward_ball / CAR_MAX_SPEED
        return 0

class StarterReward(RewardFunction):
    """
    A composite reward function for the starting game phase,
    incorporating various reward functions such as for goals, touch velocity,
    and kickoff actions.
    """
    def __init__(self):
        super().__init__()
        self.goal_reward = 10
        self.boost_weight = 0.1
        self.rew = CombinedReward(
            (
                EventReward(
                    team_goal=self.goal_reward,
                    concede=-self.goal_reward,
                    demo=self.goal_reward / 3
                ),
                TouchVelChange(),
                VelocityBallToGoalReward(),
                VelocityPlayerToBallReward(),
                JumpTouchReward(min_height=120),
                KickoffReward(boost_punish=False)
            ),
            (
                1.0, 1.5, 0.075, 0.075, 2.0, 0.1
            )
        )

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return self.rew.get_reward(player, state, previous_action)

class SwiftGroundDribbleReward:
    """
    A reward function that encourages fast ground dribbling,
    rewarding players based on their speed and proximity to the ball while grounded.
    """
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
    """
    A reward function that penalizes players for being too close to their teammates,
    encouraging better team spacing.
    """
    def __init__(self, min_spacing: float = 1000) -> None:
        super().__init__()
        self.min_spacing = clamp(math.inf, 0.0000001, min_spacing)

    def reset(self, initial_state: GameState):
        pass

    def spacing_reward(self, player: PlayerData, state: GameState) -> float:
        reward = 0
        for p in state.players:
            if (
                hasattr(p, 'team_num')
                and hasattr(p, 'car_id')
                and hasattr(p, 'is_demoed')
                and hasattr(p, 'car_data')
            ):
                if (
                    p.team_num == player.team_num
                    and p.car_id != player.car_id
                    and not player.is_demoed
                    and not p.is_demoed
                ):
                    separation = distance(player.car_data.position, p.car_data.position)
                    if separation < self.min_spacing:
                        separation_factor = separation / self.min_spacing
                        reward -= 1 - separation_factor
        return reward

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return self.spacing_reward(player, state)

class ThreeManRewards(RewardFunction):
    """
    A reward function that rewards players for maintaining good
    spacing with their teammates and applies additional rewards for specific roles
    during gameplay.
    """
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
    """
    A reward function that rewards players for touching the ball,
    with adjustments based on the ball's height (aerial rewards).
    """
    def __init__(self, aerial_weight=0.):
        self.aerial_weight = aerial_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched:
            # Default just rewards 1, set aerial weight to reward more depending on ball height
            height_factor = (state.ball.position[2] + BALL_RADIUS) / (2 * BALL_RADIUS)
            return height_factor ** self.aerial_weight
        return 0

class TouchBallTweakedReward(RewardFunction):
    """
    A modified reward function that rewards players for touching the ball,
    considering factors like height, proximity to enemies, and velocity change.
    """
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
                closest_dist = min(closest_dist, dist)
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
    """
    A reward function that calculates the change in the ball's
    velocity before and after a player touches the ball.
    The reward is based on the magnitude of this change, normalized by a constant.
    """
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
    """
    A reward function that rewards a player based on their velocity towards the opponent's goal,
    considering a leeway range and adjusting for the player's position and team.
    """
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
    """
    A reward function that rewards a player for maintaining a
    defensive position relative to the ball,
    based on the player's proximity to the opposing goal and the ball's velocity.
    """
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
    """
    A reward function that calculates the velocity of the ball towards the opponent's goal,
    with optional adjustments for scalar projection or specific goal orientation.
    """
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
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

        # Regular component velocity
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / BALL_MAX_SPEED
        return float(np.dot(norm_pos_diff, norm_vel))

class VelocityPlayerToBallReward(RewardFunction):
    """
    A reward function that rewards a player based on their velocity towards the ball,
    using either a regular dot product or a scalar projection to determine the reward.
    """
    def __init__(self, use_scalar_projection=False):
        super().__init__()
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        vel = player.car_data.linear_velocity
        pos_diff = state.ball.position - player.car_data.position
        if self.use_scalar_projection:
            # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
            # Max value should be max_speed / ball_radius = 2300 / 92.75 = 24.8
            # Used to guide the agent towards the ball
            inv_t = math.scalar_projection(vel, pos_diff)
            return inv_t

        # Regular component velocity
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / CAR_MAX_SPEED
        return float(np.dot(norm_pos_diff, norm_vel))

class VelocityReward(RewardFunction):
    """
    A simple reward function based on the player's velocity,
    either rewarding or penalizing based on the velocity.
    """
    def __init__(self, negative=False):
        super().__init__()
        self.negative = negative

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        velocity = np.linalg.norm(player.car_data.linear_velocity)
        return velocity / CAR_MAX_SPEED * (1 - 2 * self.negative)

class VersatileBallVelocityReward(RewardFunction):
    """
    A composite reward function that adapts based on whether the player is on offense or defense,
    utilizing offensive and defensive velocity rewards for the ball.
    """
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

        return self.offensive_reward.get_reward(player, state, previous_action)

class WallTouchReward(RewardFunction):
    """
    A reward function that rewards players for touching the ball while it is above a certain height,
    typically when it makes contact with the wall during a game.
    """
    def __init__(self, min_height=92, exp=0.2):
        self.min_height = min_height
        self.exp = exp
        self.max = float('inf')

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and player.on_ground and state.ball.position[2] >= self.min_height:
            return (clamp(self.max, 0.0001, state.ball.position[2] - 92) ** self.exp)-1

        return 0

class ZeroSumReward(RewardFunction):
    """
    A reward function that calculates rewards for each player using a zero-sum approach,
    considering team rewards, individual rewards, and penalties based on opponent performance.
    """
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
        reward = individual_rewards * (1-team_spirit) + avg_team_reward
        * team_spirit - avg_opp_reward * opp_scale
        '''

        # Get the individual rewards from each player while also adding
        # them to that team's reward list
        individual_rewards = {}
        team_reward_lists = [ [], [] ]
        for player in state.players:
            if is_final:
                reward = self.child_reward.get_final_reward(player, state, None)
            else:
                reward = self.child_reward.get_reward(player, state, None)
            individual_rewards[player.car_id] = reward
            team_reward_lists[int(player.team_num)].append(reward)

        # If a team has no players, add a single 0 to their team rewards
        # so the average doesn't break
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
    def get_reward_multi(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray, is_final
    ) -> float:
        # If this is the first get_reward call this step,
        # we need to update the rewards for all players
        if self._update_next:
            self.update(state, is_final)
            self._update_next = False
        return self._rewards_cache[player.car_id]

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return self.get_reward_multi(player, state, previous_action, False)

    def get_final_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return self.get_reward_multi(player, state, previous_action, True)
