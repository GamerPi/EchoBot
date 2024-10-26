##helpful information for later use
##class game_object:
    # This object holds information about the current match
##    def __init__(self):
##        self.time = 0
##        self.time_remaining = 0
##        self.overtime = False
##        self.round_active = False
##        self.kickoff = False
##        self.match_ended = False
##        self.friend_score = 0
##        self.foe_score = 0
##        self.gravity = Vector()
##
##    def update(self, team, packet):
##        game = packet.game_info
##        self.time = game.seconds_elapsed
##        self.time_remaining = game.game_time_remaining
##        self.overtime = game.is_overtime
##        self.round_active = game.is_round_active
##        self.kickoff = game.is_kickoff_pause
##        self.match_ended = game.is_match_ended
##        self.friend_score = packet.teams[team].score
##        self.foe_score = packet.teams[not team].score
##        self.gravity.z = game.world_gravity_z



import logging
import math
import numpy as np
import os
import psutil
import rlgym
import signal
import time

from abc import ABC, abstractmethod ##test abstractmethod for state system
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.logging_utils import get_logger
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlbot.utils.structures.quick_chats import QuickChats ##correct quickchat location
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward
from rlbot.utils.structures.ball_prediction_struct import BallPrediction
from rlbot.utils.structures.game_data_struct import GameTickPacket, PlayerInfo
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from stable_baselines3 import PPO
from typing import Optional
from util.ball_prediction_analysis import predict_future_goal  # Importing the prediction function
from util.orientation import Orientation, relative_location  # Importing the necessary orientation functions
from util.sequence import Sequence, ControlStep  # Importing the Sequence and ControlStep classes
from util.spikes import SpikeWatcher  # Importing SpikeWatcher from spikes.py
from util.vec import Vec3

# Set up logging
logger = get_logger("EchoBotLogs") ##RLBot implementation of logging

# Abstract base class for states
class BaseState(ABC):
    """Base class that all states should inherit from."""

    @abstractmethod
    def is_viable(self, agent, packet):
        """Determines whether the state can run at the moment."""
        pass

    @abstractmethod
    def get_output(self, agent, packet):
        """Gets called every frame by the StateHandler until it returns None."""
        pass

# Define specific states
class GetBoost(BaseState):
    def is_viable(self, agent, packet):
        return packet.game_cars[agent.index].boost < 80

    def get_output(self, agent, packet):
        car = packet.game_cars[agent.index]
        boost_pads = packet.game_boost_pads

        # Check for active boost pads
        if not boost_pads:
            return None # No boost pads to target 

        # Find the closest boost pad
        closest_boost = min(
            (pad for pad in boost_pads if pad.is_active),
            key=lambda pad: Vec3(car.physics.location).dist(Vec3(pad.location)),
            default=None
        )

        if closest_boost:
            return agent.go_to_location(Vec3(closest_boost.location))
        return None  # Still searching for boost


class SaveNet(BaseState):
    def is_viable(self, agent, packet):
        return agent.is_ball_heading_to_own_goal(packet)

    def get_output(self, agent, packet):
        return agent.defend_goal(packet)


class TakeShot(BaseState):
    def is_viable(self, agent, packet):
        return agent.can_take_shot(packet)

    def get_output(self, agent, packet):
        return agent.take_shot(packet)


class ChaseBall(BaseState):
    def is_viable(self, agent, packet):
        return True  # Always viable if no other state is viable

    def get_output(self, agent, packet):
        return agent.chase_ball(packet)
    
class StateHandler:
    def __init__(self, agent):
        self.agent = agent
        self.current_state = None
        self.prev_frame_score = (0, 0)

    def select_state(self, packet):
        """Chooses the first viable state (determined from is_viable)."""
        states = [
            SaveNet(),
            TakeShot(),
            GetBoost(),
            ChaseBall()
        ]

        for state in states:
            if state.is_viable(self.agent, packet):
                self.current_state = state  # Set the current state
                return state
        
        return ChaseBall()  # Default state if none are viable

    @staticmethod
    def get_goal_score(packet):
        """Returns a tuple of (blue team goals, orange team goals)."""
        return packet.teams[0].score, packet.teams[1].score

    def get_output(self, packet):
        """Returns the output from the current state and selects a new state if needed."""
        current_frame_score = self.get_goal_score(packet)

        # Reset state if a goal is scored
        if current_frame_score != self.prev_frame_score:
            self.current_state = None
            self.prev_frame_score = current_frame_score

        # Select a new state if the current one is None
        if self.current_state is None:
            self.current_state = self.select_state(packet)

        state_output = self.current_state.get_output(self.agent, packet)

        # Return the controller if the state is still running
        if state_output is not None:
            return state_output

        # Reset and recurse if the state finished
        self.current_state = None
        return self.get_output(packet)


def limit_to_safe_range(value: float) -> float:
    """Ensure values are within the safe range of -1 to 1."""
    return max(-1, min(1, value))

def steer_toward_target(car: PlayerInfo, target: Vec3) -> float:
    """Calculate the steering direction towards a target."""
    relative = relative_location(Vec3(car.physics.location), Orientation(car.physics.rotation), target)
    angle = math.atan2(relative.y, relative.x)
    return limit_to_safe_range(angle * 5)

# Create RLGym environment and setup PPO model
def create_rlgym_env(reward_fn=VelocityBallToGoalReward(), terminal_conditions=None):
    if terminal_conditions is None:
        terminal_conditions = [TimeoutCondition(300), GoalScoredCondition()]

    return rlgym.make(
        obs_builder=AdvancedObs(),  # Advanced observation model for PPO
        reward_fn=reward_fn,
        terminal_conditions=terminal_conditions
    )

# Train or load PPO model
def train_or_load_model(num_iterations):
    env = create_rlgym_env()  # Create the environment
    
    try:
        model = PPO.load("trained_model", env=env)  # Provide the environment when loading
        logger.info("Loaded pre-trained model.")
    except FileNotFoundError:
        logger.warning("No pre-trained model found, training from scratch...")
        model = PPO("MlpPolicy", env, verbose=1)

    # Training loop
    for i in range(num_iterations):
        logger.info(f"Starting iteration {i + 1} of {num_iterations}...")
        model.learn(total_timesteps=10000)  # Adjust timesteps per iteration as needed

    model.save("trained_model")  # Save the model after training
    logger.info("Model saved as 'trained_model'.")

    return model

# Map PPO action to Rocket League controls
def map_action_to_controls(action):
    controller_state = SimpleControllerState()

    # Action mapping logic
    controller_state.throttle = float(action[0])  # Convert to float
    controller_state.steer = np.clip(float(action[1]), -1.0, 1.0)  # Ensure steer is within bounds
    controller_state.boost = bool(action[2])
    controller_state.jump = bool(action[3])

    return controller_state

class EchoBot(BaseAgent):

    Supersonic_Speed = 2200 #Max speed defined

    def __init__(self, name, team, index, model: PPO):
        super().__init__(name, team, index)
        self.controller = SimpleControllerState()
        self.model = model  # PPO model for controlling bot actions
        self.initialized = False
        self.sequence = None  # Sequence to manage complex maneuvers
        self.spike_watcher = SpikeWatcher()  # Initialize SpikeWatcher
        self.state_handler = StateHandler(self)  # Initialize state handler
        self.goal_location = Vec3(0, -5100, 0) if self.team == 0 else Vec3(0, 5100, 0)


    def send_quick_chat(self, quick_chat_index):
        team_only = False  # Change to True for team-only messages
        quick_chat = self.quick_chats[quick_chat_index]
        self.send_quick_chat(self.game_interface, self.index, self.team, team_only, quick_chat)

    def initialize_agent(self):
        # Called once when the bot is spawned
        self.initialized = True
        self.sequence = None
        self.spike_watcher = SpikeWatcher()  # Initialize SpikeWatcher

    def get_observation(self, packet: GameTickPacket):
        # Extract useful features from the packet to feed to the PPO model
        car = packet.game_cars[self.index]
        ball = packet.game_ball

        # Example observation: car location, velocity, ball location
        car_location = Vec3(car.physics.location)
        car_velocity = Vec3(car.physics.velocity)
        ball_location = Vec3(ball.physics.location)

        # Build observation as a NumPy array
        observation = np.array([
            car_location.x, car_location.y, car_location.z,
            car_velocity.x, car_velocity.y, car_velocity.z,
            ball_location.x, ball_location.y, ball_location.z,
            np.sqrt((car_location.x - ball_location.x) ** 2 + (car_location.y - ball_location.y) ** 2)  # Distance to ball
        ])
        
        return observation

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        if not self.initialized:
            self.initialize_agent()

        # Check the game mode to see if it is Rumble
        is_rumble = packet.game_info.is_rumble

        # Update SpikeWatcher if in Rumble mode
        if is_rumble:
            self.spike_watcher.read_packet(packet)

        # Check if we are in a sequence and execute it if so
        if self.sequence:
            controls = self.sequence.tick(packet)
            if controls is not None:
                return controls

        # Get observation from the game
        observation = self.get_observation(packet)
        self.update_state(packet)  # Update state based on current conditions

        # Use PPO model to predict actions
        action, _ = self.model.predict(observation, deterministic=True)

        # Map action predictions to controller output (throttle, steer, etc.)
        self.controller = map_action_to_controls(action)

        try:
            # Obtain ball prediction and analyze it
            ball_prediction = self.get_ball_prediction(packet)  # Line of interest
            
            future_goal_slice = predict_future_goal(ball_prediction)
            
            # If the ball is predicted to enter the goal, adjust the strategy
            if future_goal_slice is not None:
                logger.info("The ball is predicted to enter the goal! Adjusting strategy.")
                self.controller.boost = True  # Example adjustment; you can refine this logic further
                
                # Send a quick chat about the prediction
                self.send_quick_chat(self.quick_chats.Reactions_Noooo)  # Replace with your desired quick chat option
        except AttributeError as e:
            logger.warning(f"Ball prediction failed: {e}")
            ball_prediction = None  # You can also handle this case if needed

        # Obtain ball prediction and analyze it
#        ball_prediction = self.get_ball_prediction(packet)
#        future_goal_slice = predict_future_goal(ball_prediction)
        
        # If the ball is predicted to enter the goal, adjust the strategy
        if future_goal_slice is not None:
            logger.info("The ball is predicted to enter the goal! Adjusting strategy.")
            self.controller.boost = True  # Example adjustment; you can refine this logic further
            
            # Send a quick chat about the prediction
            self.send_quick_chat(self.quick_chats.Reactions_Noooo)  # Replace with your desired quick chat option

        # Get the car's current state
        car = packet.game_cars[self.index]

        # Calculate the steering direction towards the ball or goal based on the predicted trajectory
        target_location = ball_prediction.slices[0].physics.location if ball_prediction.num_slices > 0 else goal_location
        self.controller.steer = steer_toward_target(car, target_location)

        # If in Rumble mode and the bot is spiking the ball, you could modify the bot's actions here
        if is_rumble and self.spike_watcher.carrying_car == car:
            logger.info("The bot is carrying the ball with spikes!")
            # Example behavior modification: aim towards the goal with spiked ball
            self.controller.throttle = 1.0  # Full throttle when carrying the ball
            self.controller.steer = steer_toward_target(car, goal_location)  # Aim towards the goal

        # Log the current state for debugging
        logger.info(f"Current state: {self.state}")

        # Delegate to the state handler
        output = self.state_handler.get_output(packet)

        # If no output, fall back to PPO model
        if output is None:
            observation = self.get_observation(packet)
            action, _ = self.model.predict(observation, deterministic=True)
            self.controller = self.map_action_to_controls(action)
        else:
            self.controller = output

        return self.controller
    
    def map_action_to_controls(self, action):
        controller_state = SimpleControllerState()
        controller_state.throttle = float(action[0])
        controller_state.steer = np.clip(float(action[1]), -1.0, 1.0)
        controller_state.boost = bool(action[2])
        controller_state.jump = bool(action[3])
        return controller_state

    def go_to_location(self, location: Vec3):
        #GPT logic for going to boost. REVISIT for going for demos
        car = self.game_interface.get_car(self.index)  # Get the current car state
        car_location = Vec3(car.physics.location)  # Get the car's current position
        car_velocity = Vec3(car.physics.velocity)  # Get the car's current velocity

        # Calculate the vector to the target location
        direction_to_target = location - car_location
        distance_to_target = direction_to_target.magnitude()  # Get the distance to the target location
        direction_to_target_normalized = direction_to_target.normalized() if distance_to_target > 0 else None

        # Normalize the direction vector
#        if distance_to_target > 0:
#            direction_to_target_normalized = direction_to_target.normalized()
#        else:
#            return SimpleControllerState()  # Already at the target location

        # Calculate the desired steering angle
        steering_adjustment = steer_toward_target(car, location)

        # Calculate the car's current speed
        current_speed = car_velocity.magnitude()

        throttle = 1.0 if distance_to_target > 200 else max(0.3, 1.0 - (distance_to_target / 200))
        # Throttle control: Accelerate if far from the target, decelerate if close
#        if distance_to_target > 200:
#            throttle = 1.0  # Full throttle if far away
#        else:
#            throttle = max(0.0, 1.0 - (distance_to_target / 200))  # Gradually reduce throttle as it gets closer

        # Boost logic
        use_boost = current_speed < self.Supersonic_Speed and car.boost > 20  # Use boost to reach supersonic speed

        # Check if the car is already supersonic
        if current_speed >= self.Supersonic_Speed:
            throttle = 1.0  # Maintain full throttle when supersonic

        # Create the controller state
        controller_state = SimpleControllerState()
        controller_state.throttle = throttle
        controller_state.steer = steering_adjustment
        controller_state.boost = use_boost

        return controller_state

    def is_ball_heading_to_own_goal(self, packet):
        """Logic to predict if the ball is heading towards the bot's goal."""
        ball_location = Vec3(packet.game_ball.physics.location)
        ball_velocity = Vec3(packet.game_ball.physics.velocity)
        goal_location = Vec3(0, -5100, 0) if self.team == 0 else Vec3(0, 5100, 0)
        
        relative_velocity = ball_location - goal_location
        return relative_velocity.y < 0 and ball_velocity.y < 0  # Ball is moving towards the own goal

    def defend_goal(self, packet):
        #GPT logic for basic goal defense. REVISIT for aerial defense, prejumps, backboard saves, squishy save for style
        car = packet.game_cars[self.index]  # Get the bot's car state
        ball_location = Vec3(packet.game_ball.physics.location)  # Get the ball's position
        car_location = Vec3(car.physics.location)  # Get the car's position

        # Define the goal location based on the bot's team
        goal_location = Vec3(0, -5100, 0) if self.team == 0 else Vec3(0, 5100, 0)

        # Calculate the distance from the car to the goal
        distance_to_goal = car_location.dist(goal_location)

        # Check if the ball is near the goal
        # Optimized target position logic
        if abs(ball_location.y) > 4600:  
            target_position = Vec3(ball_location.x, goal_location.y, 0)
        else:
            target_position = Vec3(goal_location.x, goal_location.y, 0)

        # Calculate the steering direction to the target position
        steer = steer_toward_target(car, target_position)

        # Set throttle and boost based on distance to the goal
        if distance_to_goal > 500:  # If far from the goal
            throttle = 1.0  # Full throttle to reach the goal quickly
        else:
            throttle = 0.5  # Slow down as we approach

        # Boost logic
        use_boost = car.boost > 30 and distance_to_goal > 500  # Use boost if far from the goal

        # Create the controller state to return
        controller_state = SimpleControllerState()
        controller_state.throttle = throttle
        controller_state.steer = steer
        controller_state.boost = use_boost

        return controller_state

    def can_take_shot(self, packet):
        #GPT code to determine if bot can shoot or if it's beneficial not to. REVISIT to give the bot choices if it should dribble, aerial play, wall pinch or else
        car = packet.game_cars[self.index]  # Get the bot's car state
        ball_location = Vec3(packet.game_ball.physics.location)  # Get the ball's position
        car_location = Vec3(car.physics.location)  # Get the car's position

        # Define some parameters
        shot_range = 2000  # Maximum distance to consider for a shot
        opponent_threshold = 1500  # Minimum distance from an opponent to take a shot
        goal_location = Vec3(0, -5100, 0) if self.team == 0 else Vec3(0, 5100, 0)  # Goal location based on team

        # Check if the ball is within a reasonable distance to take a shot
        if car_location.dist(ball_location) > shot_range:
            return False  # Too far to take a shot

        # Check if the ball is in front of the bot
        if not is_ball_in_front(car, ball_location):
            return False  # The ball is behind the bot

        # Check if there are opponents close by
        opponents = [opponent for opponent in packet.game_cars if opponent.team != self.team]
        if any(car_location.dist(Vec3(opponent.physics.location)) < opponent_threshold for opponent in opponents):
            return False  # An opponent is too close to take a shot

        # Check for clear line to the goal
        if not has_clear_line_to_goal(car, ball_location, goal_location, packet):
            return False  # No clear line to the goal

        # If all checks are passed, the bot can take a shot
        return True
    
    def is_ball_in_front(self, car, ball_location):
        """Check if the ball is in front of the car."""
        car_forward = Vec3(car.physics.rotation).forward()  # Get the car's forward vector
        car_to_ball = (ball_location - Vec3(car.physics.location)).normalize()
        return car_forward.dot(car_to_ball) > 0  # Return true if the ball is in front of the car

    def has_clear_line_to_goal(self, car, ball_location, goal_location, packet):
        """Check if there is a clear line to the goal."""
        # Simple line-of-sight check
        for opponent in packet.game_cars:
            if opponent.team != car.team:  # Check only opponents
                if intersects(car, ball_location, goal_location, opponent):
                    return False  # An opponent blocks the line to the goal
        return True

#intersects(car, ball_location, goal_location, opponent)
    def intersects(self, line_start, line_end, opponent):
        """Determine if a line from the car to the goal intersects with the opponent."""
        # A very simplistic intersection check; can be expanded with actual physics collision checks
        opponent_location = Vec3(opponent.physics.location)
        # Check if the opponent is within a certain distance of the line
        #distance_to_line = distance_from_point_to_line(opponent_location, ball_location, goal_location)
        distance_to_line = self.distance_from_point_to_line(opponent_location, line_start, line_end)
        return distance_to_line < 200  # Arbitrary threshold for intersection

#distance_from_point_to_line(point, line_start, line_end)
    def distance_from_point_to_line(self, point, line_start, line_end):
        """Calculate the distance from a point to a line segment."""
        # Vector AB
        ab = line_end - line_start
        # Vector AC
        ac = point - line_start
        # Calculate the area of the triangle formed by points A, B, and C
        area = abs(ab.x * ac.y - ab.y * ac.x)
        # Length of line AB
        length_ab = ab.length()
        return area / length_ab if length_ab > 0 else 0  # Avoid division by zero


    def take_shot(self, packet):
        #GPT code for a basic shot, REVISIT as im sure it has no idea how physics works and how power shots work, determine what shot is better and execute, opponent distance?
        car = packet.game_cars[self.index]  # Get the bot's car state
        ball_location = Vec3(packet.game_ball.physics.location)  # Get the ball's position
        car_location = Vec3(car.physics.location)  # Get the car's position
        goal_location = Vec3(0, -5100, 0) if self.team == 0 else Vec3(0, 5100, 0)  # Goal location based on team

        # Determine the distance to the goal
        distance_to_goal = car_location.dist(goal_location)

        # Calculate a shot direction towards the goal
        shot_direction = (goal_location - ball_location).normalize()

        # Basic decision making
        shot_power = 1.0  # Default power; will adjust based on distance
        if distance_to_goal < 1000:
            shot_power = 1.0  # Full power for shots close to the goal
        elif distance_to_goal < 2000:
            shot_power = 0.8  # Moderate power for mid-range shots
        else:
            shot_power = 0.5  # Less power for long-distance shots

        # Create a controller state for the shot
        controller_state = SimpleControllerState()
        controller_state.throttle = shot_power  # Set throttle for shot power
        controller_state.steer = shot_direction.x  # Steer towards the goal
        controller_state.pitch = 0  # Level the car
        controller_state.yaw = 0  # Keep the car facing forward (or adjust as needed)
        controller_state.roll = 0  # No roll for a simple shot

        # Implement a basic kick for the ball
        if ball_location.dist(car_location) < 200:  # If close enough to hit the ball
            controller_state.jump = True  # Jump to give a lift to the shot
            if car.boost > 30:  # Check if there's enough boost
                controller_state.boost = True  # Use boost to enhance shot power

        return controller_state


    def chase_ball(self, packet):
        car = packet.game_cars[self.index]
        ball_location = Vec3(packet.game_ball.physics.location)
        car_location = Vec3(car.physics.location)
        team_goal_location = Vec3(0, -5100, 0) if self.team == 0 else Vec3(0, 5100, 0)
        
        # Calculate distances and directions
        distance_to_ball = car_location.dist(ball_location)
        direction_to_ball = (ball_location - car_location).normalize()
        direction_to_goal = (team_goal_location - car_location).normalize()
        
        # Initialize controller state
        controller_state = SimpleControllerState()
        
        # Find closest opponent to the ball
        closest_opponent_distance = float('inf')
        for opponent in packet.game_cars:
            if opponent.team != self.team:
                opponent_distance_to_ball = Vec3(opponent.physics.location).dist(ball_location)
                closest_opponent_distance = min(closest_opponent_distance, opponent_distance_to_ball)
        
        # Decide when to boost: if we have enough boost and are significantly further from the ball than our opponent
        if car.boost > 50 or (car.boost > 20 and distance_to_ball < closest_opponent_distance + 300):
            controller_state.boost = True
        
        # Adjust steering: stay defensive if opponent is closer to the ball than we are
        if closest_opponent_distance < distance_to_ball and closest_opponent_distance < 1000:
            controller_state.steer = direction_to_goal.x  # Adjust towards goal side defensively
        else:
            controller_state.steer = direction_to_ball.x  # Steer towards the ball if we're clear

        # Apply throttle
        controller_state.throttle = 1.0
        
        return controller_state

    def get_ball_prediction(self, packet: GameTickPacket, time_horizon: float = 2.0) -> Optional[Vec3]:
        #GPT code for ball prediction. May already be defined REVISIT TO LOOK FOR IT
        ball_prediction = packet.ball_prediction  # Get the basic ball prediction

        if ball_prediction is None or ball_prediction.num_slices == 0:
            return None  # No prediction available

        # Iterate through the predicted ball slices to find the one closest to the time horizon
        for i in range(ball_prediction.num_slices):
            ball_slice = ball_prediction.slices[i]  # Get each future ball state
            if ball_slice.game_seconds >= packet.game_info.seconds_elapsed + time_horizon:
                # Return the predicted position at the time_horizon (convert to Vec3 for custom use)
                return Vec3(ball_slice.physics.location)

        # If no slice matches the time horizon, return the latest available prediction
        return Vec3(ball_prediction.slices[-1].physics.location) if ball_prediction.num_slices > 0 else None


    def start_sequence(self, sequence: Sequence):
        """ Start a new sequence of actions. """
        self.sequence = sequence

# Load or train the model, then instantiate the bot with the model
if __name__ == "__main__":
    while True:  # Loop until a valid input is received
        mode = input("Would you like to (1) Train the bot or (2) Play against the bot? (Enter 1 or 2): ")
        if mode not in ['1', '2']:
            logger.error("Invalid input! Please enter 1 for training or 2 for playing against the bot.")
            continue  # Prompt the user again
        break  # Exit the loop if valid input is received

    if mode == '1':  # If training mode
        while True:  # Loop until a valid input is received
            try:
                num_iterations = int(input("Enter the number of training iterations (positive integer): "))
                if num_iterations <= 0:  # Check for negative or zero
                    logger.warning("Please enter a positive integer greater than 0.")
                    continue  # Prompt the user again
                break  # Exit the loop if valid input is received
            except ValueError:
                logger.error("Invalid input! Please enter a valid positive integer.")

        model = train_or_load_model(num_iterations)

        # Instantiate the bot with the trained model
        bot = EchoBot(name="EchoBot", team=0, index=0, model=model)

        logger.info("Training complete. Exiting the game...")

        def exit_game():
            """ Exit the Rocket League game cleanly. """
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] == 'RocketLeague.exe':
                    proc.terminate()  # Terminate the game process
                    logger.info("Rocket League has been terminated.")

        exit_game()  # Call the exit function
        import sys
        sys.exit(0)  # Exit the script

    elif mode == '2':  # If playing against the bot
        logger.info("Launching GUI to play against the bot...")
        os.system("python run_gui.py")  # Launch the GUI script
