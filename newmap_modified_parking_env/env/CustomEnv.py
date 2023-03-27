from abc import abstractmethod
from gym import Env
import numpy as np
from gym.envs.classic_control import rendering

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation, observation_factory
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Landmark, Obstacle

from gym.envs.classic_control import rendering

class GoalEnv(Env):
    """
    Interface for A goal-based environment.

    This interface is needed by agents such as Stable Baseline3's Hindsight Experience Replay (HER) agent.
    It was originally part of https://github.com/openai/gym, but was later moved
    to https://github.com/Farama-Foundation/gym-robotics. We cannot add gym-robotics to this project's dependencies,
    since it does not have an official PyPi package, PyPi does not allow direct dependencies to git repositories.
    So instead, we just reproduce the interface here.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError


class CustomEnv(AbstractEnv, GoalEnv):
    """
    A continuous control environment.

    It implements a reach-type task, where the agent observes their position and speed and must
    control their acceleration and steering so as to reach a given goal.

    Credits to Munir Jojo-Verge for the idea and initial implementation.
    """

    # For parking env with GrayscaleObservation, the env need
    # this PARKING_OBS to calculate the reward and the info.
    # Bug fixed by Mcfly(https://github.com/McflyWZX)
    PARKING_OBS = {"observation": {
            "type": "KinematicsGoal",
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False
        }}

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.observation_type_parking = None

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "KinematicsGoal",
                "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": False
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 1, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -114514,
            "steering_range": np.deg2rad(45),
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 500,
            "screen_width": 1200,
            "screen_height": 1000,
            "centering_position": [0.5, 0.5],
            "scaling": 7,
            "controlled_vehicles": 1
        })
        return config

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        super().define_spaces()
        self.observation_type_parking = observation_factory(self, self.PARKING_OBS["observation"])

    def _info(self, obs, action) -> dict:
        info = super(CustomEnv, self)._info(obs, action)
        if isinstance(self.observation_type, MultiAgentObservation):
            success = tuple(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        else:
            obs = self.observation_type_parking.observe()
            success = self._is_success(obs['achieved_goal'], obs['desired_goal'])
        info.update({"is_success": success})
        return info

    def _reset(self):
        self._create_road()
        self._create_vehicles()

    def _create_road(self, spots: int = 15) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        net = RoadNetwork()
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        net.add_lane("a", "b", StraightLane([-32, 60], [-32, 60-4], width=22, line_types=lt))
                
        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])
        
        for y in [-60, 60]:
            obstacle = Obstacle(self.road, [0, y])
            obstacle.LENGTH, obstacle.WIDTH = (90, 1)
            obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)
        for x in [-45, 45]:
            obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
            obstacle.LENGTH, obstacle.WIDTH = (120, 1)
            obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
            self.road.objects.append(obstacle)
        
        #lane = self.np_random.choice(self.road.network.lanes_list())
        self.goal = Landmark(self.road, [0, 0])
        self.road.objects.append(self.goal)
        #self.addWall()

        #return
        
        net.add_lane("a", "b", StraightLane([-43, -60], [-43, 60], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([-21, -60], [-21, -56], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([-21, -56], [45, -56], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([-21, 60], [-21, 56], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([-21, 56], [45, 56], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([-17.4355, -40], [17.5645, -40], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([-17.4355, 40], [17.5645, 40], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([43, -25], [43, 25], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([23, -25], [23, 25], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([7.3108, 18.5], [7.3108, -7.5], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([7.3108, 18.5], [-23, 1], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([7.3108, -7.5], [-23, -25], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([-23, 1], [-23, 25], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([-17.4355, -40], [-23, -25], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([17.5645, -40], [23, -25], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([-17.4355, 40], [-23, 25], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([17.5645, 40], [23, 25], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([37.5645, -40], [43, -25], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([37.5645, 40], [43, 25], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([37.5645, -40], [45, -40], width=0, line_types=lt))
        net.add_lane("a", "b", StraightLane([37.5645, 40], [45, 40], width=0, line_types=lt))
        
    def addWall(self):
        obstacle = Obstacle(self.road, [-43.0, -59.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -57.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -55.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -53.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -51.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -49.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -47.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -45.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -43.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -41.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -39.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -37.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -35.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -33.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -31.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -29.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -27.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -25.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -23.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -21.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -19.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -17.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -15.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -13.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -11.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -9.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -7.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -5.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -3.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, -1.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 1.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 3.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 5.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 7.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 9.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 11.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 13.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 15.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 17.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 19.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 21.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 23.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 25.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 27.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 29.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 31.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 33.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 35.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 37.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 39.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 41.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 43.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 45.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 47.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 49.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 51.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 53.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 55.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 57.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-43.0, 59.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [-21.0, -59.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-21.0, -57.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [-21.0, 57.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-21.0, 59.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [43.0, -24.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, -22.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, -20.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, -18.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, -16.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, -14.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, -12.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, -10.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, -8.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, -6.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, -4.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, -2.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 0.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 2.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 4.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 6.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 8.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 10.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 12.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 14.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 16.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 18.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 20.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 22.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [43.0, 24.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [23.0, -24.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, -22.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, -20.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, -18.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, -16.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, -14.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, -12.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, -10.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, -8.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, -6.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, -4.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, -2.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 0.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 2.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 4.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 6.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 8.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 10.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 12.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 14.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 16.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 18.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 20.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 22.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [23.0, 24.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [7.3108, -6.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.3108, -4.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.3108, -2.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.3108, -0.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.3108, 1.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.3108, 3.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.3108, 5.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.3108, 7.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.3108, 9.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.3108, 11.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.3108, 13.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.3108, 15.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.3108, 17.5], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [-23.0, 2.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-23.0, 4.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-23.0, 6.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-23.0, 8.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-23.0, 10.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-23.0, 12.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-23.0, 14.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-23.0, 16.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-23.0, 18.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-23.0, 20.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-23.0, 22.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-23.0, 24.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [-20.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-18.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-16.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-14.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-12.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-10.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-8.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-6.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-4.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-2.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [0.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [2.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [4.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [6.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [8.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [10.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [12.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [14.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [16.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [18.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [20.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [22.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [24.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [26.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [28.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [30.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [32.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [34.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [36.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [38.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [40.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [42.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [44.0, -56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [-20.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-18.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-16.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-14.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-12.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-10.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-8.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-6.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-4.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-2.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [0.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [2.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [4.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [6.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [8.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [10.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [12.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [14.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [16.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [18.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [20.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [22.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [24.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [26.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [28.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [30.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [32.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [34.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [36.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [38.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [40.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [42.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [44.0, 56.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [-16.4355, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0000000000000018, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-14.4355, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-12.4355, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-10.4355, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-8.4355, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (1.9999999999999991, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-6.4355, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-4.4355, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-2.4355, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-0.4355, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [1.5645, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (1.9999999999999998, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [3.5645, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [5.5645, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.5645, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.000000000000001, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [9.5645, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [11.5645, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [13.5645, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [15.5645, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (1.9999999999999982, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [17.0645, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (1.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [-16.4355, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0000000000000018, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-14.4355, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-12.4355, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-10.4355, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-8.4355, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (1.9999999999999991, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-6.4355, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-4.4355, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-2.4355, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-0.4355, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [1.5645, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (1.9999999999999998, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [3.5645, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [5.5645, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [7.5645, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.000000000000001, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [9.5645, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [11.5645, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [13.5645, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [15.5645, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (1.9999999999999982, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [17.0645, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (1.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [38.5645, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [40.5645, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [42.5645, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [44.2823, -40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (1.4354999999999976, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [38.5645, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [40.5645, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [42.5645, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [44.2823, 40.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (1.4354999999999976, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [6.4688, 18.0139], heading= 0.5236000487892992)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401520609003, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [4.7849, 17.0417], heading= 0.5236000462178609)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401607211494, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [3.101, 16.0694], heading= 0.5236000532431504)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401570609104, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [1.417, 15.0972], heading= 0.5236000487893002)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401520609007, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-0.2669, 14.125], heading= 0.5236000462178594)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401607211477, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-1.9508, 13.1528], heading= 0.5236000487893001)
        obstacle.LENGTH, obstacle.WIDTH = (1.944440152060901, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-3.6348, 12.1806], heading= 0.5236000532431512)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401570609113, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-5.3187, 11.2083], heading= 0.5236000462178593)
        obstacle.LENGTH, obstacle.WIDTH = (1.944440160721148, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-7.0026, 10.2361], heading= 0.5236000487893002)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401520609003, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-8.6866, 9.2639], heading= 0.5236000487893)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401520609012, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-10.3705, 8.2917], heading= 0.5236000462178596)
        obstacle.LENGTH, obstacle.WIDTH = (1.944440160721147, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-12.0544, 7.3194], heading= 0.5236000532431511)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401570609116, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-13.7384, 6.3472], heading= 0.5236000487892996)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401520609007, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-15.4223, 5.375], heading= 0.5236000462178598)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401607211494, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-17.1062, 4.4028], heading= 0.5236000487892999)
        obstacle.LENGTH, obstacle.WIDTH = (1.944440152060899, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-18.7902, 3.4306], heading= 0.5236000532431517)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401570609104, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-20.4741, 2.4583], heading= 0.523600046217859)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401607211503, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-22.158, 1.4861], heading= 0.5236000487893001)
        obstacle.LENGTH, obstacle.WIDTH = (1.944440152060899, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [6.4688, -7.9861], heading= 0.5236000487893)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401520609012, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [4.7849, -8.9583], heading= 0.5236000462178593)
        obstacle.LENGTH, obstacle.WIDTH = (1.944440160721148, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [3.101, -9.9306], heading= 0.5236000532431512)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401570609113, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [1.417, -10.9028], heading= 0.5236000487893002)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401520609007, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-0.2669, -11.875], heading= 0.5236000462178594)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401607211477, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-1.9508, -12.8472], heading= 0.5236000487893001)
        obstacle.LENGTH, obstacle.WIDTH = (1.944440152060901, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-3.6348, -13.8194], heading= 0.5236000532431512)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401570609113, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-5.3187, -14.7917], heading= 0.5236000462178593)
        obstacle.LENGTH, obstacle.WIDTH = (1.944440160721148, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-7.0026, -15.7639], heading= 0.5236000487893002)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401520609003, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-8.6866, -16.7361], heading= 0.5236000487892992)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401520609003, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-10.3705, -17.7083], heading= 0.5236000462178612)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401607211488, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-12.0544, -18.6806], heading= 0.5236000532431511)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401570609116, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-13.7384, -19.6528], heading= 0.5236000487892992)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401520609003, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-15.4223, -20.625], heading= 0.5236000462178592)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401607211486, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-17.1062, -21.5972], heading= 0.5236000487892997)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401520608987, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-18.7902, -22.5694], heading= 0.5236000532431515)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401570609102, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-20.4741, -23.5417], heading= 0.5236000462178603)
        obstacle.LENGTH, obstacle.WIDTH = (1.944440160721152, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-22.158, -24.5139], heading= 0.5236000487892997)
        obstacle.LENGTH, obstacle.WIDTH = (1.9444401520608987, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [-17.7833, -39.0625], heading= -1.215566409942541)
        obstacle.LENGTH, obstacle.WIDTH = (1.999858042813601, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-18.4788, -37.1875], heading= -1.2155664099425394)
        obstacle.LENGTH, obstacle.WIDTH = (1.9998580428136024, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-19.1744, -35.3125], heading= -1.2155664099425394)
        obstacle.LENGTH, obstacle.WIDTH = (1.9998580428136024, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-19.87, -33.4375], heading= -1.215566409942541)
        obstacle.LENGTH, obstacle.WIDTH = (1.999858042813601, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-20.5655, -31.5625], heading= -1.2155664099425394)
        obstacle.LENGTH, obstacle.WIDTH = (1.9998580428136024, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-21.2611, -29.6875], heading= -1.2155664099425394)
        obstacle.LENGTH, obstacle.WIDTH = (1.9998580428136024, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-21.9567, -27.8125], heading= -1.215566409942541)
        obstacle.LENGTH, obstacle.WIDTH = (1.999858042813601, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-22.6522, -25.9375], heading= -1.2155664099425394)
        obstacle.LENGTH, obstacle.WIDTH = (1.9998580428136024, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [17.9042, -39.0625], heading= 1.2231471914147594)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723418, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [18.5837, -37.1875], heading= 1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [19.2631, -35.3125], heading= 1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [19.9425, -33.4375], heading= 1.2231471914147594)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723418, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [20.622, -31.5625], heading= 1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [21.3014, -29.6875], heading= 1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [21.9808, -27.8125], heading= 1.2231471914147594)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723418, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [22.6603, -25.9375], heading= 1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [-17.7833, 39.0625], heading= 1.215566409942541)
        obstacle.LENGTH, obstacle.WIDTH = (1.999858042813601, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-18.4788, 37.1875], heading= 1.2155664099425394)
        obstacle.LENGTH, obstacle.WIDTH = (1.9998580428136024, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-19.1744, 35.3125], heading= 1.2155664099425394)
        obstacle.LENGTH, obstacle.WIDTH = (1.9998580428136024, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-19.87, 33.4375], heading= 1.215566409942541)
        obstacle.LENGTH, obstacle.WIDTH = (1.999858042813601, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-20.5655, 31.5625], heading= 1.2155664099425394)
        obstacle.LENGTH, obstacle.WIDTH = (1.9998580428136024, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-21.2611, 29.6875], heading= 1.2155664099425394)
        obstacle.LENGTH, obstacle.WIDTH = (1.9998580428136024, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-21.9567, 27.8125], heading= 1.215566409942541)
        obstacle.LENGTH, obstacle.WIDTH = (1.999858042813601, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-22.6522, 25.9375], heading= 1.2155664099425394)
        obstacle.LENGTH, obstacle.WIDTH = (1.9998580428136024, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [17.9042, 39.0625], heading= -1.2231471914147594)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723418, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [18.5837, 37.1875], heading= -1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [19.2631, 35.3125], heading= -1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [19.9425, 33.4375], heading= -1.2231471914147594)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723418, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [20.622, 31.5625], heading= -1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [21.3014, 29.6875], heading= -1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [21.9808, 27.8125], heading= -1.2231471914147594)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723418, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [22.6603, 25.9375], heading= -1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [37.9042, -39.0625], heading= 1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [38.5837, -37.1875], heading= 1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [39.2631, -35.3125], heading= 1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [39.9425, -33.4375], heading= 1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [40.622, -31.5625], heading= 1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [41.3014, -29.6875], heading= 1.2231471914147578)
        obstacle.LENGTH, obstacle.WIDTH = (1.994306976472343, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [41.9808, -27.8125], heading= 1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [42.6603, -25.9375], heading= 1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [37.9042, 39.0625], heading= -1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [38.5837, 37.1875], heading= -1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [39.2631, 35.3125], heading= -1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [39.9425, 33.4375], heading= -1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [40.622, 31.5625], heading= -1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [41.3014, 29.6875], heading= -1.2231471914147578)
        obstacle.LENGTH, obstacle.WIDTH = (1.994306976472343, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [41.9808, 27.8125], heading= -1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [42.6603, 25.9375], heading= -1.2231471914147611)
        obstacle.LENGTH, obstacle.WIDTH = (1.9943069764723405, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -59.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -57.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -55.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -53.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -51.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -49.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -47.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -45.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -43.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -41.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -39.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -37.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -35.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -33.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -31.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -29.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -27.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -25.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -23.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -21.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -19.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -17.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -15.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -13.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -11.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -9.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -7.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -5.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -3.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, -1.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 1.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 3.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 5.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 7.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 9.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 11.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 13.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 15.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 17.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 19.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 21.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 23.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 25.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 27.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 29.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 31.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 33.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 35.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 37.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 39.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 41.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 43.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 45.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 47.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 49.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 51.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 53.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 55.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 57.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [45.0, 59.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -59.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -57.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -55.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -53.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -51.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -49.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -47.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -45.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -43.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -41.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -39.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -37.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -35.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -33.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -31.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -29.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -27.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -25.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -23.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -21.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -19.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -17.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -15.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -13.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -11.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -9.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -7.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -5.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -3.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, -1.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 1.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 3.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 5.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 7.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 9.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 11.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 13.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 15.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 17.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 19.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 21.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 23.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 25.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 27.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 29.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 31.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 33.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 35.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 37.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 39.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 41.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 43.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 45.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 47.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 49.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 51.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [-45.0, 53.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 55.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 57.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-45.0, 59.0], heading= 1.5707963267948966)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [-44.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-42.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-40.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-38.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-36.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-34.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-32.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-30.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-28.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-26.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-24.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-22.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-20.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-18.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-16.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-14.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-12.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-10.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-8.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-6.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-4.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-2.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [0.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [2.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [4.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [6.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [8.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [10.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [12.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [14.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [16.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [18.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [20.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [22.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [24.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [26.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [28.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [30.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [32.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [34.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [36.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [38.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [40.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [42.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [44.0, -60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

        obstacle = Obstacle(self.road, [-44.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-42.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-40.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-38.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-36.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-34.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-32.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-30.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-28.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-26.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-24.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-22.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-20.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-18.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-16.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-14.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-12.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-10.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-8.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-6.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-4.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [-2.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [0.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [2.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [4.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [6.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [8.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [10.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [12.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [14.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [16.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [18.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [20.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [22.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [24.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [26.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [28.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [30.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [32.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [34.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [36.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [38.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [40.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [42.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)


        obstacle = Obstacle(self.road, [44.0, 60.0], heading= 0)
        obstacle.LENGTH, obstacle.WIDTH = (2.0, 1)
        obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
        self.road.objects.append(obstacle)

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for i in range(self.config["controlled_vehicles"]):
            #vehicle = self.action_type.vehicle_class(self.road, [i*20, 0], 2*np.pi*self.np_random.rand(), 0)
            vehicle = self.action_type.vehicle_class(self.road, [32, -50],0, 0)
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)



    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        """
        Proximity to the goal is rewarded

        We use a weighted p-norm

        :param achieved_goal: the goal that was achieved
        :param desired_goal: the goal that was desired
        :param dict info: any supplementary information
        :param p: the Lp^p norm used in the reward. Use p<1 to have high kurtosis for rewards in [0, 1]
        :return: the corresponding reward
        """
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)

    def _reward(self, action: np.ndarray) -> float:
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        return sum(self.compute_reward(agent_obs['achieved_goal'], agent_obs['desired_goal'], {})
                     for agent_obs in obs)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        time = self.steps >= self.config["duration"]
        crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = self.observation_type_parking.observe()
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        return time or crashed or success

    
class ParkingEnvActionRepeat(CustomEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})


