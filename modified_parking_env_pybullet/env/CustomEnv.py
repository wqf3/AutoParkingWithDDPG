from abc import abstractmethod
from gym import Env
from gym.envs.registration import register
import numpy as np

import os
import time

import pybullet as blt
import pybullet_data

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import MultiAgentObservation, observation_factory
from highway_env.road.lane import StraightLane, LineType
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.objects import Landmark, Obstacle

class GoalEnv(Env):
    @abstractmethod
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> float:
        raise NotImplementedError


class CustomEnv(AbstractEnv, GoalEnv):
    PARKING_OBS = {"observation": {
            "type": "KinematicsGoal",
            "features": ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
            "scales": [100, 100, 5, 5, 1, 1],
            "normalize": False
        }}

    def __init__(self, config: dict = None) -> None:
        
        self.observation_type_parking = None
        self.stepCnt = 0
        
        self.client = blt.connect(blt.GUI) #DIRECT
        blt.resetSimulation()
        time.sleep(1. / 240.)
        blt.setGravity(0, 0, -9.81)

        #blt.setRealTimeSimulation(1)

        self.goal = [5.375,3.45]
        self.targetOrien = np.pi / 2
        self.steps = 0
        #blt.resetDebugVisualizerCamera(cameraDistance = 3,cameraYaw = 0,cameraPitch = 0,cameraTargetPosition = [0, 3, 0.5])
        super().__init__(config)

        
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
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
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

    def rendering(self) -> None:
        blt.stepSimulation(self.client)
        #blt.stepSimulation()
        time.sleep(1. / 240.)

    def reset(self):
        self.steps = 0
        blt.resetSimulation(self.client)
        blt.setGravity(0, 0, -9.81)

        self.ground = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\scene.urdf", 
                basePosition=[0, 0, 0.005], useFixedBase=10)

        self.wallN = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wallh.urdf", 
                basePosition=[0, 6, 0.005], useFixedBase=10)

        self.wallS = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wallh.urdf", 
                basePosition=[0, -6, 0.005], useFixedBase=10)

        self.wallE = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wallv.urdf", 
                basePosition=[4.5, 0, 0.005], useFixedBase=10)

        self.wallW = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wallv.urdf", 
                basePosition=[-4.5, 0, 0.005], useFixedBase=10)

        #self.pkWall1 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\parkingWall.urdf", 
        #        basePosition=[5.375, 2.45, 0.005], useFixedBase=10)

        #self.pkWall2 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\parkingWall.urdf", 
        #        basePosition=[5.375, 4.45, 0.005], useFixedBase=10)

        #self.pkWall2 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\parkingWall.urdf", 
        #        basePosition=[-5.375, 4.45, -0.23], useFixedBase=10)

        self.wall0 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall0.urdf",basePosition=[-4.3,0.0, 0.005], useFixedBase=10)
        self.wall1 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall1.urdf",basePosition=[-2.1,-5.8, 0.005], useFixedBase=10)
        self.wall2 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall2.urdf",basePosition=[1.2,-5.6, 0.005], useFixedBase=10)
        self.wall3 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall3.urdf",basePosition=[-2.1,5.8, 0.005], useFixedBase=10)
        self.wall4 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall4.urdf",basePosition=[1.2,5.6, 0.005], useFixedBase=10)
        self.wall5 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall5.urdf",basePosition=[0.00645,-4.0, 0.005], useFixedBase=10)
        self.wall6 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall6.urdf",basePosition=[0.00645,4.0, 0.005], useFixedBase=10)
        self.wall7 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall7.urdf",basePosition=[4.3,0.0, 0.005], useFixedBase=10)
        self.wall8 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall8.urdf",basePosition=[2.3,0.0, 0.005], useFixedBase=10)
        self.wall9 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall9.urdf",basePosition=[0.7310800000000001,0.55, 0.005], useFixedBase=10)
        self.wall10 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall10.urdf",basePosition=[-0.7844599999999999,0.975, 0.005], useFixedBase=10)
        self.wall11 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall11.urdf",basePosition=[-0.7844599999999999,-1.625, 0.005], useFixedBase=10)
        self.wall12 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall12.urdf",basePosition=[-2.3,1.3, 0.005], useFixedBase=10)
        self.wall13 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall13.urdf",basePosition=[-2.02178,-3.25, 0.005], useFixedBase=10)
        self.wall14 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall14.urdf",basePosition=[2.02822,-3.25, 0.005], useFixedBase=10)
        self.wall15 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall15.urdf",basePosition=[-2.02178,3.25, 0.005], useFixedBase=10)
        self.wall16 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall16.urdf",basePosition=[2.02822,3.25, 0.005], useFixedBase=10)
        self.wall17 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall17.urdf",basePosition=[4.02823,-3.25, 0.005], useFixedBase=10)
        self.wall18 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall18.urdf",basePosition=[4.02823,3.25, 0.005], useFixedBase=10)
        self.wall19 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall19.urdf",basePosition=[4.12823,-4.0, 0.005], useFixedBase=10)
        self.wall20 = blt.loadURDF(r".\modified_parking_env_pybullet\env\scene\wall\wall20.urdf",basePosition=[4.12823,4.0, 0.005], useFixedBase=10)
        
        self.goalPos = np.array([self.goal[0], self.goal[1], 0.0, 0.0, np.cos(self.targetOrien), np.sin(self.targetOrien)])
        
        self.vehicle = CustomEnvPybulletVehicle(client = self.client)
        self.vehicleModel = self.vehicle.vehicleModel
        vehicleObservation = self.vehicle.GetObservation(self.goalPos)
        self.stepCnt = 0
        
        return vehicleObservation

    def getDistance(self, pos):
        return np.sqrt(pow(pos[0] - self.goal[0], 2) + pow(pos[1] - self.goal[1], 2))

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict, p: float = 0.5) -> float:
        return -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.config["reward_weights"])), p)

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self.compute_reward(achieved_goal, desired_goal, {}) > -self.config["success_goal_reward"]

    def isCollised(self):
        pt1 =  blt.getContactPoints(self.vehicleModel,self.wallN)
        pt2 =  blt.getContactPoints(self.vehicleModel,self.wallS)
        pt3 =  blt.getContactPoints(self.vehicleModel,self.wallE)
        pt4 =  blt.getContactPoints(self.vehicleModel,self.wallW)
        wpt0 = blt.getContactPoints(self.vehicleModel,self.wall0)
        wpt1 = blt.getContactPoints(self.vehicleModel,self.wall1)
        wpt2 = blt.getContactPoints(self.vehicleModel,self.wall2)
        wpt3 = blt.getContactPoints(self.vehicleModel,self.wall3)
        wpt4 = blt.getContactPoints(self.vehicleModel,self.wall4)
        wpt5 = blt.getContactPoints(self.vehicleModel,self.wall5)
        wpt6 = blt.getContactPoints(self.vehicleModel,self.wall6)
        wpt7 = blt.getContactPoints(self.vehicleModel,self.wall7)
        wpt8 = blt.getContactPoints(self.vehicleModel,self.wall8)
        wpt9 = blt.getContactPoints(self.vehicleModel,self.wall9)
        wpt10= blt.getContactPoints(self.vehicleModel,self.wall10)
        wpt11= blt.getContactPoints(self.vehicleModel,self.wall11)
        wpt12= blt.getContactPoints(self.vehicleModel,self.wall12)
        wpt13= blt.getContactPoints(self.vehicleModel,self.wall13)
        wpt14= blt.getContactPoints(self.vehicleModel,self.wall14)
        wpt15= blt.getContactPoints(self.vehicleModel,self.wall15)
        wpt16= blt.getContactPoints(self.vehicleModel,self.wall16)
        wpt17= blt.getContactPoints(self.vehicleModel,self.wall17)
        wpt18= blt.getContactPoints(self.vehicleModel,self.wall18)
        wpt19= blt.getContactPoints(self.vehicleModel,self.wall19)
        wpt20= blt.getContactPoints(self.vehicleModel,self.wall20)

        return (len(pt1) or len(pt2) or len(pt3) or len(pt4) or len(wpt1 ) or len(wpt2 ) or len(wpt3 ) or len(wpt4 ) or len(wpt0 ) 
                or len(wpt5 ) or len(wpt6 ) or len(wpt7 ) or len(wpt8 ) or len(wpt9 ) 
                or len(wpt10) or len(wpt11) or len(wpt12) or len(wpt13) or len(wpt14) 
                or len(wpt15) or len(wpt16) or len(wpt17) or len(wpt18) or len(wpt19) 
                or len(wpt20) )

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the goal is reached."""
        time = self.steps >= self.config["duration"]

        crashed = self.isCollised()
        obs = self.vehicle.GetObservation(self.goalPos)
        obs = obs if isinstance(obs, tuple) else (obs,)
        success = all(self._is_success(agent_obs['achieved_goal'], agent_obs['desired_goal']) for agent_obs in obs)
        return time or crashed or success

    def step(self, action):
        self.steps += 1
        self.vehicle.ActionExecution(action)  # 小车执行动作
        blt.stepSimulation()
        observation = self.vehicle.GetObservation(self.goalPos)  # 获取小车状态

        position = np.array(observation['observation'][:2])
        distance = self.getDistance(position)
        reward = self.compute_reward(observation['observation'], observation['desired_goal'], None)

        #print(f'dis: {distance}, reward: {reward}, center: {self.goal}, pos: {observation}')

        self.done = self._is_terminal()
        self.success = self._is_success(observation['observation'], observation['desired_goal'])


        info = {'is_success': self.success}

        return observation, reward, self.done, info

    
class CustomEnvActionRepeat(CustomEnv):
    def __init__(self):
        super().__init__({"policy_frequency": 1, "duration": 20})



class CustomEnvPybulletVehicle():

    def __init__(self,
                client,
                basePos = [-3.2, -4.5, 0.1],
                baseOrien = [0, 0, np.pi/2],
                actionSteps = 40):

        blt.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.actionSteps = actionSteps
        self.client = client

        self.vehicleModel = blt.loadURDF(fileName=r"racecar\racecar.urdf", basePosition=basePos, 
                                   baseOrientation=blt.getQuaternionFromEuler(baseOrien),
                                   globalScaling=2)

        self.wheels = [2, 3]
        self.steering = [4, 6]
        inactive_wheels = [3, 5, 7]


        for wheel in inactive_wheels:
            blt.setJointMotorControl2(self.vehicleModel, wheel, blt.VELOCITY_CONTROL, targetVelocity=0, force=0)


    def ActionExecution(self, action):
        
        velocity = action[0]
        targetOrien = action[1]

        for i in range(self.actionSteps):
            for wheel in self.wheels:
                blt.setJointMotorControl2(self.vehicleModel,
                                        wheel,
                                        blt.VELOCITY_CONTROL,
                                        targetVelocity = -velocity*5,
                                        force=10)

            for steer in self.steering:
                blt.setJointMotorControl2(self.vehicleModel,
                                        steer,
                                        blt.POSITION_CONTROL,
                                        targetPosition = targetOrien)

            blt.stepSimulation(self.client)

    def GetObservation(self,goalPos):
        position, angle = blt.getBasePositionAndOrientation(self.vehicleModel) 
        angle = blt.getEulerFromQuaternion(angle)
        velocity = blt.getBaseVelocity(self.vehicleModel)[0]

        position = [position[0], position[1]]
        velocity = [velocity[0], velocity[1]]
        orientation = [np.cos(angle[2]), np.sin(angle[2])]

        observation = np.array(position + velocity + orientation)  

        obsDict = {'observation':observation,'achieved_goal':observation,'desired_goal':goalPos}

        return obsDict
        