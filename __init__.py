import gym
import highway_env
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def Train(envName:str = 'parking-v0',trainTime:int = int(1e6) ,saveDir:str = r"DDPG\model", logDir:str = r"DDPG\log") -> None:
    '''
    Training model with DDPG.
    Parameter(s):
        - envName:str -> Training environment. 
                         Default: parking-v0 (from highway-env). 
                         modified_parking_env (from authors) is recommended.
        - saveDir:str -> Directory of the model.
                         Default: "DDPG/model".
        - logDir :str -> Directory of the log.
                         Default: "DDPG/log".
                         To activate log: tensorboard.exe --logdir={Absolute path of log directory(NOT A FILE)}
        - trainTime:int -> Total time step of training.
                           Default: int(1e6).
    Return:
        None
    '''
    env = gym.make(envName)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG('MultiInputPolicy', env,
                  policy_kwargs=dict(net_arch=[256, 256]),
                  learning_rate=5e-4,
                  buffer_size=15000,
                  learning_starts=200,
                  batch_size=256,
                  gamma=0.8,
                  action_noise=action_noise,
                  train_freq=1,
                  gradient_steps=-1,
                  verbose=1,
                  tensorboard_log=logDir)
    model.learn(trainTime)
    model.save(saveDir)

def Eval(envName:str = 'parking-v0', modelDir:str = r"DDPG\model") -> None:
    '''
    Showing the result of training.
    Parameter(s):
        - envName :str -> Training environment. 
                          Default: parking-v0 (from highway-env). 
                          modified_parking_env (from authors) is recommended.
                          Should be same as training.
        - modelDir:str -> Directory of the model.
                          Default: "DDPG/model".
                          A pre-trained model "Modified_Parking_DDPG\model" is recommended.
    Return:
        None
    '''
    env = gym.make(envName)
    model = DDPG.load(modelDir)
    while True:
        done = truncated = False
        obs = env.reset()
        while not (done):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()

if __name__ == '__main__':
    import modified_parking_env
    Train()
    Eval()
else:
    from GymParking import modified_parking_env