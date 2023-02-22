import time
import gym
import gym_driving
from stable_baselines3 import PPO
from finite_state_machine_lib.FSM import FSM
from finite_state_machine_lib.State import State
from termcolor import colored

REST_TIME = 1/500

def drive_correct(li):
    model, obs, env = li
    print(colored("corr", "green"), end='\r')
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    time.sleep(REST_TIME) 
    if not done:
        return False, [model, obs, env]
    return True, [True, model, obs, env]
    

def drive_incorrect(li):
    model, obs, env = li
    print(colored("incorr", "red"), end='\r')
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    time.sleep(REST_TIME) 
    if not done:
        return False, [model, obs, env]
    return True, [False, model, obs, env]


def switcher(li):
    print()
    origin, model, _,  env = li
    obs = env.reset()
    return origin, [model, obs, env]


if __name__ == "__main__":
    print("Hello World")
    env = gym.make('Driving-v0')
    model = PPO("MlpPolicy", env).load("logs/final_model")
    obs = env.reset()

    fsm = FSM()
    corr = State(drive_correct)
    incorr = State(drive_incorrect)
    switch = State(switcher)
    corr.add_transition(True, switch)
    corr.add_transition(False, corr)
    incorr.add_transition(True, switch)
    incorr.add_transition(False, incorr)
    switch.add_transition(True, incorr)
    switch.add_transition(False, corr)
    fsm.add_states([corr, incorr, switch])
    fsm.run([model, obs, env])

    env.close()


