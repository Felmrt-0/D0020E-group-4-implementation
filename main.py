import time
import datetime
import gym
import gym_driving
from stable_baselines3 import PPO
from finite_state_machine_lib.FSM import FSM
from finite_state_machine_lib.State import State
from termcolor import colored

from signal import signal, SIGINT

REST_TIME = 1/500

class normalTest:

    def __init__(self):
        print("Hello World")
        env = gym.make('Driving-v0')
        model = PPO("MlpPolicy", env).load("logs/final_model")
        obs = env.reset()

        fsm = FSM()
        corr = State(self.drive_correct)
        incorr = State(self.drive_incorrect)
        switch = State(self.switcher)
        corr.add_transition(True, switch)
        corr.add_transition(False, corr)
        incorr.add_transition(True, switch)
        incorr.add_transition(False, incorr)
        switch.add_transition(True, incorr)
        switch.add_transition(False, corr)
        fsm.add_states([corr, incorr, switch])
        fsm.run([model, obs, env])

        env.close()

    @staticmethod
    def drive_correct(li):
        model, obs, env = li
        print(colored("corr", "green"), end='\r')
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        time.sleep(REST_TIME) 
        if not done:
            return False, [model, obs, env]
        return True, [True, model, obs, env]
        
    @staticmethod
    def drive_incorrect(li):
        model, obs, env = li
        print(colored("incorr", "red"), end='\r')
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        time.sleep(REST_TIME) 
        if not done:
            return False, [model, obs, env]
        return True, [False, model, obs, env]

    @staticmethod
    def switcher(li):
        print()
        origin, model, _,  env = li
        obs = env.reset()
        return origin, [model, obs, env]


class DBTest:
    def __init__(self):
        signal(SIGINT, self.signal_handler)

        self.env = gym.make('Driving-v0')
        model = PPO("MlpPolicy", self.env).load("logs/final_model")
        obs = self.env.reset()

        fsm = FSM()
        fsm.create_database()
        self.db = fsm.get_database()
        corr = State(self.drive_correct, static_parameter=fsm.get_database())
        incorr = State(self.drive_incorrect, static_parameter=fsm.get_database())
        switch = State(self.switcher)
        corr.add_transition(True, switch)
        corr.add_transition(False, corr)
        incorr.add_transition(True, switch)
        incorr.add_transition(False, incorr)
        switch.add_transition(True, incorr)
        switch.add_transition(False, corr)
        fsm.add_states([corr, incorr, switch])
        self.table = "DBTest"
        fsm.run([model, obs, self.env, "DBTest"])

        self.env.close()

    def signal_handler(self, SIGINT, frame):
        self.env.close()
        print(self.db.print_everything(self.table))
        self.db.custom_query("DELETE from " + self.table + " WHERE time > 0")
        exit(SIGINT)

    def drive_correct(self, db, li):
        model, obs, env, table = li
        print(colored("corr", "green"), end='\r')
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        self.db_write(db, table, obs[0:2], "corr")
        time.sleep(REST_TIME)
        if not done:
            return False, [model, obs, env, table]
        return True, [True, model, obs, env, table]
        
    def drive_incorrect(self, db, li):
        model, obs, env, table = li
        print(colored("incorr", "red"), end='\r')
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        self.db_write(db, table, obs[0:2], "incorr")
        time.sleep(REST_TIME)
        if not done:
            return False, [model, obs, env, table]
        return True, [False, model, obs, env, table]

    def switcher(self, li):
        print()
        origin, model, _,  env, table = li
        obs = env.reset()
        return origin, [model, obs, env, table]

    def db_write(self, db, table, info, tag):
        x, y = info
        data = {
            "measurement": table,
            "tags": {
                "Source": tag,
            },
            "time": datetime.datetime.now(),
            "fields": {
                "X-DistToGoal": x,
                "Y-DistToGoal": y
            }
        }
        db.insert([data])


if __name__ == "__main__":
    #test = normalTest()
    test = DBTest()

