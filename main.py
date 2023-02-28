import time
import datetime
import gym
import gym_driving
from stable_baselines3 import PPO
from finite_state_machine_lib.FSM import FSM
from finite_state_machine_lib.State import State
from finite_state_machine_lib.Logic import Logic
from termcolor import colored

from signal import signal, SIGINT

REST_TIME = 1/500


class NormalTest:

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


class NestTest():
    def __init__(self):
        global REST_TIME
        while True:
            loop = input("How many bad drives?\n")
            try:
                loop = int(loop)
                break
            except:
                continue
        while True:
            speed = input("How fast?\nSlow: 1\nMedium: 2\nFast: 3\nBreakneck: 9\n")
            match speed:
                case "1":
                    REST_TIME = 1/250
                    break
                case "2":
                    REST_TIME = 1/500
                    break
                case "3":
                    REST_TIME = 1/1000
                    break
                case "9":
                    REST_TIME = 0
                    break
                case _:
                    continue

        env = gym.make('Driving-v0')
        model = PPO("MlpPolicy", env).load("logs/final_model")
        obs = env.reset()

        mainFSM = FSM()
        stateM1 = State(self.drive_correct)
        stateFSM = State(self.miniFSM)
        stateM2 = State(self.drive_correct)
        stateM3 = State(self.drive_correct, ending=True)

        stateM1.add_transition(False, stateM1)
        stateM1.add_transition(True, stateFSM)
        stateFSM.add_transition(True, stateM2)
        stateM2.add_transition(False, stateM2)
        stateM2.add_transition(True, stateM3)

        mainFSM.add_states([stateM1, stateFSM, stateM2, stateM3])

        mainFSM.set_current_state(stateM1)
        mainFSM.run([model, obs, env, loop])

    def miniFSM(self, li):
        model, obs, env, loop = li
        subFSM = FSM()
        stateS1 = State(self.drive_incorrect)
        stateS2 = State(self.drive_incorrect, ending=True)

        subFSM.add_states([stateS1, stateS2])
        logic = Logic()
        logic.greater_than_limit(0)
        stateS1.add_transition(logic, stateS1)
        stateS1.add_transition(0, stateS1)
        stateS1.add_transition(-1, stateS2)

        if loop > 0:
            subFSM.set_current_state(stateS1)
        else:
            subFSM.set_current_state(stateS2)
        _, li = subFSM.run(li)
        return True, li

    def drive_correct(self, li):
        if len(li) < 4:
            print(li)
        model, obs, env, loop = li
        print(colored("corr", "green"), end='\r')
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        time.sleep(REST_TIME)
        if not done:
            return False, [model, obs, env, loop]
        obs = env.reset()
        return True, [model, obs, env, loop]

    def drive_incorrect(self, li):
        model, obs, env, loop = li
        if loop <= 0:
            return -1, li
        print(colored("incorr", "red"), end='\r')
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        time.sleep(REST_TIME)
        if not done:
            return False, [model, obs, env, loop]
        obs = env.reset()
        return loop-1, [model, obs, env, loop-1]


if __name__ == "__main__":
    #test = NormalTest()
    #test = DBTest()
    test = NestTest()

