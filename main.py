import time
import datetime
import gym
import gym_driving
from stable_baselines3 import PPO
from finite_state_machine_lib.FSM import FSM
from finite_state_machine_lib.State import State
from finite_state_machine_lib.Logic import Logic
from finite_state_machine_lib.CustomExceptions import DatabaseTableEmpty
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
        print()
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
        model, obs, env, loop = li
        print(colored("corr", "green"), end='\r')
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        time.sleep(REST_TIME)
        if not done:
            return False, [model, obs, env, loop]
        obs = env.reset()
        print()
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
        print()
        return loop-1, [model, obs, env, loop-1]


class NestDBTest:
    def __init__(self):
        global REST_TIME
        signal(SIGINT, self.signal_handler)


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
                    REST_TIME = 1 / 250
                    break
                case "2":
                    REST_TIME = 1 / 500
                    break
                case "3":
                    REST_TIME = 1 / 1000
                    break
                case "9":
                    REST_TIME = 0
                    break
                case _:
                    continue

        self.env = gym.make('Driving-v0')
        model = PPO("MlpPolicy", self.env).load("logs/final_model")
        obs = self.env.reset()

        mainFSM = FSM()
        mainFSM.create_database()
        self.db = mainFSM.get_database()
        stateM1 = State(self.drive_correct, static_parameter=self.db)
        stateFSM = State(self.miniFSM, static_parameter=self.db)
        stateM2 = State(self.drive_correct, static_parameter=self.db)
        stateM3 = State(self.drive_correct, static_parameter=self.db, ending=True)

        stateM1.add_transition(False, stateM1)
        stateM1.add_transition(True, stateFSM)
        stateFSM.add_transition(True, stateM2)
        stateM2.add_transition(False, stateM2)
        stateM2.add_transition(True, stateM3)

        mainFSM.add_states([stateM1, stateFSM, stateM2, stateM3])
        self.table = "NestDBTest"
        mainFSM.set_current_state(stateM1)
        mainFSM.run([model, obs, self.env, loop])

        self.signal_handler(SIGINT, False)

    def signal_handler(self, SIGINT, frame):
        self.env.close()
        print()
        print(self.db.print_everything(self.table))
        self.db.custom_query("DELETE from " + self.table + " WHERE time > 0")
        exit(SIGINT)

    def miniFSM(self, db, li):
        model, obs, env, loop = li
        subFSM = FSM()
        stateS1 = State(self.drive_incorrect, static_parameter=db)
        stateS2 = State(self.drive_incorrect, static_parameter=db, ending=True)

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

    def drive_correct(self, db, li):
        global REST_TIME
        model, obs, env, loop = li
        print(colored("corr", "green"), end='\r')
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        self.db_write(db, self.table, obs[0:2], "corrMain", "")
        time.sleep(REST_TIME)
        if not done:
            return False, [model, obs, env, loop]
        obs = env.reset()
        print()
        return True, [model, obs, env, loop]

    def drive_incorrect(self, db, li):
        global REST_TIME
        model, obs, env, loop = li
        if loop <= 0:
            return -1, li
        print(colored("incorr", "red"), end='\r')
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        self.db_write(db, self.table, obs[0:2], "incorrSub", loop)
        time.sleep(REST_TIME)
        if not done:
            return False, [model, obs, env, loop]
        obs = env.reset()
        print()
        return loop - 1, [model, obs, env, loop - 1]

    def db_write(self, db, table, info, tag, loop):
        x, y = info
        data = {
            "measurement": table,
            "tags": {
                "Source": tag,
                "Loop": loop
            },
            "time": datetime.datetime.now(),
            "fields": {
                "X-DistToGoal": x,
                "Y-DistToGoal": y
            }
        }
        db.insert([data])


class BabyDriverTest:
    def __init__(self):
        global REST_TIME
        self.db_exists = False
        signal(SIGINT, self.signal_handler)

        while True:
            tmpStr = "How much intervention? (the smaller the number, the more often it'll trigger)\n"
            try:
                self.level_of_intervention = int(input(tmpStr))
                break
            except:
                continue

        self.env = gym.make('Driving-v0')
        model = PPO("MlpPolicy", self.env).load("logs/final_model")
        obs = self.env.reset()

        fsm = FSM()
        fsm.create_database()
        self.db = fsm.get_database()
        self.db_exists = True

        steeringWheels = State(self.drive_correct)
        freeDrive = State(self.drive_incorrect)
        safetyCheck = State(self.parentalCheck, static_parameter=self.db)

        logic = Logic()
        logic.greater_than_limit(0)
        freeDrive.add_transition(logic, freeDrive)
        freeDrive.add_transition(Logic().less_than(1), safetyCheck)
        freeDrive.add_transition(True, freeDrive)
        safetyCheck.add_transition(True, freeDrive)
        safetyCheck.add_transition(False, steeringWheels)
        steeringWheels.add_transition(True, freeDrive)

        fsm.add_states([steeringWheels, freeDrive, safetyCheck])
        fsm.set_current_state(safetyCheck)
        self.table = "BabyDriverTest"
        fsm.run([model, obs, self.env, self.level_of_intervention])

        self.signal_handler(0, False)

    def signal_handler(self, sigint, frame):
        if not self.db_exists:
            exit(sigint)
        self.env.close()
        print()
        print(self.db.print_everything(self.table))
        self.db.custom_query("DELETE from " + self.table + " WHERE time > 0")
        exit(sigint)

    def drive_correct(self, li):    # run log pos
        global REST_TIME
        model, obs, env = li
        print(colored("Daddy", "green"), end='\r')
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        time.sleep(REST_TIME)
        print()
        if not done:
            return True, [model, obs, env, self.level_of_intervention]
        obs = env.reset()
        return True, [model, obs, env, self.level_of_intervention]

    def drive_incorrect(self, li):
        global REST_TIME
        model, obs, env, loi = li
        print(colored("Baby", "red"), end='\r')
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        time.sleep(REST_TIME)
        if not done:
            return loi-1, [model, obs, env, loi-1]
        obs = env.reset()
        print()
        return True, [model, obs, env, self.level_of_intervention]

    def log_position(self, db, info, tag):
        x, y = info
        data = {
            "measurement": self.table,
            "tags": {
                "Switching to": tag,
            },
            "time": datetime.datetime.now(),
            "fields": {
                "X-DistToGoal": x,
                "Y-DistToGoal": y
            }
        }
        db.insert([data])

    def parentalCheck(self, db, li):
        model, obs, env, loi = li
        new_x, new_y = obs[0:2]
        try:
            headers, data = db.get_latest_rows(self.table, 1)
        except DatabaseTableEmpty:
            self.log_position(db, [new_x, new_y], "Baby")
            return True, [model, obs, env, self.level_of_intervention]

        xc, yc = None, None
        for i, column in enumerate(headers):
            if column == "X-DistToGoal":
                xc = i
            elif column == "Y-DistToGoal":
                yc = i
        assert xc is not None and yc is not None, "Couldn't find the correct columns. Please check the names"

        old_x, old_y = None, None
        for i, column in enumerate(data[0]):
            if i == xc:
                old_x = column
            elif i == yc:
                old_y = column

        if BabyDriverTest.pythagoras(new_x, new_y) > BabyDriverTest.pythagoras(old_x, old_y):
            print()
            self.log_position(db, [new_x, new_y], "Daddy")
            return False, [model, obs, env]
        else:
            self.log_position(db, [new_x, new_y], "Baby")
            return True, [model, obs, env, self.level_of_intervention]

    @staticmethod
    def pythagoras(x, y):
        import math
        return math.sqrt(x**2+y**2)


if __name__ == "__main__":
    while True:
        try:
            print("Select test to run")
            print("1: NormalTest")
            print("2: DBTest")
            print("3: NestTest")
            print("4: NestDBTest")
            print("5: BabyDriverTest")
            inp = int(input())
        except ValueError:
            continue
        match inp:
            case 1:
                test = NormalTest()
                break
            case 2:
                test = DBTest()
                break
            case 3:
                test = NestTest()
                break
            case 4:
                test = NestDBTest()
                break
            case 5:
                test = BabyDriverTest()
                break
            case _:
                print("\nEnter a number from 1-5\n")


