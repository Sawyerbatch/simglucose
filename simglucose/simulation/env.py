from simglucose.patient.t1dpatient import Action
from simglucose.analysis.risk import risk_index
import pandas as pd
from datetime import timedelta
import logging
from collections import namedtuple
from simglucose.simulation.rendering import Viewer
import gym
from gym import spaces
import numpy as np

# try:
# from rllab.envs.base import Step
# except ImportError:
_Step = namedtuple("Step", ["observation", "reward", "done", "info"])

def Step(observation, reward, done, **kwargs):
    """
    Convenience method creating a namedtuple with the results of the
    environment.step method.
    Put extra diagnostic info in the kwargs
    """
    return _Step(observation, reward, done, kwargs)


Observation = namedtuple('Observation', ['CGM'])#, 'dCGM']) # inserire derivata
logger = logging.getLogger(__name__)


def risk_diff(BG_last_hour):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
        return risk_prev - risk_current


class T1DSimEnv(object):
    def __init__(self, patient, sensor, pump, scenario):
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        # self._reset()
        # self.observation_space = spaces.Box(0, 4, shape=(1,), dtype=int)
        # self.action_space = spaces.Discrete(2)
        
    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action):
        # current action 
        patient_action = self.scenario.get_action(self.time) # stato interno del paziente che avanza
        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus
        CHO = patient_action.meal
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)
        
        # CGM_prev = self.sensor.measure(self.patient)
        
        # State update
        self.patient.step(patient_mdl_act) # avanzamento dello stato del paziente

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient) # CGM al tempo t+1
        # dCGM = CGM - CGM_prev
        # aggiungere valore per derivata?
        # print(CHO, insulin, BG, CGM, dCGM, '\n')
        return CHO, insulin, BG, CGM #dCGM # aggiungere valore per derivata?



    def step(self, action, reward_fun=risk_diff):
        '''
        action is a namedtuple with keys: basal, bolus
        '''
        CHO = 0.0
        insulin = 0.0
        BG = 0.0
        CGM = 0.0
        # dCGM = 0.0
        # CGM_old = 0.0
        # aggiungere valore per derivata
        
        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action) # tmp_dCGM
            # mini_step fornisce il delta dei valori
            CHO += tmp_CHO / self.sample_time
            insulin += tmp_insulin / self.sample_time
            BG += tmp_BG / self.sample_time
            # dCGM += tmp_dCGM / self.sample_time # aggiungere derivata
            # dCGM = tmp_CGM # self.sample_time
            CGM += tmp_CGM / self.sample_time
            # print('index', _, BG, CGM, dCGM)
        
        # dCGM = tmp_dCGM / self.sample_time
        dCGM = CGM - self.CGM_hist[-1]
            

        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)

        # Record current action
        self.CHO_hist.append(CHO)
        self.insulin_hist.append(insulin)

        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        self.dCGM_hist.append(dCGM) # aggiungere derivata?
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)

        # Compute reward, and decide whether game is over
        window_size = int(60 / self.sample_time)
        BG_last_hour = self.CGM_hist[-window_size:]
        reward = reward_fun(BG_last_hour)
        done = BG < 70 or BG > 350
        # obs = Observation(CGM=CGM, dCGM=dCGM) # aggiungere derivata
        obs = np.array(Observation([CGM, dCGM]))
        
        return Step(observation=obs,
                    reward=reward,
                    done=done,
                    sample_time=self.sample_time,
                    patient_name=self.patient.name,
                    meal=CHO,
                    patient_state=self.patient.state,
                    time=self.time,
                    bg=BG,
                    lbgi=LBGI,
                    hbgi=HBGI,
                    risk=risk)

    # def _reset(self):
    #     self.sample_time = self.sensor.sample_time
    #     self.viewer = None

    #     BG = self.patient.observation.Gsub
    #     horizon = 1
    #     LBGI, HBGI, risk = risk_index([BG], horizon)
    #     CGM = self.sensor.measure(self.patient)
    #     dCGM = 0.0
    #     self.time_hist = [self.scenario.start_time]
    #     self.BG_hist = [BG]
    #     self.CGM_hist = [CGM] 
    #     self.dCGM_hist = [dCGM] # aggiungere derivata?
    #     self.risk_hist = [risk]
    #     self.LBGI_hist = [LBGI]
    #     self.HBGI_hist = [HBGI]
    #     self.CHO_hist = []
    #     self.insulin_hist = []

    # def reset(self):
    #     self.patient.reset()
    #     self.sensor.reset()
    #     self.pump.reset()
    #     self.scenario.reset()
    #     self._reset()
    #     CGM = self.sensor.measure(self.patient)
    #     dCGM = 0.0
    #     obs = Observation(CGM=CGM, dCGM=dCGM) # aggiungere derivata
    #     return Step(observation=obs,
    #                 reward=0,
    #                 done=False,
    #                 sample_time=self.sample_time,
    #                 patient_name=self.patient.name,
    #                 meal=0,
    #                 patient_state=self.patient.state,
    #                 time=self.time,
    #                 bg=self.BG_hist[0],
    #                 lbgi=self.LBGI_hist[0],
    #                 hbgi=self.HBGI_hist[0],
    #                 risk=self.risk_hist[0])
    
    # for gym 0.21
    def reset(self):
        # print('sto usando reset')
        self.patient.reset()
        self.sensor.reset()
        self.pump.reset()
        self.scenario.reset()
        # self._reset()
              
        self.sample_time = self.sensor.sample_time
        self.viewer = None

        BG = self.patient.observation.Gsub
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)
        CGM = self.sensor.measure(self.patient)
        dCGM = 0.0
        self.time_hist = [self.scenario.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM] 
        self.dCGM_hist = [dCGM] # aggiungere derivata?
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.CHO_hist = []
        self.insulin_hist = []
        
        # self._agent_location = 1
        # observation = self._agent_location = 1
        
        CGM = self.sensor.measure(self.patient)
        dCGM = 0.0
        # obs = Observation(CGM=CGM, dCGM=dCGM)
        obs = np.array(Observation([CGM, dCGM])) # aggiungere derivata
        self.ritorno = Step(observation=obs,
                    reward=0,
                    done=False,
                    sample_time=self.sample_time,
                    patient_name=self.patient.name,
                    meal=0,
                    patient_state=self.patient.state,
                    time=self.time,
                    bg=self.BG_hist[0],
                    lbgi=self.LBGI_hist[0],
                    hbgi=self.HBGI_hist[0],
                    risk=self.risk_hist[0])
        # print(self.ritorno)
        # # print(type(obs))
        # print(obs.shape)
        
        return self.ritorno

    
    # def _get_obs(self):
    #     return self._agent_location
    
    def render(self, close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario.start_time, self.patient.name)

        self.viewer.render(self.show_history())

    def show_history(self):
        df = pd.DataFrame()
        df['Time'] = pd.Series(self.time_hist)
        df['BG'] = pd.Series(self.BG_hist)
        df['CGM'] = pd.Series(self.CGM_hist)
        df['dCGM'] = pd.Series(self.dCGM_hist)
        df['CHO'] = pd.Series(self.CHO_hist)
        df['insulin'] = pd.Series(self.insulin_hist)
        df['LBGI'] = pd.Series(self.LBGI_hist)
        df['HBGI'] = pd.Series(self.HBGI_hist)
        df['Risk'] = pd.Series(self.risk_hist)
        df = df.set_index('Time')
        return df

# cl = T1DSimEnv()
# print(cl.reset())