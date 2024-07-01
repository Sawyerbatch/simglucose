from simglucose.patient.t1dpatient import Action
from simglucose.analysis.risk import risk_index
import pandas as pd
from datetime import timedelta
import logging
from collections import namedtuple
from simglucose.simulation.rendering import Viewer
import numpy as np
try:
    from rllab.envs.base import Step
except ImportError:
    _Step = namedtuple("Step", ["observation", "reward", "done", "info"])

    def Step(observation, reward, done, **kwargs):
        """
        Convenience method creating a namedtuple with the results of the
        environment.step method.
        Put extra diagnostic info in the kwargs
        """
        return _Step(observation, reward, done, kwargs)


Observation = namedtuple("Observation", ["CGM", 'dCGM', 'IOB', 'h_zone', 'food'])
logger = logging.getLogger(__name__)


def risk_diff(BG_last_hour):
    # print('REWARD BASE')
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
        return risk_prev - risk_current


class T1DSimEnv_MARL(object):
    def __init__(self, patient, sensor, pump, scenario, reward_fun, training, folder, morty_cap, rick_cap): #n_steps,
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        self.reward_fun = reward_fun
        self.training = training
        self.folder = folder
        self.morty_cap = morty_cap
        self.rick_cap = rick_cap
        self._reset()
        self.N = 180 # durata dell'effetto dell'insulina (minuti)
        
    def a(self, k, N):
      if k > N:
        return 0
      else:
        return (N - k)/N
        
    def IOB_fun(self, t, Ins, N):
      IOB = 0
      N_min = np.minimum(N, len(Ins))
      for k in range(N_min):
          IOB += self.a(k,N)*Ins[t-k]
      return IOB

    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action):
        # current action
        patient_action = self.scenario.get_action(self.time)
        # print('patient action', patient_action)
        # print(self.scenario)
        basal = self.pump.basal(action.basal)
        bolus = self.pump.bolus(action.bolus)
        insulin = basal + bolus
        CHO = patient_action.meal
        # print('CHO', CHO)
        patient_mdl_act = Action(insulin=insulin, CHO=CHO)
        # print(self.patient)
        # State update
        self.patient.step(patient_mdl_act)

        # next observation
        BG = self.patient.observation.Gsub
        CGM = self.sensor.measure(self.patient)

        return CHO, insulin, BG, CGM
    

    def step(self, action, reward_fun=risk_diff):
        """
        action is a namedtuple with keys: basal, bolus
        """
        CHO = 0.0
        insulin = 0.0
        BG = 0.0
        CGM = 0.0

        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action)
            CHO += tmp_CHO / self.sample_time
            insulin += tmp_insulin / self.sample_time
            BG += tmp_BG / self.sample_time
            CGM += tmp_CGM / self.sample_time

        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)
        
        IOB = self.IOB_fun(0, self.insulin_hist, self.N)
        self.IOB = float(IOB)
        
        self.dCGM = CGM - self.CGM_hist[-1]
        
        self.h_zone = int(self.time.hour/2)
        
        if CHO > 0:
            self.food = True
        else:
            self.food = False

        # Record current action
        self.CHO_hist.append(CHO)
        self.insulin_hist.append(insulin)

        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)

        # Compute reward, and decide whether game is over
        window_size = int(60 / self.sample_time)
        BG_last_hour = self.CGM_hist[-window_size:]
        reward = reward_fun(BG_last_hour)
        # print('ACTION', action)
        # print('REWARD', reward)
        # if self.training:
        # done = BG < 40 or BG > 350
        done = BG < 40 or BG > 600
        # else:
        #     done = BG < 40 or BG > 600
        # print('DOOOOOOOOOOOOOOOOOOOOONEEEE', BG, done)
        obs = Observation(CGM=CGM,dCGM=self.dCGM, IOB=self.IOB, 
                          h_zone=self.h_zone, food=self.food)

        return Step(
            observation=obs,
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
            risk=risk,
            insulin=insulin
        )

    def _reset(self):
        self.sample_time = self.sensor.sample_time
        self.viewer = None
        

        BG = self.patient.observation.Gsub
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)
        CGM = self.sensor.measure(self.patient)
        self.time_hist = [self.scenario.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM]
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.CHO_hist = []
        self.insulin_hist = []

    def reset(self):
        self.patient.reset()
        self.sensor.reset()
        self.pump.reset()
        self.scenario.reset()
        self._reset()
        CGM = self.sensor.measure(self.patient)
        
        
        # MODIFICA obs aggiuntive
        self.dCGM = 0.0
        self.h_zone = int(self.scenario.start_time.hour/2)
        self.food = False
        self.CHO = 0.0
        self.IOB = 0.0
        
        obs = Observation(CGM=CGM, dCGM=self.dCGM, IOB=self.IOB, 
                          h_zone=self.h_zone, food=self.food)
        
        return Step(
            observation=obs,
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
            risk=self.risk_hist[0],
            
        )
    

    def render(self, close=False):
        if close:
            self._close_viewer()
            return

        if self.viewer is None:
            self.viewer = Viewer(self.scenario.start_time, self.patient.name)

        self.viewer.render(self.show_history())

    def _close_viewer(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def show_history(self):
        df = pd.DataFrame()
        df["Time"] = pd.Series(self.time_hist)
        df["BG"] = pd.Series(self.BG_hist)
        df["CGM"] = pd.Series(self.CGM_hist)
        df["CHO"] = pd.Series(self.CHO_hist)
        df["insulin"] = pd.Series(self.insulin_hist)
        df["LBGI"] = pd.Series(self.LBGI_hist)
        df["HBGI"] = pd.Series(self.HBGI_hist)
        df["Risk"] = pd.Series(self.risk_hist)
        df = df.set_index("Time")
        # print('DDDDFFFFFF', df)
        return df
