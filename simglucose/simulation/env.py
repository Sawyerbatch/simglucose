from simglucose.patient.t1dpatient import Action
from simglucose.analysis.risk import risk_index
from simglucose.actuator.pump import InsulinPump
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

N = 180 # durata dell'effetto dell'insulina (minuti)

# funzione che pesa sempre meno il contributo dell'insulina più k è alto, 
# ovvero più si va a misurare effetti risalenti a k minuti prima
# a(0 minuti prima) = 1, il massimo, mentre a(N minuti prima) = 0, dove N=180
def a(k, N):
  if k > N:
    return 0
  else:
    return (N - k)/N



def Step(observation, reward, done, **kwargs):
    """
    Convenience method creating a namedtuple with the results of the
    environment.step method.
    Put extra diagnostic info in the kwargs
    """
    return _Step(observation, reward, done, kwargs)


Observation = namedtuple('Observation', ['CGM'])#, 'dCGM']) # inserire derivata
logger = logging.getLogger(__name__)
# aggiungere allo spazio delle osservazioni zona oraria(h_zone[int da 1 a 12]) e alimentazione(food[bool])

def risk_diff(BG_last_hour):
    if len(BG_last_hour) < 2:
        return 0
    else:
        _, _, risk_current = risk_index([BG_last_hour[-1]], 1)
        _, _, risk_prev = risk_index([BG_last_hour[-2]], 1)
        return risk_prev - risk_current


class T1DSimEnv(object):
    def __init__(self, patient, sensor, pump, scenario, strategy):
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        self.strategy = strategy
        # self.reset()
        # self.env, _, _, _ = self.create_env_from_random_state(scenario)
        # # self._reset()
        # self.INSULIN_PUMP_HARDWARE = 'Insulet'
        # pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        # ub = self.env.pump._params['max_basal']
        # ub = 0.5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,5))
        
        self.action_space = spaces.Box(low=0., high=0.08, shape=(1,2))
        # self.action_space = spaces.Box(low=np.array([0.,0.]), high=np.array([ub,4.]), shape=(1,2))
        self.metadata = {'render.modes': ['human']}
        
    # definisco l'Insulin On Board (vedi anche eq. (3) paper "A New Glycemic closed-loop control based on Dyna-Q for Type-1-Diabetes" )
    # Ins: array in cui ad ogni indice corrisponde un minuto
    def IOB_fun(self, t, Ins, N):
      IOB = 0
      N_min = np.minimum(N, len(Ins))
      for k in range(N_min):
          IOB += a(k,N)*Ins[t-k]
      return IOB
  
    # def moving_average(self, lst, window_size):
    #     moving_average_list = []
    #     for i in range(len(lst)):
    #         if i + window_size <= len(lst):
    #             moving_average = sum(lst[i:i + window_size]) / window_size
    #         else:
    #             moving_average = sum(lst[i:]) / len(lst[i:])
    #         moving_average_list.append(moving_average)
    #     return moving_average_list
    
    # def moving_average_2(self, lst, window_size):
    #     moving_average_list = []
    #     for i in range(len(lst) - window_size + 1):
    #         moving_average = sum(lst[i:i + window_size]) / window_size
    #         moving_average_list.append(moving_average)
    #     return ([0.0]*(window_size-1)) + moving_average_list
    
    def moving_average(self, lst, window_size):
        moving_average_list = []
        window_len = min(len(lst), window_size)
        for i in range(window_len):
            moving_average_head = sum(lst[0:i+1]) / (i+1)
            moving_average_list.append(moving_average_head)
        # if window_size > len(lst):            
        for i in range(len(lst) - window_size):
            moving_average = sum(lst[i:i + window_size]) / window_size
            moving_average_list.append(moving_average)
        
        return moving_average_list
    
    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action):
        # current action 
        patient_action = self.scenario.get_action(self.time) # stato interno del paziente che avanza
        
        if self.strategy == 'PID':
            basal = self.pump.basal(action.basal)
            bolus = self.pump.bolus(action.bolus)
            # basal = self.pump.basal(action[0])
            # basal = self.pump.basal(action) # for moving average
            # bolus = self.pump.bolus(action[1])
            # insulin = basal
            insulin = basal + bolus
            CHO = patient_action.meal
            patient_mdl_act = Action(insulin=insulin, CHO=CHO)
        
        elif self.strategy == 'BBC':
            basal = self.pump.basal(action.basal)
            bolus = self.pump.bolus(action.bolus)
            # basal = self.pump.basal(action[0])
            # basal = self.pump.basal(action) # for moving average
            # bolus = self.pump.bolus(action[1])
            # insulin = basal
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
        # insulin = 0.0
        insulin = [0.0]
        BG = 0.0
        CGM = 0.0
        # dCGM = 0.0
        # CGM_old = 0.0
        # aggiungere valore per derivata
        
        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action) # BBC
            # tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action[0]) # PPO
            
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
        h_zone = int(self.time.hour/2)
        if CHO > 0:
            food = True
        else:
            food = False
                   
        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)

        # Record current action
        self.CHO_hist.append(CHO)
        self.insulin_hist.append(insulin)
        
        # last hour insulin
        if len(self.insulin_hist) >= 480: 
            insulin_integral = np.sum(self.insulin_hist[-480:])
        else:
            insulin_integral = np.sum(self.insulin_hist)
            
        self.insulin_24h.append(insulin_integral)
        
        # insulin_array = np.array(self.insulin_hist, dtype=object)
        # self.insulin_array = np.array(self.insulin_hist)
        IOB = self.IOB_fun(0, self.insulin_hist, N)
        IOB = float(IOB)
        # IOB = self.IOB_fun(0, self.insulin_array, N)
        # IOB = self.IOB_fun(0, insulin_array, N) # è un array ma serve un float
        difference = (self.time - self.scenario.start_time).total_seconds()
        minutes, _ = divmod(difference, 60)
        # print('vvvvvvvvvvvvvvvvvvvvvvvv')
        print('insulin:',insulin)
        print(minutes)
        # print(IOB)
        # if minutes > 2:
        #     IOB = float(IOB[int(minutes)])
        # else:
        #     IOB = 0.0
        # IOB = 0.0
        
        # media mobile insulina
        # df_insulina = pd.DataFrame(self.insulin_hist, columns=['insulina'])
        # if len insulina >= 30:
        #     ins_mean = df_insulina.rolling(30).mean()
        # self.ins_mean_hist.append(ins_mean)
        self.IOB_hist.append(IOB)
        # print(ins_mean)
        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        
        self.dCGM_hist.append(dCGM) # aggiungere derivata?
        self.h_zone_hist.append(h_zone)
        self.food_hist.append(food)
        
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)
        

        # Compute reward, and decide whether game is over
        window_size = int(60 / self.sample_time)
        BG_last_hour = self.CGM_hist[-window_size:]
        reward = reward_fun(BG_last_hour)
        done = BG < 70 or BG > 350
        # obs = Observation(CGM=CGM, dCGM=dCGM) # aggiungere derivata
        # obs = np.array(Observation([CGM, dCGM, h_zone, food, IOB]))
        
        # obs = Observation([CGM, dCGM])
        obs = Observation(CGM=CGM)
        print(obs)
        
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
        h_zone = int(self.scenario.start_time.hour/2)
        food = False
        CHO = 0.0
        IOB = 0.0
        # insulin = 0.0
        insulin = np.array([0.0])
        # ins_mean = 0.0
        ins_mean = np.array([0.0])
        

        self.time_hist = [self.scenario.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM] 
        
        self.dCGM_hist = [dCGM] # aggiungere derivata?
        self.h_zone_hist = [h_zone]
        self.food_hist = [food]
        
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.CHO_hist = [CHO]
        self.insulin_hist = [insulin]
        self.insulin_24h = [insulin]
        self.ins_mean_hist = [ins_mean]
        self.IOB_hist = [IOB]
        
        
        # self._agent_location = 1
        # observation = self._agent_location = 1
        
        CGM = self.sensor.measure(self.patient)
        dCGM = 0.0
        # obs = Observation(CGM=CGM, dCGM=dCGM)
        # obs = np.array(Observation([CGM, dCGM, h_zone, food, IOB])) # aggiungere derivata
        # obs = Observation([CGM, dCGM])
        obs = Observation(CGM=CGM)
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
        print(self.ritorno)
        # # print(type(obs))
        # print(obs.shape)
        
        return self.ritorno #BBC
        # return obs #PPO
    
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
        df['h_zone'] = pd.Series(self.h_zone_hist)
        df['food'] = pd.Series(self.food_hist)
        df['CHO'] = pd.Series(self.CHO_hist)
        df['insulin'] = pd.Series(self.insulin_hist)
        df['insulin_integral'] = pd.Series(self.insulin_24h)
        window = 30
        df['ins_mean'] = pd.Series(self.moving_average(self.insulin_hist, window))
        df['LBGI'] = pd.Series(self.LBGI_hist)
        df['HBGI'] = pd.Series(self.HBGI_hist)
        df['Risk'] = pd.Series(self.risk_hist)
        df = df.set_index('Time')
        return df
    
    

class PPOSimEnv(object):
    def __init__(self, patient, sensor, pump, scenario):
        self.patient = patient
        self.sensor = sensor
        self.pump = pump
        self.scenario = scenario
        
        # self.env, _, _, _ = self.create_env_from_random_state(scenario)
        # # self._reset()
        # self.INSULIN_PUMP_HARDWARE = 'Insulet'
        # pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        # ub = self.env.pump._params['max_basal']
        # ub = 0.5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,5))
        
        # cap di accordo con il paziente
        df_cap = pd.read_excel('C:\\GitHub\simglucose\Simulazioni_RL\Risultati\Strategy\paz_cap.xlsx', index_col=None)
        # df_strategy = pd.read_excel('C:\\GitHub\simglucose\Simulazioni_RL\Risultati\Strategy\strategy.xlsx')
        self.paziente = df_cap['paziente'][0]
        # print(self.patient)
        cap = df_cap.loc[df_cap['paziente']==self.paziente].iloc[:,1]
        cap = cap.iloc[0]
        # cap = cap.iloc
        print('\ncap insulina per paziente '+self.paziente+': '+str(cap))
        self.action_space = spaces.Box(low=0., high=cap, shape=(1,2))
        
        # cap statico
        # self.action_space = spaces.Box(low=0., high=0.08, shape=(1,2))
        # self.action_space = spaces.Box(low=np.array([0.,0.]), high=np.array([ub,4.]), shape=(1,2))
        self.metadata = {'render.modes': ['human']}
        
    # definisco l'Insulin On Board (vedi anche eq. (3) paper "A New Glycemic closed-loop control based on Dyna-Q for Type-1-Diabetes" )
    # Ins: array in cui ad ogni indice corrisponde un minuto
    def IOB_fun(self, t, Ins, N):
      IOB = 0
      N_min = np.minimum(N, len(Ins))
      for k in range(N_min):
          IOB += a(k,N)*Ins[t-k]
      return IOB
  
    # def moving_average(self, lst, window_size):
    #     moving_average_list = []
    #     for i in range(len(lst)):
    #         if i + window_size <= len(lst):
    #             moving_average = sum(lst[i:i + window_size]) / window_size
    #         else:
    #             moving_average = sum(lst[i:]) / len(lst[i:])
    #         moving_average_list.append(moving_average)
    #     return moving_average_list
    
    # def moving_average_2(self, lst, window_size):
    #     moving_average_list = []
    #     for i in range(len(lst) - window_size + 1):
    #         moving_average = sum(lst[i:i + window_size]) / window_size
    #         moving_average_list.append(moving_average)
    #     return ([0.0]*(window_size-1)) + moving_average_list
    
    def moving_average(self, lst, window_size):
        moving_average_list = []
        window_len = min(len(lst), window_size)
        for i in range(window_len):
            moving_average_head = sum(lst[0:i+1]) / (i+1)
            moving_average_list.append(moving_average_head)
        # if window_size > len(lst):            
        for i in range(len(lst) - window_size):
            moving_average = sum(lst[i:i + window_size]) / window_size
            moving_average_list.append(moving_average)
        
        return moving_average_list
    
    @property
    def time(self):
        return self.scenario.start_time + timedelta(minutes=self.patient.t)

    def mini_step(self, action):
        # current action 
        patient_action = self.scenario.get_action(self.time) # stato interno del paziente che avanza
        # basal = self.pump.basal(action.basal)
        # bolus = self.pump.bolus(action.bolus)
        # basal = self.pump.basal(action[0])
        basal = self.pump.basal(action) # for moving average
        # bolus = self.pump.bolus(action[1])
        insulin = basal
        # insulin = basal + bolus
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
        # insulin = 0.0
        insulin = [0.0]
        BG = 0.0
        CGM = 0.0
        # dCGM = 0.0
        # CGM_old = 0.0
        # aggiungere valore per derivata
        
        for _ in range(int(self.sample_time)):
            # Compute moving average as the sample measurements
            # tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action) # BBC
            tmp_CHO, tmp_insulin, tmp_BG, tmp_CGM = self.mini_step(action[0]) # PPO
            
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
        h_zone = int(self.time.hour/2)
        if CHO > 0:
            food = True
        else:
            food = False
                   
        # Compute risk index
        horizon = 1
        LBGI, HBGI, risk = risk_index([BG], horizon)

        # Record current action
        self.CHO_hist.append(CHO)
        self.insulin_hist.append(insulin)
        
        # last hour insulin
        if len(self.insulin_hist) >= 480: 
            insulin_integral = np.sum(self.insulin_hist[-480:])
        else:
            insulin_integral = np.sum(self.insulin_hist)
            
        self.insulin_24h.append([insulin_integral])
        
        # BB insulin
        bb_ins_df = pd.read_csv('C:\\GitHub\simglucose\Simulazioni_RL\Risultati\Strategy\\vpatient_params.csv')
        
        u2ss = bb_ins_df.loc[bb_ins_df['Name']==self.paziente].iloc[:,16]   # unit: pmol/(L*kg)
        BW = bb_ins_df.loc[bb_ins_df['Name']==self.paziente].iloc[:,58]   # unit: kg
        basal = u2ss * BW / 6000  # unit: U/min
        
        self.insulin_BB.append([basal])
        
        # last hour insulin BB
        if len(self.insulin_BB) >= 480: 
            insulin_BB_integral = np.sum(self.insulin_BB[-480:])
        else:
            insulin_BB_integral = np.sum(self.insulin_BB)
            
        self.insulin_BB_24h.append([insulin_BB_integral])

        # insulin_array = np.array(self.insulin_hist, dtype=object)
        # self.insulin_array = np.array(self.insulin_hist)
        IOB = self.IOB_fun(0, self.insulin_hist, N)
        IOB = float(IOB)
        # IOB = self.IOB_fun(0, self.insulin_array, N)
        # IOB = self.IOB_fun(0, insulin_array, N) # è un array ma serve un float
        difference = (self.time - self.scenario.start_time).total_seconds()
        minutes, _ = divmod(difference, 60)
        # print('vvvvvvvvvvvvvvvvvvvvvvvv')
        print('insulin:',insulin)
        print(minutes)
        # print(IOB)
        # if minutes > 2:
        #     IOB = float(IOB[int(minutes)])
        # else:
        #     IOB = 0.0
        # IOB = 0.0
        
        # media mobile insulina
        # df_insulina = pd.DataFrame(self.insulin_hist, columns=['insulina'])
        # if len insulina >= 30:
        #     ins_mean = df_insulina.rolling(30).mean()
        # self.ins_mean_hist.append(ins_mean)
        self.IOB_hist.append(IOB)
        # print(ins_mean)
        # Record next observation
        self.time_hist.append(self.time)
        self.BG_hist.append(BG)
        self.CGM_hist.append(CGM)
        
        self.dCGM_hist.append(dCGM) # aggiungere derivata?
        self.h_zone_hist.append(h_zone)
        self.food_hist.append(food)
        
        self.risk_hist.append(risk)
        self.LBGI_hist.append(LBGI)
        self.HBGI_hist.append(HBGI)
        

        # Compute reward, and decide whether game is over
        window_size = int(60 / self.sample_time)
        BG_last_hour = self.CGM_hist[-window_size:]
        reward = reward_fun(BG_last_hour)
        done = BG < 70 or BG > 350
        # obs = Observation(CGM=CGM, dCGM=dCGM) # aggiungere derivata
        obs = np.array(Observation([CGM, dCGM, h_zone, food, IOB]))
        print(obs)
        # obs = Observation([CGM, dCGM])
        
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
        h_zone = int(self.scenario.start_time.hour/2)
        food = False
        CHO = 0.0
        IOB = 0.0
        # insulin = 0.0
        insulin = np.array([0.0])
        # ins_mean = 0.0
        ins_mean = np.array([0.0])
        

        self.time_hist = [self.scenario.start_time]
        self.BG_hist = [BG]
        self.CGM_hist = [CGM] 
        
        self.dCGM_hist = [dCGM] # aggiungere derivata?
        self.h_zone_hist = [h_zone]
        self.food_hist = [food]
        
        self.risk_hist = [risk]
        self.LBGI_hist = [LBGI]
        self.HBGI_hist = [HBGI]
        self.CHO_hist = [CHO]
        self.insulin_hist = [insulin]
        self.insulin_24h = [insulin]
        self.insulin_BB = [insulin]
        self.insulin_BB_24h = [insulin]
        self.ins_mean_hist = [ins_mean]
        self.IOB_hist = [IOB]
        
        
        # self._agent_location = 1
        # observation = self._agent_location = 1
        
        CGM = self.sensor.measure(self.patient)
        dCGM = 0.0
        # obs = Observation(CGM=CGM, dCGM=dCGM)
        obs = np.array(Observation([CGM, dCGM, h_zone, food, IOB])) # aggiungere derivata
        # obs = Observation([CGM, dCGM])
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
        print(self.ritorno)
        # # print(type(obs))
        # print(obs.shape)
        
        # return self.ritorno #BBC
        return obs #PPO
    
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
        df['h_zone'] = pd.Series(self.h_zone_hist)
        df['food'] = pd.Series(self.food_hist)
        df['CHO'] = pd.Series(self.CHO_hist)
        df['insulin'] = pd.Series(self.insulin_hist)
        window = 30
        df['insulin_integral'] = pd.Series(self.insulin_24h)
        df['insulin_BB'] = pd.Series(self.insulin_BB)
        df['insulin_BB_integral'] = pd.Series(self.insulin_BB_24h)
        df['ins_mean'] = pd.Series(self.moving_average(self.insulin_hist, window))
        df['LBGI'] = pd.Series(self.LBGI_hist)
        df['HBGI'] = pd.Series(self.HBGI_hist)
        df['Risk'] = pd.Series(self.risk_hist)
        df = df.set_index('Time')
        return df