import logging
import time
import os
from datetime import datetime
date_time = str(datetime.now())[:19].replace(" ", "_" ).replace("-", "" ).replace(":", "" )



pathos = True
try:
    from pathos.multiprocessing import ProcessPool as Pool
except ImportError:
    print('You could install pathos to enable parallel simulation.')
    pathos = False

logger = logging.getLogger(__name__)


class SimObj(object):
    def __init__(self,
                 env,
                 controller,
                 sim_time,
                 animate=True,
                 path=None,
                 strategy=None):
        self.env = env
        self.controller = controller
        self.sim_time = sim_time
        self.animate = animate
        self._ctrller_kwargs = None
        self.path = path
        self.strategy = strategy

    def simulate(self):
        self.controller.reset()
        if self.strategy == 'PPO':
            obs = self.env.reset() # PPO
        else:
            obs, reward, done, info = self.env.reset() # BBC

        done = False
        reward = 0
        
        
        tic = time.time()
        while self.env.time < self.env.scenario.start_time + self.sim_time:
            if self.animate:
                self.env.render()
            if self.strategy == 'PPO':
                action = self.controller.policy(obs, reward, done) # PPO
            else:
                action = self.controller.policy(obs, reward, done, **info) # BBC

            obs, reward, done, info = self.env.step(action)
        toc = time.time()
        logger.info('Simulation took {} seconds.'.format(toc - tic))

    def results(self):
        return self.env.show_history()

    def save_results(self):
        df = self.results()
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        # filename = os.path.join(self.path, str(self.env.patient.name) + '.csv')
        filename_exc = os.path.join(self.path, str(self.env.patient.name) +'_'+self.strategy+'_'+date_time+'.xlsx')
        # df.to_csv(filename)
        df.to_excel(filename_exc)

    def reset(self):
        self.env.reset()
        self.controller.reset()


def sim(sim_object):
    print("Process ID: {}".format(os.getpid()))
    print('Simulation starts ...')
    sim_object.simulate()
    sim_object.save_results()
    print('Simulation Completed!')
    return sim_object.results()


def batch_sim(sim_instances, parallel=False):
    tic = time.time()
    if parallel and pathos:
        with Pool() as p:
            results = p.map(sim, sim_instances)
    else:
        if parallel and not pathos:
            print('Simulation is using single process even though parallel=True.')
        results = [sim(s) for s in sim_instances]
    toc = time.time()
    print('Simulation took {} sec.'.format(toc - tic))
    return results
