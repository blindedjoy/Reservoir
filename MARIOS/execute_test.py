#!/bin/bash/python

import multiprocessing
import sys
import os

# necessary to add cwd to path when script run 
# by slurm (since it executes a copy)
sys.path.append(os.getcwd()) 

# get number of cpus available to job
try:
    ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
except KeyError:
    ncpus = multiprocessing.cpu_count()


from multiprocessing import set_start_method
from reservoir import *
from PyFiles.imports import *
from PyFiles.helpers import *
from PyFiles.experiment import *
from itertools import combinations
from random import randint



### Timing
import time
import timeit
class NoDaemonProcess(multiprocessing.Process):
      @property
      def daemon(self):
          return False

      @daemon.setter
      def daemon(self, value):
          pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool): #ThreadPool):#
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)
def run_experiment(inputs, n_cores = 2, cv_samples = 3):
      """
      4*4 = 16 + 

      The final form of the input dict is:

        inputs = {"target_frequency_" : ...
                  "obs_hz" : ...
                  "target_hz" : ...
                  "split" : ...
                  ""
                  }

      Reinier's example:
      {
        'leaking_rate' : (0, 1), 
        'spectral_radius': (0, 1.25),
        'regularization': (-12, 1),
        'connectivity': (-3, 0),
        'n_nodes':  (100, 1000)
      }

      """
      target_frequency_ = inputs["target_freq"]
      split_, obs_hz_, target_hz_ = inputs["split"], inputs["obs_hz"], inputs["target_hz"]


      experiment = EchoStateExperiment(size = "medium", 
                                       target_frequency = target_frequency_, 
                                       obs_hz = obs_hz_, 
                                       target_hz = target_hz_, 
                                       verbose = False)

      experiment.get_observers(method = "freq", split = split_, aspect = 0.9, plot_split = False)
      #(-12, 1)}
      #example cv args:
      cv_args = {
          'bounds' : inputs["bounds"],
          'initial_samples' : 100,
          'subsequence_length' : 250, #150 for 500
          'eps' : 1e-4,
          'cv_samples' : cv_samples, 
          'max_iterations' : 1000, 
          'scoring_method' : 'tanh',
          "n_jobs" : n_cores,
          "verbose" : True,
          "plot" : False,

      }
      experiment.RC_CV(cv_args = cv_args, model = "uniform")
      experiment.RC_CV(cv_args = cv_args, model = "exponential")


"""
things_2_combine = { 
    "obs500" :  500,
    "obs1k"  : 1000,
    "targ500" : 500,
    "targ1k" : 1000,
    "split0.5" : ,
}

lst_of_dicts = []
count = 0
for target_frequency_ in [2000, 4000]:
    for split_ in [0.5, 0.9]:
        for obs_hz_ in [ 10,  20]:
            for target_hz_ in [10, 20]:
                count += 1
                print( "set " + str(count) + ": { tf: " + str(target_frequency_) +
                       ", split: " + str(split_) + 
                       ", obs_hz: " + str(obs_hz_) + 
                       ", targ_hz: " + str(target_hz_) + "}")
                dict_spec_ = {
                              "target_freq" : target_frequency_, 
                              "split" : split_, 
                              "obs_hz" : obs_hz_,
                              "target_hz" : target_hz_
                              }
                lst_of_dicts += [dict_spec_]

#print(lst_of_dicts)

#for experiment_input in experiment_set:
#  run_experiment(experiment_input)

#parrallelized loop:
#Pool = multiprocessing.Pool(n_experiments)
#results = zip(*Pool.map(run_experiment, experiment_set))
"""

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
#https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic#:~:text=Pool%20(%20multiprocessing.,used%20for%20the%20worker%20processes.&text=The%20important%20parts%20are%20the,top%20and%20to%20call%20pool.


# THE FOLLOWING IS FROM 2011, needs to be updated: from the same stackoverflow:
"""
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
"""
  



def test():
    assert type(TEST) == bool
    if TEST == True:
      print("TEST")
      experiment_set = [
             {'target_freq': 2000, 'split': 0.5, 'obs_hz': 10, 'target_hz': 10},
             {'target_freq': 2000, 'split': 0.5, 'obs_hz': 10, 'target_hz': 20}]
      """
      experiment_set = [
           {'target_freq': 2000, 'split': 0.5, 'obs_hz': 10, 'target_hz': 10},
           {'target_freq': 2000, 'split': 0.5, 'obs_hz': 10, 'target_hz': 20},
           {'target_freq': 2000, 'split': 0.5, 'obs_hz': 10, 'target_hz': 20},
           {'target_freq': 2000, 'split': 0.5, 'obs_hz': 20, 'target_hz': 10},
           {'target_freq': 2000, 'split': 0.5, 'obs_hz': 20, 'target_hz': 20}, 
           {'target_freq': 2000, 'split': 0.9, 'obs_hz': 10, 'target_hz': 10}, 
           {'target_freq': 2000, 'split': 0.9, 'obs_hz': 10, 'target_hz': 20}, 
           {'target_freq': 2000, 'split': 0.9, 'obs_hz': 20, 'target_hz': 10}, 
           {'target_freq': 2000, 'split': 0.9, 'obs_hz': 20, 'target_hz': 20}, 
           {'target_freq': 4000, 'split': 0.5, 'obs_hz': 10, 'target_hz': 10}, 
           {'target_freq': 4000, 'split': 0.5, 'obs_hz': 10, 'target_hz': 20}, 
           {'target_freq': 4000, 'split': 0.5, 'obs_hz': 20, 'target_hz': 10}, 
           {'target_freq': 4000, 'split': 0.5, 'obs_hz': 20, 'target_hz': 20}, 
           {'target_freq': 4000, 'split': 0.9, 'obs_hz': 10, 'target_hz': 10}, 
           {'target_freq': 4000, 'split': 0.9, 'obs_hz': 10, 'target_hz': 20}, 
           {'target_freq': 4000, 'split': 0.9, 'obs_hz': 20, 'target_hz': 10}, 
           {'target_freq': 4000, 'split': 0.9, 'obs_hz': 20, 'target_hz': 20}]
      """
      print("Real Run")
      bounds = {
          'llambda' : (-12, 3), 
          'connectivity': (-3, 0), # 0.5888436553555889, 
          'n_nodes': 100,#(100, 1500),
          'spectral_radius': (0.05, 0.99),
          'regularization': (-10,-2)}
    else:
      bounds = {
                 #all are log scale except  spectral radius, leaking rate and n_nodes
                'llambda' : (-12, 3), 
                'connectivity': (-3, 0), # 0.5888436553555889, 
                'n_nodes': (100, 1500),
                'spectral_radius': (0.05, 0.99),
                'regularization': (-12, 1),
                "leaking_rate" : (0.05, 1) # we want some memory. 0 would mean no memory.
                # current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state
                }
      """
      experiment_set = [
            {'target_freq': 2000, 'split': 0.5, 'obs_hz': 500, 'target_hz': 500},
            {'target_freq': 2000, 'split': 0.5, 'obs_hz': 500, 'target_hz': 1000},
            {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1000, 'target_hz': 500},
            {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1000, 'target_hz': 1000},
            {'target_freq': 2000, 'split': 0.9, 'obs_hz': 500, 'target_hz': 500},
            {'target_freq': 2000, 'split': 0.9, 'obs_hz': 500, 'target_hz': 1000}, 
            {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1000, 'target_hz': 500}, 
            {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1000, 'target_hz': 1000}, 
            {'target_freq': 4000, 'split': 0.5, 'obs_hz': 500, 'target_hz': 500}, 
            {'target_freq': 4000, 'split': 0.5, 'obs_hz': 500, 'target_hz': 1000}, 
            {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1000, 'target_hz': 500}, 
            {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1000, 'target_hz': 1000}, 
            {'target_freq': 4000, 'split': 0.9, 'obs_hz': 500, 'target_hz': 500}, 
            {'target_freq': 4000, 'split': 0.9, 'obs_hz': 500, 'target_hz': 1000}, 
            {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1000, 'target_hz': 500}, 
            {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1000, 'target_hz': 1000}]
      """
      experiment_set = [
            {'target_freq': 2000, 'split': 0.5, 'obs_hz': 500, 'target_hz': 500},
            {'target_freq': 2000, 'split': 0.5, 'obs_hz': 500, 'target_hz': 1000},
            {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1000, 'target_hz': 500},
            {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1000, 'target_hz': 1000}]
      
      """
      
      """
      
    for experiment in experiment_set:
      experiment["bounds"] = bounds
    
    n_experiments = len(experiment_set)
    print("Creating " + str(n_experiments) + " (non-daemon) workers and jobs in main process.")
    try:
      set_start_method('forkserver')
    except RuntimeError:
      pass
    pool = MyPool(n_experiments)


    pool.map(run_experiment, experiment_set)#work, [randint(1, 5) for x in range(5)])

    pool.close()
    pool.join()
      #print(result)

#https://github.com/pytorch/pytorch/issues/3492
if __name__ == '__main__':
  

  
  
  

  #imports
  


  #set_start_method('forkserver')



  # 16 total experiments, 8 cores each --> 16 * 8 cores = 128 total cores. But first lets try some experiments.

  

  
  """
  
  """
  TEST = False #TODO: fix this so that it's a command line argument

  #set_start_method('spawn')#, force = True) # set_start_method('spawn'
  start = timeit.default_timer()
  test()
  stop = timeit.default_timer()
  print('Time: ', stop - start) 

 
#experiment.RC_CV(cv_args = cv_args, model = "hybrid")
