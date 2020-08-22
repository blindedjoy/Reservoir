#!/bin/bash/env python3

from itertools import combinations
import multiprocessing
from multiprocessing import set_start_method
import os
from PyFiles.experiment import *
from PyFiles.helpers import *
from PyFiles.imports import *
from random import randint
from reservoir import *
import sys
import time
import timeit


# necessary to add cwd to path when script run 
# by slurm (since it executes a copy)
sys.path.append(os.getcwd()) 

# get number of cpus available to job

try:
    ncpus = os.environ["SLURM_JOB_CPUS_PER_NODE"]
except KeyError:
    ncpus = multiprocessing.cpu_count()

experiment_specification = int(sys.argv[1])


accept_Specs = list(range(10))#[1, 2, 3, 4, 5, 100, 200, 300, 400, 500]


assert experiment_specification in accept_Specs




### Timing

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
def run_experiment(inputs, n_cores = int(sys.argv[2]), cv_samples = 5, size = "medium"):
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


      experiment = EchoStateExperiment(size = size, 
                                       target_frequency = target_frequency_, 
                                       obs_hz = obs_hz_, 
                                       target_hz = target_hz_, 
                                       verbose = False)

      experiment.get_observers(method = "freq", split = split_, aspect = 0.9, plot_split = False)
      
      #default arguments
      if size == "medium":
        default_presets = {
          "cv_samples" : 5,
          "max_iterations" : 4000,
          "eps" : 1e-5,
          'subsequence_length' : 250,
          "initial_samples" : 100}

      elif size == "publish":
        default_presets = {
          "cv_samples" : 5,
          "max_iterations" : 2000,
          "eps" : 1e-4,
          'subsequence_length' : 500,
          "initial_samples" : 200}

      cv_args = {
          'bounds' : inputs["bounds"],
          'scoring_method' : 'tanh',
          "n_jobs" : n_cores,
          "verbose" : True,
          "plot" : False, 
          **default_presets
      }

      for model_ in ["exponential", "uniform"]: #hybrid
        experiment.RC_CV(cv_args = cv_args, model = model_)

def test(TEST, multiprocessing = False):
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
           print("Real Run")"""
      bounds = {
          #'noise' : (-2, -4),
          'llambda' : (-3, -1), 
          'connectivity': (-3, 0), # 0.5888436553555889, 
          'n_nodes': 1000,#(100, 1500),
          'spectral_radius': (0.05, 0.99),
          'regularization': (-10,-2)}
      
      
    else:
      bounds = { #noise hyper-parameter.
                 #all are log scale except  spectral radius, leaking rate and n_nodes
                'noise' :          (-2, -4),
                'llambda' :        (-3, -1), 
                'connectivity':    (-3, 0),       # 0.5888436553555889, 
                'n_nodes':         1000,          #(100, 1500),
                'spectral_radius': (0.01, 0.99),
                'regularization':  (-3, 3),#(-12, 1),
                "leaking_rate" :   (0.01, 1) # we want some memory. 0 would mean no memory.
                # current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state
                }
      experiment_set = [  #4k, 0.5 filling in some gaps:

                          {'target_freq': 2000, 'split': 0.5, 'obs_hz': 750, 'target_hz': 500} ,
                          {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1250, 'target_hz': 500},
                          {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1500, 'target_hz': 500},
                          {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1750, 'target_hz': 500},
                          {'target_freq': 2000, 'split': 0.5, 'obs_hz': 2000, 'target_hz': 500},
                          {'target_freq': 2000, 'split': 0.5, 'obs_hz': 750, 'target_hz': 750} ,
                          {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1250, 'target_hz': 750},
                          {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1500, 'target_hz': 750},
                          {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1750, 'target_hz': 750},
                          {'target_freq': 2000, 'split': 0.5, 'obs_hz': 2000, 'target_hz': 750}
              
                          ]
      
    for experiment in experiment_set:
      experiment["bounds"] = bounds

    try:
      set_start_method('forkserver')
    except RuntimeError:
      pass
    
    n_experiments = len(experiment_set)
    exper_ = [experiment_set[experiment_specification]]

    print("Creating " + str(n_experiments) + " (non-daemon) workers and jobs in main process.")

    pool = MyPool(n_experiments)
    pool.map(run_experiment, exper_)#work, [randint(1, 5) for x in range(5)])
    pool.close()
    pool.join()


if __name__ == '__main__':

  print("Total cpus available: " + str(ncpus))
  print("RUNNING EXPERIMENT " + str(experiment_specification))

  TEST = False

  
  start = timeit.default_timer()
  test(TEST = TEST)
  stop = timeit.default_timer()
  print('Time: ', stop - start) 

 

""" ##################################### VESTIGAL CODE BELOW
  #https://github.com/pytorch/pytorch/issues/3492:
          set_start_method('spawn')#, force = True), set_start_method('forkserver')
      if experiment_specification == 1:
        
        experiment_set = [  #4k, 0.5 filling in some gaps:
                          {'target_freq': 4000, 'split': 0.5, 'target_hz': 1000, 'obs_hz': 500},
                          {'target_freq': 4000, 'split': 0.5, 'target_hz': 1500, 'obs_hz': 1000},
                          {'target_freq': 2000, 'split': 0.9, 'target_hz': 1250, 'obs_hz': 500},
                          {'target_freq': 4000, 'split': 0.5, 'target_hz': 1250, 'obs_hz': 500},
                          {'target_freq': 2000, 'split': 0.9, 'target_hz': 1250, 'obs_hz': 1000},
                          {'target_freq': 2000, 'split': 0.5, 'target_hz': 1250, 'obs_hz': 500},
                          {'target_freq': 2000, 'split': 0.5, 'target_hz': 1250, 'obs_hz': 1000},
                          {'target_freq': 4000, 'split': 0.9, 'target_hz': 1500, 'obs_hz': 500},
                          {'target_freq': 4000, 'split': 0.9, 'target_hz': 1500, 'obs_hz': 1000}
              
                          ]
      elif experiment_specification == 2: 
        # for 2k lets add some 750 target hz.
        experiment_set = [  #4k, 0.5 filling in some more gaps:
                          ]
      """