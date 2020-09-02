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

PREDICTION_TYPE = "column"


# necessary to add cwd to path when script run 
# by slurm (since it executes a copy)


sys.path.append(os.getcwd()) 

# get number of cpus available to job

try:
    ncpus = os.environ["SLURM_JOB_CPUS_PER_NODE"]
except KeyError:
    ncpus = multiprocessing.cpu_count()

experiment_specification = int(sys.argv[1])


accept_Specs = list(range(100))#[1, 2, 3, 4, 5, 100, 200, 300, 400, 500]


assert experiment_specification in accept_Specs

def liang_idx_convert(lb, ub, small = True):
    if small:
      lb = lb // 2
      ub = ub // 2
    idx_list = list(range(lb, ub + 1))
    return idx_list


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

def run_experiment(inputs, n_cores = int(sys.argv[2]), cv_samples = 5, size = "small",
                   interpolation_method = "griddata-linear"):
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
      #default arguments



      if "librosa" in inputs:
        default_presets = {
          "cv_samples" : 8,
          "max_iterations" : 3000,
          "eps" : 1e-8,
          'subsequence_length' : 250,
          "initial_samples" : 1000}
        librosa_args = { "spectrogram_path": inputs["spectrogram_path"],
                         "librosa": inputs["librosa"],
                         "spectrogram_type": inputs["spectrogram_type"]
                         }
      else:
        librosa_args = {}

      EchoArgs = { "size"    : size, 
                   "verbose" : False}

     

      

      if PREDICTION_TYPE == "column":
        train_time_idx, test_time_idx = inputs["train_time_idx"], inputs["test_time_idx"]
        
        experiment = EchoStateExperiment(size = size, 
                                         target_frequency = None,#target_frequency_, 
                                         verbose = False,
                                         prediction_type = PREDICTION_TYPE,
                                         train_time_idx = train_time_idx,
                                         test_time_idx = test_time_idx,
                                         **librosa_args)
        print("INITIALIZED")
        experiment.get_observers(method = "exact", split = split_, plot_split = False)
        default_presets["subsequence_length"] = 5

      elif  PREDICTION_TYPE == "block":
        split_  = inputs["split"]

        if "obs_freqs" in inputs:
          AddEchoArgs = {"obs_freqs" : inputs["obs_freqs"],
                         "target_freqs" : inputs["target_freqs"],
                         "prediction_type" : PREDICTION_TYPE
                        }
          EchoArgs = Merge(EchoArgs, AddEchoArgs)

        else:
          AddEchoArgs = {
                      "target_frequency" : inputs["target_freq"],
                      "obs_hz" : inputs["obs_hz"],
                      "target_hz" : inputs["target_hz"]
                      }

          EchoArgs = Merge(EchoArgs, AddEchoArgs)
        
        
        #obs_hz_, target_hz_ = inputs["obs_hz"], inputs["target_hz"]
        experiment = EchoStateExperiment( **EchoArgs, **librosa_args)
        if "obs_freqs" in inputs:
          experiment.get_observers(method = "exact", split = split_, aspect = 0.9, plot_split = False)
        else:
          experiment.get_observers(method = "freq", split = split_, aspect = 0.9, plot_split = False)
      


      if size == "small":
        default_presets = {
          "cv_samples" : 6,
          "max_iterations" : 1000,
          "eps" : 1e-5,
          'subsequence_length' : 180,
          "initial_samples" : 100}
      elif size == "medium":
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
      if TEACHER_FORCING:
        cv_args = Merge(cv_args, {"esn_feedback" : True})

      models = ["exponential", "uniform"] if PREDICTION_TYPE == "block" else ["uniform"]

      for model_ in models:#["exponential", "uniform"]: #hybrid
        experiment.RC_CV(cv_args = cv_args, model = model_)

def test(TEST, multiprocessing = False):
    assert type(TEST) == bool
    if TEST == True:
      print("TEST")

      def get_frequencies(trial = 1):
        """
        get frequency lists
        """
        if trial == 1:
            lb_targ, ub_targ, obs_hz  = 210, 560, int(320 / 2)
        elif trial == 2:
            lb_targ, ub_targ, obs_hz  = 340, 640, 280
        elif trial == 3:
            lb_targ, ub_targ, obs_hz  = 340, 350, 40


        obs_list = list(range(lb_targ-obs_hz, lb_targ, 10))
        obs_list += list(range(ub_targ, ub_targ + obs_hz, 10))
        resp_list = list(range(lb_targ, ub_targ, 10))

        return obs_list, resp_list

      obs_freqs, resp_freqs = get_frequencies(1)
      obs_freqs2, resp_freqs2 = get_frequencies(2)
      obs_freqs3, resp_freqs3 = get_frequencies(3)
      print(obs_freqs)


      if PREDICTION_TYPE == "block":
        librosa_args = {"spectrogram_path" : "custom",
                        "spectrogram_type"  : "power",#"db", #power
                        "librosa": True}
        

        experiment_set = [
               { 'split': 0.9, "obs_freqs": obs_freqs2, "target_freqs": resp_freqs2 },
               { 'split': 0.9, "obs_freqs": obs_freqs, "target_freqs": resp_freqs},
               { 'split': 0.5, "obs_freqs": obs_freqs2, "target_freqs": resp_freqs2 },
               #{'target_freq': 250, 'split': 0.5, 'obs_hz': 25, 'target_hz': 50},
               {'split': 0.5, "obs_freqs": obs_freqs, "target_freqs": resp_freqs}]

        #experiment_set = [
        #      { 'split': 0.9, "obs_freqs": obs_freqs3, "target_freqs": resp_freqs3 },
        #       #{'target_freq': 250, 'split': 0.5, 'obs_hz': 25, 'target_hz': 50},
        #       {'split': 0.5, "obs_freqs": obs_freqs3, "target_freqs": resp_freqs3}]
              
               #{'target_freq': 250, 'split': 0.5, 'obs_hz': 100, 'target_hz': 20},
               #{'target_freq': 250, 'split': 0.5, 'obs_hz': 25, 'target_hz': 50},
               #{'target_freq': 500, 'split': 0.5, 'obs_hz': 50, 'target_hz': 25}]

        experiment_set = [ Merge(experiment, librosa_args) for experiment in experiment_set]


      else:
        

        test1 = liang_idx_convert(250, 259)
        train1  = liang_idx_convert(240, 249)

        test2 = liang_idx_convert(514, 523)
        train2 = liang_idx_convert(504, 513)


        experiment_set = [
                          {'target_freq': 2000, 'split': 0.5, 'train_time_idx': train1 , 'test_time_idx': test1},
                          {'target_freq': 2000, 'split': 0.5, 'train_time_idx': train2, 'test_time_idx':  test2},
                          {'target_freq': 2000, 'split': 0.9, 'train_time_idx': train1 , 'test_time_idx': test1},
                          {'target_freq': 2000, 'split': 0.9, 'train_time_idx': train2, 'test_time_idx':  test2}
                          ]
      hi = """
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
           print("Real Run")
      """
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
      hi = """ experiment_set = [  #4k, 0.5 filling in some gaps:

                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 750, 'target_hz': 1000} ,
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1250, 'target_hz': 1000},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1500, 'target_hz': 1000},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1750, 'target_hz': 1000},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 2000, 'target_hz': 1000},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1250, 'target_hz': 1250},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1500, 'target_hz': 1250},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1750, 'target_hz': 1250},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 2000, 'target_hz': 1250},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 750, 'target_hz': 1500} ,
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1250, 'target_hz': 1500},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1500, 'target_hz': 1500},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1750, 'target_hz': 1500},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 2000, 'target_hz': 1500},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 500, 'target_hz': 1750} ,
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 750, 'target_hz': 1750} ,
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1000, 'target_hz': 1750},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1250, 'target_hz': 1750},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1500, 'target_hz': 1750},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1750, 'target_hz': 1750},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 2000, 'target_hz': 1750},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 500, 'target_hz': 2000} ,
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 750, 'target_hz': 2000} ,
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1500, 'target_hz': 2000},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 1750, 'target_hz': 2000},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 2000, 'target_hz': 2000},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 750, 'target_hz': 1000} ,
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1250, 'target_hz': 1000},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1500, 'target_hz': 1000},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1750, 'target_hz': 1000},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 2000, 'target_hz': 1000},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1500, 'target_hz': 1250},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1750, 'target_hz': 1250},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 2000, 'target_hz': 1250},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 750, 'target_hz': 1500} ,
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1250, 'target_hz': 1500},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1500, 'target_hz': 1500},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1750, 'target_hz': 1500},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 2000, 'target_hz': 1500},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1250, 'target_hz': 1750},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1500, 'target_hz': 1750},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1750, 'target_hz': 1750},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 2000, 'target_hz': 1750},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 500, 'target_hz': 2000} ,
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 750, 'target_hz': 2000} ,
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1000, 'target_hz': 2000},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1250, 'target_hz': 2000},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1500, 'target_hz': 2000},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 1750, 'target_hz': 2000},
                        {'target_freq': 2000, 'split': 0.9, 'obs_hz': 2000, 'target_hz': 2000},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 750, 'target_hz': 1250} ,
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1250, 'target_hz': 1250},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1500, 'target_hz': 1250},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1750, 'target_hz': 1250},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 2000, 'target_hz': 1250},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 750, 'target_hz': 1500} ,
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1250, 'target_hz': 1500},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1500, 'target_hz': 1500},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1750, 'target_hz': 1500},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 2000, 'target_hz': 1500},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 500, 'target_hz': 1750} ,
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 750, 'target_hz': 1750} ,
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1000, 'target_hz': 1750},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1250, 'target_hz': 1750},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1500, 'target_hz': 1750},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1750, 'target_hz': 1750},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 2000, 'target_hz': 1750},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 750, 'target_hz': 2000} ,
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1250, 'target_hz': 2000},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1500, 'target_hz': 2000},
                        {'target_freq': 4000, 'split': 0.5, 'obs_hz': 1750, 'target_hz': 2000},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 750, 'target_hz': 1250} ,
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1250, 'target_hz': 1250},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1500, 'target_hz': 1250},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1750, 'target_hz': 1250},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 2000, 'target_hz': 1250},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 750, 'target_hz': 1500} ,
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1250, 'target_hz': 1500},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1500, 'target_hz': 1500},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1750, 'target_hz': 1500},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 2000, 'target_hz': 1500},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 500, 'target_hz': 1750} ,
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 750, 'target_hz': 1750} ,
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1000, 'target_hz': 1750},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1250, 'target_hz': 1750},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1500, 'target_hz': 1750},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1750, 'target_hz': 1750},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 2000, 'target_hz': 1750},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 500, 'target_hz': 2000} ,
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 750, 'target_hz': 2000} ,
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1000, 'target_hz': 2000},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1250, 'target_hz': 2000},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1500, 'target_hz': 2000},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 1750, 'target_hz': 2000},
                        {'target_freq': 4000, 'split': 0.9, 'obs_hz': 2000, 'target_hz': 2000}
                        ]"""
      experiment_set = [  #4k, 0.5 filling in some gaps:
                          {'target_freq': 4000, 'split': 0.5, 'obs_hz': 250, 'target_hz':  500},
                          {'target_freq': 4000, 'split': 0.5, 'obs_hz': 250, 'target_hz':  750},
                          {'target_freq': 4000, 'split': 0.5, 'obs_hz': 250, 'target_hz': 1000},
                          {'target_freq': 4000, 'split': 0.5, 'obs_hz': 500, 'target_hz':  500},
                          {'target_freq': 4000, 'split': 0.5, 'obs_hz': 500, 'target_hz':  750},
                          {'target_freq': 4000, 'split': 0.5, 'obs_hz': 500, 'target_hz': 1000},
                          {'target_freq': 4000, 'split': 0.5, 'obs_hz': 750, 'target_hz':  500},
                          {'target_freq': 4000, 'split': 0.5, 'obs_hz': 750, 'target_hz':  750},
                          {'target_freq': 4000, 'split': 0.5, 'obs_hz': 750, 'target_hz': 1000},
                          {'target_freq': 4000, 'split': 0.9, 'obs_hz': 250, 'target_hz':  500},
                          {'target_freq': 4000, 'split': 0.9, 'obs_hz': 250, 'target_hz':  750},
                          {'target_freq': 4000, 'split': 0.9, 'obs_hz': 250, 'target_hz': 1000},
                          {'target_freq': 4000, 'split': 0.9, 'obs_hz': 500, 'target_hz':  500},
                          {'target_freq': 4000, 'split': 0.9, 'obs_hz': 500, 'target_hz':  750},
                          {'target_freq': 4000, 'split': 0.9, 'obs_hz': 500, 'target_hz': 1000},
                          {'target_freq': 4000, 'split': 0.9, 'obs_hz': 750, 'target_hz':  500},
                          {'target_freq': 4000, 'split': 0.9, 'obs_hz': 750, 'target_hz':  750},
                          {'target_freq': 4000, 'split': 0.9, 'obs_hz': 750, 'target_hz': 1000},
                          ]

    for experiment in experiment_set:
      experiment["bounds"] = bounds

    try:
      set_start_method('forkserver')
    except RuntimeError:
      pass
    
    n_experiments = len(experiment_set)
    exper_ = [experiment_set[experiment_specification]]

    #print("Creating " + str(n_experiments) + " (non-daemon) workers and jobs in main process.")

    pool = MyPool(n_experiments)
    pool.map(run_experiment, exper_)
    pool.close()
    pool.join()


if __name__ == '__main__':

  print("Total cpus available: " + str(ncpus))
  print("RUNNING EXPERIMENT " + str(experiment_specification))

  TEST = True

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
