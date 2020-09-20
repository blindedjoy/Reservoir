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
import multiprocessing.pool 

PREDICTION_TYPE = "block"

TEACHER_FORCING = False

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
      lb = lb - 1 #// 2
      ub = ub - 1 # // 2
    idx_list = list(range(lb, ub + 1))
    return idx_list

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
  elif trial == 4:
      lb_targ, ub_targ, obs_hz  = 60, 350, 40
  elif trial == 5:
      lb_targ, ub_targ, obs_hz  = 50, 200, 40
  if trial == 6:
      lb_targ, ub_targ, obs_hz  = 150, 560, 100
  obs_list =  list( range( lb_targ - obs_hz, lb_targ, 10))
  obs_list += list( range( ub_targ, ub_targ + obs_hz, 10))
  resp_list = list( range( lb_targ, ub_targ, 10))
  return obs_list, resp_list

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

def run_experiment(inputs, n_cores = int(sys.argv[2]), cv_samples = 5, interpolation_method = "griddata-linear"):
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
      size = inputs["size"]
      #default arguments
      print("Prediction Type: " + inputs["prediction_type"])

      ####if you imported the data via librosa this will work
      if "librosa" in inputs:
        default_presets = {
          "cv_samples" : 4,
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

      obs_inputs = {"split" : inputs["split"], "aspect": 0.9, "plot_split": False}

      if "k" in inputs:
        obs_inputs["k"] = inputs["k"]

      if PREDICTION_TYPE == "column":
        train_time_idx, test_time_idx = inputs["train_time_idx"], inputs["test_time_idx"]

        experiment_inputs = { "size" : inputs["size"],
                              "target_frequency" : None,
                              "verbose" : False,
                              "prediction_type" : inputs["prediction_type"],
                              "train_time_idx" : train_time_idx,
                              "test_time_idx" : test_time_idx,
                              **librosa_args
                            }

        print("experiment_inputs: " + str(experiment_inputs))
        experiment = EchoStateExperiment(**experiment_inputs)
        
        obs_inputs = Merge(obs_inputs, {"method" : "exact"})

        print("obs_inputs: " + str(obs_inputs))
        experiment.get_observers(**obs_inputs)
      
      elif PREDICTION_TYPE == "block":
        if "obs_freqs" in inputs:
          AddEchoArgs = { "obs_freqs" : inputs["obs_freqs"],
                          "target_freqs" : inputs["target_freqs"],
                          "prediction_type" : PREDICTION_TYPE
                        }
          EchoArgs = Merge(EchoArgs, AddEchoArgs)
        else:
          AddEchoArgs = { "target_frequency" : inputs["target_frequency"],
                          "obs_hz" : inputs["obs_hz"],
                          "target_hz" : inputs["target_hz"]
                        }
          EchoArgs = Merge( Merge(EchoArgs, AddEchoArgs), librosa_args)
        print(EchoArgs)
        experiment = EchoStateExperiment( **EchoArgs)
        ### NOW GET OBSERVERS
        method = "exact" if "obs_freqs" in inputs else "freq"

        experiment.get_observers(method = method, **obs_inputs)
      
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
          "cv_samples" : 8,
          "max_iterations" : 2000,
          "eps" : 1e-6,
          'subsequence_length' : 700,
          "initial_samples" : 300}

      if PREDICTION_TYPE == "column":
        if "subseq_len" in inputs:
          default_presets['subsequence_length'] = inputs["subseq_len"]
        else:
          default_presets['subsequence_length'] = 75

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

      models = ["exponential", "uniform"] if PREDICTION_TYPE == "block" else ["uniform"] #
      for model_ in models:
        print("Train shape: " + str(experiment.Train.shape))
        print("Test shape: " +  str(experiment.Test.shape))
        experiment.RC_CV(cv_args = cv_args, model = model_)

def test(TEST, multiprocessing = False, gap = False):
    assert type(TEST) == bool
    if TEST == True:
      print("TEST")
      if PREDICTION_TYPE == "block":
        if gap: 
          print("HA")
        else:
          print("on track")
          
          experiment_set = [
                {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 1000.0, 'target_hz': 1000.0},
                {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 500.0, 'target_hz': 1000.0},
                {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 1000.0, 'target_hz': 500.0},
                {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 500.0,  'target_hz': 500.0},
                {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 250.0,  'target_hz': 100.0},
                {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 100.0,  'target_hz': 100.0},

                {'target_frequency': 1000, "split" : 0.9, 'obs_hz': 1000.0, 'target_hz': 1000.0},
                {'target_frequency': 1000, "split" : 0.9, 'obs_hz': 500.0, 'target_hz': 1000.0},
                {'target_frequency': 1000, "split" : 0.9, 'obs_hz': 1000.0, 'target_hz': 500.0},
                {'target_frequency': 1000, "split" : 0.9, 'obs_hz': 500.0,  'target_hz': 500.0},
                {'target_frequency': 1000, "split" : 0.9, 'obs_hz': 250.0,  'target_hz': 100.0},
                {'target_frequency': 1000, "split" : 0.9, 'obs_hz': 100.0,  'target_hz': 100.0},
                ]
          #{ 'target_freq': 500, 'split': 0.5, 'obs_hz': 20,  'target_hz': 10}]
          #{ 'target_freq': 250, 'split': 0.5, 'obs_hz': 20,  'target_hz': 10}]
          #{ 'target_freq': 250, 'split': 0.5, 'obs_hz': 100, 'target_hz': 20},
          #{ 'target_freq': 250, 'split': 0.5, 'obs_hz': 25,  'target_hz': 50},
          #
          #NEXTUP:
          #[{'target_frequency': 1000, 'obs_hz': 1000.0, 'target_hz': 500.0},
          #            {'target_frequency': 1000, 'obs_hz': 500.0,  'target_hz': 500.0},
          #            {'target_frequency': 1000, 'obs_hz': 250.0,  'target_hz': 100.0},
          #            {'target_frequency': 1000, 'obs_hz': 100.0,  'target_hz': 100.0}]

          #experiment_set = [ Merge(experiment, librosa_args) for experiment in experiment_set]
        set_specific_args = {"prediction_type": "block", "size" : "publish"}
        experiment_set = [ Merge(experiment, set_specific_args) for experiment in experiment_set]

      elif PREDICTION_TYPE == "column":

        #librosa_args = {"spectrogram_path" : "publish",   #examples: 18th_cqt_low,
        #                "spectrogram_type"  : "power",    #examples: db, power
        #                "librosa": True}
        librosa_args = {}
        
        gap_start = 250
        train_width = 50
        train_width = gap_start
        test1    = liang_idx_convert(gap_start, 289)  #249 -> 288 inclusive
        train1   = liang_idx_convert(gap_start - train_width, gap_start - 1 ) #199 -> 248 inclusive

        subseq_len = int(np.array(train1).shape[0] * 0.5)
        
        gap_start2 = 514
        test2   = liang_idx_convert(gap_start2, 613) #514 -> 613 in matlab, 513 -> 612 in python
        train2  = liang_idx_convert(gap_start2 - train_width, gap_start2 - 1 )

        set_specific_args = {"prediction_type": "column", "subseq_len" : subseq_len}
        experiment_set = [
                          {'split': 0.5, 'train_time_idx': train1 , 'test_time_idx': test1},#, "k" : 100},
                          {'split': 0.5, 'train_time_idx': train2, 'test_time_idx':  test2},#, "k" : 30},
                          {'split': 0.9, 'train_time_idx': train1 , 'test_time_idx': test1}, #, "k" : 35},
                          {'split': 0.9, 'train_time_idx': train2, 'test_time_idx':  test2}#, "k" : 40}
                         ]

        experiment_set = [ Merge(experiment, set_specific_args) for experiment in experiment_set]

        if librosa_args["spectrogram_path"] == "18th_cqt_low":
          experiment_set = [ Merge(experiment, librosa_args) for experiment in experiment_set]
        
      bounds = {
                #'noise' : (-2, -4),
                'llambda' : (-3, -1), 
                'connectivity': (-3, 0), # 0.5888436553555889, 
                'n_nodes': 1000,#(100, 1500),
                'spectral_radius': (0.05, 0.99),
                'regularization': (-10,-2)
                }
    
    else:
      print("This is not a test")
      bounds = { #noise hyper-parameter.
                 #all are log scale except  spectral radius, leaking rate and n_nodes

                 #9/16/2020 based on the hyper-parameter plot we will make the following adjustments:
                 #exponential adj:
                 #llambda -> wider net: (-3.5, 0.5), noise -> larger (more general solution then): (-5, -0.5),
                 # connectivity needs to be wider as well: (-5, 0)
                 #unif adj:
                 # not going to impliment these, but connectivity clustered around 1, leaking rate around 1, spectral radius around 1
                'noise' :          (-5, -0.5),
                'llambda' :        (-4, 0), 
                'connectivity':    (-5, 0),       # 0.5888436553555889, 
                'n_nodes':         1000,          #(100, 1500),
                'spectral_radius': (0.001, 0.999),
                'regularization':  (-3, 4),#(-12, 1),
                "leaking_rate" :   (0.001, 1) # we want some memory. 0 would mean no memory.
                # current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state
                }
      experiment_set = [
                {'target_frequency': 990, "split" : 0.5, 'obs_hz': 980, 'target_hz': 980.0},
                {'target_frequency': 990, "split" : 0.5, 'obs_hz': 450.0, 'target_hz': 980.0},
                {'target_frequency': 990, "split" : 0.5, 'obs_hz': 980.0, 'target_hz': 450.0},
                {'target_frequency': 990, "split" : 0.5, 'obs_hz': 450.0,  'target_hz': 450.0},
                {'target_frequency': 990, "split" : 0.5, 'obs_hz': 250.0,  'target_hz': 80.0},
                {'target_frequency': 990, "split" : 0.5, 'obs_hz': 80.0,  'target_hz': 80.0},

                {'target_frequency': 990, "split" : 0.9, 'obs_hz': 980.0, 'target_hz': 980.0},
                {'target_frequency': 990, "split" : 0.9, 'obs_hz': 450.0, 'target_hz': 980.0},
                {'target_frequency': 990, "split" : 0.9, 'obs_hz': 980.0, 'target_hz': 450.0},
                {'target_frequency': 990, "split" : 0.9, 'obs_hz': 450.0,  'target_hz': 450.0},
                {'target_frequency': 990, "split" : 0.9, 'obs_hz': 230.0,  'target_hz': 80.0},
                {'target_frequency': 990, "split" : 0.9, 'obs_hz': 80.0,  'target_hz': 80.0},
                ]


      #librosa_args = {"spectrogram_path" : "19th_century_male_stft",
      #                "spectrogram_type" : "db",#"db", #power
      #                "librosa": True}
    
      obs_freqs, resp_freqs   = get_frequencies(1)
      obs_freqs2, resp_freqs2 = get_frequencies(2)
      obs_freqs3, resp_freqs3 = get_frequencies(3)
      obs_freqs4, resp_freqs4 = get_frequencies(4)
      obs_freqs5, resp_freqs5 = get_frequencies(5)
      obs_freqs6, resp_freqs6 = get_frequencies(6)

      experiment_set = [
             { 'split': 0.9, "obs_freqs": obs_freqs,  "target_freqs": resp_freqs  },
             { 'split': 0.9, "obs_freqs": obs_freqs2, "target_freqs": resp_freqs2 },
             { 'split': 0.9, "obs_freqs": obs_freqs3, "target_freqs": resp_freqs3 },
             { 'split': 0.9, "obs_freqs": obs_freqs4, "target_freqs": resp_freqs4 },
             { 'split': 0.9, "obs_freqs": obs_freqs5, "target_freqs": resp_freqs5 },
             { 'split': 0.9, "obs_freqs": obs_freqs6, "target_freqs": resp_freqs6 },

             { 'split': 0.7, "obs_freqs": obs_freqs,  "target_freqs": resp_freqs  },
             { 'split': 0.7, "obs_freqs": obs_freqs2, "target_freqs": resp_freqs2 },
             { 'split': 0.7, "obs_freqs": obs_freqs3,  "target_freqs": resp_freqs3  },
             { 'split': 0.7, "obs_freqs": obs_freqs4,  "target_freqs": resp_freqs4  },
             { 'split': 0.7, "obs_freqs": obs_freqs5, "target_freqs": resp_freqs5 },
             { 'split': 0.7, "obs_freqs": obs_freqs6,  "target_freqs": resp_freqs6  },

             
             { 'split': 0.5, "obs_freqs": obs_freqs,  "target_freqs": resp_freqs  },
             { 'split': 0.5, "obs_freqs": obs_freqs2, "target_freqs": resp_freqs2 },
             { 'split': 0.5, "obs_freqs": obs_freqs3,  "target_freqs": resp_freqs3  },
             { 'split': 0.5, "obs_freqs": obs_freqs4,  "target_freqs": resp_freqs4  },
             { 'split': 0.5, "obs_freqs": obs_freqs5,  "target_freqs": resp_freqs5  },
             { 'split': 0.5, "obs_freqs": obs_freqs6,  "target_freqs": resp_freqs6  },
             ]

      #set_specific_args = {"prediction_type": "block"}
      #experiment_set = [ Merge(experiment, set_specific_args) for experiment in experiment_set]


    for experiment in experiment_set:
      experiment["bounds"] = bounds
      experiment["prediction_type"] = "block"
      experiment["size"] = "publish"

    try:
      set_start_method('forkserver')
    except RuntimeError:
      pass
    
    n_experiments = len(experiment_set)
    exper_ = [experiment_set[experiment_specification]]

    #print("Creating " + str(n_experiments) + " (non-daemon) workers and jobs in main process.")
    if n_experiments > 1:
      pool = MyPool(n_experiments)
      pool.map(run_experiment, exper_)
      pool.close()
      pool.join()
    else:
      run_experiment(exper_[0])


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
        experiment_set = [
                {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 1000.0, 'target_hz': 1000.0},
                {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 500.0, 'target_hz': 1000.0},
                {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 1000.0, 'target_hz': 500.0},
                {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 500.0,  'target_hz': 500.0},
                {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 250.0,  'target_hz': 100.0},
                {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 100.0,  'target_hz': 100.0},

                {'target_frequency': 1000, "split" : 0.9, 'obs_hz': 1000.0, 'target_hz': 1000.0},
                {'target_frequency': 1000, "split" : 0.9, 'obs_hz': 500.0, 'target_hz': 1000.0},
                {'target_frequency': 1000, "split" : 0.9, 'obs_hz': 1000.0, 'target_hz': 500.0},
                {'target_frequency': 1000, "split" : 0.9, 'obs_hz': 500.0,  'target_hz': 500.0},
                {'target_frequency': 1000, "split" : 0.9, 'obs_hz': 250.0,  'target_hz': 100.0},
                {'target_frequency': 1000, "split" : 0.9, 'obs_hz': 100.0,  'target_hz': 100.0},
                ]
      elif experiment_specification == 2: 
        # for 2k lets add some 750 target hz.
        experiment_set = [  #4k, 0.5 filling in some more gaps:
                          ]

      """
