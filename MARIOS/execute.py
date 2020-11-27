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
#load tests ect.
from execute_scripts.column import *

RUN_LITE = False

# necessary to add cwd to path when script run 
# by slurm (since it executes a copy)
sys.path.append(os.getcwd()) 

PREDICTION_TYPE = "block"


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


def test(exper_type, multiprocessing = False, gap = False):
    assert type(TEST) == bool
    if TEST == True:
      print("TEST")
      if exper_type == "block":

        experiment_set = [
              {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 50.0, 'target_hz': 6.0},
              {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 1000.0, 'target_hz': 1000.0},
              {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 500.0, 'target_hz': 1000.0},
              {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 1000.0, 'target_hz': 500.0},
              {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 500.0,  'target_hz': 500.0},
              {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 250.0,  'target_hz': 100.0},
              {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 100.0,  'target_hz': 100.0}
              ]
          
          #experiment_set = [ Merge(experiment, librosa_args) for experiment in experiment_set]
        set_specific_args = {"prediction_type": "block", "size" : "publish"}
        experiment_set = [ Merge(experiment, set_specific_args) for experiment in experiment_set]

      elif exper_type == "column":
        librosa_args = {}
        
        gap_start = 250
        train_width = 50
        train_width = gap_start

        #not actually a test, we need this asap.
        zhizhuo_target1    = liang_idx_convert(gap_start, 289)  #249 -> 288 inclusive
        zhizhuo_train1   = liang_idx_convert(gap_start - train_width, gap_start - 1 ) #199 -> 248 inclusive

        subseq_len = int(np.array(zhizhuo_train1).shape[0] * 0.5)
        
        gap_start2 = 514
        zhizhuo_target2 = liang_idx_convert(gap_start2, 613) #514 -> 613 in matlab, 513 -> 612 in python
        zhizhuo_train2  = liang_idx_convert(gap_start2 - train_width, gap_start2 - 1 )


        single_column_target = liang_idx_convert(100, 101)
        single_column_train = liang_idx_convert(100 - 5, 100-1)

        print("single column target" + str(single_column_target))

        set_specific_args = {"prediction_type": "column"}
        experiment_set = [
                          {'split': 0.5, 'train_time_idx': single_column_train, 'test_time_idx': single_column_target, 'k' : 1, "subseq_len" : 3},
                          {'split': 0.5, 'train_time_idx': zhizhuo_train1 , 'test_time_idx': zhizhuo_target1, 'k' : 1, "subseq_len" : subseq_len},
                          {'split': 0.5, 'train_time_idx': zhizhuo_train2, 'test_time_idx':  zhizhuo_target2, 'k' : 1, "subseq_len" : subseq_len},
                          #{'split': 0.5, 'train_time_idx': single_column_train, 'test_time_idx': single_column_target, 'k' : 10, "subseq_len" : 3},
                          #{'split': 0.5, 'train_time_idx': zhizhuo_train1 , 'test_time_idx': zhizhuo_target1, 'k' : 10, "subseq_len" : subseq_len},#, "k" : 100},
                          #{'split': 0.5, 'train_time_idx': zhizhuo_train2, 'test_time_idx':  zhizhuo_target2, 'k' : 10, "subseq_len" : subseq_len},#, "k" : 30},
                          ]
        # {'llambda': 0.00938595717962852, 'llambda2': 0.002908498759116776, 'connectivity': 1.0, 'spectral_radius': 0.48154601180553436, 'regularization': 0.3676013152573216, 'leaking_rate': 0.7179883186221123, 'noise': 1.2589254117941673, 'n_nodes': 1000, 'random_seed': 123}

        experiment_set = [ Merge(experiment, set_specific_args) for experiment in experiment_set]
    
    elif exper_type == "freqs":
      print("This is not a test")
      #librosa_args = {"spectrogram_path" : "19th_century_male_stft",
      #                "spectrogram_type" : "db",#"db", #power
      #                "librosa": True}
      obs_freqs, resp_freqs   = get_frequencies("run_fast_publish")
      #obs_freqs, resp_freqs   = get_frequencies(1)
      obs_freqs2, resp_freqs2 = get_frequencies(2)
      obs_freqs3, resp_freqs3 = get_frequencies(3)
      obs_freqs4, resp_freqs4 = get_frequencies(4)
      obs_freqs5, resp_freqs5 = get_frequencies(5)
      obs_freqs6, resp_freqs6 = get_frequencies(6)
      obs_freqs7, resp_freqs7 = get_frequencies(7)

      experiment_set = [
             #{ 'split': 0.9, "obs_freqs": obs_freqs6, "target_freqs": resp_freqs6 },
             #{ 'split': 0.9, "obs_freqs": obs_freqs3, "target_freqs": resp_freqs3 },
             #{ 'split': 0.9, "obs_freqs": obs_freqs7, "target_freqs": resp_freqs7 },
             
             #{ 'split': 0.7, "obs_freqs": obs_freqs6, "target_freqs": resp_freqs6 },
             #{ 'split': 0.7, "obs_freqs": obs_freqs3, "target_freqs": resp_freqs3 },
             #{ 'split': 0.7, "obs_freqs": obs_freqs7, "target_freqs": resp_freqs7 },
             { 'split': 0.5, "obs_freqs": obs_freqs, "target_freqs": resp_freqs },
             { 'split': 0.5, "obs_freqs": obs_freqs6, "target_freqs": resp_freqs6 },
             { 'split': 0.5, "obs_freqs": obs_freqs3, "target_freqs": resp_freqs3 },
             { 'split': 0.5, "obs_freqs": obs_freqs7, "target_freqs": resp_freqs7 },
             ]

      #set_specific_args = {"prediction_type": "block"}
      #experiment_set = [ Merge(experiment, set_specific_args) for experiment in experiment_set]
    bounds = { #noise hyper-parameter.
               #all are log scale except  spectral radius, leaking rate and n_nodes

               #9/16/2020 based on the hyper-parameter plot we will make the following adjustments:
               #exponential adj:
               #llambda -> wider net: (-3.5, 0.5), noise -> larger (more general solution then): (-5, -0.5),
               # connectivity needs to be wider as well: (-5, 0)
               #unif adj:
               # not going to impliment these, but connectivity clustered around 1, leaking rate RUNaround 1, spectral radius around 1
              'noise' :          (-5, -0.5),
              'llambda' :        (-5, 0),
              'llambda2' :       (-5, 0), 
              'connectivity':    (-5, 0),       # 0.5888436553555889, 
              'n_nodes':         1000,          #(100, 1500),
              'spectral_radius': (0.001, 0.999),
              'regularization':  (-3, 4),#(-12, 1),
              "leaking_rate" :   (0.001, 1) # we want some memory. 0 would mean no memory.
              # current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state
              }

    if RUN_LITE == True:
      bounds = { #noise hyper-parameter.
              'noise' :          0.1, #(-5, -0.5),
              'llambda' :        (-3, -1),
              'llambda2' :       0.1, 
              'connectivity':    0.05,       # 0.5888436553555889, 
              'n_nodes':         1000,          #(100, 1500),
              'spectral_radius': (0.001, 0.999),
              'regularization':  (-3, 1),#(-12, 1),
              "leaking_rate" :   0.99 #(0.001, 1) # we want some memory. 0 would mean no memory.
              # current_state = self.leaking_rate * update + (1 - self.leaking_rate) * current_state
              }

    for experiment in experiment_set:
      experiment["bounds"] = bounds
      experiment["prediction_type"] = exper_type
      experiment["size"] = "small"
      experiment["model_type"] = "random"
      experiment["input_weight_type"] = "exponential"
      

    #if TEACHER_FORCING:
      
    
    
    exper_ = [experiment_set[experiment_specification]]
    n_experiments = len(exper_)
    #print("Creating " + str(n_experiments) + " (non-daemon) workers and jobs in main process.")
    print("N EXPERIMENTS", n_experiments)

    #if n_experiments > 1:
    #  try:
    #    set_start_method('forkserver')
    #  except RuntimeError:
    #    pass
    #  pool = MyPool(n_experiments)
    #  pool.map(run_experiment, exper_)
    #  pool.close()
    #  pool.join()
    #else:
    run_experiment(exper_[0])
    

if __name__ == '__main__':

  print("Total cpus available: " + str(ncpus))
  print("RUNNING EXPERIMENT " + str(experiment_specification) + " YOU ARE NOT RUNNING EXP TESTS RIGHT NOW")


  TEST = True #false for low frequencies, true for column.

  start = timeit.default_timer()
  test("column")
  stop = timeit.default_timer()
  print('Time: ', stop - start) 

 

""" ##################################### VESTIGAL CODE BELOW
https://github.com/pytorch/pytorch/issues/3492:
"""
