import sys
from PyFiles.experiment import *
from PyFiles.analysis import *

TEACHER_FORCING = False

def get_frequencies(trial = 1):
  """
  get frequency lists
  """
  if trial =="run_fast_publish":
      lb_targ, ub_targ, obs_hz  = 340, 350, 10
  elif trial == 1:
      lb_targ, ub_targ, obs_hz  = 210, 560, int(320 / 2)   
  elif trial == 2:
      lb_targ, ub_targ, obs_hz  = 340, 640, 280
  elif trial == 3:
      lb_targ, ub_targ, obs_hz  = 340, 350, 20#40
  elif trial == 4:
      lb_targ, ub_targ, obs_hz  = 60, 350, 40
  elif trial == 5:
      lb_targ, ub_targ, obs_hz  = 50, 200, 40
  if trial == 6:
      lb_targ, ub_targ, obs_hz  = 130, 530, 130
  if trial == 7:
      lb_targ, ub_targ, obs_hz  = 500, 900, 250
  obs_list =  list( range( lb_targ - obs_hz, lb_targ))
  obs_list += list( range( ub_targ, ub_targ + obs_hz))
  resp_list = list( range( lb_targ, ub_targ))
  return obs_list, resp_list

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


  #notes on modularizing this: column.py doesn't make sense in terms of its name. Ideally you will have block.py and column.py,
  but at the end of the day the experiment specifications should be easier to execute. Namely, they should be at the top of this file.
  }"""
  model_type = inputs["model_type"]
  size = inputs["size"]


  prediction_type = inputs["prediction_type"] 
  if "k" in inputs:
    k = inputs["k"]
  else:
    k = None
  #default arguments
  #print("Prediction Type: " + inputs["prediction_type"])

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
                     "spectrogram_type": inputs["spectrogram_type"]}
  else:
    librosa_args = {}

  EchoArgs = { "size"    : size,  "verbose" : False}

  obs_inputs = {"split" : inputs["split"], "aspect": 0.9, "plot_split": False}

  if inputs["prediction_type"] == "column":
    train_time_idx, test_time_idx = inputs["train_time_idx"], inputs["test_time_idx"]

    experiment_inputs = { "size" : inputs["size"],
                          "target_frequency" : None,
                          "verbose" : False,
                          "prediction_type" : inputs["prediction_type"],
                          "train_time_idx" : train_time_idx,
                          "test_time_idx" : test_time_idx,
                          "k" : k,
                          "model" : model_type,
                          **librosa_args}

    print("experiment_inputs: " + str(experiment_inputs))
    experiment = EchoStateExperiment(**experiment_inputs)
    
    obs_inputs = Merge(obs_inputs, {"method" : "exact"})

    print("obs_inputs: " + str(obs_inputs))
    experiment.get_observers(**obs_inputs)

  elif inputs["prediction_type"] == "block":
    if "obs_freqs" in inputs:
      AddEchoArgs = { "obs_freqs" : inputs["obs_freqs"],
                      "target_freqs" : inputs["target_freqs"],
                      "prediction_type" : inputs["prediction_type"],
                      "model" : model_type
                    }
      EchoArgs = Merge(EchoArgs, AddEchoArgs)
    else:
      AddEchoArgs = { "target_frequency" : inputs["target_frequency"],
                      "obs_hz" : inputs["obs_hz"],
                      "target_hz" : inputs["target_hz"],
                      "model" : model_type
                    }
      EchoArgs = Merge( Merge(EchoArgs, AddEchoArgs), librosa_args)
    print(EchoArgs)
    experiment = EchoStateExperiment( **EchoArgs)
    ### NOW GET OBSERVERS
    method = "exact" if "obs_freqs" in inputs else "freq"
    experiment.get_observers(method = method, **obs_inputs)

  if size == "small":
    default_presets = {
      "cv_samples" : 1,
      "max_iterations" : 1000,
      "eps" : 1e-5,
      'subsequence_length' : 180,
      "initial_samples" : 200,
      "random_seed" : None,
      "n_res": 1,
      "batch_size": 1}
  elif size == "medium":
    default_presets = {
      "cv_samples" : 1,
      "max_iterations" : 4000,
      "eps" : 1e-5,
      'subsequence_length' : 250,
      "initial_samples" : 100}
  elif size == "publish":
    default_presets = {
      "cv_samples" : 3,
      "max_iterations" : 2000,
      "eps" : 1e-6,
      "random_seed" : None,
      'subsequence_length' : 700,
      "n_res": 1,
      "initial_samples" : 300}

  if inputs["prediction_type"] == "column":
    default_presets['esn_feedback'] = True
    if "subseq_len" in inputs:
      default_presets['subsequence_length'] = inputs["subseq_len"]
    else:
      default_presets['subsequence_length'] = 75
  print("NCORES", n_cores)
  cv_args = {
      'bounds' : inputs["bounds"],
      'scoring_method' : 'tanh',
      "n_jobs" : n_cores,
      "verbose" : True,
      "plot" : False, 
      **default_presets
  }
  if model_type in ["delay_line", "cyclic"]:
    cv_args = {**cv_args, "activation_function" : "sin_sq"}

  

  #Consider combining cyclic and delay line
  if model_type == "uniform" and prediction_type == "column":
    experiment.RC_CV(cv_args = cv_args, model = "uniform")
  elif model_type in ["delay_line", "cyclic"]:
    experiment.RC_CV(cv_args = cv_args, model = model_type, input_weight_type = "uniform")
    experiment.RC_CV(cv_args = cv_args, model = model_type, input_weight_type = "exponential")
  else:
    experiment.RC_CV(cv_args = cv_args, model = "random", input_weight_type = "uniform")
    experiment.RC_CV(cv_args = cv_args, model = "random", input_weight_type = "exponential")
  

