import sys
from PyFiles.experiment import *
from PyFiles.analysis import *

TEACHER_FORCING = False

def get_frequencies(trial = 1):
  """
  get frequency lists
  """
  if trial == 1:
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
      lb_targ, ub_targ, obs_hz  = 150, 560, 100
  if trial == 7:
      lb_targ, ub_targ, obs_hz  = 400, 900, 200
  obs_list =  list( range( lb_targ - obs_hz, lb_targ, 10))
  obs_list += list( range( ub_targ, ub_targ + obs_hz, 10))
  resp_list = list( range( lb_targ, ub_targ, 10))
  return obs_list, resp_list
PREDICTION_TYPE = "block"

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
  }"""
  size = inputs["size"]
  if "k" in inputs:
    k = inputs["k"]
  else:
    k = None
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
                     "spectrogram_type": inputs["spectrogram_type"]}
  else:
    librosa_args = {}

  EchoArgs = { "size"    : size,  "verbose" : False}

  obs_inputs = {"split" : inputs["split"], "aspect": 0.9, "plot_split": False}

  if PREDICTION_TYPE == "column":
    train_time_idx, test_time_idx = inputs["train_time_idx"], inputs["test_time_idx"]

    experiment_inputs = { "size" : inputs["size"],
                          "target_frequency" : None,
                          "verbose" : False,
                          "prediction_type" : inputs["prediction_type"],
                          "train_time_idx" : train_time_idx,
                          "test_time_idx" : test_time_idx,
                          "k" : k, 
                          **librosa_args}

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
  #experiment.RC_CV(cv_args = cv_args, model = "uniform")
  experiment.RC_CV(cv_args = cv_args, model = "exponential")
  """
  models = ["exponential", "uniform"] if PREDICTION_TYPE == "block" else ["uniform"] #
  for model_ in models:
    print("Train shape: " + str(experiment.Train.shape))
    print("Test shape: " +  str(experiment.Test.shape))
    experiment.RC_CV(cv_args = cv_args, model = model_)
  """
