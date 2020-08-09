#!/bin/bash/python

from reservoir import *
from PyFiles.imports import *
from PyFiles.helpers import *
from PyFiles.experiment import *

def run_experiment(inputs, n_cores = 8, cv_samples = 5):


    tf = inputs["target_freq"]
    split, obs_hz, target_hz = inputs["split"], inputs["obs_hz"], inputs["target_hz"]


    return(0)

    experiment = EchoStateExperiment(size = "medium", 
        target_frequency = tf, 
        obs_hz = obs_hz, 
        target_hz = target_hz, 
        verbose = False)

    experiment.get_observers(method = "freq", split = split, aspect = 0.9, plot_split = False)
    bounds = {
            'llambda' : (-12, 3), 
            'connectivity': (-3, 0), # 0.5888436553555889, 
            'n_nodes': 100,#(100, 1500),
            'spectral_radius': (0.05, 0.99),
            'regularization': (-10,-2)#(-12, 1)
            #all are log scale except  spectral radius and n_nodes
    }
    #example cv args:
    cv_args = {
        'bounds' : bounds,
        'initial_samples' : 100,
        'subsequence_length' : 250, #150 for 500
        'eps' : 1e-5,
        'cv_samples' : cv_samples, 
        'max_iterations' : 1000, 
        'scoring_method' : 'tanh',
        "n_jobs" : n_cores,
        "verbose" : True,
        "plot" : False,

    }
    experiment.RC_CV(cv_args = cv_args, model = "uniform")
    #experiment.RC_CV(cv_args = cv_args, model = "exponential")
#run_experiment()


experiment_test_lst =[{'target_freq': 2000, 'split': 0.5, 'obs_hz': 30, 'target_hz': 20},
                        {'target_freq': 2000, 'split': 0.5, 'obs_hz': 30, 'target_hz': 20}]
for experiment in experiment_test_lst :
    run_experiment(experiment)
#experiment.RC_CV(cv_args = cv_args, model = "hybrid")

#TODO write a function that reveals the results:


"""
from subprocess import call as run

class RunPy(object):

	def __init__(self):
		pass

	def runPythonFile(self, path):
		run(["Python3","{}".format(path)])

if __name__ == "__main__":
	c = RunPy()
	c.runPythonFile()
c = RunPy()

for i in ["imports", "helpers", "experiment"]:
	file_path = "PyFiles/"+ i + ".py"
	print("running " + file_path)
	#c.runPythonFile(file_path)
"""
