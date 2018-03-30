import os

import numpy as np

from psyrun import Param
from psyrun.backend import LoadBalancingBackend

from cue.trials import HebbRepetitionTrial


rng = np.random.RandomState(39)
n_trials = 25
seeds = range(n_trials)


pspace = Param(seed=seeds, trial=range(n_trials))
backend = LoadBalancingBackend
pool_size = 3
max_jobs = 1


def setup(proc_id):
    os.environ['PYOPENCL_CTX'] = '0:' + str(proc_id)


def execute(trial, **kwargs):
    result = HebbRepetitionTrial().run(**kwargs)
    return result
