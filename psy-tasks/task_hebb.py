import os

import numpy as np
import pyopencl

from psyrun import Param
from psyrun.backend import LoadBalancingBackend

from cue.trials import HebbRepetitionTrial


rng = np.random.RandomState(39)
n_trials = 25
seeds = range(n_trials)


pspace = Param(extension=['forward-assoc', 'direct-assoc']) * Param(
    seed=seeds, trial=range(n_trials))
backend = LoadBalancingBackend
exclude_from_result = ['cl_context']
pool_size = 4
max_jobs = 1


def setup(proc_id):
    context = pyopencl.create_some_context(answers=[0, proc_id])
    return {'cl_context': context}


def execute(trial, **kwargs):
    result = HebbRepetitionTrial().run(progress=False, **kwargs)
    return result
