import platform

import numpy as np

from psyrun import Param
from psyrun.scheduler import Slurm, Sqsub

from cue.trials.default import CueTrial


rng = np.random.RandomState(846777)
n_trials = 100
seeds = range(100)


pspace = Param(
    seed=seeds,
    trial=range(n_trials),
    recall_duration=90.,
    min_evidence=0.02,
    noise=0.015)
    # noise=0.09,
    # min_evidence=0.02)
min_items = 1
pool_size = 1
max_jobs = 100

if platform.node().startswith('gra') or platform.node().startswith('cedar'):
    workdir = '/scratch/jgosmann/cue'
    scheduler = Slurm(workdir)
    def timelimit(name):
        if 'split' in name or 'merge' in name:
            return '0-00:59'
        else:
            return '0-06:00'
    def memory_per_cpu(name):
        if 'split' in name:
            return '300M'
        elif 'merge' in name:
            return '300M'
        else:
            return '3G'
    scheduler_args = {
        'timelimit': timelimit,
        'n_cpus': '1',
        'memory_per_cpu': memory_per_cpu,
    }

def execute(trial, **kwargs):
    kwargs['protocol'] = 'no_scopolamine'
    result = CueTrial().run(**kwargs)
    return result
