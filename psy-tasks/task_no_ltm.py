import platform

import numpy as np

from psyrun import Param
from psyrun.scheduler import Slurm

from cue.trials import CueTrial


rng = np.random.RandomState(846777)
n_trials = 100
seeds = range(100)


pspace = Param(
    seed=seeds,
    trial=range(n_trials),
    recall_duration=45.,
    noise=0.015,
    min_evidence=0.04,
    disable_ltm_recall=True)
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
            return '0-02:59'
    def memory_per_cpu(name):
        if 'split' in name:
            return '300M'
        elif 'merge' in name:
            return '60M'
        else:
            return '3G'
    scheduler_args = {
        'timelimit': timelimit,
        'n_cpus': '1',
        'memory_per_cpu': memory_per_cpu,
    }

def execute(trial, **kwargs):
    kwargs['protocol'] = 'immediate'
    result = CueTrial().run(**kwargs)
    return result
