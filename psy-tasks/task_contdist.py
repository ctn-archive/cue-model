import platform

import numpy as np

from psyrun import Param
from psyrun.backend import LoadBalancingBackend
from psyrun.scheduler import Slurm

from cue.trials import CueTrial


rng = np.random.RandomState(846777)
n_trials = 100
seeds = range(100)


pspace = Param(
    seed=seeds,
    trial=range(n_trials),
    min_evidence=0.03,
    noise=0.009,
    distractor_rate=0.4,
    ordinal_prob=0.1)
exclude_from_result = ['cl_context']
min_items = 1

if platform.node().startswith('gra') or platform.node().startswith('cedar'):
    pool_size = 1
    max_jobs = 100
    workdir = '/scratch/jgosmann/cue'
    scheduler = Slurm(workdir)
    def timelimit(name):
        if 'split' in name or 'merge' in name:
            return '0-00:10'
        else:
            return '0-12:00'
    def memory_per_cpu(name):
        if 'split' in name:
            return '300M'
        elif 'merge' in name:
            return '60M'
        else:
            return '4G'
    scheduler_args = {
        'timelimit': timelimit,
        'n_cpus': '1',
        'memory_per_cpu': memory_per_cpu,
    }
elif platform.node().startswith('ctngpu'):
    pool_size = 4
    max_jobs = 1
    backend = LoadBalancingBackend

    import pyopencl

    def setup(proc_id):
        context = pyopencl.create_some_context(answers=[0, proc_id])
        return {'cl_context': context}

def execute(trial, **kwargs):
    kwargs['protocol'] = 'contdist'
    result = CueTrial().run(progress=False, **kwargs)
    return result
