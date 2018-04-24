import functools
import traceback

import matplotlib.pyplot as plt
import numpy as np

import nengo
from nengo.builder.connection import get_eval_points, solve_for_decoders
from nengo.builder.operator import Reset
from nengo.builder.signal import Signal
from nengo.config import SupportDefaultsMixin
from nengo.params import Default, IntParam



class WeightSymmetryLR(
        nengo.learning_rules.LearningRuleType, SupportDefaultsMixin):
    modifies = 'decoders'
    seed = IntParam('seed', default=None, optional=True, readonly=True)

    def __init__(self, learning_rate=1., seed=Default):
        super(WeightSymmetryLR, self).__init__(learning_rate, size_in=1)
        self.seed = seed


class SimWeightSymmetryLearning(nengo.builder.Operator):
    def __init__(
            self, learning_rate, base_decoders, pre, decoders, scale,
            tag=None):
        super(SimWeightSymmetryLearning, self).__init__(tag=tag)

        self.learning_rate = learning_rate
        self.base_decoders = base_decoders

        self.sets = []
        self.incs = []
        self.reads = [pre, scale]
        self.updates = [decoders]

    @property
    def pre(self):
        return self.reads[0]

    @property
    def scale(self):
        return self.reads[1]

    @property
    def decoders(self):
        return self.updates[0]

    def make_step(self, signals, dt, rng):
        base_decoders = self.base_decoders
        pre = signals[self.pre]
        decoders = signals[self.decoders]
        scale = signals[self.scale]
        alpha = self.learning_rate * dt

        def step_assoc_learning():
            x = base_decoders.T @ pre
            decoders[...] += 2. * scale * alpha * (
                x.T @ x - pre @ decoders @ pre) * np.outer(pre, pre)

        return step_assoc_learning


@nengo.builder.Builder.register(WeightSymmetryLR)
def build_weight_symmetry_learning(model, weight_symmetry_learning, rule):
    if weight_symmetry_learning.seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(weight_symmetry_learning.seed)

    conn = rule.connection

    pre = model.sig[conn.pre_obj.neurons]['out']

    decoders = model.sig[conn]['weights']

    scale = Signal(np.zeros(rule.size_in), name="WeightSymmetryLearn:scale")
    model.add_op(Reset(scale))
    model.sig[rule]['in'] = scale

    encoders = model.params[conn.pre_obj].encoders
    gain = model.params[conn.pre_obj].gain
    bias = model.params[conn.pre_obj].bias

    eval_points = get_eval_points(model, conn, rng)
    targets = eval_points

    x = np.dot(eval_points, encoders.T / conn.pre_obj.radius)

    base_decoders, _ = solve_for_decoders(
        conn, gain, bias, x, targets, rng=rng)

    model.add_op(SimWeightSymmetryLearning(
        weight_symmetry_learning.learning_rate, base_decoders, pre, decoders,
        scale))



class SignalGenerator(nengo.Process):
    def __init__(self, duration, high=40.):
        super(SignalGenerator, self).__init__()
        self._whitenoise = nengo.processes.WhiteSignal(duration, high=high)

    def make_step(self, size_in, size_out, dt, rng=np.random):
        return functools.partial(
            self.sample,
            sample_whitenoise=self._whitenoise.make_step(
                size_in, size_out, dt, rng),
            sample_length=self._whitenoise.make_step(size_in, (1,), dt, rng))

    @staticmethod
    def sample(t, sample_whitenoise, sample_length):
        sampled = sample_whitenoise(t)
        return sampled / np.linalg.norm(sampled) * sample_length(t)


d = 32
n = 25
duration = 4 * 1200.

with nengo.Network(seed=1) as model:
    pre = nengo.Ensemble(
        n * d, d,
        intercepts=nengo.dists.CosineSimilarity(d + 2),
        eval_points=nengo.dists.CosineSimilarity(d + 2),
        neuron_type=nengo.LIF())
    post = nengo.Node(size_in=n * d)
    def_post = nengo.Node(size_in=d)
    def_conn = nengo.Connection(pre, def_post)

    in_signal = nengo.Node(SignalGenerator(duration), size_out=d)
    nengo.Connection(in_signal, pre)

    conn = nengo.Connection(
        pre, post,
        function=lambda x: np.zeros(n * d),
        learning_rule_type=WeightSymmetryLR(1e-13))
    nengo.Connection(nengo.Node(1.), conn.learning_rule)

    sample_every = max(duration / 600., 1e-3)
    p_decoders = nengo.Probe(conn, 'weights', sample_every=sample_every)
    p_post = nengo.Probe(post, sample_every=sample_every)
    p_def_post = nengo.Probe(def_post, sample_every=sample_every)


with nengo.Simulator(model) as sim:
    try:
        sim.run(duration)
    except FloatingPointError:
        traceback.print_exc()


W = sim.data[def_conn].weights.T @ sim.data[def_conn].weights


np.savez(
    'weightsym.npz',
    t=sim.trange(), W=W, def_conn_weights=sim.data[def_conn].weights,
    decoders=sim.data[p_decoders], post=sim.data[p_post],
    def_post=sim.data[p_def_post])
