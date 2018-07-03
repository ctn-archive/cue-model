import random

import nengo
import nengo_spa as spa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ranksums
import xarray as xr

from cue.model.amlearn import AML


class Protocol(object):
    def __init__(self, n_items=7, repeats=6, item_duration=1., seed=None):
        self.n_items = n_items
        self.repeats = repeats
        self.item_duration = item_duration

        self.l_keys = ['L' + str(i) for i in range(self.n_items)]
        self.p_keys = ['P' + str(i) for i in range(self.n_items)]
        self.lp_keys = self.l_keys + self.p_keys
        self.all_keys = self.lp_keys + ['X']

        rng_state = None
        if seed is not None:
            rng_state = random.getstate()
            random.seed(seed)

        self.screen_stimuli = self.repeats * self.lp_keys
        random.shuffle(self.screen_stimuli)
        self.test_stimuli = self.repeats * self.lp_keys
        random.shuffle(self.test_stimuli)

        if rng_state is not None:
            random.setstate(rng_state)

    @property
    def screen_duration(self):
        return len(self.screen_stimuli) * self.item_duration

    @property
    def learn_duration(self):
        return self.n_items * self.item_duration

    @property
    def test_duration(self):
        return len(self.test_stimuli) * self.item_duration

    @property
    def total_duration(self):
        return (
            1. + self.screen_duration + self.learn_duration +
            self.test_duration)

    def mask_screen(self, t):
        return t - 1. <= self.screen_duration

    def mask_test(self, t):
        return t > 1. + self.screen_duration + self.learn_duration

    @property
    def onsets_screen(self):
        return np.arange(len(self.screen_stimuli)) * self.item_duration + 1.

    @property
    def onsets_test(self):
        return (
            np.arange(len(self.test_stimuli)) * self.item_duration + 1. +
            self.screen_duration  + self.learn_duration)

    def phase(self, t):
        if t <= 1.:
            return 'init', t
        t -= 1.
        if t <= self.screen_duration:
            return 'screen', t
        t -= self.screen_duration
        if t <= self.learn_duration:
            return 'learn', t
        t -= self.learn_duration
        if t <= self.test_duration:
            return 'test', t
        t -= self.test_duration
        return 'end', t

    def learn(self, t):
        return self.phase(t)[0] == 'learn'

    def stimulus(self, stim_type, t):
        phase, phase_t = self.phase(t)
        i = int(np.ceil(phase_t / self.item_duration)) - 1
        if phase == 'init':
            stim = 'X'
        elif phase == 'screen':
            stim = self.screen_stimuli[i]
        elif phase == 'learn':
            stim = stim_type + str(i)
        elif phase == 'test':
            stim = self.test_stimuli[i]
        else:
            assert phase == 'end'
            stim = 'X'
        if stim != 'X' and not stim.startswith(stim_type):
            stim = '0'
        return stim


def create_model(protocol, vocab, seed, n_neurons=5000):
    d = vocab.dimensions
    with nengo.Network(seed=seed) as model:
        model.config[nengo.Ensemble].max_rates = nengo.dists.Uniform(10, 20)
        model.config[nengo.Ensemble].intercepts = nengo.dists.Uniform(0.1, 1.)

        pre_person = nengo.Ensemble(n_neurons, d)
        pre_landmark = nengo.Ensemble(n_neurons, d)
        post = nengo.Ensemble(
            n_neurons, d, radius=1.,
            noise=nengo.processes.FilteredNoise(
                synapse=nengo.Lowpass(0.1),
                dist=nengo.dists.Gaussian(0.01, .05)))
        nengo.Connection(pre_person, post)
        nengo.Connection(pre_landmark, post)

        err = nengo.Node(size_in=d + 2)
        nengo.Connection(pre_person, err[2:])
        nengo.Connection(pre_landmark, err[2:])
        nengo.Connection(nengo.Node(protocol.learn), err[0])
        nengo.Connection(nengo.Node(1.), err[1])

        c_person = nengo.Connection(
            pre_person, post, learning_rule_type=AML(d, 0.9),
            function=lambda x: np.zeros(d))
        nengo.Connection(err, c_person.learning_rule)
        c_landmark = nengo.Connection(
            pre_landmark, post, learning_rule_type=AML(d, 0.9),
            function=lambda x: np.zeros(d))
        nengo.Connection(err, c_landmark.learning_rule)

        stim_p = spa.Transcode(
            lambda t: protocol.stimulus('P', t), output_vocab=vocab)
        nengo.Connection(stim_p.output, pre_person)
        stim_l = spa.Transcode(
            lambda t: protocol.stimulus('L', t), output_vocab=vocab)
        nengo.Connection(stim_l.output, pre_landmark)

        p_post_spikes = nengo.Probe(post.neurons, 'spikes')
    return model, p_post_spikes


class Trial(object):
    def __init__(self, protocol, seed, d=32, n_neurons=1500):
        self.protocol = protocol
        self.vocab = spa.Vocabulary(
            d, strict=True, rng=np.random.RandomState(seed + 1))
        self.vocab.populate(';'.join(protocol.all_keys))

        self.model, self.probe = create_model(
            protocol, self.vocab, seed, n_neurons=n_neurons)
        self.dt = 0.001
        self.sim = None
        self.trange = None
        self.spikes = None

    def run(self, backend=nengo.Simulator):
        with backend(self.model, dt=self.dt) as self.sim:
            self.sim.run(self.protocol.total_duration)
        self.trange = self.sim.trange()
        self.spikes = self.sim.data[self.probe]


def slice_by_stim(t, spikes, stimuli, onsets, intervals=((-.5, .1), (.2, .8))):
    by_stimulus = {k: [] for k in set(stimuli)}
    for onset, stim in zip(onsets, stimuli):
        by_stimulus[stim].append([xr.DataArray(
            spikes[(onset + l < t) & (t <= onset + u)],
            dims=('t', 'spikes')) for l, u in intervals])
    return by_stimulus


def slice_intervals(t, spikes, onsets, interval=(-.75, 1.)):
    l, u = interval
    return [
        xr.DataArray(
            spikes[(onset + l < t) & (t <= onset + u)], dims=('t', 'spikes'))
        for onset in onsets]


def pop_response(protocol, pair_coding, t_bins, z_score, phase, selector):
    phase_mappings = {
        'screen': (protocol.onsets_screen, protocol.screen_stimuli),
        'test': (protocol.onsets_test, protocol.test_stimuli),
    }
    onsets, stimuli = phase_mappings[phase]
    return xr.concat(
        (xr.concat(
            slice_intervals(
                t_bins, z_score[k].isel(spikes=pair_coding[k]),
                [os for os, s in zip(onsets, stimuli) if selector(k, s)]),
            dim='spikes')
         for k in z_score),
        dim='spikes')


def eval_pop_responses(protocol, trial):
    # Abbreviations:
    # bl = before learning
    # al = after learning
    #  p = preferred
    # np = non-preferred
    # na = non-associated

    dt = trial.dt
    f_spikes = gaussian_filter1d(trial.spikes, 0.1 / dt, axis=0)

    mask = protocol.mask_screen(trial.trange)
    sliced_bl = slice_by_stim(
        trial.trange[mask], trial.spikes[mask], protocol.screen_stimuli,
        protocol.onsets_screen)

    mask = protocol.mask_test(trial.trange)
    sliced_al = slice_by_stim(
        trial.trange[mask], trial.spikes[mask], protocol.test_stimuli,
        protocol.onsets_test)

    frs_bl = {
        k: xr.DataArray(
            [[x.mean(dim='t') for x in intervals] for intervals in onsets],
            dims=('onsets', 'intervals', 'fr'))
        for k, onsets in sliced_bl.items()}
    frs_al = {
        k: xr.DataArray(
            [[x.mean(dim='t') for x in intervals] for intervals in onsets],
            dims=('onsets', 'intervals', 'fr'))
        for k, onsets in sliced_al.items()}

    median_frs_bl = {k: v.median(dim='onsets') for k, v in frs_bl.items()}
    vis_responsive = {
        k: [ranksums(*fr)[1] < 0.05 and bool(median_fr > 2.)
            for fr, median_fr in zip(frs_bl[k].T, median_frs_bl[k][1])]
        for k in median_frs_bl}

    stimuli = set(protocol.screen_stimuli)
    pair_coding = {}
    for stim in stimuli:
        np_stim = {'L' : 'P', 'P': 'L'}[stim[0]] + stim[1]
        na_stims = {s for s in stimuli if s not in (stim, np_stim)}
        sel = vis_responsive[stim]
        fr_bl_np = frs_bl[np_stim].isel(intervals=1, fr=sel)
        fr_al_np = frs_al[np_stim].isel(intervals=1, fr=sel)
        inc_np = fr_al_np - fr_bl_np
        fr_bl_np_before_stim = frs_bl[np_stim].isel(intervals=0, fr=sel)
        pair_coding_ = [
            ranksums(al, bl)[1] < 0.05 and
            bool(np.median(al) > np.median(bl)) and
            ranksums(bl, bl_before)[1] >= 0.05
            for bl, al, bl_before in zip(
                fr_bl_np.T, fr_al_np.T, fr_bl_np_before_stim.T)]
        for na_stim in na_stims:
            fr_bl_na = frs_bl[na_stim].isel(intervals=1, fr=sel)
            fr_al_na = frs_al[na_stim].isel(intervals=1, fr=sel)
            inc_na = fr_al_na - fr_bl_na
            keep = [
                ranksums(inp, ina)[1] < 0.05 and
                bool(np.mean(inp) > np.mean(ina))
                for inp, ina in zip(inc_np.T, inc_na.T)]
            pair_coding_ = [p and e for p, e in zip(pair_coding_, keep)]
        pair_coding[stim] = np.arange(trial.spikes.shape[1])[sel][pair_coding_]

    fr_response = xr.DataArray(
        np.mean(f_spikes.reshape(
            (-1, int(0.05 / dt), trial.spikes.shape[1])), axis=1),
        dims=('bins', 'spikes'))
    t_bins = np.mean(trial.trange.reshape((-1, int(0.05 / dt))), axis=1)

    eta = 0.1
    z_bl = {
        k: (fr_response - xr.concat(
            (x[0].mean(dim='t') for x in v), dim='trial').mean(dim='trial')) /
           (xr.concat((x[0].mean(dim='t') for x in v), dim='trial').std(
               dim='trial') + eta)
        for k, v in sliced_bl.items()}
    z_al = {
        k: (fr_response - xr.concat(
            (x[0].mean(dim='t') for x in v), dim='trial').mean(dim='trial')) /
           (xr.concat((x[0].mean(dim='t') for x in v), dim='trial').std(
               dim='trial') + eta)
        for k, v in sliced_al.items()}

    return dict(
        p_bl=pop_response(
            protocol, pair_coding, t_bins, z_bl, 'screen',
            lambda k, s: k == s),
        p_al=pop_response(
            protocol, pair_coding, t_bins, z_al, 'test', lambda k, s: k == s),
        np_bl=pop_response(
            protocol, pair_coding, t_bins, z_bl, 'screen',
            lambda k, s: k[0] != s[0] and k[1] == s[1]),
        np_al=pop_response(
            protocol, pair_coding, t_bins, z_al, 'test',
            lambda k, s: k[0] != s[0] and k[1] == s[1]),
        na_bl=pop_response(
            protocol, pair_coding, t_bins, z_bl, 'screen',
            lambda k, s: k[1] != s[1]),
        na_al=pop_response(
            protocol, pair_coding, t_bins, z_al, 'test',
            lambda k, s: k[1] != s[1]),
        pair_coding=pair_coding,
    )
