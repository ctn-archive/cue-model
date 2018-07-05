from __future__ import absolute_import

import nengo
import nengo_spa as spa
import numpy as np
import pyopencl
import pytry

from cue import Simulator
from cue.model import CUE, Vocabularies
from cue.protocols import MixedSelStimulusProvider


class MixedSelTrial(pytry.PlotTrial):
    # pylint: disable=attribute-defined-outside-init,arguments-differ

    def params(self):
        self.param("item dimensionality", item_d=256)
        self.param("context dimensionality", context_d=256)
        self.param("contextual drift rate", beta=0.62676)
        self.param("distractor rate", distractor_rate=.0)
        self.param("OSE memory decay", gamma=0.9775)
        self.param("OSE memory threshold", ose_thr=0.1)
        self.param("noise in recall", noise=0.009)
        self.param("min. recall evidence", min_evidence=0.015)
        self.param("PyOpenCL context", cl_context=None)
        self.param("debug mode", debug=False)
        self.param("show progress bar", progress=True)
        self.param("sequence", sequence=0)

    def model(self, p):
        self.stim_provider = MixedSelStimulusProvider(p.sequence)
        self.vocabs = Vocabularies(
            self.stim_provider, p.item_d, p.context_d,
            self.stim_provider.n_total_items,
            np.random.RandomState(p.seed + 1))

        with spa.Network(seed=p.seed) as model:
            model.cue = CUE(
                self.stim_provider, self.vocabs, p.beta, p.gamma,
                p.ose_thr, 1., p.noise, p.min_evidence)
            self.p = nengo.Probe(model.cue.ose.mem.mem.output, synapse=0.01)
            self.p_spikes = nengo.Probe(
                model.cue.ose.mem.mem.state_ensembles.ensembles[-1].neurons,
                'spikes')

            self.debug_probes = {
            }
            if p.debug:
                for k in self.debug_probes:
                    self.debug_probes[k] = nengo.Probe(
                        self.debug_probes[k], synapse=0.01)

        return model

    def evaluate(self, p, plt):
        context = p.cl_context
        if context is None:
            context = pyopencl.create_some_context(interactive=True)
        model = self.model(p)
        with Simulator(model, context=context, progress_bar=p.progress) as sim:
            sim.run(self.stim_provider.total_duration)

        result = {
            'intercepts': sim.data[model.cue.ose.mem.mem.state_ensembles.ensembles[-1]].intercepts,
            'encoders': sim.data[model.cue.ose.mem.mem.state_ensembles.ensembles[-1]].encoders,
            'decoded': sim.data[self.p],
            'spikes': sim.data[self.p_spikes],
            'list': self.stim_provider.l,
            'vocab_vectors': self.vocabs.items.vectors,
            'vocab_keys': list(self.vocabs.items.keys()),
            'pos_vectors': self.vocabs.positions.vectors,
            'pos_keys': list(self.vocabs.positions.keys()),
        }
        if p.debug:
            result.update(
                {k: sim.data[v] for k, v in self.debug_probes.items()})
        return result
