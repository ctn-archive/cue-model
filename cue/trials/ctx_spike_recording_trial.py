from __future__ import absolute_import

import nengo
import nengo_spa as spa
import numpy as np
import pyopencl
import pytry

from cue import Simulator
from cue.model import CUE, Vocabularies
from cue.protocols import PROTOCOLS, StimulusProvider


class SpikeRecordingTrial(pytry.NengoTrial):
    # pylint: disable=attribute-defined-outside-init,arguments-differ

    def params(self):
        self.param("item dimensionality", item_d=256)
        self.param("context dimensionality", context_d=256)
        self.param("contextual drift rate", beta=0.62676)
        self.param("distractor rate", distractor_rate=.4)
        self.param("OSE memory decay", gamma=0.9775)
        self.param("OSE memory threshold", ose_thr=0.1)
        self.param("TCM prob. to recall from beginning", ordinal_prob=.0)
        self.param("noise in recall", noise=0.015)
        self.param("min. recall evidence", min_evidence=0.04)
        self.param("protocol", protocol='immediate')
        self.param("recall duration", recall_duration=45.)
        self.param("PyOpenCL context", cl_context=None)
        self.param("show progress bar", progress=True)

    def model(self, p):
        self.proto = PROTOCOLS[p.protocol]
        self.stim_provider = StimulusProvider(
            self.proto, p.distractor_rate, p.recall_duration)
        self.vocabs = Vocabularies(
            self.stim_provider, p.item_d, p.context_d, self.proto.n_items + 3,
            np.random.RandomState(p.seed + 1))

        with spa.Network(seed=p.seed) as model:
            model.cue = CUE(
                self.stim_provider, self.vocabs, p.beta, p.gamma,
                p.ose_thr, p.ordinal_prob, p.noise, p.min_evidence)
            self.p_recalls = nengo.Probe(model.cue.output, synapse=0.01)
            self.p_pos = nengo.Probe(model.cue.output_pos, synapse=0.01)

            self.probes = {
                'ctx': nengo.Probe(
                    model.cue.tcm.current_ctx.current.mem.output, synapse=0.01),
                'ctx_spikes': nengo.Probe(
                    model.cue.tcm.current_ctx.current.mem.state_ensembles.ensembles[-1].neurons,
                    'spikes'),
            }

        return model

    def evaluate(self, p, sim, plt):
        context = p.cl_context
        if context is None:
            context = pyopencl.create_some_context(interactive=True)
        with Simulator(self.model(p), context=context, progress_bar=p.progress) as sim:
            sim.run(self.proto.duration)

        result = {
        }
        result.update(
            {k: sim.data[v] for k, v in self.probes.items()})
        return result
