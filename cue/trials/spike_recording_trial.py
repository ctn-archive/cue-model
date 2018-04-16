from __future__ import absolute_import

import nengo
import nengo_spa as spa
import numpy as np
import pytry

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
        self.param("TCM prob. to recall from beginning", ordinal_prob=.1)
        self.param("noise in recall", noise=0.015)
        self.param("min. recall evidence", min_evidence=0.04)
        self.param("protocol", protocol='immediate')
        self.param("recall duration", recall_duration=45.)

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
                'stm': nengo.Probe(
                    model.cue.ose.mem.mem.output, synapse=0.01),
                'stm_spikes': nengo.Probe(
                    model.cue.ose.mem.mem.state_ensembles.ensembles[-1].neurons,
                    'spikes'),
                'm_tf_input_cue': nengo.Probe(
                    model.cue.tcm.net_m_tf.input_cue, synapse=0.01),
                'm_tf_input_target': nengo.Probe(
                    model.cue.tcm.net_m_tf.input_target, synapse=0.01),
                'm_tf_output': nengo.Probe(
                    model.cue.tcm.net_m_tf.output, synapse=0.01),
                'm_tf_output_no_filter': nengo.Probe(
                    model.cue.tcm.net_m_tf.output, synapse=None),
            }
            for c in model.cue.tcm.net_m_tf.connections:
                if (c.pre in model.cue.tcm.net_m_tf.state.all_ensembles and
                        hasattr(c.post, 'obj') and
                        c.post.obj is model.cue.tcm.net_m_tf.output):
                    self.probes['m_tf_weights'] = nengo.Probe(c, 'weights')
                    break
            self.m_tf_post = model.cue.tcm_recall_gate.state_ensembles.ensembles[-1]

        return model

    def evaluate(self, p, sim, plt):
        sim.run(4.)

        result = {
            't': sim.trange(),
            'm_tf_post_enc': sim.data[self.m_tf_post].encoders,
            'pos': sim.data[self.p_pos],
            'recalls': sim.data[self.p_recalls],
            'vocab_vectors': self.vocabs.items.vectors,
            'vocab_keys': list(self.vocabs.items.keys()),
            'pos_vectors': self.vocabs.positions.vectors,
            'pos_keys': list(self.vocabs.positions.keys()),
        }
        result.update(
            {k: sim.data[v] for k, v in self.probes.items()})
        return result
