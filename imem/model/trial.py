from __future__ import absolute_import

import nengo
import nengo_spa as spa
import numpy as np
import pytry

from imem.model import IMem, Vocabularies
from imem.protocols import PROTOCOLS, StimulusProvider


class IMemTrial(pytry.NengoTrial):
    # pylint: disable=attribute-defined-outside-init,arguments-differ

    def params(self):
        self.param("item dimensionality", item_d=256)
        self.param("context dimensionality", context_d=256)
        self.param("contextual drift rate", beta=0.62676)
        self.param("distractor rate", distractor_rate=1.)
        self.param("OSE memory decay", gamma=0.9775)
        self.param("OSE memory threshold", ose_thr=0.2)
        self.param("TCM prob. to recall from beginning", ordinal_prob=0.2)
        self.param("noise in recall", noise=0.)
        self.param("min. recall evidence", min_evidence=0.025)
        self.param("protocol", protocol='immed')
        self.param("recall duration", recall_duration=60.)

    def model(self, p):
        self.proto = PROTOCOLS[p.protocol]
        self.stim_provider = StimulusProvider(self.proto, p.distractor_rate)
        self.vocabs = Vocabularies(
            self.stim_provider, p.item_d, p.context_d, self.proto.n_items + 3)

        with spa.Network(seed=p.seed) as model:
            model.imem = IMem(
                self.stim_provider, self.vocabs, p.beta, p.gamma,
                p.ose_thr, p.ordinal_prob, p.noise, p.min_evidence)
            self.p_recalls = nengo.Probe(model.imem.output, synapse=0.01)
            self.p_pos_ = nengo.Probe(model.imem.output_pos_, synapse=0.01)

            self.debug_probes = {
                'recall_state': model.imem.recall.state.output,
                'recall_threshold': model.imem.recall.threshold.heaviside,
                'recall_buf': model.imem.recall.buf.output,
                'pos__recall_state': model.imem.pos__recall.state.output,
                'pos__recall_buf': model.imem.pos__recall.buf.output,
                'aml_comp': model.imem.tcm.net_m_tf.compare.output,
                'ctx': model.imem.tcm.current_ctx.output,
                'ctx_update': model.imem.tcm.current_ctx.input_update_context,
                'inhib_recall': model.imem.recall.inhibit.output,
                'out_inhibit_gate': model.imem.recall.out_inhibit_gate.output,
                'recall_ctx': model.imem.tcm.net_m_ft.output,
                'recall_ctx_cue': model.imem.tcm.net_m_ft.input_cue,
                'input_pos_': model.imem.tcm.input_pos_,
                'current_ctx': model.imem.tcm.current_ctx.output,
                'input_update_ctx': model.imem.tcm.input_update_context,
                'sim_th': model.imem.sim_th.output,
                'last_item': model.imem.last_item.output,
                'ose_output': model.imem.ose.output,
                'tcm_output': model.imem.tcm.net_m_tf.output,
                'failed_recall_int': model.imem.recall.failed_recall_int,
                'failed_recall': model.imem.recall.failed_recall,
                'failed_recall_heaviside':
                    model.imem.recall.failed_recall_heaviside,
                'start_of_recall': model.imem.start_of_recall,
                'pos__state': model.imem.pos_.state.output,
                'pos__state_in': model.imem.pos_.state.input,
                'pos__inhibit_threshold': model.imem.pos_.inhibit_threshold.output,
                'pos__advance_threshold': model.imem.pos_.advance_threshold.output,
                'input_inc': model.imem.pos_.input_inc,
                'no_pos__count': model.imem.ctrl.output_no_pos__count,
                'ose_recall_gate': model.imem.ose_recall_gate.output,
                'tcm_recall_gate': model.imem.tcm_recall_gate.output,
                'pos__gate': model.imem.pos__gate.output,
                'ose': model.imem.ose.mem.output,
                'buf_input_store': model.imem.recall.buf_input_store,
                # 'out_inhib_gate_update': model.imem.recall.out_inhib_gate_update,
                # 'input_update_inhibit': model.imem.recall.input_update_inhibit,
                'sim_th_neg': model.imem.sim_th_neg,
                'sim_th_pos_': model.imem.sim_th_pos_,
            }
            if p.debug:
                for k in self.debug_probes:
                    self.debug_probes[k] = nengo.Probe(
                        self.debug_probes[k], synapse=0.01)

        return model

    def evaluate(self, p, sim, plt):
        sim.run(self.proto.duration + p.recall_duration)

        recall_vocab = self.vocabs.items.create_subset(self.stim_provider.get_all_items())
        similarity = spa.similarity(sim.data[self.p_recalls], recall_vocab)
        responses = []
        pos_itions = np.arange(self.proto.n_items)
        if self.proto.serial:
            for i in pos_itions:
                recall_phase = sim.trange() > self.proto.pres_phase_duration
                s = recall_phase & (sim.data[self.p_pos_][:, i] > 0.8)
                if np.any(s):
                    recall_for_pos_ = similarity[s][-1, :]
                else:
                    recall_for_pos_ = np.array([0.])
                if np.any(recall_for_pos_ > 0.6):
                    recalled = float(np.argmax(recall_for_pos_))
                    if len(responses) == 0 or recalled != responses[-1]:
                        responses.append(recalled)
                    else:
                        responses.append(np.nan)
                else:
                    responses.append(np.nan)
        else:
            above_threshold = similarity[np.max(similarity, axis=1) > 0.8, :]
            for x in np.argmax(above_threshold, axis=1):
                if x not in responses:
                    responses.append(float(x))
        responses = responses + (self.proto.n_items - len(responses)) * [np.nan]

        result = {
            'responses': responses,
            'pos_': sim.data[self.p_pos_],
            'recalls': sim.data[self.p_recalls],
            'pos_itions': pos_itions,
            'vocab_vectors': self.vocabs.items.vectors,
            'vocab_keys': list(self.vocabs.items.keys()),
            'pos__vectors': self.vocabs.pos_itions.vectors,
            'pos__keys': list(self.vocabs.pos_itions.keys()),
        }
        if p.debug:
            result.update(
                {k: sim.data[v] for k, v in self.debug_probes.items()})
        return result
