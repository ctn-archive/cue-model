from __future__ import absolute_import

import nengo
import nengo_spa as spa
import numpy as np
import pytry

from cue.model import CUE, Vocabularies
from cue.protocols import PROTOCOLS, StimulusProvider


class CueTrial(pytry.NengoTrial):
    # pylint: disable=attribute-defined-outside-init,arguments-differ

    def params(self):
        self.param("item dimensionality", item_d=256)
        self.param("context dimensionality", context_d=256)
        self.param("contextual drift rate", beta=0.62676)
        self.param("distractor rate", distractor_rate=.4)
        self.param("OSE memory decay", gamma=0.9775)
        self.param("OSE memory threshold", ose_thr=0.1)
        self.param("TCM prob. to recall from beginning", ordinal_prob=.1)
        self.param("noise in recall", noise=0.009)
        self.param("min. recall evidence", min_evidence=0.035)
        self.param("protocol", protocol='immed')
        self.param("recall duration", recall_duration=60.)

    def model(self, p):
        self.proto = PROTOCOLS[p.protocol]
        self.stim_provider = StimulusProvider(self.proto, p.distractor_rate)
        self.vocabs = Vocabularies(
            self.stim_provider, p.item_d, p.context_d, self.proto.n_items + 3,
            np.random.RandomState(p.seed + 1))

        with spa.Network(seed=p.seed) as model:
            model.cue = CUE(
                self.stim_provider, self.vocabs, p.beta, p.gamma,
                p.ose_thr, p.ordinal_prob, p.noise, p.min_evidence)
            self.p_recalls = nengo.Probe(model.cue.output, synapse=0.01)
            self.p_pos = nengo.Probe(model.cue.output_pos, synapse=0.01)

            self.debug_probes = {
                'recall_state': model.cue.recall.state.output,
                'recall_threshold': model.cue.recall.threshold.heaviside,
                'recall_buf': model.cue.recall.buf.output,
                'pos_recall_state': model.cue.pos_recall.state.output,
                'pos_recall_buf': model.cue.pos_recall.buf.output,
                'aml_comp': model.cue.tcm.net_m_tf.compare.output,
                'ctx': model.cue.tcm.current_ctx.output,
                'ctx_update': model.cue.tcm.current_ctx.input_update_context,
                'inhib_recall': model.cue.recall.inhibit.output,
                'out_inhibit_gate': model.cue.recall.out_inhibit_gate.output,
                'recall_ctx': model.cue.tcm.net_m_ft.output,
                'recall_ctx_cue': model.cue.tcm.net_m_ft.input_cue,
                'input_pos': model.cue.tcm.input_pos,
                'current_ctx': model.cue.tcm.current_ctx.output,
                'input_update_ctx': model.cue.tcm.input_update_context,
                'sim_th': model.cue.sim_th.output,
                'last_item': model.cue.last_item.output,
                'ose_output': model.cue.ose.output,
                'tcm_output': model.cue.tcm.net_m_tf.output,
                'failed_recall_int': model.cue.recall.failed_recall_int,
                'failed_recall': model.cue.recall.failed_recall,
                'failed_recall_heaviside':
                    model.cue.recall.failed_recall_heaviside,
                'start_of_serial_recall': model.cue.start_of_serial_recall,
                'start_of_free_recall': model.cue.start_of_free_recall,
                'pos_state': model.cue.pos.state.output,
                'pos_state_in': model.cue.pos.state.input,
                'pos_inhibit_threshold': model.cue.pos.inhibit_threshold.output,
                'pos_advance_threshold': model.cue.pos.advance_threshold.output,
                'input_inc': model.cue.pos.input_inc,
                'no_pos_count': model.cue.ctrl.output_no_pos_count,
                'ose_recall_gate': model.cue.ose_recall_gate.output,
                'tcm_recall_gate': model.cue.tcm_recall_gate.output,
                'pos_gate': model.cue.pos_gate.output,
                'ose': model.cue.ose.mem.output,
                'buf_input_store': model.cue.recall.buf_input_store,
                # 'out_inhib_gate_update': model.cue.recall.out_inhib_gate_update,
                # 'input_update_inhibit': model.cue.recall.input_update_inhibit,
                'sim_th_neg': model.cue.sim_th_neg,
                'sim_th_pos': model.cue.sim_th_pos,
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
        positions = np.arange(self.proto.n_items)
        last_recall = -1
        if self.proto.serial:
            for i in positions:
                recall_phase = sim.trange() > self.proto.pres_phase_duration
                s = recall_phase & (sim.data[self.p_pos][:, i] > 0.8)
                if np.any(s):
                    recall_for_pos = similarity[s][-1, :]
                else:
                    recall_for_pos = np.array([0.])
                if np.any(recall_for_pos > 0.6):
                    recalled = float(np.argmax(recall_for_pos))
                    if len(responses) == 0 or recalled != last_recall:
                        responses.append(recalled)
                        last_recall = recalled
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
            'pos': sim.data[self.p_pos],
            'recalls': sim.data[self.p_recalls],
            'positions': positions,
            'vocab_vectors': self.vocabs.items.vectors,
            'vocab_keys': list(self.vocabs.items.keys()),
            'pos_vectors': self.vocabs.positions.vectors,
            'pos_keys': list(self.vocabs.positions.keys()),
        }
        if p.debug:
            result.update(
                {k: sim.data[v] for k, v in self.debug_probes.items()})
        return result
