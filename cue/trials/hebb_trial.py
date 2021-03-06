from __future__ import absolute_import

import nengo
import nengo_spa as spa
import numpy as np
import pyopencl
import pytry

from cue import Simulator
from cue.analysis.neural import model_out_to_responses
from cue.model import CUE, Vocabularies
from cue.protocols import HebbRepStimulusProvider


class HebbRepetitionTrial(pytry.PlotTrial):
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
        self.param("min. recall evidence", min_evidence=0.015)
        self.param("PyOpenCL context", cl_context=None)
        self.param("debug mode", debug=False)
        self.param("show progress bar", progress=True)
        self.param("extension", extension='forward-assoc')
        self.param("weight decay", decay=.999973176)
        self.param("repetition pattern", rep_pattern='001')
        self.param("randomize position", rand_pos='None')

    def model(self, p):
        if p.rand_pos == 'None':
            rand_pos = None
        else:
            rand_pos = int(p.rand_pos)
        self.stim_provider = HebbRepStimulusProvider(
            n_total_items=9,
            n_items_per_list=9,
            n_lists=24,
            rep_pattern=[x == '1' for x in p.rep_pattern],
            rand_pos=rand_pos,
            pi=1.,
            recall_duration=60.)
        self.vocabs = Vocabularies(
            self.stim_provider, p.item_d, p.context_d,
            self.stim_provider.n_total_items,
            np.random.RandomState(p.seed + 1))

        with spa.Network(seed=p.seed) as model:
            model.cue = CUE(
                self.stim_provider, self.vocabs, p.beta, p.gamma,
                p.ose_thr, p.ordinal_prob, p.noise, p.min_evidence,
                decay=p.decay, extensions={p.extension})
            self.p_recalls = nengo.Probe(model.cue.output, synapse=0.01)
            self.p_pos = nengo.Probe(model.cue.output_pos, synapse=0.01)

            self.debug_probes = {
                'start_of_serial_recall': model.cue.start_of_serial_recall,
                'start_of_free_recall': model.cue.start_of_free_recall,
                'start_of_pres_phase': model.cue.start_of_pres_phase,
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
        with Simulator(self.model(p), context=context, progress_bar=p.progress) as sim:
            sim.run(self.stim_provider.total_duration)

        t = sim.trange()

        recall_vocab = self.vocabs.items.create_subset(self.stim_provider.get_all_items())
        responses = []
        for i in range(len(self.stim_provider.lists)):
            start = i * self.stim_provider.epoch_duration
            end = (i + 1) * self.stim_provider.epoch_duration
            sel = (start < t) & (t <= end)
            responses.append(model_out_to_responses(
                recall_vocab, t[sel] - start, sim.data[self.p_recalls][sel],
                sim.data[self.p_pos][sel], self.stim_provider))


        result = {
            'responses': responses,
            # 'pos': sim.data[self.p_pos],
            # 'recalls': sim.data[self.p_recalls],
            'vocab_vectors': self.vocabs.items.vectors,
            'vocab_keys': list(self.vocabs.items.keys()),
            'pos_vectors': self.vocabs.positions.vectors,
            'pos_keys': list(self.vocabs.positions.keys()),
            'lists': self.stim_provider.lists,
        }
        if p.debug:
            result.update(
                {k: sim.data[v] for k, v in self.debug_probes.items()})
        return result
