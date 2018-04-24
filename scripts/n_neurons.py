# coding: utf-8
from cue.model import CUE, Vocabularies
from cue.protocols import StimulusProvider, Recall

proto = Recall(
    n_items=20, pi=1., ipi=0., ri=0., lr=10., serial=False, exp_data=None)
stim_provider = StimulusProvider(proto, 0., recall_duration=60.)
vocabs = Vocabularies(stim_provider, 256, 256, proto.n_items)
model = CUE(stim_provider, vocabs, 0.6, 0.9775, 0.1, 0.1, 0.009, 0.035)
n_item_recall = model.recall.n_neurons
n_pos_recall = model.pos_recall.n_neurons

if __name__ == '__main__':
    print("TCM", model.tcm.n_neurons)
    print("ctx", model.tcm.current_ctx.n_neurons)
    print("OSE", model.ose.n_neurons)
    print("pos", model.pos.n_neurons)
    print("recall", n_item_recall + n_pos_recall)
    print("item recall", n_item_recall)
    print("pos recall", n_pos_recall)
    print("total", model.n_neurons)
