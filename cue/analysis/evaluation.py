import logging
from glob import glob
import os.path
import traceback
import warnings

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from psyrun.store import AutodetectStore
from scipy.stats import kurtosis
import seaborn as sns

from cue.analysis import analysis
from cue.analysis.conversion import convert, DataRep
from cue.analysis.io import read_exp_data
from cue.protocols import PROTOCOLS


logger = logging.getLogger(__name__)
store = AutodetectStore()


def interval_overlap(m1, l1, u1, m2, l2, u2):
    return (m1 - l1 <= m2 + u2) & (m2 - l2 <= m1 + u1)


def interval_overlap_df(main_var, df1, df2):
    df1 = df1.fillna(0.)
    df2 = df2.fillna(0.)
    return interval_overlap(
        df1[main_var], df1['ci_low'], df1['ci_upp'],
        df2[main_var], df2['ci_low'], df2['ci_upp'])


def interval_overlap_aggregate(exp_data, model_data, fn):
    m1, (l1, u1) = analysis.aggregate_measure(exp_data, fn)
    m2, (l2, u2) = analysis.aggregate_measure(model_data, fn)
    x = interval_overlap(m1, m1 - l1, u1 - m1, m2, m2 - l2, u2 - m2)
    return x


def evaluate(path):
    for proto_name, proto in PROTOCOLS.items():
        try:
            proto_path = os.path.join(path, proto_name)
            if os.path.exists(proto_path):
                if proto.exp_data is None:
                    exp_data = None
                else:
                    exp_data = read_exp_data(proto.exp_data)
                model_data = DataRep(
                    'psyrun', store.load(locate_results_file(proto_path)))

                logger.info(proto_name)

                if proto.serial:
                    fig = plt.figure(figsize=(12, 4))
                    evaluate_serial_recall(
                        proto, exp_data, model_data, fig=fig)
                else:
                    fig = plt.figure(figsize=(12, 12))
                    evaluate_free_recall(proto, exp_data, model_data, fig=fig)

                fig.suptitle(path + ', ' + proto_name)
                fig.tight_layout(rect=(.0, .0, 1., .95))
        except Exception as err:
            traceback.print_exc()
            warnings.warn(str(err))


def evaluate_serial_recall(proto, exp_data, model_data, fig=None):
    if fig is None:
        fig = plt.gcf()

    evaluate_serial_pos_curve(
        proto, exp_data, model_data, ax=fig.add_subplot(1, 2, 1))
    evaluate_transpositions(
        proto, exp_data, model_data, ax=fig.add_subplot(1, 2, 2))


def evaluate_free_recall(proto, exp_data, model_data, fig=None):
    if fig is None:
        fig = plt.gcf()

    evaluate_successful_recalls(
        proto, exp_data, model_data, ax=fig.add_subplot(3, 2, 1))
    evaluate_successful_recall_dist(
        proto, exp_data, model_data, ax=fig.add_subplot(3, 2, 2))
    evaluate_p_first_recall(
        proto, exp_data, model_data, ax=fig.add_subplot(3, 2, 3))
    evaluate_crp(
        proto, exp_data, model_data, ax=fig.add_subplot(3, 2, 4))
    evaluate_serial_pos_curve(
        proto, exp_data, model_data, strict=False, ax=fig.add_subplot(3, 2, 5))


def evaluate_successful_recalls(proto, exp_data, model_data, ax=None):
    cp = iter(sns.color_palette())

    if ax is None:
        ax = plt.gca()

    plot_successful_recalls(
        model_data, proto.n_items, color=next(cp), label="model", ax=ax)
    if exp_data is not None:
        plot_successful_recalls(
            exp_data, proto.n_items, color=next(cp), label="experimental",
            ax=ax)

    ax.set_xlim(-0.5, proto.n_items + 0.5)
    ax.set_xlabel("# successful recalls")
    ax.set_ylabel("Proportion")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()


def evaluate_successful_recall_dist(proto, exp_data, model_data, ax=None):
    ev_model_data = convert(model_data, 'success_count')
    plot_dist_stats(ev_model_data.data, ax)
    if exp_data is not None:
        ev_exp_data = convert(exp_data, 'success_count')
        evaluate_dist_overlap(ev_exp_data.data, ev_model_data.data)
        plot_dist_stats(ev_exp_data.data, ax)


def evaluate_p_first_recall(proto, exp_data, model_data, ax=None):
    if ax is None:
        ax = plt.gca()

    ev_model_data = analysis.p_first_recall(model_data)
    ev_model_data['p_first'].plot(
        marker='o', label="model", ax=ax,
        yerr=ev_model_data[['ci_low', 'ci_upp']].values.T)
    if exp_data is not None:
        ev_exp_data = analysis.p_first_recall(exp_data)
        ev_exp_data['p_first'].plot(
            marker='s', label="experimental", ax=ax,
            yerr=ev_exp_data[['ci_low', 'ci_upp']].values.T)

    logger.info(
        'p_first_recall: %i/%i', int(np.sum(interval_overlap_df(
            'p_first', ev_model_data, ev_exp_data))),
        proto.n_items)

    ax.set_xlabel("Serial position")
    ax.set_ylabel("P(first recall)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.set_ylim(bottom=0.)


def evaluate_crp(proto, exp_data, model_data, ax=None, limit=6):
    if ax is None:
        ax = plt.gca()

    with warnings.catch_warnings():
        # Warning generated by plotting and I am not sure why. It doesn't seem
        # to effect anything, though.
        warnings.filterwarnings(
            'ignore', '.*converting a masked element to nan.*')

        ev_model_data = analysis.crp(model_data)
        ev_model_data['crp'].plot(
            marker='o', label="model", ax=ax,
            yerr=np.copy(ev_model_data[['ci_low', 'ci_upp']].values.T))

        if exp_data is not None:
            ev_exp_data = analysis.crp(exp_data)
            ev_exp_data['crp'].plot(
                marker='s', label="experimental", ax=ax,
                yerr=np.copy(ev_exp_data[['ci_low', 'ci_upp']].values.T))

    assert np.all(ev_exp_data.index == ev_model_data.index)
    sel = (np.abs(ev_model_data.index) <= limit) & (ev_model_data.index != 0)
    logger.info(
        'crp: %i/%i', int(np.sum(interval_overlap_df(
            'crp', ev_model_data[sel], ev_exp_data[sel]))),
        len(ev_model_data['crp'][sel]))

    ax.set_xlim(-limit, limit)
    ax.set_ylim(bottom=0.)
    ax.set_xlabel("Lag position")
    ax.set_ylabel("CRP")
    ax.legend()


def evaluate_serial_pos_curve(
        proto, exp_data, model_data, strict=True, ax=None):
    if ax is None:
        ax = plt.gca()

    ev_model_data = analysis.serial_pos_curve(model_data, strict=strict)
    ev_model_data['correct'].plot(
        marker='o', label="model", ax=ax,
        yerr=ev_model_data[['ci_low', 'ci_upp']].values.T)
    if exp_data is not None:
        ev_exp_data = analysis.serial_pos_curve(exp_data, strict=strict)
        ev_exp_data['correct'].plot(
            marker='s', label="experimental", ax=ax,
            yerr=ev_exp_data[['ci_low', 'ci_upp']].values.T)

    logger.info(
        'serial_pos_curve: %i/%i', int(np.sum(interval_overlap_df(
            'correct', ev_model_data, ev_exp_data))),
        proto.n_items)

    ax.set_xlabel("Serial position")
    ax.set_ylabel("Recall proportion")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='best')


def evaluate_transpositions(proto, exp_data, model_data, ax=None):
    if ax is None:
        ax = plt.gca()

    data = analysis.transpositions(model_data)
    ax.bar(
        data.index, data['p_transpose'],
        width=1., color=sns.color_palette()[0],
        yerr=data[['ci_low', 'ci_upp']].values.T)
    lim = np.max(np.abs(data.index.values))
    ax.set_xlim(-lim - .5, lim + .5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel("Transposition")
    ax.set_ylabel("Response proportion")


def plot_successful_recalls(
        n_successfull, n_items, ax=None, label=None, **kwargs):
    n_successfull = convert(n_successfull, 'success_count').data
    if ax is None:
        ax = plt.gca()
    ax.hist(
        n_successfull, bins=n_items + 1, range=(-0.5, n_items + 0.5),
        density=True, alpha=0.5, label=label, **kwargs)
    if label is not None:
        label = label + ' (mean)'
    ax.axvline(x=np.mean(n_successfull.data), label=label, **kwargs)


def plot_dist_stats(data, ax=None):
    if ax is None:
        ax = plt.gca()

    mean, (mean_l, mean_u) = analysis.aggregate_measure(data, np.mean)
    std, (std_l, std_u) = analysis.aggregate_measure(data, np.std)
    kur, (kur_l, kur_u) = analysis.aggregate_measure(data, kurtosis)

    ax.errorbar(range(3), [mean, std, kur], yerr=[
        [mean - mean_l, std - std_l, kur - kur_l],
        [mean_u - mean, std_u - std, kur_u - kur]], marker='o')
    # TODO remove lines
    # TODO label plot


def evaluate_dist_overlap(exp_data, model_data):
    x = (int(interval_overlap_aggregate(exp_data, model_data, np.mean)) +
        int(interval_overlap_aggregate(exp_data, model_data, np.std)) +
        int(interval_overlap_aggregate(exp_data, model_data, kurtosis)))
    logger.info('dist_overlap: %i/3', x)


def locate_results_file(path):
    candidates = glob(os.path.join(path, 'result.*'))
    if len(candidates) < 1:
        raise FileNotFoundError("No results file found in {!r}.".format(path))
    elif len(candidates) > 1:
        warnings.warn(
            "Found multiple results file in {!r}, using {!r}.".format(
                path, candidates[0]))
    return candidates[0]
