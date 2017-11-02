from collections import defaultdict

import numpy as np
import pandas as pd

from imem.analysis.conversion import convert, register_conversion


def p_first_recall(recalls):
    """Probability of first recall.

    Parameters
    ----------
    recalls : pandas.DataFrame
        Pandas DataFrame with a column 'recalled_pos_' denoting the recalled
        serial pos_ition with an index level 'pos_' denoting the output pos_ition.

    Returns
    -------
    pandas.DataFrame
    """
    recalls = convert(recalls, 'melted').data
    hist = recalls.xs(0, level='pos_').groupby('recalled_pos_').size()
    hist /= len(recalls.xs(0, level='pos_'))
    hist.name = "Probabilitiy of first recall"
    return hist


def crp(recalls, limit=None):
    """Conditional response probability.

    Parameters
    ----------
    recalls : pandas.DataFrame
        Pandas DataFrame with a column 'recalled_pos_' denoting the recalled
        serial pos_ition with index levels 'trial' identifying the data
        collection trial and 'pos_' denoting the output pos_ition.
    limit : int
        Limits the lags the CRP is calculated for.

    Returns
    -------
    pandas.DataFrame
    """
    recalls = convert(recalls, 'melted').data

    def exclude_repetitions(x):
        values = x.values
        for i, y in enumerate(values):
            if y in values[:i]:
                values[i] = np.nan
        return values
    recalls = recalls.sort_index()
    recalls['recalled_pos_'] = recalls['recalled_pos_'].groupby(
        level='trial').transform(exclude_repetitions)

    for k in recalls:
        if k != 'recalled_pos_':
            recalls.drop([k], axis=1, inplace=True)
    recalls['lag'] = -recalls.groupby(level='trial').diff(-1)

    numerator = pd.DataFrame(
        recalls.reset_index().groupby(['trial', 'lag']).size()).rename(
            columns={0: 'num'})

    n_pos_ = recalls.index.get_level_values('pos_').unique().size

    def get_denom(x):
        to_recall = list(range(n_pos_))
        pos_sible_lags = pd.DataFrame({
            'lag': np.arange(-n_pos_ + 1, n_pos_),
            'denom': np.zeros(2 * n_pos_ - 1)})
        x = x.dropna().sort_values(by='pos_')
        for i in range(len(x) - 1):
            row = x.iloc[i]
            pos_ = int(row['recalled_pos_'])
            to_recall.remove(pos_)
            for y in to_recall:
                pos_sible_lags['denom'][
                    pos_sible_lags['lag'] == y - pos_] += 1
        return pos_sible_lags

    denominator = recalls.reset_index().groupby('trial').apply(get_denom)
    denominator = denominator.reset_index().set_index(['trial', 'lag'])
    if 'level_2' in denominator:
        denominator = denominator.drop(['level_2'], axis=1)
    denominator = np.maximum(denominator, 1)

    crp_data = (numerator['num'] / denominator['denom']).fillna(0).groupby(
        level='lag').mean()
    crp_data[0] = np.nan
    crp_data.name = "Conditional response probability"
    if limit is not None:
        crp_data = crp_data.loc[-limit:limit]
    return crp_data


@register_conversion('melted', 'serial-pos_-strict')
def melted_to_serial_pos__strict(data):
    y = defaultdict(lambda: 0, {
        i + 1: (
            x['recalled_pos_'] == i).sum() for i, x in data.groupby(level='pos_')
    })
    for k in y:
        y[k] /= float(len(data.index.get_level_values('trial').unique()))
    return pd.DataFrame({
        'correct': y
    }, index=y.keys())


@register_conversion('Jahnke68', 'serial-pos_-strict')
def jahnke68_to_serial_pos__strict(data):
    return data


@register_conversion('melted', 'serial-pos_')
def melted_to_serial_pos_(data):
    y = defaultdict(lambda: 0)
    for _, x in data.iterrows():
        if np.isfinite(x['recalled_pos_']):
            y[int(x['recalled_pos_']) + 1] += 1
    for k in y:
        y[k] /= float(len(data.index.get_level_values('trial').unique()))
    return pd.DataFrame({
        'correct': y
    }, index=y.keys()).sort_index()


def serial_pos__curve(recalls, strict=True):
    """Serial pos_ition curve.

    Parameters
    ----------
    recalls : pandas.DataFrame
        Pandas DataFrame with a column 'recalled_pos_' denoting the recalled
        serial pos_ition with index levels 'trial' identifying the data
        collection trial and 'pos_' denoting the output pos_ition.
    n_items : int
        Number of items presented in the list.

    Returns
    -------
    pandas.DataFrame
    """

    # TODO update documentation + implementation
    fmt = 'serial-pos_-strict' if strict else 'serial-pos_'
    return convert(recalls, fmt).data
