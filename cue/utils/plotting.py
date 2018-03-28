import os
import os.path

import seaborn as sns


textwidth = 390.56151 / 72.27
row_height = textwidth / 1.5 / 1.6
single_fig_size = (textwidth / 1.5, row_height)


def save_tight_fig(fig, path, **kwargs):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    fig.savefig(path, bbox_inches='tight', pad_inches=0., **kwargs)


def style():
    weight = 'light'
    sns.set('paper', 'ticks', None, rc={
        'axes.labelweight': weight,
        'axes.linewidth': 0.8,
        'axes.titleweight': weight,
        'font.cursive': ['cmsy10'],
        'font.sans-serif': ['Lato'],
        'font.weight': weight,
        'mathtext.fontset': 'custom',
    })
