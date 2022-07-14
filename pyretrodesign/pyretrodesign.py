from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt


def pyretrodesign(A, s, alpha=0.05, df=np.inf, n_sims=10000, rng=None, make_plots=False,
                  plims=None, slims=None, elims=None, fig=None, axes=None, context='notebook'):
    if A < 0:
        raise ValueError(f'A should be positive, you passed {A}.')
    if rng is None:
        # print(np.random.SeedSequence().entropy)
        # produced 217392773431744244676552396159683491339
        seed = 217392773431744244676552396159683491339
        rng = np.random.default_rng(seed)
    if df is np.inf:
        z = stats.norm.ppf(1 - (alpha / 2))
        p_hi = 1 - stats.norm.cdf(z - (A / s))
        p_lo = stats.norm.cdf(-z - (A / s))
    else:
        z = stats.t.ppf(1 - (alpha / 2), df)
        p_hi = 1 - stats.t.cdf(z - (A / s), df)
        p_lo = stats.t.cdf(-z - (A / s), df)
    power = p_hi + p_lo
    type_s = p_lo / power
    exaggeration, ex_hi, ex_lo, ey_hi, ey_lo = _calc_exaggeration(A, s, df, alpha, power)

    if make_plots:
        sign_palette = [sns.color_palette()[2], sns.color_palette()[1]]
        exag_color = sns.color_palette()[1]
        if df is np.inf:
            dist = stats.norm(loc=A / s)
        else:
            dist = stats.t(df=df, loc=(A / s))
        bottom_x = dist.ppf(0.00001)
        top_x = dist.ppf(0.99999)
        x_lo = np.linspace(bottom_x, -z, 1000)
        y_lo = dist.pdf(x_lo)
        x_med = np.linspace(-z, z, 1000)
        y_med = dist.pdf(x_med)
        x_hi = np.linspace(z, top_x, 1000)
        y_hi = dist.pdf(x_hi)

        with sns.plotting_context(context):
            if fig is None or axes is None:
                fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            else:
                [ax.clear() for ax in axes]
            ax = axes[0]
            ax.fill_between(x_lo, y_lo, color=sns.color_palette()[1])
            ax.fill_between(x_med, y_med, color=sns.color_palette()[0])
            ax.fill_between(x_hi, y_hi, color=sns.color_palette()[2])
            ylim = ax.get_ylim()
            ax.vlines([-z, z], 0, ylim[1], color='red', linestyles="--", label='z = 2.01')
            ax.set_ylim(ylim)
            ax.legend()
            ax.set_ylabel("Density")
            ax.set_xlabel("t-value")
            ax.set_title(f'Power = {power:0.3f}')
            if plims is not None:
                ax.set_xlim(plims)

            ax = axes[1]
            bar_df = [{'Sign': 'Pos', 'Percent of Sig Results': p_hi / power * 100},
                      {'Sign': 'Neg', 'Percent of Sig Results': p_lo / power * 100}
                      ]
            bar_df = pd.DataFrame(bar_df)
            ax = sns.barplot(x='Percent of Sig Results', y='Sign', data=bar_df, palette=sign_palette, ax=ax)
            ax.set_title(f"Type S rate = {type_s:0.3g}")
            if slims is not None:
                ax.set_xlim(slims)

            ax = axes[2]
            ax.fill_between(ex_hi / A, ey_hi*A/power + ey_lo*A/power, color=sns.color_palette()[2], label='Pos')
            ax.fill_between(np.abs(ex_lo / A), ey_lo*A/power, color=sns.color_palette()[1], label='Neg')
            ax.legend(title='Sign')
            ax.set_title(f'Mean Exaggeration Ratio= {exaggeration:0.3f}')
            xlims = ax.get_xlim()
            ax.set_xlim((0, xlims[1]))
            ax.set_xlabel('Exaggeration Ratio')
            if elims is not None:
                ax.set_xlim(elims)
            if fig is None or axes is None:
                fig.tight_layout()

    return power, type_s, exaggeration

def _calc_exaggeration(A, s, df, alpha, power):
    if df is np.inf:
        dist = stats.norm(loc=A, scale = s)
    else:
        dist = stats.t(df=df, loc=(A ), scale = s)
    bottom_x = dist.ppf(0.0001) * 2
    top_x = dist.ppf(0.9999) * 2
    z_plot = dist.ppf(1- (alpha / 2)) - A
    x_lo = np.linspace(bottom_x, -z_plot, 10000)
    x_hi = np.linspace(z_plot, top_x, 10000)
    y_lo = dist.pdf(x_lo)
    y_hi = dist.pdf(x_hi)
    ext_hi = x_hi / A
    eyt_hi = y_hi * A / power
    ext_lo = np.abs(x_lo / A)
    eyt_lo = y_lo * A / power
    exaggeration = np.sum(ext_hi * eyt_hi)/ np.sum(eyt_hi) + np.sum(eyt_lo * eyt_lo)/ np.sum(eyt_lo)
    # cludge for making the expectation figure
    x_lo = -x_hi[x_hi < top_x / 2]
    y_lo = dist.pdf(x_lo)
    return exaggeration, x_hi[x_hi < top_x / 2], x_lo, y_hi[x_hi < top_x / 2], y_lo