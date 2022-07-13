from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt


def pyretrodesign(A, s, alpha=0.05, df=np.inf, n_sims=10000, rng=None, make_plots=False,
                  plims=None, slims=None, elims=None):
    if rng is None:
        # print(np.random.SeedSequence().entropy)
        # produced 217392773431744244676552396159683491339
        seed = 217392773431744244676552396159683491339
        rng = np.random.default_rng(seed)
    z = stats.t.ppf(1 - (alpha / 2), df)
    p_hi = 1 - stats.t.cdf(z - (A / s), df)
    p_lo = stats.t.cdf(-z - (A / s), df)
    power = p_hi + p_lo
    if A > 0:
        type_s = p_lo / power
    else:
        type_s = p_hi / power

    # numpy standard_t doesn't seem to deal with infinite degrees of freedom correctly
    # manually cludging this for now
    if df is np.inf:
        estimate = A + (s * rng.standard_normal(n_sims))
    else:
        estimate = A + (s * rng.standard_t(df, n_sims))

    significant = np.abs(estimate) > (s * z)
    exaggeration = np.mean(np.abs(estimate)[significant]) / A

    est_df = pd.DataFrame(estimate, columns=['estimated'])
    est_df['Significant'] = significant
    est_df['Exaggeration'] = np.abs(est_df.estimated) / A
    est_df['Sign'] = np.nan
    est_df.loc[est_df.estimated > 0, 'Sign'] = 'Pos'
    est_df.loc[est_df.estimated < 0, 'Sign'] = 'Neg'

    if make_plots:
        sign_palette = [sns.color_palette()[2], sns.color_palette()[1]]
        exag_color = sns.color_palette()[1]
        if df is np.inf:
            dist = stats.norm(loc=A / s)
        else:
            dist = stats.t(df=df, loc=(A / s))
        bottom_x = dist.ppf(0.001)
        top_x = dist.ppf(0.9999)
        x_lo = np.linspace(bottom_x, -z, 1000)
        y_lo = dist.pdf(x_lo)
        x_med = np.linspace(-z, z, 1000)
        y_med = dist.pdf(x_med)
        x_hi = np.linspace(z, top_x, 1000)
        y_hi = dist.pdf(x_hi)

        with sns.plotting_context('talk'):

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
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
            ax = sns.histplot(x='Exaggeration', data=est_df.query('Significant'), hue=est_df.Sign,
                              element='step', multiple='stack', palette=sign_palette, hue_order=['Pos', 'Neg'], ax=ax)
            ax.set_title(f'Mean Exaggeration Ratio= {exaggeration:0.3f}')
            xlims = ax.get_xlim()
            ax.set_xlim((0, xlims[1]))
            ax.set_xlabel('Exaggeration Ratio')
            if elims is not None:
                ax.set_xlim(elims)
            fig.tight_layout()

    return power, type_s, exaggeration, est_df