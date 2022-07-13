# +
from scipy import stats
import numpy as np
import pandas as pd

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt
# %matplotlib inline
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from pyretrodesign import pyretrodesign
# -

#robjects.r('install.packages("retrodesign")')
robjects.r('library(retrodesign)')

# check to make sure I've grabbed the equivalent functions
A = 0.5
s = 1
alpha = 0.05
df = np.inf
z = stats.t.ppf(1 - (alpha / 2), df)
p_hi = 1 - stats.t.cdf(z - (A/s), df)
p_lo = stats.t.cdf(-z - (A/s), df)
assert np.isclose(z, robjects.r(f'qt({1-alpha/2}, Inf)'))
assert np.isclose(p_hi, robjects.r(f'1 - pt({z - (A/s)}, Inf)')[0])
assert np.isclose(p_lo, robjects.r(f'pt({(-z - (A/s))}, Inf)')[0])

# +
# test the two examples given in the paper
rres = robjects.r('retrodesign(0.1, 3.28)')
rpower = rres[0][0]
rtype_s = rres[1][0]
rexaggeration = rres[2][0]
power, type_s, exaggeration, est_df = pyretrodesign(0.1, 3.28)

assert np.isclose(rpower, power)
assert np.isclose(rtype_s, type_s)
# Becasuse it depends on simulations, this one is noisy
assert np.round(rexaggeration, 0) == np.round(exaggeration, 0)

rres = robjects.r('retrodesign(2, 8.1)')
rpower = rres[0][0]
rtype_s = rres[1][0]
rexaggeration = rres[2][0]
power, type_s, exaggeration, est_df = pyretrodesign(2, 8.1)

assert np.isclose(rpower, power)
assert np.isclose(rtype_s, type_s)
# Becasuse it depends on simulations, this one is noisy
assert np.round(rexaggeration, 0) == np.round(exaggeration, 0)
# -

# print(np.random.SeedSequence().entropy)
# produced 217392773431744244676552396159683491339
seed = 217392773431744244676552396159683491339
rng = np.random.default_rng(seed)

# ## for step by step example

A = 0.5
s = 1
alpha = 0.05
df = 50
n_sims=10000

z = stats.t.ppf(1 - (alpha / 2), df)
p_hi = 1 - stats.t.cdf(z - (A/s), df)
p_lo = stats.t.cdf(-z - (A/s), df)

z

-z - (A/s)

z - (A/s)

# +
x_lo = np.linspace(-5, -z, 1000)
y_lo = stats.t.pdf(x_lo, df, loc=(A / s))  
x_med = np.linspace(-z, z, 1000)
y_med = stats.t.pdf(x_med, df, loc=(A / s))
x_hi = np.linspace( z, 5, 1000)
y_hi = stats.t.pdf(x_hi, df, loc=(A / s))

power = p_hi + p_lo
type_s = p_lo / power


estimate = A + (s * rng.standard_t(df, n_sims))

significant = np.abs(estimate) > (s * z)
exaggeration = np.mean(np.abs(estimate)[significant]) / A

est_df = pd.DataFrame(estimate, columns=['estimated'])
est_df['Significant'] = significant
est_df['Exaggeration'] = np.abs(est_df.estimated) / A
est_df['Sign'] = np.nan
est_df.loc[est_df.estimated > 0, 'Sign'] = 'Pos'
est_df.loc[est_df.estimated < 0, 'Sign'] = 'Neg'

exaggeration = np.mean(np.abs(estimate)[significant]) / A
# -

sign_palette = [sns.color_palette()[2], sns.color_palette()[1]]
exag_color = sns.color_palette()[1]


with sns.plotting_context('talk'):

    fig, ax = plt.subplots(1)
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

with sns.plotting_context('talk'):
    bar_df = [{'Sign': 'Pos', 'Power': p_hi},
              {'Sign': 'Neg', 'Power': p_lo}
             ]
    bar_df = pd.DataFrame(bar_df)
    ax = sns.barplot(x='Power', y='Sign', data=bar_df, palette=sign_palette)
    ax.set_title(f"Type S rate = {type_s:0.3f}")

with sns.plotting_context('talk'):
    ax = sns.histplot(x='Exaggeration', data=est_df.query('Significant'), hue=est_df.Sign, 
                 element='step', multiple='stack', palette=sign_palette, hue_order=['Pos', 'Neg'])
    ax.set_title(f'Mean Exaggeration Ratio= {exaggeration:0.3f}')
    xlims = ax.get_xlim()
    ax.set_xlim((0, xlims[1]))
    ax.set_xlabel('Exaggeration Ratio')

A = 2.8
s = 1
power, type_s, exaggeration, est_df = pyretrodesign(A, s)
power, type_s, exaggeration

A = 2.8
s = 1
power, type_s, exaggeration, est_df = pyretrodesign(A, s)
power, type_s, exaggeration

power, type_s, exaggeration, est_df = pyretrodesign(0.5, 1, df=50, make_plots=True)


power, type_s, exaggeration, est_df = pyretrodesign(0.5, 1, df=50, make_plots=True)


power, type_s, exaggeration, est_df = pyretrodesign(A, s, make_plots=True)


power, type_s, exaggeration, est_df = pyretrodesign(0.1, 3.3, df=2971, make_plots=True, slims=(0,100), elims=(0,150))


power, type_s, exaggeration, est_df = pyretrodesign(0.3, 3.3, df=2971, make_plots=True, slims=(0,100), elims=(0,150))


power, type_s, exaggeration, est_df = pyretrodesign(3, 3.3, df=2971, make_plots=True, slims=(0,100), elims=(0,150))


# # Somewhat hacky code to make some widgets

from ipywidgets import interact


def widgetretrodesign(A, s, sims, alpha=0.05, df=np.inf):
    z = stats.t.ppf(1 - (alpha / 2), df)
    p_hi = 1 - stats.t.cdf(z - (A/s), df)
    p_lo = stats.t.cdf(-z - (A/s), df)
    power = p_hi + p_lo
    if A > 0:
        type_s = p_lo / power
    else:
        type_s = p_hi / power

    # numpy standard_t doesn't seem to deal with infinite degrees of freedom correctly
    # manually cludging this for now
    estimate = A + (s * sims)

    significant = np.abs(estimate) > (s * z)
    if A != 0:
        exaggeration = np.mean(np.abs(estimate)[significant]) / A
    else:
        exaggeration = np.nan
    
    est_df = pd.DataFrame(estimate, columns=['estimated'])
    est_df['Significant'] = significant
    est_df['Exaggeration'] = np.abs(est_df.estimated) / A

    return power, type_s, exaggeration, est_df


nsims=10000
sims = rng.standard_t(2971, nsims)

# +
import matplotlib.ticker as ticker

@ticker.FuncFormatter
def pct_formatter(x, pos):
    return f"{int(x/10000 * 100):d}%" 


# -

sign_palette = [sns.color_palette()[2], sns.color_palette()[3]]
exag_color = sns.color_palette()[1]
plims = (-15, 15)
slims = (0, 500)
elims = (0, 125)


def update(A=1, s=1, alpha=0.05, df=100):
    power, type_s, exaggeration, est_df = pyretrodesign(A, s, alpha, df=50, make_plots=True, plims=(-10,10))



interact(update, A=(0, 5, 0.05), s=(0.1, 5, 0.05), alpha = (0.001, 0.5, 0.001), df=(5, 5000, 1));


