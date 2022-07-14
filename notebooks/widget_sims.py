from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
# %matplotlib widget 
from matplotlib import pyplot as plt
from pyretrodesign import pyretrodesign
from ipywidgets import interact

# +
fig_exist = False
plt.ioff()
fig, axes = plt.subplots(1, 3, figsize=(10,4))
fig.canvas.toolbar_visible = False
fig.canvas.header_visible = False # Hide the Figure name at the top of the figure

power, type_s, exaggeration = pyretrodesign(A=1, s=1, alpha=0.05, df=100, make_plots=True, plims=(-10,10), fig=fig, axes=axes)
fig.tight_layout()
plt.ion()
def update(A=1, s=1, alpha=0.05, df=100):
    global fig
    global axes
    power, type_s, exaggeration = pyretrodesign(A, s, alpha, df=df, make_plots=True, plims=(-10,10), fig=fig, axes=axes)
    fig.show()
interact(update, A=(0.01, 5, 0.05), s=(0.1, 5, 0.05), alpha = (0.001, 0.5, 0.001), df=(5, 5000, 1));
# -



interact(update, A=(0, 5, 0.05), s=(0.1, 5, 0.05), alpha = (0.001, 0.5, 0.001), df=(5, 5000, 1));


