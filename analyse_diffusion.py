from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from collections import OrderedDict
from sklearn.cluster import AgglomerativeClustering
import utils.misc as m
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from pygam import LinearGAM, s
import pandas as pd
import plotnine as p9
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import os
import ruptures as rpt


words = [
#        'summary',
#        'government',
        'resilience',
        'resilient',
        'sustainable',
        'sustainability',
        'wellbeing'
        ]

categories = [
'MOD',
'cabinet_office',
'leg',
'treasury',
'home_office',
'DHSC',
'DWP',
'DE',
'MOH',
'DCMS',
'DEFRA',
'FCO',
'MOJ'
]

i_path = f'{os.getcwd()}/t_measures_output_sample'
o_path = f'{os.getcwd()}/diffusion_analysis'
colours = m.random_colours(13)

####


def plot_gam(X, y, save_path, word, size=(6, 4), colour_label=None, x_label=None, y_label=None, custom_xticks=None, ax=None, save=True, conf_int=True, alphas=(0.3, 1), font_size=20, tick_size=15, leg_size=9, n_lcol=3, lw=3):
     
    gam = LinearGAM(s(0)).fit(X, y)
#    print(gam.summary())
    if ax is None:
        ax = plt.figure(figsize=size).gca()
    if colour_label is None:
        ax.scatter(X, y, color=np.array([[0, 0, 1]]), s=10, alpha=alphas[0])
        ax.plot(X, gam.predict(X), linewidth=lw, color=np.array([0, 0, 1]), alpha=alphas[1])
    else:
        ax.scatter(X, y, label=colour_label[1], color=np.array([colour_label[0]]), s=10, alpha=alphas[0])
        ax.plot(X, gam.predict(X), linewidth=lw, label=colour_label[1], color=np.array(colour_label[0]), alpha=alphas[1])
    if conf_int:
        ax.plot(X, gam.confidence_intervals(X), color=np.array([0, 0, 1]), ls='--')
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.yaxis.offsetText.set_fontsize(tick_size)
    ax.set_title(word, fontsize=font_size)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
#    ax.set_xticks(fontsize=tick_size)
#    ax.set_yticks(fontsize=tick_size)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=font_size)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=font_size)
    if custom_xticks is not None:
        xt = ax.get_xticks()[1:-1]
        ax.set_xticks(xt)
        custom_xticks = [custom_xticks[int(i)] for i in xt]
        ax.set_xticklabels(custom_xticks)
    
    if save:
        plt.savefig(f'{save_path}/{word}.png', bbox_inches='tight')


def get_rf(d, word, categories=None):
    
    years = sorted([y for y in d])
    if categories is None:
        categories = set([cat for y in d for cat in d[y]])
    v_d = {y: 0 for y in years}
    catv_d = {y:{cat: 0 for cat in categories} for y in years}
    for y in years:
        total = 0
        count = 0
        for cat in categories:
            cat_total = sum([d[y][cat][w] for w in d[y][cat]])
            total += cat_total   
            if word in d[y][cat]:
                count += d[y][cat][word]
                catv_d[y][cat] = d[y][cat][word] / cat_total
        v_d[y] = count / total

    return v_d, catv_d

         
def main():
    
    colours = m.random_colours(len(categories))
    fig, ax = plt.subplots(len(words), 1, figsize=(6, 18))
    for i, word in enumerate(words):

        #relative frequencies
        if not os.path.exists(o_path):
            os.makedirs(o_path)
        rf_path = f'{o_path}/change_point_detection/rel_freqs'
        if not os.path.exists(rf_path):
            os.makedirs(rf_path)
        with open(f'{i_path}/cby_d.pickle', 'rb') as f_name:
           cby_d = pickle.load(f_name)    
        rf, cat_rfs = get_rf(cby_d, word)
               
        #gam
        gam_path1 = f'{o_path}/gam/rel_freqs'
        path = f'{o_path}/gam/rel_freqs/all_cats'
        if not os.path.exists(gam_path1):
            os.makedirs(gam_path1)
        if not os.path.exists(path):
            os.makedirs(path)
        y = np.array([rf[y] for y in rf])
        X = np.array([y for y in rf])
        plot_gam(X, y, gam_path1, word, ax=ax[i], save=False)
    
    fig.text(0, 0.5, 'Relative Frequency', ha='center', va='center', rotation='vertical', fontsize=20)
    fig.tight_layout()
    plt.savefig(f'{gam_path1}/test.png', bbox_inches='tight')

if __name__ == '__main__':
    main()



