from sklearn import linear_model
import statsmodels.api as sm
import scipy.stats as stats
from matplotlib.ticker import MultipleLocator
import numpy as np
from analyse_type import average_wscores, bar_chart_words
from tqdm import tqdm
import re
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import kruskal, f_oneway, mannwhitneyu, t
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import os


targets = [
        'resilience',
        'resilient',
        'sustainable',
        'sustainability',
        'wellbeing'
        ]
k_range = (2, 10)
root = '/home/gog/projects/gov_lv'
data_root = '/media/gog/external2/corpora/gov_corp'
so_path = f'{root}/s_measures_output_{k_range[0]}_{k_range[1]}'
sa_path = f'{root}/sense_analysis'
to_path = f'{root}/t_measures_output_sample'
ta_path = f'{root}/type_analysis'


def scatter_ts_topscores(targets, st_y_d, wt_d, sa_path, min_num=20, colours=[[0, 0, 1], [0.5, 0.5, 0.5]], alphas=[0.5, 0.2], x_label='Top $S^\#$', y_label='Top $W^\#$', font_size=20, tick_size=15):

    tsenses = [f'{w}_{i}' for w in targets for i in range(10)]
    categories = list(set([cat for y in st_y_d for cat in st_y_d[y]]))
    X = []
    y = []
    for cat in categories:
        senses = list(set([s for y in st_y_d for s in st_y_d[y][cat] \
                if s not in tsenses]))
        for s in senses:
            specs = [st_y_d[y][cat][s] for y in st_y_d if s in st_y_d[y][cat]]
            vols = [wt_d[y][cat][s] for y in wt_d if s in st_y_d[y][cat]]
            if len(specs) >= min_num and len(vols) >= min_num:
                X.append(max(specs))
                y.append(max(vols))
    plt.figure(figsize=(6, 6))
    plt.scatter(X, y, color=colours[1], alpha=alphas[1])
    tX = []
    ty = []
    for s in tsenses:
        for cat in categories:
            specs = [st_y_d[y][cat][s] for y in st_y_d if s in st_y_d[y][cat]]
            vols = [wt_d[y][cat][s] for y in wt_d if s in st_y_d[y][cat]]
            if specs != [] and vols != []:
                tX.append(max(specs))
                ty.append(max(vols))
    plt.scatter(tX, ty, color=colours[0], alpha=alphas[0])
    if x_label is not None:
        plt.xlabel(x_label, fontsize=font_size)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=font_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.savefig(f'{sa_path}/top_ss_sv_all_scatter.png', bbox_inches='tight')   
    t_group = [[tX[i], ty[i]] for i, _ in enumerate(tX)]
    o_group = [[X[i], y[i]] for i, _ in enumerate(X)]
 
    return t_group, o_group


def cohen_d(x, y, z=2.58):

    m1 = np.average(x)
    m2 = np.average(y)
    std1 = np.std(x)
    std2 = np.std(y)
    n1 = len(x)
    n2 = len(y)
    pooled_std = np.sqrt(
            (((n1 - 1) * std1 ** 2) + ((n2 - 1) * std2 ** 2)) / (n1 + n2 - 2)
            )
    d = (m1 - m2) / pooled_std
    sigma_d = np.sqrt(((n1 + n1) / (n1 * n2)) + (d ** 2 / (2 * (n1 + n2))))
    lb = d - z * sigma_d
    ub = d + z * sigma_d 

    return d, lb, ub, m1, m2 


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a_s', '-additional_senses', required=True)
    args = parser.parse_args()
    a_s = [int(v) for v in args.a_s.split('-')]
    
    with open(f'{so_path}/as_output_{a_s[0]}_{a_s[1]}/st_y_d.pickle', 'rb') as f_name:
        st_y_d = pickle.load(f_name)
    with open(f'{so_path}/as_output_{a_s[0]}_{a_s[1]}/wt_d.pickle', 'rb') as f_name:
        wt_d = pickle.load(f_name)

    #scatter top scores and get welch t-test effect sizes
    t_group, o_group = scatter_ts_topscores(targets, st_y_d, wt_d, sa_path)   
    p_vals = []
    #specs
    t_s = [lst[0] for lst in t_group]
    o_s = [lst[0] for lst in o_group]
    t, p = stats.ttest_ind(t_s, o_s, equal_var=False)
    d, lb, ub, m1, m2 = cohen_d(t_s, o_s)
    print((t, p, d, lb, ub, m1, m2))
    #vol
    t_v = [lst[1] for lst in t_group]
    o_v = [lst[1] for lst in o_group]
    t, p = stats.ttest_ind(t_s, o_v, equal_var=False)
    d, lb, ub, m1, m2 = cohen_d(t_v, o_v)
    print((t, p, d, lb, ub, m1, m2))

if __name__ == '__main__':
    main()



