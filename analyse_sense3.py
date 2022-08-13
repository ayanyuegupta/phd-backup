import analyse_sense as an_s
import argparse
import scipy.stats as stats
import math
import analyse_type as at
from statsmodels.regression import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import utils.misc as misc
from tqdm import tqdm
import os
import pickle

k_range = (2, 10)
targets = [
        'resilience',
        'resilient',
        'sustainable',
        'sustainability',
        'wellbeing'
        ]
name = '_'.join([str(k_range[0]), str(k_range[1])])
root = '/home/gog/projects/gov_lv'
data_root = '/media/gog/external2/corpora'
data_path = f'{data_root}/gov_corp/sample'
to_path = f'{root}/t_measures_output_sample'
so_path = f'{root}/s_measures_output_{name}'
sa_path = f'{root}/sense_analysis'
c_path = f'{data_root}/gov_corp/clusters_{name}'

years = [str(2000 + i) for i in range(21)]


def scatter_stwt_scores(st_y_d, wt_d, sa_path, target_words=[]):
    
    plt.figure(figsize=(6, 6))
    categories = list(set([cat for y in st_y_d for cat in st_y_d[y]]))
    X = []
    y = []
    tX = []
    ty = [] 
    for cat in categories:
        senses = list(set([s for y in st_y_d for s in st_y_d[y][cat]]))
        for s in senses:
            term = s.split('_')[0]
            if term in targets:
                tX.append(np.median([st_y_d[y][cat][s] for y in st_y_d if s in st_y_d[y][cat]]))
                ty.append(np.median([wt_d[y][cat][s] for y in wt_d if s in st_y_d[y][cat]]))
            else:
                X.append(np.median([st_y_d[y][cat][s] for y in st_y_d if s in st_y_d[y][cat]]))
                y.append(np.median([wt_d[y][cat][s] for y in wt_d if s in st_y_d[y][cat]]))
    plt.scatter(X, y, color=[0.5, 0.5, 0.5], alpha=0.2)
    plt.scatter(tX, ty, color=[0, 0, 1], alpha=0.2)
    plt.savefig(f'{sa_path}/st_wt_corr.png', bbox_inches='tight')
    print(stats.spearmanr(X, y))
    print(stats.spearmanr(tX, ty))

    return X, tX, y, ty


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a_s', '-additional_senses', required=True)
    args = parser.parse_args()
    a_s = [int(v) for v in args.a_s.split('-')]

    ts_path = f'{so_path}/as_output_{a_s[0]}_{a_s[1]}'
    with open(f'{ts_path}/st_y_d.pickle', 'rb') as f_name:
        st_y_d = pickle.load(f_name)
    with open(f'{ts_path}/wt_d.pickle', 'rb') as f_name:
        wt_d = pickle.load(f_name)   
   
    #scatter top st wt scores
    scatter_stwt_cores(st_y_d, wt_d, sa_path, targets=targets)
    
    #linear regressions
#    #scatter sense specificities and volatilies
#    path = f'{sa_path}/corr_ss_sv'
#    if not os.path.exists(path):
#        os.makedirs(path)
#    for year in st_y_d:
#        for cat in st_y_d[year]:
#            X = [st_y_d[year][cat][s] for s in st_y_d[year][cat]]
#            y = [wt_d[year][cat][s] for s in st_y_d[year][cat]]
#            an_s.spearman_scatter(X, y, f'{path}/{year}_{cat}.png', x_label='$S_T$', y_label='$W_T$', title=f'{year} {cat}', alpha=0.2)


if __name__ == '__main__':
    main()


