import scipy.stats as stats
import math
import analyse_type as at
from statsmodels.regression import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import analyse_diffusion as ad
import statistics
import utils.misc as mi
from tqdm import tqdm
import enchant
import os
import pickle

k_range = (2, 10)
targets = None
#targets = [
#        'resilience',
#        'resilient',
#        'sustainable',
#        'sustainability',
##        'system',
#        'wellbeing',
##        'environment',
##        'climate'
#        ]

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

name = '_'.join([str(k_range[0]), str(k_range[1])])
root = '/home/gog/projects/gov_lv'
data_root = '/media/gog/external2/corpora'
data_path = f'{data_root}/gov_corp/sample'
to_path = f'{os.getcwd()}/t_measures_output_sample'
so_path = f'{os.getcwd()}/s_measures_output_{name}'
c_path = f'{data_root}/gov_corp/clusters_{name}'

years = [str(2000 + i) for i in range(21)]

####

def avg_years(snpmi_y_d, categories=None):
    
    years = [y for y in snpmi_y_d]
    if categories is None:
        categories = set(list([cat for y in snpmi_y_d for cat in snpmi_y_d[y]]))
    snpmi_d = {cat: {} for cat in categories}
    for cat in categories:
        senses = []
        for y in years:
            senses += [s for s in snpmi_y_d[y][cat]]
        senses = list(set(senses))
        for s in tqdm(senses):
            scores = [snpmi_y_d[y][cat][s] for y in years if s in snpmi_y_d[y][cat]]
            snpmi_d[cat][s] = statistics.mean(scores)

    return snpmi_d


def rekey_snpmi_y_d(snpmi_y_d):
  
    for y in snpmi_y_d:
        snpmi_y_d[y]['home_office'] = snpmi_y_d[y].pop('home')
        snpmi_y_d[y]['cabinet_office'] = snpmi_y_d[y].pop('cabinet')

    return snpmi_y_d


def avg_senses(snpmi_d, categories=None):

    if categories is None:
        categories = [cat for cat in snpmi_d]
    cavg_s_d = {}
    for cat in categories:
        cavg_s_d[cat] = statistics.mean([snpmi_d[cat][s] for s in snpmi_d[cat]])    
    return cavg_s_d


def m_scores(snpmi_d, all_years_sc, categories=None, only_eng=True):

    if categories is None:
        categories = set(list([cat for y in snpmi_y_d for cat in snpmi_y_d[y]]))
    m_d = {cat: {} for cat in categories}
    if only_eng:
        eng_d = enchant.Dict('en_GB')
    all_years_sc['home_office'] = all_years_sc.pop('home')
    all_years_sc['cabinet_office'] = all_years_sc.pop('cabinet')
    for cat in all_years_sc:
        vocab = list(set([s.split('_')[0] for s in all_years_sc[cat] if eng_d.check(s.split('_')[0])]))
        m_d[cat] = {w: None for w in vocab}
        for w in tqdm(vocab):
            s_lst = [(s, all_years_sc[cat][s]) for s in all_years_sc[cat] if s.split('_')[0] == w]
            mf_s = sorted(s_lst, key=lambda x: x[1])[-1][0]
            m_d[cat][w] = snpmi_d[cat][mf_s]
    
    return m_d


def get_wcounts(cby_d, categories=categories):

    if categories is None:
        categories = list(set([cat for y in cby_d for cat in cby_d[y]])) 
    w_counts = {cat: {} for cat in categories}
    for cat in categories:
        vocab = []
        for y in cby_d:
            vocab += [w for w in cby_d[y][cat] if not w.isdigit()]
        vocab = list(set(vocab))
        for w in vocab:
            w_sum = 0
            for y in cby_d:
                if w in cby_d[y][cat]:
                    w_sum += cby_d[y][cat][w]
            w_counts[cat][w] = w_sum

    return w_counts


def distinctiveness_lb(w_counts, tnpmi_d, m_d, categories=categories, prop=1, percentile=0.98):

    if categories is None:
        categories = [cat for cat in w_counts]
    all_tnpmi = []
    all_m = []
    for cat in categories:
        all_tnpmi += [tnpmi_d[cat][w] for w in tnpmi_d[cat]]
        all_m += [m_d[cat][w] for w in m_d[cat]]
    min_tnpmi = sorted(all_tnpmi)[int(len(all_tnpmi) * percentile)]
    min_m = sorted(all_m)[int(len(all_m) * percentile)]
    distlb_d = {cat: {} for cat in categories}
    for cat in categories:
        top_n = sorted([(w, w_counts[cat][w]) for w in w_counts[cat]], key=lambda x: x[1], reverse=True) 
        top_n = [w for w, c in top_n[: int(len(top_n) * prop)] if w in m_d[cat] and w in tnpmi_d[cat]]
        distlb_d[cat]['tnpmi'] = len([w for w in top_n if w in tnpmi_d[cat] and tnpmi_d[cat][w] > min_tnpmi]) / len(top_n)
        distlb_d[cat]['m'] = len([w for w in top_n if w in m_d[cat] and m_d[cat][w] > min_m]) / len(top_n)

    return distlb_d
        

def spearman_scatter(X, y, path, x_label=None, y_label=None, title=None, font_size=15, tick_size=15, alpha=1, conf_int=0.99, colour=[0, 0, 1]):
    
    rho, p_val = stats.spearmanr(X, y)
    n = len(X)
    stderr = 1 / math.sqrt(n - 3)
    z_s = stats.norm.ppf(1 - (1 - conf_int) / 2)
    delta = z_s * stderr
    c_i = (
            math.tanh(math.atanh(rho) - delta),
            math.tanh(math.atanh(rho) + delta)
            ) 
    plt.rc('font', size=font_size)
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(X, y, c=np.array([colour]), alpha=alpha)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=font_size)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=font_size)
    title_string =  fr'$\rho$={round(rho, 3)}, p={round(p_val, 3)}'
    if title is None:
        plt.title(f'Spearman\'s {title_string}')
    else:
        plt.title(f'{title}, Spearman\'s {title_string}')

    plt.savefig(path, bbox_inches='tight')
    return rho, p_val, c_i



def main():
    
    sa_path = f'{root}/sense_analysis'
    if not os.path.exists(sa_path):
        os.makedirs(sa_path)
    
    with open(f'{so_path}/snpmi_y_d.pickle', 'rb') as f_name:
        snpmi_y_d = pickle.load(f_name)
    with open(f'{so_path}/snvol_d.pickle', 'rb') as f_name:
        snvol_d = pickle.load(f_name)
    with open(f'{so_path}/all_years_sc.pickle', 'rb') as f_name:
        all_years_sc = pickle.load(f_name)
    with open(f'{to_path}/tnpmi_y_d.pickle', 'rb') as f_name:
        tnpmi_y_d = pickle.load(f_name)
    with open(f'{to_path}/cby_d.pickle', 'rb') as f_name:
        cby_d = pickle.load(f_name)
#    with open(f'{to_path}/avg_dis.pickle', 'rb') as f_name:
#        avg_dis = pickle.load(f_name)

    #get M_c(t)
    snpmi_y_d = rekey_snpmi_y_d(snpmi_y_d)
    snvol_d = rekey_snpmi_y_d(snvol_d)
    snpmi_d = mi.get_items('snpmi_d', so_path, avg_years, snpmi_y_d, categories=categories)
    tnpmi_d = mi.get_items('tnpmi_d', to_path, avg_years, tnpmi_y_d, categories=None)
    m_d = mi.get_items('m_d', so_path, m_scores, snpmi_d, all_years_sc, categories=categories)
    
    #spearman correlation between sense and type specificity    
    path = f'{sa_path}/st_spec_corr'
    if not os.path.exists(path):
        os.makedirs(path)
    w_counts = get_wcounts(cby_d)
    distlb_d = distinctiveness_lb(w_counts, tnpmi_d, m_d)   
    X = [distlb_d[cat]['tnpmi'] for cat in distlb_d]
    y = [distlb_d[cat]['m'] for cat in distlb_d] 
    x_label = 'Fraction of words with high $T*$'
    y_label = 'Fraction of words with high $M*$'
    spearman_scatter(X, y, f'{path}/all_cats', x_label=x_label, y_label=y_label)
    
    #bar chart comparing categories' fractions of words with high average M*
    i_d = {cat: distlb_d[cat]['m'] for cat in categories}
    at.bar_chart(i_d, sa_path, t_bar='leg', font_size=18, y_label='Fraction of words with high $M*$') 

    #linear regression with interaction term between sense specificity and sense volatility
    path = f'{sa_path}/all_years_max_score'
    if not os.path.exists(path):
        os.makedirs(path)
    ts_d = at.top_scores(snpmi_y_d, snvol_d, path, x_label='Top $S*$', y_label='Top $W*$')
    at.top_scores_lri(ts_d, sa_path, x_label='Top $S*$', y_label='Top $W*$')


if __name__ == '__main__':
    main()

