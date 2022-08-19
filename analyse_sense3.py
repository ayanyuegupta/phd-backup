from collections import Counter
import analyse_type as ant
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


def rekey_snpmi_y_d(snpmi_y_d):
  
    for y in snpmi_y_d:
        snpmi_y_d[y]['home_office'] = snpmi_y_d[y].pop('home')
        snpmi_y_d[y]['cabinet_office'] = snpmi_y_d[y].pop('cabinet')

    return snpmi_y_d


def scatter_stwt_scores(st_y_d, wt_d, o_path, targets=[], size=(6, 6), by_cat=True, prop=1):
    
    categories = list(set([cat for y in st_y_d for cat in st_y_d[y]]))
    score_d = {'top_nv': [], 'top_tnpmi': [], 'categories': []}
    if not by_cat:
        X = []
        y = []
        tX = []
        ty = [] 
    for cat in tqdm(categories):
        senses = list(set([s for y in st_y_d for s in st_y_d[y][cat]]))
        senses = sorted(senses)[: int(len(senses) * prop)]
        if by_cat:
            X = []
            y = []
            tX = []
            ty = [] 
        for s in senses:
            term = s.split('_')[0]
            if term in targets:
                tX.append(max([st_y_d[y][cat][s] for y in st_y_d if s in st_y_d[y][cat]]))
                ty.append(max([wt_d[y][cat][s] for y in wt_d if s in st_y_d[y][cat]]))
            else:
                top_st = max([st_y_d[y][cat][s] for y in st_y_d if s in st_y_d[y][cat]])
                X.append(top_st)
                top_wt = max([wt_d[y][cat][s] for y in wt_d if s in st_y_d[y][cat]])
                y.append(top_wt)
                score_d['top_tnpmi'].append(top_st)
                score_d['top_nv'].append(top_wt)
                score_d['categories'].append(cat)
        if by_cat:
            plt.figure(figsize=size)
            plt.scatter(X, y, color=[0, 0, 1], alpha=0.2)
            plt.savefig(f'{o_path}/{cat}.png', bbox_inches='tight')
    if not by_cat:
        plt.figure(figsize=size)
        plt.scatter(X, y, color=[0.5, 0.5, 0.5], alpha=0.2)
        plt.scatter(tX, ty, color=[0, 0, 1], alpha=0.2)
        plt.savefig(f'{o_path}/all.png', bbox_inches='tight')
    print(stats.spearmanr(X, y))
    if tX != [] and ty != []:
        print(stats.spearmanr(tX, ty))

    return score_d


def tnpmi_mtscores(tnpmi_y_d, st_y_d, sc_d):

    categories = list(set([cat for y in tnpmi_y_d for cat in tnpmi_y_d[y]]))
    years = sorted([y for y in tnpmi_y_d])
    o_lst = [] 
    for cat in categories:
        vocab = list(set([w for y in tnpmi_y_d for w in tnpmi_y_d[y][cat]]))
        for w in tqdm(vocab, desc=cat):
            w_senses = [f'{w}_{i}' for i in range(10)]
            for y in years:
                ws_counts = [(s, sc_d[y][cat][s]) for s in w_senses if s in sc_d[y][cat]]
                if ws_counts != []:
                    mf_s = sorted(ws_counts, key=lambda x: x[1])[-1][0]
                    if mf_s in st_y_d[y][cat] and w in tnpmi_y_d[y][cat]:
                        m_score = st_y_d[y][cat][mf_s]
                        t_score = tnpmi_y_d[y][cat][w]
                        o_lst.append((w, (t_score, m_score)))
 
    return o_lst


def mscore_analysis(sc_d, tnpmi_y_d, st_y_d, o_path, percentile=0.98, size=(6, 6)):

    categories = list(set([cat for y in sc_d for cat in sc_d[y]]))
    X = []
    y = []
    all_ws = []
    all_ss = []
    for cat in categories:
        vocab = list(set([w for y in tnpmi_y_d for w in tnpmi_y_d[y][cat]]))
        w_scores = []
        s_scores = []
        for w in tqdm(vocab):
            w_senses = [f'{w}_{i}' for i in range(10)]
            tnpmis = [tnpmi_y_d[y][cat][w] for y in tnpmi_y_d \
                    if w in tnpmi_y_d[y][cat]]
            if tnpmis != []:
                w_scores.append(max(tnpmis))
            s_counts = ' '.join([f'{s} ' * sc_d[y][cat][s] for y in sc_d for s in w_senses \
                    if s in sc_d[y][cat]]).split()
            if s_counts != []:
                mf_sense = Counter(s_counts).most_common(1)[0][0]
                sts = [st_y_d[y][cat][mf_sense] for y in st_y_d \
                        if mf_sense in st_y_d[y][cat]]
                if sts != []:
                    s_scores.append(max(sts))
                    all_ws.append(max(tnpmis))
                    all_ss.append(max(sts))
        n_top_ws = len(sorted(w_scores)[int(len(w_scores) * percentile):])
        n_top_ss = len(sorted(s_scores)[int(len(s_scores) * percentile):])
        X.append(n_top_ws / len(vocab))
        y.append(n_top_ss / len(vocab))
    print(stats.spearmanr(X, y))
    plt.figure(figsize=size)
    plt.scatter(X, y, color=[0, 0, 1])
    plt.savefig(f'{o_path}/cat_frac.png', bbox_inches='tight')
    print(stats.spearmanr(all_ws, all_ss))
    plt.figure(figsize=size)
    plt.scatter(all_ws, all_ss, color=[0, 0, 1], alpha=0.2)
    plt.savefig(f'{o_path}/mscore_tnpmi_scatter.png', bbox_inches='tight')


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
    with open(f'{to_path}/tnpmi_y_d.pickle', 'rb') as f_name:
        tnpmi_y_d = pickle.load(f_name)
    with open(f'{so_path}/snpmi_y_d.pickle', 'rb') as f_name:
        snpmi_y_d = pickle.load(f_name)
    with open(f'{ts_path}/sc_d.pickle', 'rb') as f_name:
        sc_d = pickle.load(f_name)
    #rekey
    st_y_d = rekey_snpmi_y_d(st_y_d)
    wt_d = rekey_snpmi_y_d(wt_d)
    sc_d = rekey_snpmi_y_d(sc_d)
    snpmi_y_d = rekey_snpmi_y_d(snpmi_y_d)

#    #scatter T* and M* scores
#    #scatter 
#    tm_tpls = tnpmi_mtscores(tnpmi_y_d, st_y_d, sc_d)
#    X = [tpl[1][0] for tpl in tm_tpls]
#    y = [tpl[1][1] for tpl in tm_tpls]
#    plt.figure()
#    plt.scatter(X, y, alpha=0.2)
#    plt.savefig(f'{sa_path}/scatter_test.png', bbox_inches='tight')
#    print(stats.spearmanr(X, y))
        
    #correlation between categories' fractions of words with high tnpmi and fractions of senses 
    #with high s_t
    mscore_analysis(sc_d, tnpmi_y_d, st_y_d, sa_path)

    #scatter top st wt scores
    o_path = f'{sa_path}/stwt_corr'
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    score_d = scatter_stwt_scores(st_y_d, wt_d, o_path)
    #linear regressions
    ant.top_scores_lri(score_d, sa_path, x_label='Top $S^*_T$', y_label='Top $W^*_T$')

    #test difference between legislative and departmental sense scores
    categories = list(set([cat for y in st_y_d for cat in st_y_d[y]]))
    dep_scores = []
    leg_scores = []
    for cat in categories:
        senses = list(set([s for y in st_y_d for s in st_y_d[y][cat]]))
        for s in senses:
            if cat == 'leg':
                leg_scores.append(np.median([st_y_d[y][cat][s] for y in st_y_d \
                        if s in st_y_d[y][cat]]))
            else:
                dep_scores.append(np.median([st_y_d[y][cat][s] for y in st_y_d \
                        if s in st_y_d[y][cat]]))
    print((np.median(dep_scores), np.median(leg_scores), len(dep_scores), len(leg_scores)))
    print(stats.mannwhitneyu(dep_scores, leg_scores))


if __name__ == '__main__':
    main()


