import math
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


def scatter_stwt_scores(st_y_d, wt_d, o_path, size=(6, 6), min_n=20):
    
    categories = list(set([cat for y in st_y_d for cat in st_y_d[y]]))
    score_d = {'top_nv': [], 'top_tnpmi': [], 'categories': []}
    X = []
    y = []
    for cat in tqdm(categories):
        senses = list(set([s for y in st_y_d for s in st_y_d[y][cat]]))
        cX = []
        cy = [] 
        for s in senses:
            term = s.split('_')[0]
            sscores = [st_y_d[y][cat][s] for y in st_y_d if s in st_y_d[y][cat]]
            vscores = [wt_d[y][cat][s] for y in wt_d if s in st_y_d[y][cat]]
            if len(sscores) > min_n:
                cX.append(max(sscores))
                cy.append(max(vscores))
                X.append(max(sscores))
                y.append(max(vscores))
                score_d['top_tnpmi'].append(max(sscores))
                score_d['top_nv'].append(max(vscores))
                score_d['categories'].append(cat)
        plt.figure(figsize=size)
        plt.scatter(cX, cy, color=[0, 0, 1], alpha=0.2)
        plt.savefig(f'{o_path}/{cat}.png', bbox_inches='tight')
    plt.figure(figsize=size)
    plt.scatter(X, y, color=[0, 0, 1], alpha=0.2)
    plt.savefig(f'{o_path}/all.png', bbox_inches='tight')
    print(stats.spearmanr(X, y))

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

def spearman_ci(r, n, alpha=2.58):

    stderr = 1.0 / math.sqrt(n - 3)
    delta = alpha * stderr
    lower = math.tanh(math.atanh(r) - delta)
    upper = math.tanh(math.atanh(r) + delta)

    return lower, upper


def mscore_analysis(sc_d, tnpmi_y_d, st_y_d, o_path, percentile=0.98, size=(6, 6)):

    categories = list(set([cat for y in sc_d for cat in sc_d[y]]))
    X = []
    y = []
    all_ws = []
    all_ss = []
    for cat in tqdm(categories):
        vocab = list(set([w for y in tnpmi_y_d for w in tnpmi_y_d[y][cat]]))
        w_scores = []
        s_scores = []
        for w in vocab:
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
    r, p = stats.spearmanr(X, y)
    print(f'r={r}, p={p}, n={len(X)}')
    print(spearman_ci(r, len(X)))
    plt.figure(figsize=size)
    plt.scatter(X, y, color=[0, 0, 1])
    plt.savefig(f'{o_path}/cat_frac.png', bbox_inches='tight')
    r, p = stats.spearmanr(all_ws, all_ss)
    print(f'r={r}, p={p}, n={len(all_ws)}')
    print(spearman_ci(r, len(all_ws)))
    plt.figure(figsize=size)
    plt.scatter(all_ws, all_ss, color=[0, 0, 1], alpha=0.2)
    plt.savefig(f'{o_path}/mscore_tnpmi_scatter.png', bbox_inches='tight')


def measures_by_year(d, o_path, f_name, targets=['leg'], colour=[0, 0, 1], size=(6, 6), font_size=20, alpha=0.2, y_label=None, legend=True):
   
    font = {'size': font_size}
    plt.rc('font', size=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('xtick', labelsize=font_size-2)
    plt.rc('ytick', labelsize=font_size-2)
    plt.rc('axes', titlesize=font_size)
    
    ax = plt.figure(figsize=size).gca()
    categories = list(set([cat for year in d for cat in d[year]]))
    x = sorted([year for year in d])
    for cat in categories:
        y = [d[year][cat] for year in  x]
        if targets is not None:
            if cat in targets:
                plt.plot(x, y, c=np.array(colour), marker='o', label=cat)
            else:
                 plt.plot(x, y, linestyle='-', marker='o', c=np.array([0, 0, 0]), alpha=alpha)
        else:
            plt.plot(x, y, c=np.array(colour), marker='o', label=cat)

    ax.xaxis.get_major_locator().set_params(integer=True)
    if legend:
        plt.legend(loc='upper left', frameon=False) 
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig(f'{o_path}/{f_name}.png', bbox_inches='tight')

    return ax, categories


def delta(x, y, z=2.58):

    m1 = np.average(x)
    m2 = np.average(y)
    sd1 = np.std(x)
    sd2 = np.std(y)
    n1 = len(x)
    n2 = len(y)
    avg_sd = np.sqrt(
            0.5 * (sd1 ** 2 + sd2 ** 2)
            )
    d = (m1 - m2) / avg_sd
    sigma_d = (d ** 2) * ((sd1 ** 4 / (n1 - 1)) + (sd2 ** 4 / (n2 - 1))) / (8 * avg_sd ** 4)  \
            + (sd1 ** 2) / (avg_sd ** 2 * (n1 - 1)) + (sd2 ** 2) / (avg_sd ** 2 * (n2 - 1)) 
    lb = d - z * np.sqrt(sigma_d)
    ub = d + z * np.sqrt(sigma_d) 

    return d, lb, ub, m1, m2 


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
    with open(f'{so_path}/snvol_d.pickle', 'rb') as f_name:
        snvol_d = pickle.load(f_name)
    with open(f'{ts_path}/sc_d.pickle', 'rb') as f_name:
        sc_d = pickle.load(f_name)
    #rekey
    st_y_d = rekey_snpmi_y_d(st_y_d)
    wt_d = rekey_snpmi_y_d(wt_d)
    sc_d = rekey_snpmi_y_d(sc_d)
    snpmi_y_d = rekey_snpmi_y_d(snpmi_y_d)
    snvol_d = rekey_snpmi_y_d(snvol_d)

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
    ant.top_scores_lri(score_d, sa_path, x_label='Top $S^\#$', y_label='Top $W^\#$')
    
    #welch t-test for diff between leg dep top W^#
    df = pd.DataFrame.from_dict(score_d)
    x = df.loc[df['categories'] == 'leg']['top_tnpmi']
    y = df.loc[df['categories'] == 'dep']['top_tnpmi']
    #specificity
    t, p = stats.ttest_ind(x, y, equal_var=False)
    d, lb, ub, m1, m2 = delta(x, y)
    
    print((t, p, d, lb, ub, m1, m2, f'{len(x) + len(y)}'))

    #difference between legislative and departmental median sense scores over time
    med_d = {y: {} for y in st_y_d}
    for y in st_y_d:
        for cat in st_y_d[y]:
            med_d[y][cat] = np.average([st_y_d[y][cat][s] for s in st_y_d[y][cat]])
    measures_by_year(med_d, sa_path, 'med_ss_time', y_label='Average $S^\#$')


if __name__ == '__main__':
    main()


