import numpy as np
from analyse_type import average_wscores, bar_chart_words
from tqdm import tqdm
from analyse_sense import spearman_scatter
import re
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import kruskal, ttest_ind, mannwhitneyu
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


def natural_key(string_):
    
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def get_senses_vocab(tsnpmi_y_d):

    categories = sorted(list(set([cat for y in tsnpmi_y_d for cat in tsnpmi_y_d[y]])))
    senses = []
    for y in tsnpmi_y_d:
        senses += [s for cat in tsnpmi_y_d[y] for s in tsnpmi_y_d[y][cat]]
    senses = sorted(list(set(senses)), key=natural_key)
    vocab = list(set([s.split('_')[0] for s in senses]))
    
    return senses, vocab, categories


def avg_scores(senses, categories, tsnpmi_y_d):
    
    d = {s: [0 for i, _ in enumerate(categories)] for s in senses}   
    for s in d:
        for i, cat in enumerate(categories):
            d[s][i] = np.average([tsnpmi_y_d[y][cat][s] for y in tsnpmi_y_d \
                    if s in tsnpmi_y_d[y][cat]])

    return d
  

def avg_scores_df(w, d, senses, categories):
    
    senses = [s for s in senses if s.split('_')[0] == w]
    o_d = {}
    for s in senses:
        o_d[s] = d[s]
    df = pd.DataFrame.from_dict(o_d, orient='index', columns=categories)

    return df


def score_heatmap(df, w, path, colours=None):

    fig = plt.figure()
    ax = sns.heatmap(df, center=0)
    plt.title(w)
    
    plt.savefig(f'{path}/{w}_heatmap.png', bbox_inches='tight')


def test_tscat_kw(w, senses, tsnpmi_y_d, categories, adjust_pvals=True, alpha=0.05, method='fdr_bh'):
    
    senses = [s for s in senses if s.split('_')[0] == w]
    p_vals = []
    for s in senses:
        groups = []
        for cat in categories:
            groups.append([tsnpmi_y_d[y][cat][s] for y in tsnpmi_y_d if s in tsnpmi_y_d[y][cat]])
        groups = [g for g in groups if g != []]
        H, p = kruskal(*groups)
        p_vals.append(p)
    if adjust_pvals:
        p_adjusted = multipletests(p_vals, alpha=alpha, method=method)
        kw_d = {s: p_adjusted[1][i] for i, s in enumerate(senses)}
        kw_df = pd.DataFrame.from_dict(kw_d, orient='index', columns=['adjusted p values'])
    else:
        kw_d = {s: p_vals[i] for i, s in enumerate(senses)}
        kw_df = pd.DataFrame.from_dict(kw_d, orient='index', columns=['p values'])
    
    return kw_df 


def test_tsfreq_kw(w, rf_d, years):
    
    all_years = {y: i for i, y in enumerate([2000 + i for i in range(21)])}
    groups = [rf_d[w][s][all_years[years[0]]: all_years[years[-1]]] for s in rf_d[w]]
    H, p = kruskal(*groups)

    return H, p

def test_tscat_u(test_lst, tsnpmi_y_d, categories, adjust_pvals=True, alpha=0.01, method='fdr_bh'):
    
    p_vals = []
    for tpl in test_lst:
        o_cats = [cat for cat in categories if cat not in tpl[1]]
        t_group = []
        o_group = []
        for s in tpl[0]:
            t_group += [tsnpmi_y_d[y][cat][s] for y in tsnpmi_y_d for cat in tpl[1] \
                    if s in tsnpmi_y_d[y][cat]]
            o_group += [tsnpmi_y_d[y][cat][s] for y in tsnpmi_y_d for cat in o_cats \
                    if s in tsnpmi_y_d[y][cat]]
        U1, p = mannwhitneyu(t_group, o_group)
        p_vals.append(p)
    if adjust_pvals:
        p_adjusted = multipletests(p_vals, alpha=alpha, method=method) 
        u_d = {f'{", ".join(tpl[0])} {", ".join(tpl[1])}': p_adjusted[1][i] for i, tpl in enumerate(test_lst)}
        u_df = pd.DataFrame.from_dict(u_d, orient='index', columns=['adjusted p values'])
    else:
        u_d = {f'{", ".join(tpl[0])} {", ".join(tpl[1])}': p_vals[i] for i, tpl in enumerate(test_lst)}
        u_df = pd.DataFrame.from_dict(u_d, orient='index', columns=['p values'])
    
    return u_df


def test_ts_u(targets, snpmi_y_d):
    
    categories = list(set([cat for y in snpmi_y_d for cat in snpmi_y_d[y]]))
    t_group = [snpmi_y_d[y][targets[1]][targets[0]] for y in snpmi_y_d \
            if targets[0] in snpmi_y_d[y][targets[1]]]
    o_group = []
    for cat in categories:
        if cat != targets[1]:
            o_group += [snpmi_y_d[y][cat][targets[0]] for y in snpmi_y_d \
                    if targets[0] in snpmi_y_d[y][cat]]
    U1, p = mannwhitneyu(t_group, o_group)
    
    return U1, p


def scatter_ts_topscores(tsnpmi_y_d, tsnvol_d, snpmi_y_d, snvol_d, sa_path, min_num=260, colours=[[0, 0, 1], [0.5, 0.5, 0.5]], alphas=[0.5, 0.2], x_label='Top $S*$', y_label='Top $W*$', font_size=20, tick_size=15):

    tsenses = []
    senses = []
    for y in tsnpmi_y_d:
        tsenses += [s for cat in tsnpmi_y_d[y] for s in tsnpmi_y_d[y][cat]]
        senses += [s for cat in snpmi_y_d[y] for s in snpmi_y_d[y][cat]]
    tsenses = list(set(tsenses))
    senses =  list(set([s for s in senses if s not in tsenses]))
    X = []
    y = []
    for s in tqdm(senses):
        specs = [snpmi_y_d[y][cat][s] for y in snpmi_y_d for cat in snpmi_y_d[y] \
                if s in snpmi_y_d[y][cat]]
        vols = [snvol_d[y][cat][s] for y in snvol_d for cat in snvol_d[y] \
                if s in snvol_d[y][cat]]
        if len(specs) >= min_num and len(vols) >= min_num:
            X.append(max(specs))
            y.append(max(vols))
    plt.figure(figsize=(6, 6))
    plt.scatter(X, y, color=colours[1], alpha=alphas[1])
    tX = []
    ty = []
    for s in tsenses:
        specs = [tsnpmi_y_d[y][cat][s] for y in tsnpmi_y_d for cat in tsnpmi_y_d[y] \
                if s in tsnpmi_y_d[y][cat]]
        vols = [tsnvol_d[y][cat][s] for y in tsnvol_d for cat in tsnvol_d[y] \
                if s in tsnpmi_y_d[y][cat]]
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
    
    spearman_scatter(tX, ty, f'{sa_path}/ts_tops_scatter.png', x_label=x_label, y_label=y_label, font_size=font_size, tick_size=tick_size)

    t_group = [[tX[i], ty[i]] for i, _ in enumerate(tX)]
    o_group = [[X[i], y[i]] for i, _ in enumerate(X)]
 
    return t_group, o_group


def bar_charts(targets, tnpmi_d, o_path, size=(6, 18), y_label=None, font_size=20, tick_size=15):

    fig, ax = plt.subplots(len(targets), 1, figsize=size)
    for i, w in enumerate(targets):
        x = sorted([cat for cat in tnpmi_d])
        y = [tnpmi_d[cat][w] for cat in x]
        x = [cat.split('_')[0] if '_' in cat else cat for cat in x]
        ax[i].bar(x, y, color=[0, 0, 1])
        ax[i].set_title(w, fontsize=font_size)
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax[i].tick_params(axis='both', which='major', labelsize=tick_size, labelrotation=90)
        ax[i].yaxis.offsetText.set_fontsize(tick_size)
    if y_label is not None:
        fig.text(0, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=font_size)
    fig.tight_layout()
    plt.savefig(f'{o_path}/target_bar_charts.png', bbox_inches='tight')


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#r31b0b1c0fec3-4
def utest_zscore(U, nx, ny):
    
    N = nx + ny
    z = (U - (nx * ny) / 2 + 0.5) / np.sqrt(nx * ny * (N + 1) / 12)
    
    return z, N


def rosenthal_corr(z, N):

    r = abs(z) / np.sqrt(N)
    print(f'Rosenthal correlation: {r}')

    return r


def common_lang(U1, nx, ny):
    
    f = U1 / (nx * ny)
    print(f'Common language: {f}')

    return f


def top_sscores_esizes_utest(t_g, o_g):
    
    U1, p = mannwhitneyu(t_g, o_g)
    print((np.median(t_g), np.median(o_g)))
    print(p)
    #rosenthal correlation
    nx, ny = len(t_g), len(o_g)
    U2 = nx*ny - U1
    U = min(U1, U2)
    z, N = utest_zscore(U, nx, ny)
    r = rosenthal_corr(z, N)
    #common language
    f = common_lang(U1, nx, ny)
    
    return p


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a_s', '-additional_senses', required=True)
    args = parser.parse_args()
    a_s = [int(v) for v in args.a_s.split('-')]

    with open(f'{to_path}/tnpmi_y_d.pickle', 'rb') as f_name:
        tnpmi_y_d = pickle.load(f_name)
    o_path = f'{ta_path}/bar_charts'
    tnpmi_d = average_wscores(tnpmi_y_d)
    bar_charts(targets, tnpmi_d, o_path, y_label='Average $T*$')

    with open(f'{so_path}/as_output_{a_s[0]}_{a_s[1]}/tsnpmi_y_d.pickle', 'rb') as f_name:
        tsnpmi_y_d = pickle.load(f_name)
    with open(f'{so_path}/as_output_{a_s[0]}_{a_s[1]}/tsnvol_d.pickle', 'rb') as f_name:
        tsnvol_d = pickle.load(f_name)
    with open(f'{so_path}/snpmi_y_d.pickle', 'rb') as f_name:
        snpmi_y_d = pickle.load(f_name)
    with open(f'{so_path}/snvol_d.pickle', 'rb') as f_name:
        snvol_d = pickle.load(f_name)
    
    #scatter top scores and get U-test effect sizes
    t_group, o_group = scatter_ts_topscores(tsnpmi_y_d, tsnvol_d, snpmi_y_d, snvol_d, sa_path)   
    p_vals = []
    #specs
    t_s = [lst[0] for lst in t_group]
    o_s = [lst[0] for lst in o_group]
    p_vals.append(top_sscores_esizes_utest(t_s, o_s))
    #vol
    t_v = [lst[1] for lst in t_group]
    o_v = [lst[1] for lst in o_group]
    p_vals.append(top_sscores_esizes_utest(t_v, o_v))   
    #adjust p-vals
    p_adjusted = multipletests(p_vals, alpha=0.01, method='fdr_bh')    
    print(f'adjusted p-values: \n{p_adjusted}')
    
    #test target vocab tnpmis leg vs dep
    with open(f'{to_path}/tnpmi_y_d.pickle', 'rb') as f_name:
        tnpmi_y_d = pickle.load(f_name)
    l_ts = [tnpmi_y_d[y]['leg'][w] for y in tnpmi_y_d for w in targets if w in tnpmi_y_d[y]['leg']]
    deps = list(set([cat for y in tnpmi_y_d for cat in tnpmi_y_d[y] if cat != 'leg']))
    d_ts = []
    for cat in deps:
        d_ts += [tnpmi_y_d[y][cat][w] for y in tnpmi_y_d for w in targets if w in tnpmi_y_d[y][cat]]
    U1, p = mannwhitneyu(l_ts, d_ts)
    print((U1, p))
    print((np.median(l_ts), np.median(d_ts)))

    #heatmaps
    o_path = f'{sa_path}/snpmi_heatmaps/{a_s[0]}_{a_s[1]}'
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    senses, vocab, categories = get_senses_vocab(tsnpmi_y_d)
    d = avg_scores(senses, categories, tsnpmi_y_d)
    for w in vocab:
        df = avg_scores_df(w, d, senses, categories)
        score_heatmap(df, w, o_path)
        kw_df = test_tscat_kw(w, senses, tsnpmi_y_d, categories)
#        print(kw_df)
    
    test_lst = [
            (['sustainability_4'], ['treasury']),
            (['sustainability_5'], ['treasury']),
            (['sustainability_8'], ['leg']),
            (['resilience_6'], ['DEFRA']),
            (['resilience_7'], ['DE']),
            (
                [
                'sustainable_6', 
                'sustainable_5',
                'sustainable_2'
                ],
                [
                    'DEFRA',
                    'MOH',
                    'FCO',
                    ]
                ),
            (['wellbeing_4'], ['DCMS'])
            ]

    u_df = test_tscat_u(test_lst, tsnpmi_y_d, categories)
    print(u_df)

    #test differences in specificities of one sense across two groups of categories
    U1, p = test_ts_u(('resilience_3', 'cabinet'), tsnpmi_y_d)        
    print(U1, p)
    
    ts_path = f'{so_path}/as_output_{a_s[0]}_{a_s[1]}'
    with open(f'{ts_path}/rf_d.pickle', 'rb') as f_name:
        rf_d = pickle.load(f_name) 
    H, p = test_tsfreq_kw('resilience', rf_d, [2015 + i for i in range(6)])
    print(H, p)

if __name__ == '__main__':
    main()



