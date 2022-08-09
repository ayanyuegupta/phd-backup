from sklearn import linear_model
import statsmodels.api as sm
import scipy.stats as stats
from matplotlib.ticker import MultipleLocator
import numpy as np
from analyse_type import average_wscores, bar_chart_words
from tqdm import tqdm
from analyse_sense import spearman_scatter
import re
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import kruskal, f_oneway, mannwhitneyu, normaltest
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


def scatter_ts_topscores(tsnpmi_y_d, tsnvol_d, snpmi_y_d, snvol_d, sa_path, min_num=10, colours=[[0, 0, 1], [0.5, 0.5, 0.5]], alphas=[0.5, 0.2], x_label='Top $S*$', y_label='Top $W*$', font_size=20, tick_size=15):

    tsenses = []
    senses = []
    for y in tsnpmi_y_d:
        tsenses += [s for cat in tsnpmi_y_d[y] for s in tsnpmi_y_d[y][cat]]
        senses += [s for cat in snpmi_y_d[y] for s in snpmi_y_d[y][cat]]
    tsenses = list(set(tsenses))
    senses =  list(set([s for s in senses if s not in tsenses]))
    categories = list(set([cat for y in tsnpmi_y_d for cat in tsnpmi_y_d[y]]))
    X = []
    y = []
    for s in tqdm(senses):
        for cat in categories:
            specs = [snpmi_y_d[y][cat][s] for y in snpmi_y_d if s in snpmi_y_d[y][cat]]
            vols = [snvol_d[y][cat][s] for y in snvol_d if s in snvol_d[y][cat]]
            if len(specs) >= min_num and len(vols) >= min_num:
                X.append(max(specs))
                y.append(max(vols))
    plt.figure(figsize=(6, 6))
    plt.scatter(X, y, color=colours[1], alpha=alphas[1])
    tX = []
    ty = []
    for s in tsenses:
        for cat in categories:
            specs = [tsnpmi_y_d[y][cat][s] for y in tsnpmi_y_d if s in tsnpmi_y_d[y][cat]]
            vols = [tsnvol_d[y][cat][s] for y in tsnvol_d if s in tsnpmi_y_d[y][cat]]
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
    print(f'Medians: {(np.median(t_g), np.median(o_g))}')
    print(f'p = {p}')
    #rosenthal correlation
    nx, ny = len(t_g), len(o_g)
    U2 = nx*ny - U1
    U = min(U1, U2)
    z, N = utest_zscore(U, nx, ny)
    r = rosenthal_corr(z, N)
    #common language
    f = common_lang(U1, nx, ny)
    
    return p


def natural_key(string_):
    
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def get_senses_vocab(tsnpmi_y_d, drop_cat=None):

    categories = sorted(list(set([cat for y in tsnpmi_y_d for cat in tsnpmi_y_d[y] if cat != drop_cat])))
    senses = []
    for y in tsnpmi_y_d:
        senses += [s for cat in tsnpmi_y_d[y] for s in tsnpmi_y_d[y][cat]]
    senses = sorted(list(set(senses)), key=natural_key)
    vocab = sorted(list(set([s.split('_')[0] for s in senses])))
     
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


def barcharts_heatmaps(tnpmi_d, avg_d, targets, senses, categories, o_path, size=(12, 18), vmin=-0.18, vmax=0.18, font_size=20, tick_size=15, drop_cat='leg', y_label='Average $T*$'):

    fig, axs = plt.subplots(5, 2, figsize=size)
    plt.rc('font', size=font_size)
    cbar_ax = fig.add_axes([0.9, 0.48, 0.015, 0.1])
    for i, w in enumerate(targets):
        #heatmaps
        df = avg_scores_df(w, avg_d, senses, categories)
        df_rows = df.index.values
        df_cols = df.columns
        g = sns.heatmap(df, 
                center=0, 
                ax=axs[i][1],  
                cbar=i == 0, 
                vmin=vmin, 
                vmax=vmax,
                cbar_ax=None if i else cbar_ax, 
                cbar_kws={'label': 'Average $S*$'}) 
#        axs[i][1].set_ylabel(w, fontsize=font_size + 5)
        axs[i][1].set_title(w, fontsize=font_size)
        axs[i][1].set_yticks(np.arange(0.5, len(df_rows)))
        axs[i][1].set_yticklabels([s.split('_')[1] for s in df_rows], fontsize=tick_size)
        axs[i][1].set_xticklabels(list(df_cols), fontsize=tick_size)
        #bar charts
        x = sorted([cat for cat in tnpmi_d if cat != drop_cat])
        y = [tnpmi_d[cat][w] for cat in x]
        x = [cat.split('_')[0] if '_' in cat else cat for cat in x]
        axs[i][0].bar(x, y, color=[0, 0, 1])
        axs[i][0].set_title(w, fontsize=font_size)
        axs[i][0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axs[i][0].tick_params(axis='both', which='major', labelsize=tick_size, labelrotation=90)
        axs[i][0].yaxis.offsetText.set_fontsize(tick_size)
#        axs[i][0].set_ylabel('Average $T*$', fontsize=font_size)
    fig.text(0, 0.525, y_label, ha='center', va='center', rotation='vertical', fontsize=font_size)
    fig.tight_layout(rect=(0, 0, 0.9, 1))    
    
    plt.savefig(f'{o_path}/test.png', bbox_inches='tight')


def rekey_snpmi_y_d(snpmi_y_d):
  
    for y in snpmi_y_d:
        snpmi_y_d[y]['home_office'] = snpmi_y_d[y].pop('home')
        snpmi_y_d[y]['cabinet_office'] = snpmi_y_d[y].pop('cabinet')

    return snpmi_y_d

def get_sc_d(sc_path, years=[2000 + i for i in range(21)]):

    sc_d = {y: None for y in years}
    for y in years:
        with open(f'{sc_path}/{y}_sc.pickle', 'rb') as f_name:
            sc_d[y] = pickle.load(f_name)

    return sc_d


def tnpmi_mscores(tnpmi_y_d, snpmi_y_d, sc_d):

    categories = list(set([cat for y in tnpmi_y_d for cat in tnpmi_y_d[y]]))
    years = sorted([y for y in tnpmi_y_d])
    o_lst = [] 
    for cat in categories:
        vocab = list(set([w for y in tnpmi_y_d for w in tnpmi_y_d[y][cat]]))
        for w in tqdm(vocab, desc=cat):
            w_senses = [f'{w}_{i}' for i in range(10)]
            for y in years:
                ws_counts = [(s, sc_d[y][s]) for s in w_senses if s in sc_d[y]]
                if ws_counts != []:
                    mf_s = sorted(ws_counts, key=lambda x: x[1])[-1][0]
                    if mf_s in snpmi_y_d[y][cat] and w in tnpmi_y_d[y][cat]:
                        m_score = snpmi_y_d[y][cat][mf_s]
                        t_score = tnpmi_y_d[y][cat][w]
                        o_lst.append((w, (t_score, m_score)))
 
    return o_lst  


def tnpmi_tsnpmi_lr(tnpmi_y_d, tsnpmi_y_d, targets, senses, categories):
    
    d = {}
    for w in targets:
        w_senses = [s for s in senses if s.split('_')[0] == w]
        #each word's tnpmi
        d[w] = [tnpmi_y_d[y][cat][w] if w in tnpmi_y_d[y][cat] else None \
                for y in tnpmi_y_d for cat in tnpmi_y_d[y]]
        #snpmi
        for s in w_senses:
            d[s] = [tsnpmi_y_d[y][cat][s] if s in tsnpmi_y_d[y][cat] else None \
                    for y in tnpmi_y_d for cat in tnpmi_y_d[y]]
    df = pd.DataFrame.from_dict(d)
    for key in d:
        df[key] = df[key].fillna(df[key].median())
    regr = linear_model.LinearRegression()
    X = df[senses]
    y = df[targets]
    regr.fit(X, y)
    print(regr.coef_)
    print(regr.score(X, y))  
   
   
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a_s', '-additional_senses', required=True)
    args = parser.parse_args()
    a_s = [int(v) for v in args.a_s.split('-')]

    
    with open(f'{so_path}/as_output_{a_s[0]}_{a_s[1]}/tsnpmi_y_d.pickle', 'rb') as f_name:
        tsnpmi_y_d = pickle.load(f_name)
    with open(f'{so_path}/as_output_{a_s[0]}_{a_s[1]}/tsnvol_d.pickle', 'rb') as f_name:
        tsnvol_d = pickle.load(f_name)
    with open(f'{so_path}/snpmi_y_d.pickle', 'rb') as f_name:
        snpmi_y_d = pickle.load(f_name)
    with open(f'{so_path}/snvol_d.pickle', 'rb') as f_name:
        snvol_d = pickle.load(f_name)
    
#    #scatter top scores and get U-test effect sizes
#    t_group, o_group = scatter_ts_topscores(tsnpmi_y_d, tsnvol_d, snpmi_y_d, snvol_d, sa_path)   
#    p_vals = []
#    #specs
#    t_s = [lst[0] for lst in t_group]
#    o_s = [lst[0] for lst in o_group]
#    print('Specificity')
#    p_vals.append(top_sscores_esizes_utest(t_s, o_s))
#    #vol
#    t_v = [lst[1] for lst in t_group]
#    o_v = [lst[1] for lst in o_group]
#    print('Volatility')
#    p_vals.append(top_sscores_esizes_utest(t_v, o_v))   
#    #adjust p-vals
#    p_adjusted = multipletests(p_vals, alpha=0.01, method='fdr_bh')    
#    print(f'adjusted p-values: \n{p_adjusted}')
#    
#    #test target vocab tnpmis leg vs dep
#    with open(f'{to_path}/tnpmi_y_d.pickle', 'rb') as f_name:
#        tnpmi_y_d = pickle.load(f_name)
#    l_ts = [tnpmi_y_d[y]['leg'][w] for y in tnpmi_y_d for w in targets if w in tnpmi_y_d[y]['leg']]
#    deps = list(set([cat for y in tnpmi_y_d for cat in tnpmi_y_d[y] if cat != 'leg']))
#    d_ts = []
#    for cat in deps:
#        d_ts += [tnpmi_y_d[y][cat][w] for y in tnpmi_y_d for w in targets if w in tnpmi_y_d[y][cat]]
#    U1, p = mannwhitneyu(l_ts, d_ts)
#    print('Test differences between departmental and legislation target vocab scores:')
#    print((U1, p))
#    print((np.median(l_ts), np.median(d_ts)))

    #heatmaps
    o_path = f'{sa_path}/snpmi_heatmaps/{a_s[0]}_{a_s[1]}'
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    with open(f'{to_path}/tnpmi_y_d.pickle', 'rb') as f_name:
        tnpmi_y_d = pickle.load(f_name)
    senses, _, categories = get_senses_vocab(tsnpmi_y_d, drop_cat='leg')
    avg_d = avg_scores(senses, categories, tsnpmi_y_d)
#    sense_heatmaps(targets, senses, avg_d, categories, o_path)
    #tnpmi bar charts
#    o_path = f'{ta_path}/bar_charts'
    tnpmi_d = average_wscores(tnpmi_y_d)
#    tnpmi_bar_charts(targets, tnpmi_d, o_path, drop_cat='leg') 
    barcharts_heatmaps(tnpmi_d, avg_d, targets, senses, categories, o_path)
    
    #tests
    #m score tnpmi correlation
    sc_path = f'{so_path}/as_output_{a_s[0]}_{a_s[1]}/sense_counts' 
    sc_d = get_sc_d(sc_path)
    tsnpmi_y_d = rekey_snpmi_y_d(tsnpmi_y_d)
    tm_tpls = tnpmi_mscores(tnpmi_y_d, tsnpmi_y_d, sc_d)
    X = [tpl[1][0] for tpl in tm_tpls]
    y = [tpl[1][1] for tpl in tm_tpls]
    print((stats.spearmanr(X, y), f'n={len(X)}'))
  
    #multiple linear regression correlating tnpmi with each sense
    tnpmi_tsnpmi_lr(tnpmi_y_d, tsnpmi_y_d, targets, senses, categories)    

if __name__ == '__main__':
    main()



