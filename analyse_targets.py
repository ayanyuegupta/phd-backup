import re
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import kruskal, f_oneway, mannwhitneyu
import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle
import os


k_range = (2, 10)
root = '/home/gog/projects/gov_lv'
data_root = '/media/gog/external2/corpora/gov_corp'
so_path = f'{root}/s_measures_output_{k_range[0]}_{k_range[1]}'
sa_path = f'{root}/sense_analysis'


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


def test_tsenses_kw(w, senses, tsnpmi_y_d, categories, adjust_pvals=True, alpha=0.01, method='bonferroni'):
    
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
        kw_df = pd.DataFrame.from_dict(kw_d, orient='index', columns=['adjusted p_values'])
    else:
        kw_d = {s: p_vals[i] for i, s in enumerate(senses)}
        kw_df = pd.DataFrame.from_dict(kw_d, orient='index', columns=['p_values'])
    
    return kw_df
    

def test_tsenses_u(s, t_cats, tsnpmi_y_d, categories):
    
    o_cats = [cat for cat in categories if cat not in t_cats]
    t_group = [tsnpmi_y_d[y][cat][s] for y in tsnpmi_y_d for cat in t_cats \
            if s in tsnpmi_y_d[y][cat]]
    o_group = [tsnpmi_y_d[y][cat][s] for y in tsnpmi_y_d for cat in o_cats \
            if s in tsnpmi_y_d[y][cat]]
    U1, p = mannwhitneyu(t_group, o_group)

    print((U1, p))


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a_s', '-additional_senses', required=True)
    args = parser.parse_args()
    a_s = [int(v) for v in args.a_s.split('-')]

    with open(f'{so_path}/as_output_{a_s[0]}_{a_s[1]}/tsnpmi_y_d.pickle', 'rb') as f_name:
        tsnpmi_y_d = pickle.load(f_name)
    with open(f'{so_path}/as_output_{a_s[0]}_{a_s[1]}/tsnvol_d.pickle', 'rb') as f_name:
        tsnvol_d = pickle.load(f_name)
    
    o_path = f'{sa_path}/snpmi_heatmaps/{a_s[0]}_{a_s[1]}'
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    senses, vocab, categories = get_senses_vocab(tsnpmi_y_d)
    d = avg_scores(senses, categories, tsnpmi_y_d)
    for w in vocab:
        df = avg_scores_df(w, d, senses, categories)
        score_heatmap(df, w, o_path)
        kw_df = test_tsenses_kw(w, senses, tsnpmi_y_d, categories)
        print(kw_df)
    
    test_tsenses_u('sustainability_4', ['treasury'], tsnpmi_y_d, categories)
    test_tsenses_u('sustainability_5', ['treasury'], tsnpmi_y_d, categories)
    test_tsenses_u('sustainability_8', ['leg'], tsnpmi_y_d, categories)


if __name__ == '__main__':
    main()



