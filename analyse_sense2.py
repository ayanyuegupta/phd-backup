import utils.misc as misc
import scipy.stats as stats
import math
import analyse_type as a_t
import analyse_sense as a_s
from tqdm import tqdm
from scipy.stats import kruskal, f_oneway, mannwhitneyu
import os
import pickle


k_range = (2, 10)
root = '/home/gog/projects/gov_lv'
data_root = '/media/gog/external2/corpora/gov_corp'

to_path = f'{root}/t_measures_output_sample'
so_path = f'{root}/s_measures_output_{k_range[0]}_{k_range[1]}'
sa_path = f'{root}/sense_analysis'

d_path = f'{data_root}/sample'
sc_path = f'{so_path}/sense_counts'
c_path = f'{data_root}/clusters_{k_range[0]}_{k_range[1]}'

years = [2000 + i for i in range(21)]



def rekey_snpmi_y_d(snpmi_y_d):
  
    for y in snpmi_y_d:
        snpmi_y_d[y]['home_office'] = snpmi_y_d[y].pop('home')
        snpmi_y_d[y]['cabinet_office'] = snpmi_y_d[y].pop('cabinet')

    return snpmi_y_d


def get_leg_dep_m(cby_d, snpmi_y_d, all_years_sc, prop=1, percentile=0.98):
    
    dep = []
    leg = []
    categories = list(set([cat for y in snpmi_y_d for cat in snpmi_y_d[y]]))
    for cat in categories:
        for y in tqdm(snpmi_y_d, desc=cat):
            vocab = list(set([w.split('_')[0] for w in snpmi_y_d[y][cat]]))
            vocab = sorted(vocab, reverse=True)[: int(len(vocab) * prop)]
            m_scores = []
            for w in vocab:
                w_senses = [f'{w}_{i}' for i in range(10)]
                s_lst = [(s, snpmi_y_d[y][cat][s]) for s in w_senses if s in snpmi_y_d[y][cat]]
                mf_s = sorted(s_lst, key=lambda x: x[1])[-1][0]
                m_scores.append(snpmi_y_d[y][cat][mf_s])
            idx = int(len(m_scores) * percentile)
            if cat == 'leg':
                dep.append(len(sorted(m_scores)[idx:]) / len(vocab))
            else:
                leg.append(len(sorted(m_scores)[idx:]) / len(vocab))
    
    return dep, leg


def get_sc_d(sc_path, years=[2000 + i for i in range(21)]):

    sc_d = {y: None for y in years}
    for y in years:
        with open(f'{sc_path}/{y}_sc.pickle', 'rb') as f_name:
            sc_d[y] = pickle.load(f_name)

    return sc_d


def top_scores_tm(tnpmi_y_d, snpmi_y_d, sc_d, min_years=20):

    categories = list(set([cat for y in tnpmi_y_d for cat in tnpmi_y_d[y]]))
    years = sorted([y for y in tnpmi_y_d])
    o_lst = [] 
    for cat in categories:
        vocab = list(set([w for y in tnpmi_y_d for w in tnpmi_y_d[y][cat]]))
        for w in tqdm(vocab, desc=cat):
            w_senses = [f'{w}_{i}' for i in range(10)]
            for y in years:
                vocab = list(set([w for w in tnpmi_y_d[y][cat]]))
                ws_counts = [(s, sc_d[y][s]) for s in w_senses if s in sc_d[y]]
                if ws_counts != []:
                    mf_s = sorted(ws_counts, key=lambda x: x[1])[-1][0]
                    if mf_s in snpmi_y_d[y][cat] and w in tnpmi_y_d[y][cat]:
                        m_score = snpmi_y_d[y][cat][mf_s]
                        t_score = tnpmi_y_d[y][cat][w]
                        o_lst.append((w, (t_score, m_score)))
 
    return o_lst


def main():
    
    #test difference between departmental and legislative fractions of words scoring high M*
    with open(f'{so_path}/snpmi_y_d.pickle', 'rb') as f_name:
        snpmi_y_d = pickle.load(f_name)
    snpmi_y_d = rekey_snpmi_y_d(snpmi_y_d)
    with open(f'{so_path}/all_years_sc.pickle', 'rb') as f_name:
        all_years_sc = pickle.load(f_name)
    with open(f'{to_path}/cby_d.pickle', 'rb') as f_name:
        cby_d = pickle.load(f_name)
    categories = list(set([cat for y in snpmi_y_d for cat in snpmi_y_d[y]])) 
    dep, leg = get_leg_dep_m(cby_d, snpmi_y_d, all_years_sc)  
    U1, p = mannwhitneyu(dep, leg)
    print(p)

    #correlation between top M* and top T* word level
    with open(f'{to_path}/tnpmi_y_d.pickle', 'rb') as f_name:
        tnpmi_y_d = pickle.load(f_name)
    sc_d = get_sc_d(sc_path)
    tm_tpls = misc.get_items('tm', sa_path, top_scores_tm, tnpmi_y_d, snpmi_y_d, sc_d)
    print(len(tm_tpls))
    X = [tpl[1][0] for tpl in tm_tpls]
    y = [tpl[1][1] for tpl in tm_tpls]
    rho, p_val, c_i = a_s.spearman_scatter(X, y, f'{sa_path}/corr_tm.png', font_size=20, alpha=0.1, x_label='$T*$', y_label='$M_t*$')
    print(c_i)

    

if __name__ == '__main__':

    main()
