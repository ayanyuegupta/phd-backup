import scipy.stats as stats
from tqdm import tqdm
import os
import pickle
import utils.misc as  misc
import matplotlib.pyplot as plt
import argparse


years = [2000 + i for i in range(21)]
k_range = (2, 10)
root = '/home/gog/projects/gov_lv'
to_path = f'{root}/t_measures_output_sample'
so_path = f'{root}/s_measures_output_{k_range[0]}_{k_range[1]}'
sc_path = f'{so_path}/sense_counts'
sa_path = f'{root}/sense_analysis'


def rekey_snpmi_y_d(snpmi_y_d):
  
    for y in snpmi_y_d:
        snpmi_y_d[y]['home_office'] = snpmi_y_d[y].pop('home')
        snpmi_y_d[y]['cabinet_office'] = snpmi_y_d[y].pop('cabinet')

    return snpmi_y_d


def get_counts(so_path, ts_path, years):

    tsc_d = {}
    sc_d = {}
    for y in years:
        with open(f'{ts_path}/sense_counts/{y}_scc.pickle', 'rb') as f_name:
            tsc_d[y] = pickle.load(f_name)
        with open(f'{so_path}/sense_counts/{y}_scc.pickle', 'rb') as f_name:
            sc_d[y] = pickle.load(f_name)

    return tsc_d, sc_d


def merge_counts(tsc_d, sc_d):

    target_senses = list(set([s for y in tsc_d for cat in tsc_d[y] for s in tsc_d[y][cat]]))
    for y in sc_d:
        for cat in sc_d[y]:
            for s in target_senses:
                if s in tsc_d[y][cat]:
                    sc_d[y][cat][s] = tsc_d[y][cat][s]

    return sc_d


def st(categories, sc_yd):

    st_d = {cat: {} for cat in sc_yd}
    for cat in tqdm(categories):
        for s in sc_yd[cat]:
            term = s.split('_')[0]
            term_senses = [f'{term}_{i}' for i in range(10)]
            fsenses_cterm = sum([sc_yd[cat][s] for s in term_senses \
                    if s in sc_yd[cat]])
            #P(s | c, term)
            p_s_given_cterm = sc_yd[cat][s] / fsenses_cterm
            #P(s | term)
            fs_term = sum([sc_yd[cat][s] for cat in categories \
                    if s in sc_yd[cat]])
            fsenses_term = 0
            for sense in term_senses:
                fsenses_term += sum([sc_yd[cat][sense] for cat in categories \
                        if sense in sc_yd[cat]])
            p_s_given_term = fs_term / fsenses_term
            st_d[cat][s] = p_s_given_cterm / p_s_given_term
            #add in code for normalisation

    return st_d


def st_by_year(sc_d):
    
    st_y_d = {}
    categories = list(set([cat for y in sc_d for cat in sc_d[y]]))
    for y in sc_d:
        st_y_d[y] = st(categories, sc_d[y])

    return st_y_d


def wt(sc_d):

    categories = list(set([cat for y in sc_d for cat in sc_d[y]]))
    wt_d = {y: {cat: {} for cat in categories} for y in years}
    for cat in categories:
        for y in tqdm(sc_d):
            for s in sc_d[y][cat]:
                term = s.split('_')[0]
                term_senses = [f'{term}_{i}' for i in range(10)]
                #P(s_c | t, term_c)
                fsenses_termc = sum([sc_d[y][cat][s] for y in sc_d for s in term_senses \
                        if s in sc_d[y][cat]])
                p_sc_given_ttermc = sc_d[y][cat][s] / fsenses_termc
                #P(s_c | term_c)
                fs_termc = sum([sc_d[y][cat][s] for y in sc_d \
                        if s in sc_d[y][cat]])
                fsenses_termc = 0
                for sense in term_senses:
                    fsenses_termc += sum([sc_d[y][cat][sense] for y in sc_d \
                            if sense in sc_d[y][cat]])
                p_s_given_termc = fs_termc / fsenses_termc
                wt_d[y][cat][s] = p_sc_given_ttermc / p_s_given_termc
                #add in code for normalisation

    return wt_d


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
                        if m_score < 1:
                            t_score = tnpmi_y_d[y][cat][w]
                            o_lst.append((w, (t_score, m_score)))
 
    return o_lst


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-a_s', '-additional_senses', required=True)
    args = parser.parse_args()
    a_s = [int(v) for v in args.a_s.split('-')]

    #get counts
    ts_path = f'{so_path}/as_output_{a_s[0]}_{a_s[1]}'
    tsc_d, sc_d = misc.get_items(['tsc_d', 'sc_d'], ts_path, get_counts, so_path, ts_path, years)
    #merge counts
    sc_d = merge_counts(tsc_d, sc_d)

    #get scores
    st_y_d = misc.get_items('st_y_d', ts_path, st_by_year, sc_d)
    wt_d = misc.get_items('wt_d', ts_path, wt, sc_d) 
    #rekey
    st_y_d = rekey_snpmi_y_d(st_y_d)
    wt_d = rekey_snpmi_y_d(wt_d)
    sc_d = rekey_snpmi_y_d(sc_d)
    tsc_d = rekey_snpmi_y_d(tsc_d)
    #scatter T* and M_t scores
    with open(f'{to_path}/tnpmi_y_d.pickle', 'rb') as f_name:
        tnpmi_y_d = pickle.load(f_name)
    tm_tpls = tnpmi_mtscores(tnpmi_y_d, st_y_d, sc_d)
    X = [tpl[1][0] for tpl in tm_tpls]
    y = [tpl[1][1] for tpl in tm_tpls]
    plt.figure()
    plt.scatter(X, y, alpha=0.5)
    plt.savefig(f'{sa_path}/scatter_test.png', bbox_inches='tight')
    print(stats.spearmanr(X, y))


if __name__ == '__main__':
    main()

