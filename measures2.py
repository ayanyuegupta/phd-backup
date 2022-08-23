import math
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


def st_by_year(sc_d, min_count=0):

    categories = list(set([cat for y in sc_d for cat in sc_d[y]]))
    st_y_d = {y: {cat: {} for cat in categories} for y in years}
    for y in sc_d:
        for cat in tqdm(categories):
            for s in sc_d[y][cat]:
                term = s.split('_')[0]
                term_senses = [f'{term}_{i}' for i in range(10)]
                n_senses = len([s for s in term_senses if s in sc_d[y][cat]])
                if n_senses > 1 and sc_d[y][cat][s] > min_count:
                    #P(s | c, term)
                    fsenses_cterm = sum([sc_d[y][cat][s] for s in term_senses \
                            if s in sc_d[y][cat]])
                    p_s_given_cterm = sc_d[y][cat][s] / fsenses_cterm
                    #P(s | term)
                    fs_term = sum([sc_d[y][cat][s] for cat in categories \
                            if s in sc_d[y][cat]])
                    fsenses_term = 0
                    for sense in term_senses:
                        fsenses_term += sum([sc_d[y][cat][sense] for cat in categories \
                                if sense in sc_d[y][cat]])
                    p_s_given_term = fs_term / fsenses_term
                    pmi = math.log(p_s_given_cterm / p_s_given_term)
                    #normalise
                    #P((s | T), c)
                    p_sgterm_c = sc_d[y][cat][s] / fsenses_term
                    if -math.log(p_sgterm_c) != 0:
                        st_y_d[y][cat][s] = pmi / -math.log(p_sgterm_c)
    
    return st_y_d


def wt(sc_d, min_count=0):

    categories = list(set([cat for y in sc_d for cat in sc_d[y]]))
    wt_d = {y: {cat: {} for cat in categories} for y in years}
    for cat in categories:
        for y in tqdm(sc_d):
            for s in sc_d[y][cat]:
                term = s.split('_')[0]
                term_senses = [f'{term}_{i}' for i in range(10)]
                n_senses = len([s for s in term_senses if s in sc_d[y][cat]])
                if n_senses > 1 and sc_d[y][cat][s] > min_count:
                    #P(s_c | t, term_c)
                    fsenses_termc = sum([sc_d[y][cat][s] for s in term_senses \
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
                    pmi = math.log(p_sc_given_ttermc / p_s_given_termc)
                    #normalise
                    #P((s_c | T_c), t)
                    p_scgiventermc_t = sc_d[y][cat][s]/fsenses_termc
                    if -math.log(p_scgiventermc_t) != 0:
                        wt_d[y][cat][s] = pmi / -math.log(p_scgiventermc_t)

    return wt_d


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
    

if __name__ == '__main__':
    main()

