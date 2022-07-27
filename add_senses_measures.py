import numpy as np
import math
from tqdm import tqdm
import measures as m
import argparse
from collections import OrderedDict
from analyse_diffusion import plot_gam
import utils.misc as misc
from measures import scounts_by_year 
import matplotlib.pyplot as plt
import pickle
import os


years = [2000 + i for i in range(21)]
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
sa_path = f'{root}/sense_analysis'
so_path = f'{root}/s_measures_output_{k_range[0]}_{k_range[1]}'
c_path = f'{data_root}/clusters_{k_range[0]}_{k_range[1]}'


def get_counts(so_path, ts_path, years):

    tsc_d = {}
    sc_d = {y: {} for y in years}
    for y in years:
        with open(f'{ts_path}/sense_counts/{y}_scc.pickle', 'rb') as f_name:
            tsc_d[y] = pickle.load(f_name)
        with open(f'{so_path}/sense_counts/{y}_scc.pickle', 'rb') as f_name:
            sc_d[y] = pickle.load(f_name)

    return tsc_d, sc_d



def target_rfs(targets, sc_d, tsc_d, a_s):
    
    o_d = {w: {} for w in targets}
    for w in targets:
        w_senses = [f'{w}_{i}' for i in range(a_s[-1])]
        o_d[w] = {s: [0 for i, _ in enumerate(sc_d)] for s in w_senses}
        for i, year in enumerate(sc_d):
            total = sum([sc_d[year][cat][w] for cat in sc_d[year] for w in sc_d[year][cat]])
            t_senses = list(set([s for cat in tsc_d[year] for s in tsc_d[year][cat]]))
            for s in w_senses:
                if s in t_senses: 
                    ts_total = sum([tsc_d[year][cat][s] for cat in tsc_d[year] if s in tsc_d[year][cat]])
                    o_d[w][s][i] =  ts_total / total
    
    return o_d
                         

def plot_sense_rfs(targets, rf_d, path, a_s, min_cdiff=0.7, leg_size=15, n_lcol=2, size=(6, 18), font_size=20, tick_size=15, y_label=None):
        
    colours = misc.random_colours(a_s[1], min_diff=min_cdiff)
    fig, ax = plt.subplots(len(targets), 1, figsize=size)
    for i, w in enumerate(targets):
#        ax = plt.figure(figsize=size).gca()
        for j, s in enumerate(rf_d[w]):
            if not all(j == 0 for j in rf_d[w][s]):
                plot_gam(years, rf_d[w][s], f'{path}/{w}.png', w, colour_label=(colours[j], s.split('_')[1]), ax=ax[i], save=False, conf_int=False, font_size=font_size, tick_size=tick_size)
#                plt.plot(years, rf_d[w][s], color=colours[i]
        #        plt.savefig(f'{path}/{w}.png', bbox_inches='tight')
    if y_label is not None:
        fig.text(0, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=font_size)
    fig.tight_layout()
    lines_labels = [a.get_legend_handles_labels() for a in ax]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    leg = fig.legend(np.unique(labels), loc='upper right', prop={'size': leg_size}, framealpha=1, ncol=n_lcol)

    plt.savefig(f'{path}/test.png', bbox_inches='tight')


def ts_npmi(categories, tsc_yd, sc_yd):

    tsnpmi_d = {cat: {} for cat in tsc_yd}
    s_fs = sum([sc_yd[cat][s] for cat in sc_yd for s in sc_yd[cat]])
    for cat in categories:
        #P(sense | c)
        s_fcs = sum([sc_yd[cat][s] for s in sc_yd[cat]])
        for s in tsc_yd[cat]:
            p_ts_given_c = tsc_yd[cat][s] / s_fcs
            #P(sense)
            f_ts = sum([tsc_yd[cat][s] for cat in categories if s in tsc_yd[cat]]) 
            p_ts = f_ts / s_fs
            #sense pmi
            tspmi = math.log(p_ts_given_c / p_ts)
            #normalise
            #P(sense, cat)
            p_tsc = -math.log(tsc_yd[cat][s] / s_fs)
            tsnpmi_d[cat][s] = tspmi / p_tsc

    return tsnpmi_d


def tsnpmi_by_year(tsc_d, sc_d):
    
    tsnpmi_y_d = {}
    categories = list(set([cat for y in tsc_d for cat in tsc_d[y]]))
    for y in tsc_d:
        tsnpmi_y_d[y] = ts_npmi(categories, tsc_d[y], sc_d[y])
    
    return tsnpmi_y_d


def ts_nvol(tsc_d, sc_d):

    categories = list(set([cat for y in tsc_d for cat in tsc_d[y]]))
    tsnvol_d = {y: {cat: {} for cat in categories} for y in years}
    for cat in categories:
        s_fs = sum([sc_d[y][cat][s] for y in sc_d for s in sc_d[y][cat]])
        for y in tsc_d:
            #P(sense_c | t)
            s_fts = sum([sc_d[y][cat][s] for s in sc_d[y][cat]])
            for s in tsc_d[y][cat]:
                p_ts_given_t = tsc_d[y][cat][s] / s_fts
                #P(sense_c)
                f_ts = sum([tsc_d[year][cat][s] for year in tsc_d if s in tsc_d[year][cat]])
                p_ts = f_ts / s_fs
                #vol
                tsvol = math.log(p_ts_given_t / p_ts)
                #normalise
                #P(sense, t)
                p_tst = -math.log(tsc_d[y][cat][s] / s_fs)
                tsnvol_d[y][cat][s] = tsvol / p_tst
    
    return tsnvol_d

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-a_s', '-additional_senses', required=True)
    args = parser.parse_args()
    a_s = [int(v) for v in args.a_s.split('-')]
   
    #get target sense counts
    ts_path = f'{so_path}/as_output_{a_s[0]}_{a_s[1]}'
    if not os.path.exists(ts_path):
        os.makedirs(ts_path)
    all_years_senses, all_years_sc = misc.get_items(['all_years_senses', 'all_years_sc'], ts_path, scounts_by_year, c_path, ts_path, a_s=a_s)
    tsc_d, sc_d = misc.get_items(['tsc_d', 'sc_d'], ts_path, get_counts, so_path, ts_path, years)

    #relative frequencies
    rf_d = misc.get_items('rf_d', ts_path, target_rfs, targets, sc_d, tsc_d, a_s)
    path = f'{sa_path}/rf_plots/{a_s[0]}_{a_s[1]}'
    if not os.path.exists(path):
        os.makedirs(path)
    plot_sense_rfs(targets, rf_d, path, a_s)

    #get spec, vol measures
    tsnpmi_y_d = misc.get_items('tsnpmi_y_d', ts_path, tsnpmi_by_year, tsc_d, sc_d)
    tsnvol_d = misc.get_items('tsnvol_d', ts_path, ts_nvol, tsc_d, sc_d)
    
  
if __name__ == '__main__':
    main()
