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
    sc_d = {}
    for y in years:
        with open(f'{so_path}/sense_counts/{y}_scc.pickle', 'rb') as f_name:
            sc_d[y] = pickle.load(f_name)
        with open(f'{ts_path}/sense_counts/{y}_scc.pickle', 'rb') as f_name:
            tsc_d[y] = pickle.load(f_name) 

    return sc_d, tsc_d


def target_rfs(targets, sc_d, tsc_d, a_s):
    
    o_d = {w: {} for w in targets}
    for w in targets:
        w_senses = [f'{w}_{i}' for i in range(a_s[-1] * 2)]
        o_d[w] = {s: [0 for i, _ in enumerate(sc_d)] for s in w_senses}
        for i, year in enumerate(sc_d):
            total = sum([sc_d[year][cat][s] for cat in sc_d[year] for s in sc_d[year][cat]])
            t_senses = list(set([s for cat in tsc_d[year] for s in tsc_d[year][cat]]))
            for s in w_senses:
                if s in t_senses: 
                    ts_total = sum([tsc_d[year][cat][s] for cat in tsc_d[year] if s in tsc_d[year][cat]])
                    o_d[w][s][i] =  ts_total / total
    
    return o_d
                         

def plot_sense_rfs(rf_d, path, a_s, min_cdiff=0.6, leg_size=8, n_lcol=3):
        
    colours = misc.random_colours(a_s[1] * 2, min_diff=min_cdiff)
    for w in targets:
        ax = plt.figure().gca()
        for i, s in enumerate(rf_d[w]):
            if not all(i == 0 for i in rf_d[w][s]):
                plot_gam(years, rf_d[w][s], f'{path}/{w}.png', w, colour_label=(colours[i], s), ax=ax, conf_int=False, y_label='Relative Frequency')
#                plt.plot(years, rf_d[w][s], color=colours[i])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        leg = plt.legend(by_label.values(), by_label.keys(), loc='upper left', prop={'size': leg_size}, framealpha=0, ncol=n_lcol)
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh.set_sizes([50])
        plt.savefig(f'{path}/{w}.png', bbox_inches='tight')


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
    sc_d, tsc_d = get_counts(so_path, ts_path, years)

    #relative frequencies
    rf_d = target_rfs(targets, sc_d, tsc_d, a_s)
    path = f'{sa_path}/rf_plots'
    if not os.path.exists(path):
        os.makedirs(path)
    plot_sense_rfs(rf_d, path, a_s)


if __name__ == '__main__':
    main()
