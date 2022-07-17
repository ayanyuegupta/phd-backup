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
a_s = (10, 15)
root = '/home/gog/projects/gov_lv'
data_root = '/media/gog/external2/corpora/gov_corp'
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


def target_rfs(targets, sc_d, tsc_d):
    
    o_d = {w: {} for w in targets}
    for w in targets:
        w_senses = [f'{w}_{i}' for i in range(a_s[-1])]
        o_d[w] = {s: [0 for i, _ in enumerate(sc_d)] for s in w_senses}
        for i, year in enumerate(sc_d):
            total = 0
            for cat in sc_d[year]:
                total += sum([sc_d[year][cat][s] for s in sc_d[year][cat] \
                        if not w == s.split('_')[0]])            
            for s in w_senses:
                t_senses = list(set([s for cat in tsc_d[year] for s in tsc_d[year][cat]]))
                if s in t_senses: 
                    ts_total = sum([tsc_d[year][cat][s] for cat in tsc_d[year] if s in tsc_d[year][cat]])
                    o_d[w][s][i] =  ts_total / total
        print(o_d)
        quit()
                         

def main():
    
    #get target sense counts
    ts_path = f'{so_path}/ts_output_{a_s[0]}_{a_s[1]}'
    if not os.path.exists(ts_path):
        os.makedirs(ts_path)
    all_years_senses, all_years_sc = misc.get_items(['all_years_senses', 'all_years_sc'], ts_path, scounts_by_year, c_path, ts_path, a_s=a_s)
    sc_d, tsc_d = get_counts(so_path, ts_path, years)

    #relative frequencies
    target_rfs(targets, sc_d, tsc_d)


if __name__ == '__main__':

    main()
