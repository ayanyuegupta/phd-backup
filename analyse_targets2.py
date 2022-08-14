import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats.contingency import margins
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd
import utils.misc as misc
import argparse
import os
import pickle


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


def contingency_table(w, tsc_d, drop_cats=['leg']):

    senses = sorted(list(set([s for y in tsc_d for cat in tsc_d[y] for s in tsc_d[y][cat] \
            if s.split('_')[0] == w])))
    categories = sorted(list(set([cat for y in tsc_d for cat in tsc_d[y] \
            if cat not in drop_cats])))
    d = {s: [] for s in senses}
    for cat in categories:
        for s in senses:
            d[s].append(sum([tsc_d[y][cat][s] for y in tsc_d if s in tsc_d[y][cat]]))
    df = pd.DataFrame.from_dict(data=d, orient='index', columns=categories)
    
    return df


def get_pval(z_score, alpha=0.01, tails=1):
    
    return stats.norm.sf(abs(z_score))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-a_s', '-additional_senses', required=True)
    args = parser.parse_args()
    a_s = [int(v) for v in args.a_s.split('-')]
    
    #get counts
    ts_path = f'{so_path}/as_output_{a_s[0]}_{a_s[1]}'
    with open(f'{ts_path}/tsc_d.pickle', 'rb') as f_name:
        tsc_d = pickle.load(f_name)
    
    #contingency tables
    for w in targets:
        ct = contingency_table(w, tsc_d)
        c, p, dof, expected = chi2_contingency(ct)
        std_res = sm.stats.Table(ct).standardized_resids.round(2)
        p_vals = std_res.applymap(get_pval)
        print(p)
        print(ct)
        print(std_res)
        print(p_vals)
        

if __name__ == '__main__':
    main()

