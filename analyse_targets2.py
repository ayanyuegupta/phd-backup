from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import mannwhitneyu
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


def rekey_snpmi_y_d(snpmi_y_d):
  
    for y in snpmi_y_d:
        snpmi_y_d[y]['home_office'] = snpmi_y_d[y].pop('home')
        snpmi_y_d[y]['cabinet_office'] = snpmi_y_d[y].pop('cabinet')

    return snpmi_y_d


def contingency_table(w, tsc_d, drop_cats=[]):

    if type(w) == str:
        lst = sorted(list(set([s for y in tsc_d for cat in tsc_d[y] for s in tsc_d[y][cat] \
                if s.split('_')[0] == w])))
    elif type(w) == list:
        lst = w
    categories = sorted(list(set([cat for y in tsc_d for cat in tsc_d[y] \
            if cat not in drop_cats])))
    d = {s: [] for s in lst}
    for cat in categories:
        for s in lst:
            d[s].append(sum([tsc_d[y][cat][s] for y in tsc_d if s in tsc_d[y][cat]]))
    df = pd.DataFrame.from_dict(data=d, orient='index', columns=categories).T
    
    return df


def get_pval(z_score, bonferroni=1):
    
    return stats.norm.sf(abs(z_score)) * bonferroni


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-a_s', '-additional_senses', required=True)
    args = parser.parse_args()
    a_s = [int(v) for v in args.a_s.split('-')]
    
    #get word counts
    with open(f'{to_path}/cby_d.pickle', 'rb') as f_name:
        cby_d = pickle.load(f_name)

    #word counts contingency table
    tfd_path = f'{ta_path}/freq_dists'
    if not os.path.exists(tfd_path):
        os.makedirs(tfd_path)
    ct = contingency_table(targets, cby_d)
    c, tp, dof, expected = chi2_contingency(ct)
    std_res = sm.stats.Table(ct).standardized_resids.round(2)
    p_vals = std_res.applymap(get_pval)
    df = pd.concat([ct, std_res])
    df.to_csv(f'{tfd_path}/{"_".join(targets)}.csv')
    print(ct)
    print(pd.DataFrame(data=expected, columns=ct.columns))
    print(std_res)

    #get sense counts
    ts_path = f'{so_path}/as_output_{a_s[0]}_{a_s[1]}'
    with open(f'{ts_path}/tsc_d.pickle', 'rb') as f_name:
        tsc_d = pickle.load(f_name)
    tsc_d = rekey_snpmi_y_d(tsc_d)
    
    #sense counts departmental contingency tables
    sfd_path = f'{sa_path}/freq_dists'
    if not os.path.exists(sfd_path):
        os.makedirs(sfd_path)
    s_pvals = []
    s_tp = []
    for w in targets:
        ct = contingency_table(w, tsc_d, drop_cats=['leg'])
        c, p, dof, expected = chi2_contingency(ct)
        std_res = sm.stats.Table(ct).standardized_resids.round(2)
        s_pvals.append(std_res.applymap(get_pval).T)
        df = pd.concat([ct, std_res])
        df.to_csv(f'{sfd_path}/{w}.csv')
        s_tp.append(p) 
        print(ct)
        print(pd.DataFrame(data=expected, columns=ct.columns))
        print(std_res)
    s_pvals = pd.concat(s_pvals)

    #adjust residual p values
    p_vals_flat = p_vals.to_numpy().flatten()
    s_pvals_flat = s_pvals.to_numpy().flatten()
    all_pvals = np.concatenate((p_vals_flat, s_pvals_flat))
    adj_pvals = multipletests(all_pvals, method='fdr_bh')[1]
    t_apvals = pd.DataFrame(data=adj_pvals[: len(p_vals_flat)].reshape(p_vals.shape), columns=p_vals.columns)
    s_apvals = pd.DataFrame(data=adj_pvals[len(p_vals_flat):].reshape(s_pvals.shape), columns=s_pvals.columns)
    t_apvals = t_apvals.rename(index={i: p_vals.index[i] for i, _ in enumerate(p_vals.index)})
    s_apvals = s_apvals.rename(index={i: s_pvals.index[i] for i, _ in enumerate(s_pvals.index)})
    t_apvals.to_csv(f'{tfd_path}/p_vals.csv')
    s_apvals.to_csv(f'{sfd_path}/p_vals.csv')
    print(t_apvals)
    print(s_apvals)
    print([tp] + s_tp)

#    #wellbeing senses contingency table
#    o_path = f'{sa_path}/freq_dists/leg'
#    if not os.path.exists(o_path):
#        os.makedirs(o_path)
#    ct = contingency_table('wellbeing', tsc_d)
#    c, p, dof, expected = chi2_contingency(ct)
#    std_res = sm.stats.Table(ct).standardized_resids.round(2)
#    p_vals = std_res.applymap(get_pval)
#    df = pd.concat([ct, std_res, p_vals])
#    df.to_csv(f'{o_path}/wellbeing.csv')
#    print(p)
#    print(ct)
#    print(pd.DataFrame(data=expected, columns=ct.columns))
#    print(std_res)
#    print(p_vals)


if __name__ == '__main__':
    main()

