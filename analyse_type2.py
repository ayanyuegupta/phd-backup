import numpy as np
from analyse_type import average_wscores
import pandas as pd
import os
import pickle
import statistics
from scipy.stats import kruskal, f_oneway, mannwhitneyu

root = '/home/gog/projects/gov_lv'
t_path = f'{root}/t_measures_output_sample'
o_path = f'{root}/type_analysis'

target_words = [
#        'resilience',
#        'resilient',
#        'sustainable',
#        'sustainability',
#        'wellbeing',
        'paragraph',
        'subsection',
        'subparagraph',
        'section',
        ]

def main():
    
    with open(f'{t_path}/dis_d.pickle', 'rb') as f_name:
        dis_d = pickle.load(f_name)
    with open(f'{t_path}/dyn_d.pickle', 'rb') as f_name:
        dyn_d = pickle.load(f_name)

    #test leg vs. dep distinctiveness
    dep = []
    dep_cats = list(set([cat for y in dis_d for cat in dis_d[y] if cat != 'leg']))
    for year in dis_d:
        dep.append(statistics.mean([dis_d[year][cat] for cat in dep_cats]))
    leg = [dis_d[year]['leg'] for year in dis_d]
    U1, p = mannwhitneyu(dep, leg)
    print(p)
    
    #test leg vs. dep subsection indexing terms
    with open(f'{t_path}/tnpmi_y_d.pickle', 'rb') as f_name:
        tnpmi_y_d = pickle.load(f_name)
    avg_d = average_wscores(tnpmi_y_d) 
    dep = []
    leg = []
    for cat in avg_d:
        if cat != 'leg':
            dep += [avg_d[cat][w] for w in target_words]
        else:
            leg += [avg_d[cat][w] for w in target_words]
    U1, p = mannwhitneyu(dep, leg)
    print(p)


if __name__ == '__main__':
    main()
