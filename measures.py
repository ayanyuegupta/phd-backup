import matplotlib.pyplot as plt
import enchant
from statistics import mean
import statistics
import statsmodels.api as sm
import pandas as pd
import numpy as np
import os
import spacy
import re
import pickle
import math
import re
import random
from collections import Counter
from tqdm import tqdm
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import utils.misc as misc


k_range = (2, 10)
targets = None
#targets = [
#        'resilience',
#        'resilient',
#        'sustainable',
#        'sustainability',
##        'system',
#        'wellbeing',
##        'environment',
##        'climate'
#        ]

categories = [
'MOD',
'cabinet_office',
'leg',
'treasury',
'home_office',
'DHSC',
'DWP',
'DE',
'MOH',
'DCMS',
'DEFRA',
'FCO',
'MOJ'
]

root = '/home/gog/projects/gov_lv'
data_root = '/media/gog/external2/corpora'
data_path = f'{data_root}/gov_corp/sample'
i_path = f'{data_root}/gov_corp/sample/merged.pickle'
o_path = f'{os.getcwd()}/t_measures_output_{data_path.split("/")[-1]}' 
so_path = f'{os.getcwd()}/s_measures_output_{k_range[0]}_{k_range[1]}'
c_path = f'{data_root}/gov_corp/clusters_{k_range[0]}_{k_range[1]}'
t_path = f'{root}/type_analysis/test_scatters'

years = [str(2000 + i) for i in range(21)]


#word counts
def sum_counts(counts):

    all_counts = {}
    for category in counts:
        for w in counts[category]:
            if w not in all_counts:
                all_counts[w] = counts[category][w]
            else:
                all_counts[w] += counts[category][w]

    return all_counts


def get_counts(i_path, categories=categories, prop=None, stopwords=True, regex_pattern='[^A-Za-z0-9\'.\s]+'):
    
    print('Getting word counts...') 
    if stopwords:
        sp = spacy.load('en_core_web_sm')
        stopwords = sp.Defaults.stop_words
#        with open(f'{os.getcwd()}/gov_lv/stopwords.pickle', 'rb') as f_name:
#            stopwords = pickle.load(f_name)
        stopwords = list(stopwords) + \
                [w.capitalize() for w in stopwords] + \
                [w.upper() for w in stopwords] 
    counts = {} 
    with open(i_path, 'rb') as f:
        data = pickle.load(f)  
    if categories is None:
        categories = [cat for cat in data]
    for category in categories:
        category_words = []
        sequences = [ntuple[-1] for ntuple in data[category]]
        for sequence in tqdm(sequences, desc=category):              
            if regex_pattern is not None:
                remove = re.compile(regex_pattern)
                sequence = re.sub(remove, '', sequence)
                sequence = sequence.replace(',','').lower()
            sequence = sequence.split()
            if stopwords:
                sequence = [item for item in sequence if item not in stopwords]
            category_words += sequence  
        counts[category] = Counter(category_words)
        if prop is not None:
            n = int(prop * len(counts[category]))
            counts[category] = {tpl[0]: tpl[1] for tpl in counts[category].most_common(n)}       
    all_counts = sum_counts(counts) 

    return counts, all_counts


def counts_by_year(data_path, years):

    cby_d = {}
    files = [f_name for f_name in os.listdir(data_path) if f_name.split('.')[0] in years]
    for f_name in tqdm(files, desc='Years completed...'): 
        year = int(f_name.split('.')[0])
        i_path = f'{data_path}/{f_name}'
        counts, _ = get_counts(i_path)
        cby_d[year] = counts

    return cby_d


#investigate cut off points
def cut_off_point(cby_d, cut_off=0.01, clean=True):

    categories = list(set([cat for y in cby_d for cat in cby_d[y]]))
#    eng_d = enchant.Dict('en_GB')
    counts = {}
    for y in tqdm(cby_d):
        for cat in categories:
            for w in cby_d[y][cat]:
                if w not in counts:
                    counts[w] = cby_d[y][cat][w]
                else:
                    counts[w] += cby_d[y][cat][w]
    counts = list(counts.items())
    if clean:
#        counts = [(w, count) for w, count in counts if eng_d.check(w) \
#                and not any(ch.isdigit() for ch in w)]
        counts = [(w, count) for w, count in counts if not any(ch.isdigit() for ch in w)] 
    n_occs = sum([count for w, count in counts])
    n_voc = len([w for w, count in counts])
    min_count = sorted(counts, key=lambda x: x[1])[0]
    print('Full dataset counts:')
    print(f'Number of occurrences: {n_occs}, Vocabulary size: {n_voc}')
    print(f'Minimum word frequency: {min_count}')
    
    o_d = {y: {cat: {} for cat in categories} for y in cby_d}
    counts = sorted(counts, reverse=True, key=lambda x: x[1])[:int(len(counts) * cut_off)]
    for w, _ in tqdm(counts):
        for y in cby_d:
            for cat in cby_d[y]:
                if w in cby_d[y][cat]:
                    o_d[y][cat][w] = cby_d[y][cat][w]
    n_occs = sum([count for w, count in counts])
    n_voc = len([w for w, count in counts])
    min_count = counts[-1]
    print(f'\nTop {round(cut_off * 100, 2)}%:')
    print(f'Number of occurrences: {n_occs}, Vocabulary size: {n_voc}')
    print(f'Minimum word frequency: {min_count}')
    
    return o_d


#word measures        
def get_specs(cby_d, normalise=True):
    
    categories = list(set([cat for y in cby_d for cat in cby_d[y]]))
    o_d = {y: {cat: {} for cat in categories} for y in cby_d}
    for y in cby_d:
        vocab = [w for cat in cby_d[y] for w in cby_d[y][cat]]
        vocab = list(set(vocab))
        sumwc_fcw = sum([cby_d[y][cat][w] for cat in cby_d[y] for w in cby_d[y][cat]])
        for w in tqdm(vocab, desc=f'{y}'):
            #P(t)
            sumc_fct = sum([cby_d[y][cat][w] for cat in categories if w in cby_d[y][cat]])
            Pt = sumc_fct / sumwc_fcw
            #P(t|c)
            for cat in categories:
                if w in cby_d[y][cat]:
                    fct = cby_d[y][cat][w]
                    sumw_fcw =sum([cby_d[y][cat][word] for word in cby_d[y][cat]])
                    Ptgc = fct / sumw_fcw
                    spec = math.log(Ptgc / Pt)
                    o_d[y][cat][w] = spec 
                    if normalise:
                        #P(t, c)
                        Ptc = fct / sumwc_fcw
                        o_d[y][cat][w] = spec / -math.log(Ptc)
    
    return o_d


def get_vols(cby_d, normalise=True):

    categories = list(set([cat for y in cby_d for cat in cby_d[y]]))
    o_d = {y: {cat: {} for cat in categories} for y in cby_d}
    for cat in categories:
        vocab = [w for y in cby_d for w in cby_d[y][cat]]
        vocab = list(set(vocab))
        sumwy_fyw = sum([cby_d[y][cat][w] for y in cby_d for w in cby_d[y][cat]])
        for w in tqdm(vocab, desc=f'{cat}'):
            #P(t_c)
            sumy_ft_c = sum([cby_d[y][cat][w] for y in cby_d if w in cby_d[y][cat]])
            Pt_c = sumy_ft_c / sumwy_fyw
            #P(t_c|y)
            for y in cby_d:
                if w in cby_d[y][cat]:
                    fyt_c= cby_d[y][cat][w]
                    sumw_fyw_c = sum([cby_d[y][cat][word] for word in cby_d[y][cat]])
                    Pt_cgy = fyt_c / sumw_fyw_c
                    vol = math.log(Pt_cgy / Pt_c)
                    o_d[y][cat][w] = vol
                    if normalise:
                        #P(t_c, y)
                        Pt_cy = fyt_c / sumwy_fyw
                        o_d[y][cat][w] = vol / -math.log(Pt_cy)

    return o_d


#sense counts
def load_senses(path, a_s=None):
    
    if a_s is None:
        path = f'{path}/senses'
    else:
        path = f'{path}/added_senses_{a_s[0]}_{a_s[1]}'
    with open(path, 'r') as f_name:
        senses = f_name.read()
    senses = senses.split('\n')
    senses = (sense.split('\t') for sense in senses if len(sense) > 1)

    return senses


def sense_counts(senses):

    all_senses = 0
    s_counts = {}
    scounts_cat = {}
    for sense in senses:
        cat = sense[0].split('_')[0]
        sense = '_'.join(sense[1:])
        if cat not in scounts_cat:
            scounts_cat[cat] = {}
        if sense not in s_counts:
            s_counts[sense] = 1
        else:
            s_counts[sense] += 1
        if sense not in scounts_cat[cat]:
            scounts_cat[cat][sense] = 1
        else:
            scounts_cat[cat][sense] += 1
        all_senses += 1

    return all_senses, s_counts, scounts_cat


def scounts_by_year(c_path, o_path, a_s=None):

    paths = [f'{c_path}/{path}' for path in os.listdir(c_path) if 'senses' in path]
    all_years_senses = 0 
    all_years_sc = {}
    for path in tqdm(paths):
        year = int(path.split('/')[-1].split('_')[0])
        senses = load_senses(path, a_s=a_s)
        s_path = f'{o_path}/sense_counts'
        if not os.path.exists(s_path):
            os.makedirs(s_path)
        all_senses, s_counts, scounts_cat = misc.get_items([f'{year}_as', f'{year}_sc', f'{year}_scc'], s_path, sense_counts, senses)
        all_years_senses += all_senses
        for cat in scounts_cat:
            if cat not in all_years_sc:
                all_years_sc[cat] = {}
            for sense in scounts_cat[cat]:
                if sense not in all_years_sc[cat]:
                    all_years_sc[cat][sense] = scounts_cat[cat][sense]
                else:
                    all_years_sc[cat][sense] += scounts_cat[cat][sense]

    return all_years_senses, all_years_sc


#sense measures
def sense_npmi(path, year, o_path):
    
    #sense pmi = log(P(sense | c) / P(sense))
    senses = load_senses(path)
    s_path = f'{o_path}/sense_counts'
    if not os.path.exists(s_path):
        os.makedirs(s_path)
    all_senses, s_counts, scounts_cat = misc.get_items([f'{year}_as', f'{year}_sc', f'{year}_scc'], s_path, sense_counts, senses)
    snpmi_d = {cat: {} for cat in scounts_cat}
    for cat in scounts_cat:
        all_senses_cat = sum([scounts_cat[cat][sense] for sense in scounts_cat[cat]])
        for sense in scounts_cat[cat]:
            #P(sense | c), P(sense)
            p_sense_given_c = scounts_cat[cat][sense] / all_senses_cat
            p_sense = s_counts[sense] / all_senses
            #sense pmi
            x = math.log(p_sense_given_c / p_sense)
            #normalise
            y = -math.log(scounts_cat[cat][sense] / all_senses)
            snpmi_d[cat][sense] = x / y

    return snpmi_d
     
    
def snpmi_by_year(c_path, o_path):
    
    snpmi_y_d = {}
    paths = [f'{c_path}/{path}' for path in os.listdir(c_path) if 'senses' in path]
    for path in tqdm(paths):
        year = int(path.split('/')[-1].split('_')[0])
        snpmi_y_d[year] = sense_npmi(path, year, o_path)

    return snpmi_y_d


def sense_nvol(c_path, o_path, all_years_sc):
    
    #sense vol = log(P(sense_c | t) / P(sense_c))
    paths = [f'{c_path}/{path}' for path in os.listdir(c_path) if 'senses' in path]
    years = [int(path.split('/')[-1].split('_')[0]) for path in paths ]
    snvol_d = {year: {} for year in years}
    for path in tqdm(paths):
        #get counts
        year = int(path.split('/')[-1].split('_')[0])
        senses = load_senses(path)
        s_path = f'{o_path}/sense_counts'
        if not os.path.exists(s_path):
            os.makedirs(s_path)
        all_senses, s_counts, scounts_cat = misc.get_items([f'{year}_as', f'{year}_sc', f'{year}_scc'], s_path, sense_counts, senses)
        for cat in scounts_cat:
            snvol_d[year][cat] = {} 
            all_senses_cat = sum([scounts_cat[cat][sense] for sense in scounts_cat[cat]])
            ally_alls_c = sum([all_years_sc[cat][sense] for sense in all_years_sc[cat]])
            for sense in scounts_cat[cat]:
                #P(sense_c | t), P(sense_c)
                p_sensec_given_t = scounts_cat[cat][sense] / all_senses_cat
                p_sensec = all_years_sc[cat][sense] / ally_alls_c
                #sense volatility
                x = math.log(p_sensec_given_t / p_sensec)
                #normalise
                y = -math.log(scounts_cat[cat][sense] / ally_alls_c)
                snvol_d[year][cat][sense] = x / y

    return snvol_d


#category level measures
def measure_expression(expression, d, only_eng=False):
    
    expression = [w for w in expression.split() if w in d]
    if only_eng:
        eng_d = enchant.Dict('en_GB')
        expression = [w for w in expression if eng_d.check(w)]
    if len(expression) > 0:    
        value = mean([d[w] for w in expression])
        return value
    else:
        return None


def cat_scores(data_path, tnpmi_y_d, nv_d, categories=categories):

    dis_d = {}
    dyn_d = {}
    if categories is None:
        categories = list(set([cat for y in tnpmi_y_d for cat in tnpmi_y_d[y]]))
    for y in tnpmi_y_d:
        dis_d[y] = {}
        dyn_d[y] = {}
        with open(f'{data_path}/{y}.pickle', 'rb') as f_name:
            data = pickle.load(f_name)
        for cat in tqdm(categories, desc=str(y)):
            tnpmi_values = []
            nv_values = []
            for _, seq in tqdm(data[cat], desc=cat):
                u_tnpmi = measure_expression(seq, tnpmi_y_d[y][cat])
                if u_tnpmi is not None:
                    tnpmi_values.append(u_tnpmi)
                u_v = measure_expression(seq, nv_d[y][cat])
                if u_v is not None:
                    nv_values.append(u_v)
            dis_d[y][cat] = mean(tnpmi_values)
            dyn_d[y][cat] = mean(nv_values)

    return dis_d, dyn_d


def average(d, categories=categories):
    
    if categories is None:
        categories = list(set([cat for year in d for cat in d[year]]))
    avg_d = {}
    for cat in categories:
        avg_d[cat] = mean([d[year][cat] for year in d])
    print(avg_d)

    return avg_d


#other
def test_scatters(spec_d, vol_d, t_path):

    for year in spec_d:
        for cat in spec_d[year]:
            X = [spec_d[year][cat][w] for w in spec_d[year][cat]]
            y = [vol_d[year][cat][w] for w in spec_d[year][cat]]
            plt.figure(figsize=(6, 6))
            plt.scatter(X, y, color=[0, 0, 1], alpha=0.2)
            plt.title(f'{year} {cat}')
            plt.xlabel('Normalised Specificity')
            plt.ylabel('Normalised Volatility')
            plt.savefig(f'{t_path}/{year}_{cat}_test.png', bbox_inches='tight')


def main():
    
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    if not os.path.exists(so_path):
        os.makedirs(so_path)
    
#    #Get counts by year
#    cby_d = misc.get_items('cby_d', o_path, counts_by_year, data_path, years)
#    cby_d = cut_off_point(cby_d)
#    quit()
#
#    #word measures
#    tnpmi_y_d = misc.get_items('tnpmi_y_d', o_path, get_specs, cby_d)
#    nv_d = misc.get_items('nv_d', o_path, get_vols, cby_d)       
#    if not os.path.exists(t_path):
#        os.makedirs(t_path)
##    test_scatters(tnpmi_y_d, nv_d, t_path)
#
#    #Distinctiveness and dynamicity
#    dis_d, dyn_d = misc.get_items(['dis_d', 'dyn_d'], o_path, cat_scores, data_path, tnpmi_y_d, nv_d)
#    avg_dis = misc.get_items('avg_dis', o_path, average, dis_d)
#    avg_dyn = misc.get_items('avg_dyn', o_path, average, dyn_d)
  
    #sense counts
    all_years_senses, all_years_sc = misc.get_items(['all_years_senses', 'all_years_sc'], so_path, scounts_by_year, c_path, so_path)
    #SNPMI by year
    snpmi_y_d = misc.get_items('snpmi_y_d', so_path, snpmi_by_year, c_path, so_path)

    #Normalised sense volatility
    snvol_d = misc.get_items('snvol_d', so_path, sense_nvol, c_path, so_path, all_years_sc)
     

if __name__ == '__main__':
    main()



