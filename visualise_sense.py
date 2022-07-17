import utils.misc as mi
import statistics
import random
import seaborn as sns
import math
from tqdm import tqdm
import matplotlib.ticker as ticker
import measures as m
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

#colour_seed = random.randint(0, 100)
colour_seed = 83
annotate = None
#annotate = [
#        'resilience_18',
#        'resilient_20'
#        ]
k_range = (35, 45)
words = [
        'resilience',
        'resilient',
        'sustainable',
        'sustainability',
        'wellbeing'
        ]

targets = [
        'resilience',
        'resilient',
        'sustainable',
        'sustainability',
        'wellbeing'
        ]

if words is not None:
    name = '_'.join(words + [str(k_range[0]), str(k_range[1])])
else:
    name = '_'.join([str(None), str(k_range[0]), str(k_range[1])])
root = '/home/gog/projects/gov_lv'
data_root = '/media/gog/external2/corpora/gov_corp'
i_path = f'{root}/s_measures_output_{name}'
c_path = f'{data_root}/clusters_{name}'
d_path = f'{data_root}/sample'
sc_path = f'{i_path}/sense_counts'
o_path = f'{root}/visualisations'
rf_path = f'{o_path}/relative_frequencies'
rfs_path = f'{o_path}/relative_frequencies_senses'
ss_path = f'{o_path}/sense_scatters/{name}'

####


def plot_rf(x, y, path, size=(10, 5), font_size=15, y_label='Relative Frequency', f_name='plot'):
    
    font = {'size': font_size}
    plt.rc('font', size=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('xtick', labelsize=font_size-2)
    plt.rc('ytick', labelsize=font_size-2)
    plt.rc('axes', titlesize=font_size) 
    ax = plt.figure(figsize=size).gca()
    for item in y:
        plt.plot(x, y[item], marker='o', label=item)
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel(y_label)
    plt.savefig(f'{path}/{f_name}.png', bbox_inches='tight')
    plt.close()
   

def wfreq_by_year(cby_d, words, rf_path, c=None):
    
    if c is None:
        categories = list(set([cat for year in cby_d for cat in cby_d[year]]))
    else:
        categories = c
    x = sorted([year for year in cby_d])
    y = {w: [] for w in words}
    for year in cby_d:    
        for w in words:
            n = 0
            f = 0
            for cat in categories:
                d = cby_d[year][cat]
                n += sum([d[w] for w in d])
                if w in d:                   
                    f += d[w]
            y[w].append(f / n)
    
    words = '_'.join(words) 
    f_name = f'{c}_{words}'
    plot_rf(x, y, rf_path, f_name=f_name)   


def get_sc_d(sc_path):
    
    x = sorted(list(set([int(f_name.split('_')[0]) for f_name in os.listdir(sc_path)])))
    y_d = {year: None for year in x}
    for year in x:
        with open(f'{sc_path}/{year}_scc.pickle', 'rb') as f_name:
            d = pickle.load(f_name)
        y_d[year] = d

    return y_d


def sfreq_by_year(y_d, word, rfs_path, c=['DE']):    
        
    if c is None:
        categories = list(set([cat for year in y_d for cat in y_d[year]]))
    else:
        categories = c
        for i, _ in enumerate(c):
            if '_' in c[i]:
                c[i] = c[i].split('_')[0]
    senses = []
    for year in y_d:
        d = y_d[year]
        senses += [sense for cat in d for sense in d[cat] if word == sense.split('_')[0]] 
    senses = list(set(senses))
    x = sorted([year for year in y_d])
    y = {s: [] for s in senses}
    for year in y_d:    
        for s in senses:
            n = 0
            f = 0
            for cat in categories:
                d = y_d[year][cat]
                n += sum([d[s] for s in d])
                if s in d:                   
                    f += d[s]
            y[s].append(f / n)
    
    f_name = f'{c}_{word}'
    plot_rf(x, y, rfs_path, f_name=f_name)


def get_all_rf(y_d, path, senses=False, w=None):
    
    if w is None:
        print('Input word or list of words.')
    categories = list(set([cat for year in y_d for cat in y_d[year]]))
    for cat in tqdm(categories):
        if not senses:
            wfreq_by_year(y_d, w, path, c=[cat])
        else:
            sfreq_by_year(y_d, w, path, c=[cat])


def get_vals(y_d, senses, cat, year=None):
    
    vals = []
    for s in senses:
        if year is None:
            scores = []
            for y in y_d:
                if cat in y_d[y]:
                    if s in y_d[y][cat]:
                        scores.append(y_d[y][cat][s])
            if len(scores) > 0:
                vals.append(statistics.mean(scores))
        else:
            if s in y_d[year][cat]:
                vals.append(y_d[year][cat][s])
    
    return vals


def random_colour():

    col = [0, 0, 0]
    for i, _ in enumerate(col):
        col[i] = round(random.random(), 1)

    return col


def scatter_senses(snpmi_y_d, snvol_d, category, ss_path, targets=None, year=None, size=(6, 6), colour=[0.3, 0.3, 0.3], highlight=[1, 0, 0], font_size=20, x_label=None, y_label=None, title=None, alpha=0.5, annotate=True, legend=False, colour_seed=colour_seed, s=40):
    
    font = {'size': font_size}
    plt.rc('font', size=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('xtick', labelsize=font_size-2)
    plt.rc('ytick', labelsize=font_size-2)
    plt.rc('axes', titlesize=font_size)
    plt.figure(figsize=size)
    if '_' in category:
        cat = category.split('_')[0]
    else:
        cat = category
    senses = []
    if year is None:
        for y in snpmi_y_d:
            if cat in snpmi_y_d[y]:
                senses += [s for s in snpmi_y_d[y][cat]]
    else:
        senses = [s for s in snpmi_y_d[year][cat]]
    senses = list(set(senses)) 
    x = get_vals(snpmi_y_d, senses, cat, year=year)
    y = get_vals(snvol_d, senses, cat, year=year)
    plt.scatter(x, y, c=np.array([colour]), alpha=alpha, s=s) 

    if targets is not None:
        random.seed(colour_seed)
        c_d = {w: random_colour() for w in targets}
        for w in targets:
            senses2 = [s for s in senses if s.split('_')[0] == w]
            x2 = get_vals(snpmi_y_d, senses2, cat, year=year)
            y2 = get_vals(snvol_d, senses2, cat, year=year)
            plt.scatter(x2, y2, c=np.array([c_d[w]]), alpha=1, label=w, s=s)
    
    if annotate is not None:
        if all(type(item) == str for item in annotate):
            if all(type(item) == str for item in annotate):
                for i, sense in enumerate(senses):
                    if any(w == sense for w in annotate):
                        plt.annotate(sense, (x[i], y[i]), fontsize=font_size-5)
                        plt.scatter(x[i], y[i], alpha=1, s=s*1.5, color=np.array([highlight]))
        else:
            pairs = [(senses[i], x[i], y[i]) for i, _ in enumerate(senses)]
            top_pairs1 = sorted(pairs, key=lambda x: x[1], reverse=True)
            top_pairs2 = sorted(pairs, key=lambda x: x[2], reverse=True)
            if all(type(item) == tuple for item in annotate):
                for si, vi in annotate:
                    if si <= len(top_pairs1) - 1:
                        plt.annotate(top_pairs1[si][0], (top_pairs1[si][1], top_pairs1[si][2]), fontsize=font_size-5)
                        plt.scatter(top_pairs1[si][1], top_pairs1[si][2], s=s*1.5, color=np.array([highlight]))
                    if vi <= len(top_pairs2) - 1:
                        plt.annotate(top_pairs2[vi][0], (top_pairs2[vi][1], top_pairs2[vi][2]), fontsize=font_size-5)
                        plt.scatter(top_pairs2[vi][1], top_pairs2[vi][2], s=s*1.5, color=np.array([highlight]))
            if all(type(item) == int for item in annotate):
                for i in annotate:
                    if i <= len(top_pairs1) - 1:
                        plt.annotate(top_pairs1[i][0], (top_pairs1[i][1], top_pairs1[i][2]), fontsize=font_size-5)
                        plt.scatter(top_pairs1[i][1], top_pairs1[i][2], s=s*1.5, color=np.array([highlight]))
                    if i <= len(top_pairs2) - 1:
                        plt.annotate(top_pairs2[i][0], (top_pairs2[i][1], top_pairs2[i][2]), fontsize=font_size-5)
                        plt.scatter(top_pairs2[i][1], top_pairs2[i][2], s=s*1.5, color=np.array([highlight]))

    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)
    if title is not None:
        plt.title(title)
    if legend:
        plt.legend(loc='upper left', frameon=True, framealpha=0.3, fontsize=font_size / 1.5)
    if targets is not None:
        path = f'{ss_path}/{"_".join(targets)}'
    else:
        path = f'{ss_path}/{targets}'
    if not os.path.exists(path):
        os.makedirs(path)
    if annotate is not None:
        if all(type(i) != str for i in annotate):
            annotate = [str(i) for i in annotate]
        plt.savefig(f'{path}/{"_".join(annotate)}_{year}_{category}.png', bbox_inches='tight')
    else:
        plt.savefig(f'{path}/{annotate}_{year}_{category}.png', bbox_inches='tight')


def scatter_all(snpmi_y_d, snvol_d, targets=targets, annotate=None, alpha=0.5):

    categories = list(set([cat for year in snpmi_y_d for cat in snpmi_y_d[year]]))
    for year in tqdm(snpmi_y_d):
        for category in tqdm(categories):
            if category in snpmi_y_d[year]:
                scatter_senses(snpmi_y_d, snvol_d, category, ss_path, targets=targets, year=year, x_label='Specificity', y_label='Volatility', title=category, annotate=annotate, legend=True, alpha=alpha)


def scatter_all_cat(snpmi_y_d, snvol_d, targets=targets, annotate=None, alpha=0.5):

    categories = list(set([cat for year in snpmi_y_d for cat in snpmi_y_d[year]]))
    for category in tqdm(categories):
        scatter_senses(snpmi_y_d, snvol_d, category, ss_path, targets=targets, x_label='Specificity', y_label='Volatility', title=category, annotate=annotate, legend=True, alpha=alpha)

def get_centroid(x, y):

    arr = np.array([[x[i], y[i]] for i, _ in enumerate(x)])
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    
    return sum_x/length, sum_y/length


def c_variance(x, y, centroid):

    distances = [np.sqrt((x[i] - centroid[0]) ** 2 + (y[i] - centroid[1]) ** 2) for i, _ in enumerate(x)]
    distances = [distance ** 2 for distance in distances]
    variance = sum(distances) / (len(distances) - 1)
    
    return variance


def plot_var(snpmi_y_d, snvol_d, words, ss_path, size=(10, 6), font_size=20, width=0.8, y_label=None):

    font = {'size': font_size}
    plt.rc('font', size=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('xtick', labelsize=font_size-2)
    plt.rc('ytick', labelsize=font_size-2)
    plt.rc('axes', titlesize=font_size)
    plt.figure(figsize=size)
    
    labels = list(set([cat for year in snpmi_y_d for cat in snpmi_y_d[year]]))
    n_bars = len(targets)
    b_width = width / n_bars
    fig = plt.figure(figsize=size)
    senses = []
    for y in snpmi_y_d:
        for cat in snpmi_y_d[y]:
            senses += [s for s in snpmi_y_d[y][cat]]
    senses = list(set(senses))

    for i, w in enumerate(targets):
        senses2 = [s for s in senses if s.split('_')[0] == w]
        offset = (i - n_bars / 2) * b_width + b_width / 2
        variances = []
        t_labels = []
        for cat in labels:
            x2 = get_vals(snpmi_y_d, senses2, cat, None)
            y2 = get_vals(snvol_d, senses2, cat, None)
            if len(x2) > 0 and len(y2) > 0:
                #get centroids and variance from centroids
                centroid = get_centroid(x2, y2)
                variances.append(c_variance(x2, y2, centroid))
                t_labels.append(cat)
        x = np.arange(len(t_labels))
        plt.bar(x + offset, variances, b_width, label=w)
    
    plt.xticks(x, labels)
    plt.legend(loc='upper left', frameon=False, fontsize=font_size-5)     
    plt.ylabel('Variance')
    plt.savefig(f'{ss_path}/test.png', bbox_inches='tight')


def see_senses(targets, c_path, d_path, o_path):
    
    d = {w: [] for w in targets}
    years = [y.split('.')[0] for y in os.listdir(d_path) if any(char.isdigit() for char in y)]
    for y in years:
        with open(f'{d_path}/{y}.pickle', 'rb') as f_name:
            data = pickle.load(f_name)
        with open(f'{c_path}/{y}_senses/senses', 'r') as f_name:
            senses = f_name.read()
        senses = senses.split('\n')
        senses = (s.split('\t') for s in senses if len(s) > 1 and any(s.split('\t')[1] == w for w in targets)) 
        for s in senses:
            location = s[0].split('_')
            cat = location[:-2]
            if len(cat) > 1:
                cat = '_'.join(cat)
            else:
                cat = cat[0]
            index = int(location[-1])
            word = s[1]
            sense = s[2]
            d[word].append((sense, data[cat][index]))
    d2 = {w: None for w in targets}
    for w in d:
        senses = [tpl[0] for tpl in d[w]]
        d2[w] = {sense: [] for sense in senses}
        for tpl in d[w]:
            sense = tpl[0]
            d2[w][sense].append(tpl[1])
    d = d2
    seq_path = f'{o_path}/sense_seq'
    if not os.path.exists(seq_path):
        os.makedirs(seq_path)
    for w in d:
        with open(f'{seq_path}/{w}.txt', 'w') as f_name:
            for s in d[w]:
                f_name.write(f'\n\n')
                for seq in d[w][s]:
                    f_name.write(f'\n~{s}: {seq[1]}, {seq[0]}')

def main():
    
    print(f'Colour seed: {colour_seed}')

    if not os.path.exists(o_path):
        os.makedirs(o_path)
    if not os.path.exists(rf_path):
        os.makedirs(rf_path)
    if not os.path.exists(rfs_path):
        os.makedirs(rfs_path)
    if not os.path.exists(ss_path):
        os.makedirs(ss_path)

#    #plot relative frequencies of words for each year
#    with open(f'{i_path}/cby_d.pickle', 'rb') as f_name:
#        cby_d = pickle.load(f_name)
#    wfreq_by_year(cby_d, ['resilience', 'sustainable', 'wellbeing'], rf_path)
#    get_all_rf(cby_d, rf_path, w=['resilience', 'sustainable', 'wellbeing'])
#
#    #plot relative frequencies of senses for a word for each year   
#    y_d = get_sc_d(sc_path)
#    for w in s_words:
#        get_all_rf(y_d, rfs_path, senses=True, w=w)
#    
#    #plot SNPMI and sense volatilities for each sense of each word in a list for a category
    with open(f'{i_path}/snpmi_y_d.pickle', 'rb') as f_name:
        snpmi_y_d = pickle.load(f_name)
    with open(f'{i_path}/snvol_d.pickle', 'rb') as f_name:
        snvol_d = pickle.load(f_name)
    
    scatter_all(snpmi_y_d, snvol_d, annotate=annotate)
    scatter_all_cat(snpmi_y_d, snvol_d, annotate=annotate)
    
#    #plot sense variation per word
#    plot_var(snpmi_y_d, snvol_d, words, ss_path)
#
#    #save senses to .txt file
#    see_senses(targets, c_path, d_path, o_path)    

if __name__ == '__main__':
    main()
