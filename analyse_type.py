import enchant
from scipy import stats
import scipy
from statsmodels.stats.outliers_influence import summary_table
from sklearn.preprocessing import StandardScaler
from statsmodels.regression import linear_model
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import statsmodels.api as sm
import statsmodels.formula.api as smf
import utils.misc as misc
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


root = '/home/gog/projects/gov_lv'
i_path = f'{root}/t_measures_output_sample'
o_path = f'{root}/type_analysis'
b_path = f'{o_path}/bar_charts'
wls_path = f'{o_path}/wl_scatters'
wh_path_tnpmi = f'{o_path}/wl_histograms_tnpmi'
wh_path_vol = f'{o_path}/wl_histograms_vol'

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

target_cats = None


####


def scatter(d_x, d_y, o_path, f_name, exclude=None, targets=None, exclude_labels=None, title=None, size=(6, 6), colour=[0, 0, 0], highlight=[0, 1, 0], font_size=20, x_label=None, y_label=None, alpha=0.1):
    
    font = {'size': font_size}
    plt.rc('font', size=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('xtick', labelsize=font_size-2)
    plt.rc('ytick', labelsize=font_size-2)
    plt.rc('axes', titlesize=font_size)
    a_size = font_size - 5

    if exclude is None:
        labels = [cat for cat in d_x]
    else:
        labels = [cat for cat in d_x if cat not in exclude]
    plt.figure(figsize=size)
    x_tpls = [(cat, d_x[cat]) for cat in labels]
    y_tpls = [(cat, d_y[cat]) for cat in labels]
    x = [tpl[-1] for tpl in x_tpls]
    y = [tpl[-1] for tpl in y_tpls]
    rho, p_val = stats.spearmanr(x, y)
    plt.scatter(x, y, c=np.array([colour]), alpha=alpha)
    if exclude_labels is not None:
        x2 = [d_x[cat] for cat in labels if cat not in exclude_labels]
        y2 = [d_y[cat] for cat in labels if cat not in exclude_labels]
        plt.scatter(x2, y2, c=np.array([highlight]), alpha=1)
    if targets is not None:
        pairs = [(x_tpls[i][0], x_tpls[i][1], y_tpls[i][1]) for i, _ in enumerate(x_tpls)] 
        top_pairs = []
        for index in targets:
            if index <= len(pairs) - 1:
                top_pairs.append(sorted(pairs, key=lambda x: x[1], reverse=True)[index])
                top_pairs.append(sorted(pairs, key=lambda x: x[2], reverse=True)[index])
        x3 = [tpl[1] for tpl in top_pairs]
        y3 = [tpl[2] for tpl in top_pairs]
        top_labels = [tpl[0] for tpl in top_pairs]
        plt.scatter(x3, y3, c=np.array([highlight]), alpha=1)
    if title is not None:
        string = fr'$\rho$={round(rho, 3)}, p={round(p_val, 3)}'
        plt.title(f'{title}, Spearman\'s {string}')    
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    for i, label in enumerate(labels):
        if exclude_labels is None and targets is None:
            plt.annotate(label, (x[i], y[i]), fontsize=a_size)
        if exclude_labels is not None:
            if label not in exclude_labels:
                plt.annotate(label, (x[i], y[i]), fontsize=a_size)
        if targets is not None:
            if label in top_labels:
                plt.annotate(label, (x[i], y[i]), fontsize=a_size)
    plt.savefig(f'{o_path}/{f_name}.png', bbox_inches='tight')


def scatter_words(d_x, d_y, wls_path, target_words=None, target_cats=None, targets=None, alpha=0.1, font_size=20, size=(6, 6), t_cat=None, x_label=None, y_label=None, title=None):
   
    eng_d = enchant.Dict('en_GB')    
    if target_cats is None:
        categories = list(set([cat for year in d_x for cat in d_x[year]]))
    else:
        categories = target_cats
    if target_words is not None:
        words = '_'.join(target_words)
    elif targets is not None:
        words = '_'.join([str(v) for v in targets])
    else:
        words = 'None'
    path = f'{wls_path}/{words}'
    if not os.path.exists(path):
        os.makedirs(path)
    for year in tqdm(d_x):
        for cat in categories:
            x = d_x[year][cat]
            y = d_y[year][cat]
            x = {w: score for w, score in x.items() \
                    if not any(char.isdigit() for char in w) \
                    and eng_d.check(w) \
                    and len(w) > 2}
            y = {w: score for w, score in y.items() \
                    if not any(char.isdigit() for char in w) \
                    and eng_d.check(w) \
                    and len(w) > 2}
            if target_words is not None:
                    exclude_labels = [w for w in x if w not in target_words]
            else:
                exclude_labels = None 
            scatter(x, y, path, f'{year}_{cat}', exclude_labels=exclude_labels, targets=targets, title=f'{year} {cat}', x_label='$T*$', y_label='$V*$', colour=[0.5, 0.5, 0.5], highlight=[0, 0, 1], alpha=alpha, font_size=font_size)


def spearman_scatter(X, y, path, x_label=None, y_label=None, title=None, font_size=20, tick_size=18, alpha=1):
    
    rho, p_val = stats.spearmanr(X, y)
#    plt.rc('font', size=font_size)
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(X, y, c=np.array([[0, 0, 1]]), alpha=alpha)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=font_size)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=font_size)
    title_string =  fr'$\rho$={round(rho, 3)}, p={round(p_val, 3)}'
    if title is None:
        plt.title(f'Spearman\'s {title_string}')
    else:
        plt.title(f'{title}, Spearman\'s {title_string}')

    plt.savefig(path, bbox_inches='tight')
    fig.clf()


def bar_chart(avg_d, o_path, t_bar=None, size=(6, 6), font_size=20, tick_size=18, width=0.8, y_label=None, colour=[0, 0, 1]):

    font = {'size': font_size}
    plt.rc('font', size=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('xtick', labelsize=tick_size)
    plt.rc('ytick', labelsize=tick_size)
    plt.rc('axes', titlesize=font_size)
    plt.figure(figsize=size)

    x = np.arange(len(avg_d))
    y = [avg_d[cat] for cat in avg_d]
    plt.bar(x, y, width=width, color=colour)
    
    if t_bar is None:
        labels = [cat.split('_')[0] for cat in avg_d]
    else:
        labels = []
        for cat in avg_d:
            if cat != t_bar:
                labels.append('')
            else:
                labels.append(cat.split('_')[0])
    plt.xticks(x, labels)
    if y_label is not None:
        plt.ylabel(y_label)
        plt.savefig(f'{o_path}/{y_label}.png', bbox_inches='tight')
    else:
        plt.savefig(f'{b_path}/bar_chart.png', bbox_inches='tight')


def average_wscores(y_d):
    
    categories = list(set([cat for y in y_d for cat in y_d[y]]))
    avg_d = {cat: {} for cat in categories}
    for cat in tqdm(categories):
        vocab = list(set([w for y in y_d for w in y_d[y][cat]]))
        for w in vocab:
            score = 0
            n = 0
            for y in y_d:
                if w in y_d[y][cat]:
                    n += 1
                    score += y_d[y][cat][w]
            avg_d[cat][w] = score / n

    return avg_d


def bar_chart_words(target_words, avg_d, b_path, t_bar=None, size=(12, 6), width=0.8, font_size=20, y_label=None):
    
    font = {'size': font_size}
    plt.rc('font', size=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('xtick', labelsize=font_size-2)
    plt.rc('ytick', labelsize=font_size-2)
    plt.rc('axes', titlesize=font_size)

    labels = [cat for cat in avg_d if all(w in avg_d[cat] for w in target_words)]
    x = np.arange(len(labels))
    n_bars = len(target_words)
    b_width = width / n_bars
    fig = plt.figure(figsize=size)
    for i, w in enumerate(target_words):
        offset = (i - n_bars / 2) * b_width + b_width / 2
        y = [avg_d[cat][w] for cat in labels]
        plt.bar(x + offset, y, b_width, label=w)
        
    f_name = '_'.join(target_words)
    plt.legend(loc='upper left', frameon=False, fontsize=font_size-5)
    if t_bar is None:
        labels = [cat.split('_')[0] for cat in labels]
    else:
        lst = []
        for cat in labels:
            if cat != t_bar:
                lst.append('')
            else:
                lst.append(cat.split('_')[0])
        labels = lst
    plt.xticks(x, labels)
    if y_label is not None:
        plt.ylabel(y_label)
        plt.savefig(f'{b_path}/{f_name}_{y_label}.png', bbox_inches='tight') 
    else:
        plt.savefig(f'{b_path}/{f_name}.png', bbox_inches='tight')


def truncate(number, digits) -> float:
    
    stepper = 10.0 ** digits
    
    return math.trunc(stepper * number) / stepper


def measures_by_year(d, o_path, f_name, targets=['leg'], colour=[0, 0, 1], size=(6, 6), font_size=20, alpha=0.2, y_label=None, legend=True):
   
    font = {'size': font_size}
    plt.rc('font', size=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('xtick', labelsize=font_size-2)
    plt.rc('ytick', labelsize=font_size-2)
    plt.rc('axes', titlesize=font_size)
    
    ax = plt.figure(figsize=size).gca()
    categories = list(set([cat for year in d for cat in d[year]]))
    x = sorted([year for year in d])
    for cat in categories:
        y = [d[year][cat] for year in  x]
        if targets is not None:
            if cat in targets:
                plt.plot(x, y, c=np.array(colour), marker='o', label=cat)
            else:
                 plt.plot(x, y, linestyle='-', marker='o', c=np.array([0, 0, 0]), alpha=alpha)
        else:
            plt.plot(x, y, c=np.array(colour), marker='o', label=cat)

    ax.xaxis.get_major_locator().set_params(integer=True)
    if legend:
        plt.legend(loc='upper left', frameon=False) 
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig(f'{o_path}/{f_name}.png', bbox_inches='tight')

    return ax, categories


def scatter_all_cats(tnpmi_y_d, nv_d, path, categories=None, size=(6, 6), x_label=None, y_label=None, alpha=0.01):
    
    if categories is None:
        categories = list(set([cat for y in tnpmi_y_d for cat in tnpmi_y_d[y]]))
    colours = misc.random_colours(len(categories))
    for year in tnpmi_y_d:
        fig = plt.figure(figsize=size)
        for i, cat in enumerate(categories):
            vocab = set([w for w in tnpmi_y_d[year][cat]]).intersection([w for w in nv_d[year][cat]])
            X = [tnpmi_y_d[year][cat][w] for w in vocab]
            y = [nv_d[year][cat][w] for w in vocab]
            plt.scatter(X, y, label=cat, c=np.array([colours[i]]), alpha=alpha)
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)
        leg = plt.legend(loc='upper left', frameon=False, prop={'size': 8}) 
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        plt.savefig(f'{path}/{year}.png', bbox_inches='tight')


def top_scores(tnpmi_y_d, nv_d, o_path, min_years=20, alpha=0.2, x_label='Top $T*$', y_label='Top $V*$'):

    categories = list(set([cat for y in tnpmi_y_d for cat in tnpmi_y_d[y]]))
    ts_d = {'top_nv': [], 'top_tnpmi': [], 'categories': []}
    for cat in categories:
        vocab = list(set([w for y in tnpmi_y_d for w in tnpmi_y_d[y][cat]]))
        X = []
        y = []
        for w in vocab:
            specs = [tnpmi_y_d[y][cat][w] for y in tnpmi_y_d if w in tnpmi_y_d[y][cat]]
            vols = [nv_d[y][cat][w] for y in nv_d if w in tnpmi_y_d[y][cat]]
            if len(specs) > min_years:
                max_spec = max(specs)
                max_vol = max(vols)
                X.append(max_spec)
                y.append(max_vol)
                ts_d['top_nv'].append(max_vol)
                ts_d['top_tnpmi'].append(max_spec)
                ts_d['categories'].append(cat)
        spearman_scatter(X, y, f'{o_path}/{cat}.png', alpha=alpha, x_label=x_label, y_label=y_label)
        
    return ts_d


#https://stats.stackexchange.com/questions/410934/how-to-interpret-the-results-of-multiple-regression-when-two-dummy-coded-predict#:~:text=The%20general%20interpretation%20of%20the,holding%20other%20independent%20variables%20constant.
#https://stats.stackexchange.com/questions/471124/how-to-qualitatively-understand-interaction-terms
def top_scores_lri(ts_d, o_path, ci_alpha=0.01, colours=[[0.5, 0.5, 0.5], [0, 0, 1]], font_size=20, size=(6, 6), scatter=True, x_label='Top $T*$', y_label='Top $V*$'):
    
    for i, cat in enumerate(ts_d['categories']):
        if cat != 'leg':
            ts_d['categories'][i] = 'dep'
    df = pd.DataFrame.from_dict(ts_d)
    dummies = pd.get_dummies(df['categories'])
    df = pd.concat([df, dummies], axis=1)
    df.drop(['categories', 'dep'], inplace=True, axis=1)
    df['top_tnpmi*leg'] = df['top_tnpmi'] * df['leg'] 
    X = df.drop('top_nv', axis=1)
    X = sm.add_constant(X)
    y = df['top_nv']
    model = linear_model.OLS(y, X).fit()
    print(model.summary())
    print(model.conf_int(0.01))
    print(model.pvalues)
    #https://gist.github.com/josef-pkt/1417e0473c2a87e14d76b425657342f5
    predicted = model.predict()
    pred = model.get_prediction()
    pred_df = pred.summary_frame(alpha=ci_alpha)
    df['predictions'] = model.predict()
    df['pi_lower'] = pred_df['obs_ci_lower']
    df['pi_upper'] = pred_df['obs_ci_upper']
    dep_df = df.loc[df['leg'] == 0]
    leg_df = df.loc[df['leg'] == 1]

    plt.figure(figsize=size)
    x = [dep_df['top_tnpmi'].min(), dep_df['top_tnpmi'].max()]
    y = [dep_df['predictions'].min(), dep_df['predictions'].max()]
    pi_l = [dep_df['pi_lower'].min(), dep_df['pi_lower'].max()]
    pi_u = [dep_df['pi_upper'].min(), dep_df['pi_upper'].max()]
    plt.plot(x, y, label='Department', color=colours[0])
    plt.plot(x, pi_l, '--', color=colours[0], alpha=0.5)
    plt.plot(x, pi_u, '--', color=colours[0], alpha=0.5)
    x = [leg_df['top_tnpmi'].min(), leg_df['top_tnpmi'].max()]
    y = [leg_df['predictions'].min(), leg_df['predictions'].max()]
    pi_l = [leg_df['pi_lower'].min(), leg_df['pi_lower'].max()]
    pi_u = [leg_df['pi_upper'].min(), leg_df['pi_upper'].max()]
    plt.plot(x, y, label='Legislation', color=colours[1])
    plt.plot(x, pi_l, '--', color=colours[1], alpha=0.5)
    plt.plot(x, pi_u, '--', color=colours[1], alpha=0.5)
    plt.legend(loc='upper left', frameon=False, fontsize=font_size) 
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.xticks(fontsize=font_size-5)
    plt.yticks(fontsize=font_size-5)
    plt.savefig(f'{o_path}/lr_test.png', bbox_inches='tight') 
    if scatter:
        plt.figure(figsize=size)
        plt.scatter(dep_df['top_tnpmi'], dep_df['top_nv'], label='Department', color=colours[0], alpha=0.2)
        plt.scatter(leg_df['top_tnpmi'], leg_df['top_nv'], label='Legislation', color=colours[1], alpha=0.2)
        leg = plt.legend(loc='upper left', frameon=False, fontsize=font_size) 
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        plt.xlabel(x_label, fontsize=font_size)
        plt.ylabel(y_label, fontsize=font_size)
        plt.savefig(f'{o_path}/sv_all_scatter.png', bbox_inches='tight')


def main():
    
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    if not os.path.exists(wls_path):
        os.makedirs(wls_path)
    if not os.path.exists(wh_path_tnpmi):
        os.makedirs(wh_path_tnpmi)
    if not os.path.exists(wh_path_vol):
        os.makedirs(wh_path_vol)
    if not os.path.exists(b_path):
        os.makedirs(b_path)
    corr_path = f'{o_path}/correlations'
    if not os.path.exists(corr_path):
        os.makedirs(corr_path)

    #plot N(c), D(c)
    with open(f'{i_path}/avg_dis.pickle', 'rb') as f_name:
        avg_dis = pickle.load(f_name)
    with open(f'{i_path}/avg_dyn.pickle', 'rb') as f_name:
        avg_dyn = pickle.load(f_name)
    scatter(avg_dis, avg_dyn, o_path, 'dis_dyn', x_label='$N(c)$', y_label='$D(c)$')   
    bar_chart(avg_dis, o_path, t_bar='leg', y_label='$N(c)$')
    bar_chart(avg_dyn, o_path, t_bar='leg', y_label='$D(c)$')

    #plot N(c_t), D(c_t)    
    with open(f'{i_path}/dis_d.pickle', 'rb') as f_name:
        dis_d = pickle.load(f_name)
    with open(f'{i_path}/dyn_d.pickle', 'rb') as f_name:
        dyn_d = pickle.load(f_name) 
    dis_plot = measures_by_year(dis_d, o_path, 'dis_by_year', y_label='$N(c_t)$')
    dyn_plot = measures_by_year(dyn_d, o_path, 'dyn_by_year', y_label='$D(c_t)$')
    
    #plot TNPMI, volatility for each word
    with open(f'{i_path}/tnpmi_y_d.pickle', 'rb') as f_name:
        tnpmi_y_d = pickle.load(f_name)
    with open(f'{i_path}/nv_d.pickle', 'rb') as f_name:
        nv_d = pickle.load(f_name)
    scatter_words(tnpmi_y_d, nv_d, wls_path, target_cats=target_cats, targets=[10, 50, 3500])
#    scatter_words(tnpmi_y_d, nv_d, wls_path, target_words=target_words)
   
#    #plot scores for target words   
    avg_d = average_wscores(tnpmi_y_d) 
    bar_chart_words(target_words, avg_d, b_path, t_bar='leg', y_label='Average $T*$')

    #correlation between specificity and volatility
    #category level
#    X = [avg_dis[cat] for cat in avg_dis]
#    y = [avg_dyn[cat] for cat in avg_dyn]
#    spearman_scatter(X, y, f'{corr_path}/dis_dyn_corr.png', x_label='Distinctiveness', y_label='Dynamicity')
#
    #word level
#    path = f'{corr_path}/all'
#    if not os.path.exists(path):
#        os.makedirs(path)
#    for year in tnpmi_y_d:
#        for cat in tnpmi_y_d[year]:
#            vocab = set([w for w in tnpmi_y_d[year][cat]]).intersection([w for w in nv_d[year][cat]])
#            X = [tnpmi_y_d[year][cat][w] for w in vocab]
#            y = [nv_d[year][cat][w] for w in vocab]
#            spearman_scatter(X, y, f'{path}/{year}_{cat}.png', title=f'{year} {cat}', alpha=0.2)
    
    path = f'{corr_path}/all_years_max_score'
    if not os.path.exists(path):
        os.makedirs(path)
    ts_d = top_scores(tnpmi_y_d, nv_d, path)

    #linear regression with interaction effect
    top_scores_lri(ts_d, o_path)   


if __name__ == '__main__':
    main()


