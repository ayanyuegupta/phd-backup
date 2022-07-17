from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from collections import OrderedDict
from sklearn.cluster import AgglomerativeClustering
import utils.misc as m
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from pygam import LinearGAM, s
import pandas as pd
import plotnine as p9
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import os
import ruptures as rpt


words = [
#        'summary',
#        'government',
        'resilience',
        'resilient',
        'sustainable',
        'sustainability',
        'wellbeing'
        ]

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

i_path = f'{os.getcwd()}/t_measures_output_complete'
o_path = f'{os.getcwd()}/diffusion_analysis'
colours = m.random_colours(13)

####


def plot_gam(X, y, save_path, word, size=(6, 3), colour_label=None, x_label=None, y_label=None, custom_xticks=None, rotation=0, ax=None, conf_int=True, alphas=(1, 1)):
     
    gam = LinearGAM(s(0)).fit(X, y)
#    print(gam.summary())
    if ax is None:
        ax = plt.figure(figsize=size).gca()
    if colour_label is None:
        ax.scatter(X, y, color=np.array([[0, 0, 1]]), s=10, alpha=alphas[0])
        ax.plot(X, gam.predict(X), color=np.array([1, 0, 1]), alpha=alphas[1])
    else:
        ax.scatter(X, y, label=colour_label[1], color=np.array([colour_label[0]]), s=10, alpha=alphas[0])
        ax.plot(X, gam.predict(X), label=colour_label[1], color=np.array(colour_label[0]), alpha=alphas[1])
    if conf_int:
        ax.plot(X, gam.confidence_intervals(X), color=np.array([1, 0, 1]), ls='--')
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.title(word)
    plt.xticks(rotation=rotation, ha='right')
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if custom_xticks is not None:
        xt = ax.get_xticks()[1:-1]
        ax.set_xticks(xt)
        custom_xticks = [custom_xticks[int(i)] for i in xt]
        ax.set_xticklabels(custom_xticks)
    
    if ax is None:
        plt.savefig(f'{save_path}/{word}.png', bbox_inches='tight')


def get_rf(d, word, categories=None):
    
    years = sorted([y for y in d])
    if categories is None:
        categories = set([cat for y in d for cat in d[y]])
    v_d = {y: 0 for y in years}
    catv_d = {y:{cat: 0 for cat in categories} for y in years}
    for y in years:
        total = 0
        count = 0
        for cat in categories:
            cat_total = sum([d[y][cat][w] for w in d[y][cat]])
            total += cat_total   
            if word in d[y][cat]:
                count += d[y][cat][word]
                catv_d[y][cat] = d[y][cat][word] / cat_total
        v_d[y] = count / total

    return v_d, catv_d


def plot_cat_rfs(catv_d, o_path, word, colours=None, categories=None, size=(6, 6)):

    if categories is None:
        categories = list(set([cat for y in catv_d for cat in catv_d[y]]))
    ax = plt.figure(figsize=size).gca()
    if colours is None:
        colours = m.random_colours(len(categories))
    for i, cat in enumerate(categories):
        data = [(y, catv_d[y][cat]) for y in catv_d if catv_d[y][cat] is not None]
        X = [y for y, _ in data]
        y = [val for _, val in data]
        plot_gam(X, y, o_path, word, ax=ax, colour_label=(colours[i], cat), conf_int=False, alphas=(0.5, 1)) 
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    leg = plt.legend(by_label.values(), by_label.keys(), loc='upper left', prop={'size': 8}, framealpha=0, ncol=1)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh.set_sizes([50])

    plt.savefig(f'{o_path}/{word}.png', bbox_inches='tight') 


def cpd(v_d, o_path, word, model='rbf', y_label=None, pen=0.25, size=(6, 3)):
    
    points = np.array([v_d[y] for y in v_d])
    algo = rpt.Pelt(model=model).fit(points)
    result = algo.predict(pen=pen)
    fig, ax = rpt.display(points, result, figsize=size)
    plt.plot(points, marker='o')
    ax = ax[0]
    x = fig.gca().get_xticks()[1:-1]
    years = [y for y in v_d if y % len(x) == 0]
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    plt.title(word)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig(f'{o_path}/{word}.png', bbox_inches='tight')


def spec_var(tnpmi_y_d, word):

    var_d = {}
    categories = list(set([cat for y in tnpmi_y_d for cat in tnpmi_y_d[y]]))
    for y in tnpmi_y_d:
        scores = []
        for cat in categories:
            if word in tnpmi_y_d[y][cat]:
                scores.append(tnpmi_y_d[y][cat][word])
        if len(scores) > 1:
            var_d[y] = np.var(scores)
        else:
            var_d[y] = None
    
    return var_d
  

def plot_lr(X, y, save_path, word, size=(6, 3), x_label=None, y_label=None, alpha=1):

    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2).fit()
    
    #https://www.statsmodels.org/dev/_modules/statsmodels/stats/outliers_influence.html#OLSInfluence.summary_table
    st, data, ss2 = summary_table(est, alpha=0.05)
    fittedvalues = data[:,2]
    predict_mean_se  = data[:,3]
    predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T
    predict_ci_low, predict_ci_upp = data[:,6:8].T

#    print(f'\n{words}')
#    print(est.summary())
    p_val = f'{est.pvalues[1]:.3e}'
    r2 = round(est.rsquared, 3)
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    plt.plot(X, fittedvalues, color=np.array([1, 0, 1]))
    plt.plot(X, predict_ci_low, color=np.array([1, 0, 1]), ls='--')
    plt.plot(X, predict_ci_upp, color=np.array([1, 0, 1]), ls='--')
    plt.scatter(X, y, color=np.array([[0, 0, 1]]), s=10, alpha=alpha)

    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    plt.title(f'{word}, $R^2$={r2}, p={p_val}')
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig(f'{save_path}/{word}.png', bbox_inches='tight')

 
def plot_cat_specs(tnpmi_y_d, word, path, size=(6, 3), alphas=(1, 0.3), m_sizes=(20, 10), loc='lower left', l_size=7.5, clusters=None, target_labels=(1, 2), y_label=None, title=None, colours=None, lw=2.5):
    
    years = [y for y in tnpmi_y_d]
    categories = list(set([cat for y in tnpmi_y_d for cat in tnpmi_y_d[y]]))
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111)
    for i, cat in enumerate(categories):
        data = []
        for y in tnpmi_y_d:
            if word in tnpmi_y_d[y][cat]:
                score = tnpmi_y_d[y][cat][word]
                data.append((y, score))
        data = sorted(data)
        if clusters is not None:
            years = sorted([year for year in clusters if clusters[year] is not None])
            X = np.array([tpl[0] for tpl in data if tpl[0] in years])
            y = np.array([tpl[1] for tpl in data if tpl[0] in years])
            labels = [clusters[year][cat] for year in X if cat in clusters[year]]
            for j, _ in enumerate(labels):
                if j + 1 < len(labels):
                    X_pair = np.array([X[j], X[j + 1]])
                    y_pair = np.array([y[j], y[j + 1]])
                    label_pair = [labels[j], labels[j + 1]]
                    if label_pair == [target_labels[0], target_labels[0]]:
                        if colours is not None:
                            plt.plot(X_pair, y_pair, color=colours[i], lw=lw, alpha=alphas[0])
                        else:
                            plt.plot(X_pair, y_pair, lw=lw, alpha=alphas[0])
                    elif label_pair == [target_labels[1], target_labels[1]] or \
                            label_pair == [target_labels[1], target_labels[0]] or \
                            label_pair == [target_labels[0], target_labels[1]]:
                        if colours is not None:
                            plt.plot(X_pair, y_pair, color=colours[i], lw=lw, alpha=alphas[1])
                        else:
                            plt.plot(X_pair, y_pair, lw=lw, alpha=alphas[1])
                    else:
                        plt.plot(X_pair, y_pair, color=[0.5, 0.5, 0.5], lw=lw, alpha=alphas[1] / 2)
                if labels[j] == target_labels[0]:
                    plt.scatter(X[j], y[j], s=m_sizes[0], label=cat, color=colours[i], alpha=alphas[0])
                elif labels[j] == target_labels[1]:
                    plt.scatter(X[j], y[j], s=m_sizes[1], label=cat, color=colours[i], alpha=alphas[1])

        else:
            X = np.array([tpl[0] for tpl in data])
            y = np.array([tpl[1] for tpl in data])
            if colours is not None:
                plt.plot(X, y, label=cat, color=colours[i])
            else:
                plt.plot(X, y, label=cat)
    if title is not None:
        plt.title(title)
    else:
        title = word
    if y_label is not None:
        plt.ylabel(y_label)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    leg = plt.legend(by_label.values(), by_label.keys(), loc=loc, prop={'size':l_size}, framealpha=0, ncol=3)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
        lh.set_sizes([50])
    ax.xaxis.get_major_locator().set_params(integer=True)

    plt.savefig(f'{path}/{title}.png', bbox_inches='tight')


def hcluster_scores(tnpmi_y_d, w, n_clusters=2, target_labels=(1, 2)):
    
    c_d = {y: {} for y in tnpmi_y_d}
    s_d = {y: {} for y in tnpmi_y_d}
    for y in tnpmi_y_d:
        d = tnpmi_y_d[y]
        scores = sorted([(cat, [d[cat][w]]) for cat in d if w in d[cat]], key=lambda x: x[1])
        if len(scores) >= n_clusters:
            X = np.array([tpl[1] for tpl in scores])
            cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
            cluster.fit_predict(X)
#            if cluster.labels_[-1] != 1:
            t_label1 = cluster.labels_[-1]
            t_label2 = [label for label in cluster.labels_ if label != t_label1][-1]
            new_labels = np.zeros(len(cluster.labels_), dtype=np.int64)
            for i, _ in enumerate(cluster.labels_):
                if cluster.labels_[i] == t_label1:
                    new_labels[i] = target_labels[0]
                if cluster.labels_[i] == t_label2:
                    new_labels[i] = target_labels[1]
            cluster.labels_ = new_labels
            s_d[y]['clusters'] = cluster.labels_
            s_d[y]['ranks'] = [tpl[0] for tpl in scores]
            s_d[y]['scores'] = [tpl[1][0] for tpl in scores]
            for i, _ in enumerate(scores):
                c_d[y][scores[i][0]] = cluster.labels_[i]
        else:
            c_d[y] = None 
    
    return c_d, s_d


def cluster_table(s_d, categories):

    years = sorted([y for y in s_d if s_d[y] != {}])
    table = {cat: [] for cat in categories}
    for y in years:
        missing = [cat for cat in categories if cat not in s_d[y]['ranks']]
        for i, c in enumerate(s_d[y]['clusters']):
            if c == 1 or c == 2:
                table[s_d[y]['ranks'][i]].append(c)
            else:
                table[s_d[y]['ranks'][i]].append(None)
        for cat in missing:
            table[cat].append(None)
    table = [(tpl, len([i for i in tpl[1] if i == 1])) for tpl in list(table.items())]
    table = [tpl[0] for tpl in sorted(table, key=lambda x: x[1], reverse=True)]
    table = [(tpl, len([i for i in tpl[1] if i is None])) for tpl in table]
    table = dict([tpl[0] for tpl in sorted(table, key=lambda x: x[1])])
    df = pd.DataFrame.from_dict(table, orient='index', columns=years)

    return df


def cluster_heatmap(df, word, path, colours=None):

    fig = plt.figure()
    if colours is None:
        colours = ((1, 0, 1, 1), (1, 0, 1, 0.5))
    cmap = LinearSegmentedColormap.from_list('Custom', colours, len(colours))
    ax = sns.heatmap(df, annot=True, cmap=cmap, cbar=False)
    colourbar = ax.collections[0].colorbar
    
    plt.title(word)
    plt.savefig(f'{path}/{word}_heatmap.png', bbox_inches='tight')

            
def main():
    
    colours = m.random_colours(len(categories))
    for word in words:

        #relative frequencies
        if not os.path.exists(o_path):
            os.makedirs(o_path)
        rf_path = f'{o_path}/change_point_detection/rel_freqs'
        if not os.path.exists(rf_path):
            os.makedirs(rf_path)
        with open(f'{i_path}/cby_d.pickle', 'rb') as f_name:
           cby_d = pickle.load(f_name)    
        rf, cat_rfs = get_rf(cby_d, word)
               
        #gam
        gam_path1 = f'{o_path}/gam/rel_freqs'
        path = f'{o_path}/gam/rel_freqs/all_cats'
        if not os.path.exists(gam_path1):
            os.makedirs(gam_path1)
        if not os.path.exists(path):
            os.makedirs(path)
        y = np.array([rf[y] for y in rf])
        X = np.array([y for y in rf])
        plot_gam(X, y, gam_path1, word, y_label='Relative Frequency')
        plot_cat_rfs(cat_rfs, path, word, colours=colours, categories=categories)
 
        #change point detection
#        cpd(rf, rf_path, words, y_label='Relative Frequency')
       
        
        #specificity variances
        tnpmi_path = f'{o_path}/gam/tnpmi_var'
        if not os.path.exists(tnpmi_path):
            os.makedirs(tnpmi_path)
        with open(f'{i_path}/tnpmi_y_d.pickle', 'rb') as f_name:
            tnpmi_y_d = pickle.load(f_name)
#        var_d = spec_var(tnpmi_y_d, word)
#        data = sorted([(y, var_d[y]) for y in var_d if var_d[y] is not None])
#        X = np.array([tpl[0] for tpl in data])
#        y = np.array([tpl[1] for tpl in data])
#        plot_gam(X, y, tnpmi_path, word, y_label='Specificity Variance')
        
        #plot specificities over time and cluster yearly specificities
        cat_specs_path = f'{o_path}/line_plots/clustered_cat_specs'
        if not os.path.exists(cat_specs_path):
            os.makedirs(cat_specs_path)
        n_clusters = 4
        c_d, s_d = hcluster_scores(tnpmi_y_d, word, n_clusters=n_clusters)
        plot_cat_specs(tnpmi_y_d, word, cat_specs_path, clusters=c_d, colours=colours, y_label='Specificity', title=f'{word}, n_clusters={n_clusters}')                    
        
        hm_path = f'{o_path}/heat_maps'
        if not os.path.exists(hm_path):
            os.makedirs(hm_path)
        df = cluster_table(s_d, categories)
        cluster_heatmap(df, word, hm_path)         
        

if __name__ == '__main__':
    main()



