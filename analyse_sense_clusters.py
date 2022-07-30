import itertools
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import pickle
import os


k_range = (2, 10)
targets = [
        'resilience',
        'resilient',
        'sustainable',
        'sustainability',
        'wellbeing'
        ]
root = '/home/gog/projects/gov_lv'
data_root = '/media/gog/external2/corpora/gov_corp'
sa_path = f'{root}/sense_analysis'
data_path = f'{data_root}/sample'
so_path = f'{root}/s_measures_output_{k_range[0]}_{k_range[1]}'
sc_path = f'{so_path}/sense_counts'
c_path = f'{data_root}/clusters_{k_range[0]}_{k_range[1]}'


def see_senses(targets, c_path, d_path, o_path, a_s=None):
    
    d = {w: [] for w in targets}
    years = [y.split('.')[0] for y in os.listdir(d_path) if any(char.isdigit() for char in y)]
    for y in years:
        with open(f'{d_path}/{y}.pickle', 'rb') as f_name:
            data = pickle.load(f_name)
        if a_s is None:
            sense_path = f'{c_path}/{y}_senses/senses'
        else:
            sense_path = f'{c_path}/{y}_senses/added_senses_{a_s[0]}_{a_s[1]}'
        with open(sense_path, 'r') as f_name:
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
        senses = sorted([tpl[0] for tpl in d[w]])
        d2[w] = {sense: [] for sense in senses}
        for tpl in d[w]:
            sense = tpl[0]
            d2[w][sense].append(tpl[1])
    d = d2
    seq_path = f'{o_path}/sense_seq'
    if not os.path.exists(seq_path):
        os.makedirs(seq_path)
    for w in d:
        if a_s is None:
            o_path = f'{seq_path}/{w}.txt'
        else:
            o_path = f'{seq_path}/{w}_{a_s[0]}_{a_s[1]}.txt'
        with open(o_path, 'w') as f_name:
            for s in d[w]:
                f_name.write(f'\n\n')
                for seq in d[w][s]:
                    f_name.write(f'\n~{s}: {seq[1]}, {seq[0]}')
    
    return d


def extract_topn_from_vector(feature_names, sorted_items, topn):
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


#https://github.com/matejMartinc/scalable_semantic_shift/blob/a105c8409db0996c99f0df11d40c35017eb3337c/interpretation.py#L85
def sense_keywords(d, o_path, re_pattern='[^a-zA-Z. ]', max_df=0.8, mf_prop=1, topn=10, ngram_range=(1, 2), add_stopwords=['yes', 'no', 'yesno'], stopdocs=['Control_Room']):
    
    regex = re.compile(re_pattern)
    sp = spacy.load('en_core_web_sm')
    stopwords = sp.Defaults.stop_words
    if add_stopwords is not None:
        stopwords = list(stopwords) + add_stopwords
    kw_d = {}
    for w in d:
        senses = sorted([s for s in d[w]])
        clusters = []
        for s in senses:
            doc = ''.join([tpl[1] for tpl in d[w][s] if not any(item in tpl[0] for item in stopdocs)]).lower()
            doc = regex.sub('', doc).split()
            clusters.append(doc)
        vocab = list(itertools.chain.from_iterable(clusters))
        v_size = len(set([w for w in vocab if w not in stopwords]))
        clusters = [' '.join(c) for c in clusters]
        tfidf_transformer = TfidfVectorizer(smooth_idf=True, use_idf=True, ngram_range=ngram_range, max_df=max_df, max_features=int(mf_prop * v_size), stop_words=stopwords)
        tfidf_transformer.fit(clusters)
        feature_names = tfidf_transformer.get_feature_names()
        kw_d[w] = {}
        for i, cluster in enumerate(clusters):
            tf_idf_vector = tfidf_transformer.transform([cluster])
            tuples = zip(tf_idf_vector.tocoo().col, tf_idf_vector.tocoo().data)
            sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
            keywords = extract_topn_from_vector(feature_names, sorted_items, topn*5)
            keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
            keywords = [x[0] for x in keywords]
            #filter unigrams that appear in ngrams and remove duplicates
            all_ngrams = " ".join([kw for kw in keywords if len(kw.split()) > 1])
            already_in = set()
            filtered_keywords = []
            for kw in keywords:
                if len(kw.split()) < ngram_range[1] and kw in all_ngrams:
                    continue
                elif len(set(kw.split())) < len(kw.split()):
                    continue
                else:
                    if kw not in already_in and kw != w:
                        filtered_keywords.append(kw)
                        already_in.add(kw)
            kw_d[w][i] = filtered_keywords[:topn]

    kw_dfs = {w: pd.DataFrame.from_dict(kw_d[w], orient='index') for w in kw_d}
    kw_df = pd.concat(kw_dfs)
    print(kw_dfs)
    o_path = f'{o_path}/sense_keywords'
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    kw_df.to_csv(f'{o_path}/keywords_ngram_range={ngram_range[0]}_{ngram_range[1]}.csv')

    return kw_df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-a_s', '-additional_senses')
    args = parser.parse_args()
    if args.a_s is not None:
        a_s = [int(v) for v in args.a_s.split('-')]
    else:
        a_s = args.a_s

    d = see_senses(targets, c_path, data_path, sa_path, a_s=a_s)
    kw_df = sense_keywords(d, sa_path)

if __name__ == '__main__':
    main()

