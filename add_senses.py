from cluster_match import Matcher
import argparse
import torch
from transformers import BertTokenizer, BertModel
import pickle
import os


years = [2000 + i for i in range(21)]
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

SEED = 0
batch_size = 32
dropout_rate = 0.25
bert_dim = 768

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        if a_s is None:
            o_path = f'{seq_path}/{w}.txt'
        else:
            o_path = f'{seq_path}/{w}_{a_s[0]}_{a_s[1]}.txt'
        with open(o_path, 'w') as f_name:
            for s in d[w]:
                f_name.write(f'\n\n')
                for seq in d[w][s]:
                    f_name.write(f'\n~{s}: {seq[1]}, {seq[0]}')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-a_s', '-additional_senses', required=True)
    args = parser.parse_args()
    a_s = [int(v) for v in args.a_s.split('-')]

    #load model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model_name = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model = Matcher(tokenizer, model_name)

    as_path = f'{c_path}/added_centroids_{a_s[0]}_{a_s[1]}'
    vocab = [f_name.split('.')[0] for f_name in os.listdir(as_path)]
    centroids_d = model.load_centroids(vocab, c_path, added_centroids=a_s)
    vocab = set(centroids_d.keys())
    for y in years:
        s_path = f'{c_path}/{y}_senses'
        with open(f'{s_path}/v_sequences.pickle', 'rb') as f_name:
            v_sequences = pickle.load(f_name)
        sentences = []
        for w in vocab:
            sentences += v_sequences[w]
        batched_data, batched_words, batched_masks, batched_users = model.get_batches(sentences, batch_size)    
        
        #DO NOT SET added_centroids TO None (KEYWORD ARG added_centroids DEFAULTS TO None) 
        #OR SENSES FROM INITIAL WSI RUN WILL BE DELETED AND REPLACED BY ADDITIONAL SENSES
        model.get_embeddings_and_match(batched_data, batched_words, batched_masks, batched_users, centroids_d, s_path, added_centroids=a_s)

    see_senses(targets, c_path, data_path, sa_path, a_s=a_s)


if __name__ == '__main__':

    main()
