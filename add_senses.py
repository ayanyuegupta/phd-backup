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
    targets = [f_name.split('.')[0] for f_name in os.listdir(as_path)]
    centroids_d = model.load_centroids(targets, c_path, added_centroids=a_s)
    for y in years:
        s_path = f'{c_path}/{y}_senses'
        with open(f'{s_path}/v_sequences.pickle', 'rb') as f_name:
            v_sequences = pickle.load(f_name)
        vocab = [w for w in v_sequences]
        sentences = []
        for w in vocab:
            sentences += v_sequences[w]
        sentences = list(set(sentences))
        sentences = [(tpl[0], tpl[1].lower(), tpl[1]) for tpl in sentences if any(w in tpl[1].lower() for w in targets)] 
        batched_data, batched_words, batched_masks, batched_users = model.get_batches(sentences, batch_size)    
        
        #DO NOT SET added_centroids TO None (KEYWORD ARG added_centroids DEFAULTS TO None) 
        #OR SENSES FROM INITIAL WSI RUN WILL BE DELETED AND REPLACED BY ADDITIONAL SENSES
        model.get_embeddings_and_match(batched_data, batched_words, batched_masks, batched_users, centroids_d, s_path, added_centroids=a_s)

#    d = see_senses(targets, c_path, data_path, sa_path, a_s=a_s)
#    sense_keywords(d)

if __name__ == '__main__':
    main()
