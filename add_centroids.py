from cluster_train import Clusterer
from cluster_match import Matcher 
from sklearn.cluster import KMeans
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import sklearn
from tqdm import tqdm
import torch
from torch import nn
import os
import pickle
from transformers import BertTokenizer, BertModel
import argparse


k_range = (2, 10)
data_root = '/media/gog/external2/corpora'
words = [
        'resilience',
        'resilient',
        'sustainable',
        'sustainability',
        'wellbeing'
        ]
i_path = f'{data_root}/gov_corp/clusters_{k_range[0]}_{k_range[1]}'
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

    with open(f'{i_path}/train_samples.pickle', 'rb') as f_name:
        train_samples = pickle.load(f_name)
    
    #get target samples
    train_samples = {w: train_samples[w] for w in words}
    
    #get additional centroids
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model_name = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model = Clusterer(tokenizer, model_name)
    for word in train_samples:
        sentences = train_samples[word]
        batched_data, batched_words, batched_masks, batched_users = model.get_batches(sentences, batch_size)
        embeddings, do_wordpiece = model.get_embeddings(batched_data, batched_words, batched_masks, batched_users, word)
        data = model.group_wordpiece(embeddings, word, do_wordpiece)
        centroids = model.cluster_embeddings(data, lamb=10000, a_s=a_s)
        c_path = f'{i_path}/added_centroids_{a_s[0]}_{a_s[1]}'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        np.save(f'{c_path}/{word}.npy', centroids)


if __name__ == '__main__':
    main()
