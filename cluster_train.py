from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import numpy as np
from collections import Counter
import os
from tqdm import tqdm
import pickle
import utils.misc as mi
import measures as m
import random


k_range = (2, 10)
yearly_sample_size = 150

root = '/home/gog/projects/gov_lv'
data_root = '/media/gog/external2/corpora/gov_corp'
data_path = f'{data_root}/sample'
i_path = f'{data_path}/merged.pickle'
o_path1 = f'{root}/t_measures_output_{data_path.split("/")[-1]}'

name = '_'.join([str(k_range[0]), str(k_range[1])])
o_path2 = f'{data_root}/clusters_{name}'

SEED = 0
batch_size = 32
dropout_rate = 0.25
bert_dim = 768

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


####


def get_vocabulary(counts, all_counts, prop=0.5, num_cat=6, min_freq=500, s_words=None):
    
    for cat in counts:
        counts[cat] = {w: counts[cat][w] for w in counts[cat] \
                if not any(char.isdigit() for char in w) \
                and len(w) > 2}
        size = len(counts[cat])
        counts[cat] = dict(Counter(counts[cat]).most_common(int(prop * size)))
        counts[cat] = [w for w in counts[cat]]
    
    all_counts = [w for w in all_counts if all_counts[w] >= min_freq]
    vocab = []
    count = 0
    for w in tqdm(all_counts, desc='Getting vocabulary...'):
        for cat in counts:
            if w in counts[cat]:
                count += 1
        if count >= num_cat:
            vocab.append(w)
        count = 0

    if s_words is not None:
        vocab = [w for w in vocab if w in s_words]
        if len(vocab) != len(s_words):
            print('Some of the selected words are not in the vocabulary:\n')
            print(f'{vocab}, {s_words}')
            quit()
    
    return vocab
                    

def get_sequences(data, vocab, max_len=100):
    
    print('Getting sequences...')
    v_sequences = {w: [] for w in vocab}
    for cat in data:
        for i, tpl in tqdm(enumerate(data[cat]), desc=cat, total=len(data[cat])):
            words = tpl[1].split()
            length = len(words)
            year = tpl[0].split('/')[-1].split('-')[0]
            ID = f'{cat}_{year}_{i}'
            for w in words:
                if w in vocab and length <= max_len:
                    v_sequences[w].append((ID, tpl[1], tpl[0]))

    for w in v_sequences:
        v_sequences[w] = list(set(v_sequences[w]))

    return v_sequences


def get_all_seq(data_path, o_path2, vocab, year_range=(2000, 2020)):
    
    years = [i for i in range(year_range[0], year_range[1] + 1, 1)]
    for year in tqdm(years):
        with open(f'{data_path}/{year}.pickle', 'rb') as f_name:
            data = pickle.load(f_name)
        o_path = f'{o_path2}/{year}_senses'
        if not os.path.exists(o_path):
            os.makedirs(o_path)
        v_sequences = mi.get_items('v_sequences', o_path, get_sequences, data, vocab)


def get_samples(o_path2, vocab, yearly_sample_size, min_sample_size=500):
    
    #v_sequences = {w: (ID, tpl)}
    print(f'Sampling... (Yearly sample size = {yearly_sample_size})')
    train_samples = {w: [] for w in vocab}
    paths = [name.path for name in os.scandir(o_path2) if 'senses' in name.path]
    for path in tqdm(paths):
        with open(f'{path}/v_sequences.pickle', 'rb') as f_name:
            v_sequences = pickle.load(f_name)
        for w in v_sequences:
            if len(v_sequences[w]) > yearly_sample_size:
                train_samples[w] += random.sample(v_sequences[w], yearly_sample_size)
            else:
                train_samples[w] += v_sequences[w]
    train_samples = {w: samples for w, samples in train_samples.items() if len(samples) > min_sample_size}

    return train_samples


class Clusterer():

    def __init__(self, tokenizer, model_name):
        
        self.tokenizer = tokenizer
        self.model = model_name
        self.model.eval()
        self.model.to(device)
             
    
    def get_batches(self, sentences, max_batch):

        # each item in these lists is a sentence
        all_data = [] # indexed tokens, or word IDs
        all_words = [] # tokenized_text, or original words
        all_masks = [] 
        all_users = []
        for sentence in sentences: 
            marked_text = sentence[1] 
            tokenized_text_all = self.tokenizer.tokenize(marked_text)
            for i in range(0, len(tokenized_text_all), 510):
                tokenized_text = tokenized_text_all[i:i+510]
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                indexed_tokens = self.tokenizer.build_inputs_with_special_tokens(indexed_tokens)
                all_data.append(indexed_tokens)
                all_words.append(tokenized_text) 
                all_masks.append(list(np.ones(len(indexed_tokens))))
                all_users.append(sentence[0])
    
        lengths = np.array([len(l) for l in all_data])
        ordering = np.argsort(lengths)
        # each item in these lists is a sentence
        ordered_data = [None for i in range(len(all_data))]
        ordered_words = [None for i in range(len(all_data))]
        ordered_masks = [None for i in range(len(all_data))]
        ordered_users = [None for i in range(len(all_data))]
        for i, ind in enumerate(ordering): 
            ordered_data[i] = all_data[ind]
            ordered_words[i] = all_words[ind]
            ordered_masks[i] = all_masks[ind]
            ordered_users[i] = all_users[ind]
        # each item in these lists is a batch of sentences
        batched_data = []
        batched_words = []
        batched_masks = []
        batched_users = []
        i = 0
        current_batch = max_batch
        while i < len(ordered_data): 
            batch_data = ordered_data[i:i+current_batch]
            batch_words = ordered_words[i:i+current_batch]
            batch_mask = ordered_masks[i:i+current_batch]
            batch_users = ordered_users[i:i+current_batch]
    
            max_len = max([len(sent) for sent in batch_data])
            for j in range(len(batch_data)): 
                blen = len(batch_data[j])
                for k in range(blen, max_len): 
                    batch_data[j].append(0)
                    batch_mask[j].append(0)
            batched_data.append(torch.LongTensor(batch_data))
            batched_words.append(batch_words)
            batched_masks.append(torch.FloatTensor(batch_mask))
            batched_users.append(batch_users)
            i += current_batch
            if max_len > 100: 
                current_batch = 12
            if max_len > 200: 
                current_batch = 6
        return batched_data, batched_words, batched_masks, batched_users
    
    
    def get_embeddings(self, batched_data, batched_words, batched_masks, batched_users, word): 
        
        word = word.lower()
        ret = [] 
        do_wordpiece = True
        for b in range(len(batched_data)):
            # each item in these lists/tensors is a sentence
            tokens_tensor = batched_data[b].to(device)
            atten_tensor = batched_masks[b].to(device)
            words = batched_words[b]
            users = batched_users[b]
            with torch.no_grad():
                _, _, encoded_layers = self.model(tokens_tensor, attention_mask=atten_tensor, token_type_ids=None)
            for sent_i in range(len(words)): 
                for token_i in range(len(words[sent_i])):
                    if batched_masks[b][sent_i][token_i] == 0: continue
                    w = words[sent_i][token_i]
                    next_w = ''
                    if (token_i + 1) < len(words[sent_i]): 
                        next_w = words[sent_i][token_i+1]
                    if w != word and '##' not in w and '##' not in next_w: continue
                    if w == word: 
                        do_wordpiece = False
                    hidden_layers = [] 
                    for layer_i in range(1, 5):
                        vec = encoded_layers[-layer_i][sent_i][token_i]
                        hidden_layers.append(vec)
                    # concatenate last four layers
                    rep = torch.cat((hidden_layers[0], hidden_layers[1], 
                                hidden_layers[2], hidden_layers[3]), 0) 
                    ret.append((w, rep.cpu().numpy().reshape(1, -1)[0]))
        
        return (ret, do_wordpiece)
    
    
    def group_wordpiece(self, embeddings, word, do_wordpiece): 
        '''
        - puts together wordpiece vectors
        - only piece them together if embeddings does not 
        contain the vocab word of interest 
        - filters vectors so we only have vectors for the word of interest
        '''
        word = word.lower()
        data = []
        prev_w = (None, None)
        ongoing_word = []
        ongoing_rep = []
        for i, tup in enumerate(embeddings):
            if not do_wordpiece: 
                if tup[0] == word: 
                    data.append(tup[1])
            else: 
                if tup[0].startswith('##'): 
                    if not prev_w[0].startswith('##'): 
                        ongoing_word.append(prev_w[0])
                        ongoing_rep.append(prev_w[1])
                    ongoing_word.append(tup[0][2:])
                    ongoing_rep.append(tup[1])
                else:
                    if ''.join(ongoing_word) == word: 
                        data.append(np.mean(ongoing_rep, axis=0).flatten())
                    ongoing_word = []
                    ongoing_rep = []
            prev_w = tup
        if ''.join(ongoing_word) == word: 
            data.append(np.mean(ongoing_rep, axis=0).flatten())
        np.random.shuffle(data)

#        return np.array(data)[:500]
        return np.array(data)


    def cluster_embeddings(self, data, ID=None, dim_reduct=None, rs=SEED, lamb=10000, finetuned=False): 
    
        ks = range(k_range[0], k_range[1])
        centroids = {} 
        rss = np.zeros(len(ks))
        for i, k in tqdm(enumerate(ks), total=(len(ks))):
            km = KMeans(k, random_state=rs)
            km.fit(data)
            rss[i] = km.inertia_
            centroids[k] = km.cluster_centers_
        crits = []
        for i in range(len(ks)): 
            k = ks[i] 
            crit = rss[i] + lamb*k
            crits.append(crit)
        best_k = np.argmin(crits)
        
        return centroids[ks[best_k]] 


def main():
    
    if not os.path.exists(o_path1):
        os.makedirs(o_path1)
    if not os.path.exists(o_path2):
        os.makedirs(o_path2)

    #get data
    counts, all_counts = mi.get_items(['counts', 'all_counts'], o_path1, m.get_counts, i_path, regex_pattern='[^A-Za-z0-9\'.\s]+')
    vocab = mi.get_items('vocab', o_path2, get_vocabulary, counts, all_counts)
    get_all_seq(data_path, o_path2, vocab)
    train_samples = mi.get_items('train_samples', o_path2,  get_samples, o_path2, vocab, yearly_sample_size) 

    #load model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model_name = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model = Clusterer(tokenizer, model_name)
    
    #get centroids
    for word in train_samples:
        sentences = train_samples[word]
        batched_data, batched_words, batched_masks, batched_users = model.get_batches(sentences, batch_size)
        embeddings, do_wordpiece = model.get_embeddings(batched_data, batched_words, batched_masks, batched_users, word)
        data = model.group_wordpiece(embeddings, word, do_wordpiece)
        centroids = model.cluster_embeddings(data, lamb=10000)
        c_path = f'{o_path2}/centroids'
        if not os.path.exists(c_path):
            os.makedirs(c_path)
        np.save(f'{c_path}/{word}.npy', centroids)


if __name__ == '__main__':
    main()
            
        
    


