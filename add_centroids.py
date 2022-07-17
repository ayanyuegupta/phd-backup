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


a_s = (10, 15)
k_range = (2, 10)
data_root = '/media/gog/external2/corpora'
i_path = f'{data_root}/gov_corp/clusters_{k_range[0]}_{k_range[1]}'
words = [
        'resilience',
        'resilient',
        'sustainable',
        'sustainability',
        'wellbeing'
        ]
o_path = f'{i_path}/added_centroids_{a_s[0]}_{a_s[1]}'

SEED = 0
batch_size = 32
dropout_rate = 0.25
bert_dim = 768

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cluster_embeddings(data, kr, ID=None, dim_reduct=None, rs=SEED, lamb=10000, finetuned=False): 
    
        ks = range(kr[0], kr[1])
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


class ClusterRetriever():

    def __init__(self, tokenizer, model_name):

        self.tokenizer = tokenizer
        self.model = model_name
        self.model.eval()
        self.model.to(device)


    def get_batches(self, sentences, max_batch, test=False): 
        
        print('Batching...')
        # each item in these lists is a sentence
        all_data = [] # indexed tokens, or word IDs
        all_words = [] # tokenized_text, or original words
        all_masks = [] 
        all_users = []
        for sentence in tqdm(sentences):
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


    def load_centroids(self, vocab, path, rs=SEED): 
        
        centroids_folder = f'{path}/centroids' 
        centroids_d = {}
        for w in tqdm(sorted(vocab)):
            if f'{w}.npy' not in os.listdir(centroids_folder): continue
            centroids = np.load(f'{centroids_folder}/{w}.npy', allow_pickle=True)
            centroids_d[w] = centroids

        return centroids_d


    def batch_match_rc(self, cluster_d, centroids_d, data_dict):

        for tok in data_dict:
            rep_list = data_dict[tok]
            IDs = []
            reps = []
            for tup in rep_list:
                IDs.append(tup[0])
                reps.append(tup[1])
            reps = np.array(reps)
            centroids = centroids_d[tok]
            assert reps.shape[1] == centroids.shape[1]
            sims = cosine_similarity(reps, centroids) # IDs x n_centroids
            labels = np.argmax(sims, axis=1)
            for i, _ in enumerate(IDs):
                cluster_d[tok].append((reps[i], labels[i])) 
            

    def get_embeddings_and_match(self, batched_data, batched_words, batched_masks, batched_users, clusters_d, centroids_d): 
        
#        outfile = open(f'{s_path}/senses', 'w')
        vocab = set(centroids_d.keys())
        ret = []
        print("Getting embeddings for batched_data of length", len(batched_data))

        # variables for grouping wordpiece vectors
        prev_w = (None, None, None)
        ongoing_word = []
        ongoing_rep = []
        data_dict = defaultdict(list) # { word : [(user, rep)] }
        for b in tqdm(range(len(batched_data))):
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
                    if w not in vocab and '##' not in w and '##' not in next_w: continue 
                    # get vector 
                    hidden_layers = [] 
                    for layer_i in range(1, 5):
                        vec = encoded_layers[-layer_i][sent_i][token_i]
                        hidden_layers.append(vec)
                    # concatenate last four layers
                    vector = torch.cat((hidden_layers[0], hidden_layers[1], 
                                hidden_layers[2], hidden_layers[3]), 0)
                    vector = vector.cpu().numpy().reshape(1, -1)[0]
                    # piece together wordpiece vectors if necessary
                    if not w.startswith('##'):
                        if len(ongoing_word) == 0 and prev_w[0] is not None:
                            data_dict[prev_w[0]].append((prev_w[1], prev_w[2]))
                        elif prev_w[0] is not None: 
                            rep = np.array(ongoing_rep)
                            rep = np.mean(rep, axis=0).flatten()
                            tok = ''
                            for t in ongoing_word: 
                                if t.startswith('##'): t = t[2:]
                                tok += t
                            if tok in vocab: 
                                data_dict[tok].append((prev_w[1], rep))
                        ongoing_word = []
                        ongoing_rep = []
                    else: 
                        if len(ongoing_word) == 0 and prev_w[0] is not None: 
                            ongoing_word.append(prev_w[0])
                            ongoing_rep.append(prev_w[2])
                        ongoing_word.append(w)
                        ongoing_rep.append(vector)
                    prev_w = (w, users[sent_i], vector)
            if b % 10 == 0: 
                # do centroid matching in batches
#                self.batch_match(outfile, centroids_d, data_dict)
                self.batch_match_rc(clusters_d, centroids_d, data_dict)
                data_dict = defaultdict(list)
        # fencepost 
        if len(ongoing_word) == 0 and prev_w[0] is not None: 
            data_dict[prev_w[0]].append((prev_w[1], prev_w[2]))
        elif prev_w[0] is not None: 
            rep = np.array(ongoing_rep)
            rep = np.mean(rep, axis=0).flatten()
            tok = ''
            for t in ongoing_word: 
                if t.startswith('##'): t = t[2:]
                tok += t
            if tok in vocab: 
                data_dict[tok].append((prev_w[1], rep))
#        self.batch_match(outfile, centroids_d, data_dict)
        self.batch_match_rc(clusters_d, centroids_d, data_dict)
#        outfile.close()


def main():

    with open(f'{i_path}/train_samples.pickle', 'rb') as f_name:
        train_samples = pickle.load(f_name)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model_name = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model = ClusterRetriever(tokenizer, model_name)
    
    #get centroids and target samples
    centroids_d = model.load_centroids(words, i_path)
    train_samples = {w: train_samples[w] for w in words}
    
    #match target train samples to centroids to reconstruct target clusters
    clusters_d = {w: [] for w in words}
    with open(f'{i_path}/targets.txt', 'w') as f_name:
        f_name.write(f' '.join(words))
    for w in train_samples:
        sentences = train_samples[w]
        batched_data, batched_words, batched_masks, batched_users = model.get_batches(sentences, batch_size)
        #get embeddings of matched samples
        model.get_embeddings_and_match(batched_data, batched_words, batched_masks, batched_users, clusters_d, centroids_d)
    
    #run k-means on reconstructed target clusters to add more centroids
    if not os.path.exists(o_path):
        os.makedirs(o_path)
    for w in clusters_d:
        labels = list(set([tpl[-1] for tpl in clusters_d[w]]))
        kr = (
                int(a_s[0] / len(labels)),
                int(a_s[1] / len(labels))
                )
        centroids = []
        for label in labels:
            data = [tpl[0] for tpl in clusters_d[w] if tpl[1] == label]
            centroids.append(cluster_embeddings(data, kr, lamb=10000))
        centroids = np.concatenate(centroids)
        np.save(f'{o_path}/{w}.npy', centroids) 


if __name__ == '__main__':

    main()
