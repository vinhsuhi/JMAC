from os.path import join
import pandas as pd
import numpy as np
import os
from src.knowledgegraph import KnowledgeGraph
import numpy as np
from src.utils import get_language_list, get_graph
import string
import time
import logging




def _get_desc_input(kg_names, data_path, logger): 
    '''
    Revised!
    '''
    word_embed = 'wiki-news-300d-1M.vec'
    start = time.time()

    entities = []
    for kg_name in kg_names:
        file = open("{}/entity/{}.tsv".format(data_path, kg_name), 'r')
        for line in file:
            entities.append(line.strip())
        file.close()

    name_triples = []
    for i in range(len(entities)):
        name_triples.append((i, -1, entities[i]))

    names = pd.DataFrame(name_triples)
    names.iloc[:, 2] = names.iloc[:, 2].str.replace(r'[{}]+'.format(string.punctuation), '').str.split(' ')
    
    # load word embedding
    with open(word_embed, 'r') as f:
        w = f.readlines()
        w = pd.Series(w[1:])
    # w is a dataframe of embeddings of all words (loaded from pretrained embeddings)
    we = w.str.split(' ')
    word = we.apply(lambda x: x[0])
    w_em = we.apply(lambda x: x[1:])
    logger.info('concat word embeddings')
    word_em = np.stack(w_em.values, axis=0).astype(np.float)
    word_em = np.append(word_em, np.zeros([1, 300]), axis=0)
    logger.info('convert words to ids')
    w_in_desc = []
    for l in names.iloc[:, 2].values:
        w_in_desc += l
    # w_in_desc is the list of all words that only appear in entity names. 
    w_in_desc = pd.Series(list(set(w_in_desc)))

    # un_logged_words is all words that does not have any embeddings. 
    un_logged_words = w_in_desc[~w_in_desc.isin(word)] 
    un_logged_id = len(word)

    # all_word is a pd frame, contain all word in word as well as un_logged_words; all the un_logged_words have the same index
    all_word = pd.concat([pd.Series(word.index, word.values), pd.Series([un_logged_id, ] * len(un_logged_words), index=un_logged_words)])

    def lookup_and_padding(x):
        default_length = 4
        ids = list(all_word.loc[x].values) + [all_word.iloc[-1], ] * default_length
        return ids[:default_length]

    logger.info('look up desc embeddings')
    names.iloc[:, 2] = names.iloc[:, 2].apply(lookup_and_padding)

    # entity-desc-embedding dataframe
    e_desc_input = pd.DataFrame(np.repeat([[un_logged_id, ] * 4], len(entities), axis=0),
                                range(len(entities)))

    e_desc_input.iloc[names.iloc[:, 0].values] = np.stack(names.iloc[:, 2].values)

    logger.info('generating desc input costs time: {:.4f}s'.format(time.time() - start))
    name_embeds = word_em[e_desc_input.values]
    name_embeds = np.sum(name_embeds, axis=1)

    return name_embeds


class ParseData(object):
    '''
    Revised!
    '''
    def __init__(self, args, logger):
        self.data_path = args.data_path
        self.data_entity = self.data_path + "/entity/"
        self.data_kg = self.data_path + "/kg/"
        self.data_align_train = self.data_path + "/seed_train_pairs/"
        self.data_align_test = self.data_path + "/seed_test_pairs/"
        self.args = args

        self.target_kg = args.target_language
        self.kg_names = get_language_list(self.data_path, logger) # all kg names, sorted
        self.num_kgs = len(self.kg_names)

    def apply_noise(self, array, noise_pc):
        '''
        Revised!
        '''
        dim = array.shape[1]
        num_change_each = int(noise_pc * dim)
        if num_change_each == 0:
            return array

        mean = np.mean(array)
        std = np.std(array)

        for i in range(len(array)):
            to_change = np.random.randint(0, dim, num_change_each)
            value = np.random.normal(mean, std, num_change_each)
            array[i][to_change] = value 
        
        return array


    def load_data(self, noise_rate=0, logger=None):
        '''
        Revised!
        '''
        ent_name_path = '{}/ent_name_emb.npy'.format(self.data_path)
        if not os.path.exists(ent_name_path):
            logger.info('Compute entity name embedding...')
            entity_name_emb = _get_desc_input(self.kg_names, self.data_path)
            np.save(ent_name_path, entity_name_emb)
        else:
            logger.info('Loading embedding name from file...')
            entity_name_emb = np.load(ent_name_path)

        entity_name_emb = self.apply_noise(entity_name_emb, noise_rate)
        kg_object_dict, seeds_train, seeds_test = self.create_KG_objects_and_alignment() 
        self.num_relations = kg_object_dict[self.target_kg].num_relation * self.num_kgs 
        return kg_object_dict, seeds_train, seeds_test, entity_name_emb


    def load_all_to_all_seed_align_links(self):
        '''
        Revised!
        '''

        seeds_train = {}
        for f in os.listdir(self.data_align_train):  # e.g. 'el-en.tsv'
            lang1 = f[0:2]
            lang2 = f[3:5]
            links = pd.read_csv(join(self.data_align_train, f), sep='\t',header=None).values.astype(int)  # [N,2] ndarray
            seeds_train[(lang1, lang2)] = links # still be original index

        seeds_test = {}
        for f in os.listdir(self.data_align_test):  # e.g. 'el-en.tsv'
            lang1 = f[0:2]
            lang2 = f[3:5]
            links = pd.read_csv(join(self.data_align_test, f), sep='\t',header=None).values.astype(int)  # [N,2] ndarray
            seeds_test[(lang1, lang2)] = links # still be original index
        return seeds_train, seeds_test # dict of numpy array!!


    def create_KG_objects_and_alignment(self):
        '''
        Revised!
        '''
        entity_base = 0
        relation_base = 0
        kg_objects_dict = {} 

        for lang in self.kg_names:
            kg_train_data, kg_val_data, kg_test_data, entity_num, relation_num = self.load_kg_data(lang) 

            if lang == self.target_kg:
                is_supporter_kg = False
            else:
                is_supporter_kg = True

            kg_each = KnowledgeGraph(lang, kg_train_data, kg_val_data, kg_test_data, entity_num, relation_num, is_supporter_kg,
                                     entity_base, relation_base, self.args.device)

            entity_base += entity_num            
            relation_base += relation_num
            kg_each.upper_entity_base = entity_base
            kg_each.upper_relation_base = relation_base
            kg_objects_dict[lang] = kg_each

        self.num_entities = entity_base

        for lang in self.kg_names:
            if lang == self.target_kg:
                is_target_KG = True
            else:
                is_target_KG = False
            kg_lang = kg_objects_dict[lang]
            edge_index, edge_type =  get_graph(self.data_path, lang, is_target_KG)
            kg_lang.edge_index = edge_index # numpy array
            kg_lang.edge_type = edge_type

        seeds_train, seeds_test = self.load_all_to_all_seed_align_links() # None, links, None # not include base index yet!

        return kg_objects_dict, seeds_train, seeds_test


    def load_kg_data(self, language):
        '''
        Revised!
        '''
        train_df = pd.read_csv(join(self.data_kg, language + '-train.tsv'), sep='\t', header=None,names=['v1', 'relation', 'v2'])
        val_df = pd.read_csv(join(self.data_kg, language + '-val.tsv'), sep='\t', header=None,names=['v1', 'relation', 'v2'])
        test_df = pd.read_csv(join(self.data_kg, language + '-test.tsv'), sep='\t', header=None,names=['v1', 'relation', 'v2'])

        f = open(self.data_entity + language + '.tsv')
        lines = f.readlines()
        f.close()

        entity_num = len(lines)

        relation_list = [line.rstrip() for line in open(join(self.data_path, 'relations.txt'))]
        relation_num = len(relation_list) + 1

        triples_train = train_df.values.astype(np.int)
        triples_val = val_df.values.astype(np.int)
        triples_test = test_df.values.astype(np.int)

        return triples_train, triples_val, triples_test, entity_num, relation_num

