from random import sample
import time
import numpy as np
import pandas as pd 
import string
from tqdm import tqdm
import torch 
import os

from modules.finding.evaluation import valid, test, early_stop
from modules.helper.helper import *
from modules.load.data_loader import *

from models.basic_model import BasicModel
from models.jmac_model import JMAC_MODEL
from modules.utils.util import get_neg
from collections import defaultdict as ddict



"""
JMAC for OpenEA dataset!
"""
class Trainer(BasicModel):
    def __init__(self):
        super().__init__()
        self.prox_map_loss = []
        self.trans_map_loss = []
        self.prox_acc = []
        self.trans_acc = []
        self.entropy = -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def completion_data_processing(self, triple_list):
        '''
        Revised!
        Prepare a dataloader for the completion-component encoder

        Parameters:
        triple_list: list of KG triples

        Returns:
        -------
        data_loader: dataloader object
        '''
        
        triples = []
        triple_dict = ddict(set)
        self.logger.info("Preparing triples for completion training")
        for triple in triple_list:
            head, rel, tail = triple[0], triple[1], triple[2]
            triple_dict[(head, rel)].add(tail)
            triple_dict[(tail , rel + self.num_rel)].add(head) # adding inverse edges
            triples.append([head, rel, tail])
            triples.append([tail, rel + self.num_rel, head]) # adding inverse edges
            
        triple_dict = {k: list(v) for k, v in triple_dict.items()}
        dataset_class = TrainDataset(triples, self.num_ent, self.all_source_index, self.all_target_index, triple_dict, 25)
        data_loader = self.get_data_loader(dataset_class, self.args.completion_batch_size)
        return data_loader


    def get_data_loader(self, dataset_class, batch_size, shuffle=True):
        '''
        Revised!
        '''
        return  DataLoader(
                dataset_class,
                batch_size      = batch_size,
                shuffle         = shuffle,
                num_workers     = max(0, 12)
            )    


    def alignment_data_processing(self, triple_list):
        """
        1. Load pretrained embeddings of entity names
        2. Add inverse edges

        Returns:
        -------

        edge_index: Torch LongTensor, size = (2, num_edges * 2)
            [source entities], [target entities]
        edge_type: Torch LongTensor, size = (num_edge * 2,)
            [edge_types]
        """
        edge_index = [[ele[0], ele[2]] for ele in triple_list]
        edge_type = [ele[1] for ele in triple_list]

        # adding inverse relations
        edge_index_inversed = [[tail, head] for head, tail in edge_index]
        edge_type_inversed = [rel + self.num_rel for rel in edge_type]
        edge_index += edge_index_inversed
        edge_type += edge_type_inversed
        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type


    def init(self):
        """
        Preprocess data for the two encoders
        Define the two channel models
        Define the two channel optimizers
        """

        ###################### KG information #######################
        self.entities = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set
        self.relations = self.kgs.kg1.relations_set | self.kgs.kg2.relations_set
        self.source_relations = sorted(list(self.kgs.kg1.relations_set)) # 0 2 4 6 8 --> 358
        self.target_relations = sorted(list(self.kgs.kg2.relations_set)) # 1 3 5 7 9 --> 331
        self.num_rel, self.num_ent = len(self.relations), len(self.entities)
        # list of inverse relations
        self.source_rel_inverse = [ele + self.num_rel for ele in self.source_relations]
        self.target_rel_inverse = [ele + self.num_rel for ele in self.target_relations]
        self.all_src_rel_index = self.source_relations + self.source_rel_inverse
        self.all_trg_rel_index = self.target_relations + self.target_rel_inverse
        self.all_source_index = sorted(list(self.kgs.kg1.entities_set)) # 2 4 6 8 ..
        self.all_target_index = sorted(list(self.kgs.kg2.entities_set)) # 1 3 5 7 ..

        self.source_id2idx = {ele: i for i, ele in enumerate(self.all_source_index)} # 2:0 4:1
        self.target_id2idx = {ele: i for i, ele in enumerate(self.all_target_index)} # 1:0 3:1 

        
        self.train_links = np.array(self.kgs.train_links) # train_seeds; Numpy array
        self.triple_list = self.kgs.kg1.relation_triples_list + self.kgs.kg2.relation_triples_list
        ent_info_att = self.load_ent_name_emb()
        # Models and Optimizers        
        self.jmac_model = JMAC_MODEL(self.num_ent, self.num_rel, ent_info_att, args=self.args).to(self.device)
        self.alignment_optimizer = torch.optim.Adam(self.jmac_model.parameters(), \
                                                    lr=self.args.alignment_learning_rate)
        self.completion_optimizer = torch.optim.Adam(self.jmac_model.parameters(), \
                                                    lr=self.args.completion_learning_rate)


    def train_completion_step(self, i, edge_index, edge_type):
        neg_size = self.args.num_negative
        start = time.time()
        triple_losses = []
        flag = 0
        length = 0
        for triple, neg in self.completion_data_loader:
            self.completion_optimizer.zero_grad()
            triple = triple.to(self.device)
            sub, rel, obj = triple[:, 0], triple[:, 1], triple[:, 2]
            if flag == 0:
                length = len(sub)
                flag = 1
            elif len(sub) != length:
                continue
            neg = neg.view(-1)
            neg = torch.LongTensor(neg).to(self.device)
            sub = sub.repeat(neg_size + 1)
            rel = rel.repeat(neg_size + 1)
            tail = torch.cat((obj, neg))
            data = {"batch_h": sub, "batch_r": rel, "batch_t": tail}
            
            loss = self.jmac_model.completion_loss(data, edge_index, edge_type, self.feeddict)
            
            if loss == 0:
                continue
            loss.backward()
            self.completion_optimizer.step()
            triple_losses.append(loss.item())            
        
        self.logger.info("Epoch: {}, Transition loss : {:.4f}, cost time: {:.4f}s".format(i, np.mean(triple_losses), time.time() - start))
        self.trans_map_loss.append(np.mean(triple_losses))
        
        

    def train_alignment_step(self, i, edge_index, edge_type):
        start = time.time()
        self.alignment_optimizer.zero_grad()
        batch_loss = self.jmac_model.alignment_loss(self.feeddict, edge_index, edge_type) 
        batch_loss.backward()
        self.alignment_optimizer.step()
        self.logger.info('Epoch: {}, alignment loss: {:.4f}, cost time: {:.4f}s'.format(i, batch_loss.item(),
                                                                                        time.time() - start))
        self.prox_map_loss.append(batch_loss.item())



    def training(self):
        """
        Train the jmac model!
        """
        stop = False
        edge_index, edge_type = self.alignment_data_processing(self.triple_list)

        for i in range(1, self.args.max_epoch + 1, self.args.eval_freq):

            for j in range(self.args.eval_freq): 
                step = i + j
                self.jmac_model.train()
                
                if step == 1 or step % 3 == 1:
                    output, _ = self.jmac_model.get_emb(edge_index, edge_type, pyt=True)
                    entropy, simi, _ = self.compute_alignment_quality(output, self.kgs.valid_entities1 + self.kgs.test_entities1, self.kgs.valid_entities2 + self.kgs.test_entities2)
                    
                    additional_pairs = []
                    if self.entropy == -1:
                        self.entropy = entropy
                    else:
                        sample_percent = (self.entropy - entropy) / self.entropy * self.args.pair_sample_weight
                        if sample_percent < 0:
                            sample_percent = 0
                            self.entropy = entropy
                        try:
                            num_pairs = int(sample_percent * len(self.kgs.valid_entities1 + self.kgs.test_entities1))
                        except:
                            num_pairs = 0
                        
                        if num_pairs > 0:
                            max_values = simi.max(dim=1)[0]
                            rows_index = max_values.multinomial(num_pairs, replacement=False)
                            rows = simi[rows_index]
                            cols_index = rows.max(dim=1)[1].tolist()
                            
                            src_nodes = np.array(self.kgs.valid_entities1 + self.kgs.test_entities1)
                            trg_nodes = np.array(self.kgs.valid_entities2 + self.kgs.test_entities2)
                            src_selected_nodes = src_nodes[rows_index]
                            trg_selected_nodes = trg_nodes[cols_index]
                            
                            # Update links here
                            additional_pairs = [[src_selected_nodes[lindex], trg_selected_nodes[lindex]] for lindex in range(len(src_selected_nodes))]
                            additional_pairs = np.array(additional_pairs)

                    if len(additional_pairs):
                        new_train_links = np.concatenate((self.train_links, additional_pairs), axis=0)
                    else:
                        new_train_links = self.train_links
                    
                    neg2_left = get_neg(new_train_links[:, 1], output, self.args.num_negative)
                    neg_right = get_neg(new_train_links[:, 0], output, self.args.num_negative)
                    
                    train_num = len(new_train_links)
                    neg_num = self.args.num_negative
                    
                    pos = np.ones((train_num, neg_num)) * (new_train_links[:, 0].reshape((train_num, 1))) # 
                    neg_left = pos.reshape((train_num * neg_num,))
                    pos = np.ones((train_num, neg_num)) * (new_train_links[:, 1].reshape((train_num, 1)))
                    neg2_right = pos.reshape((train_num * neg_num,)) # np x nn
                                        
                    self.feeddict = {"neg_left": neg_left,
                                        "neg_right": neg_right,
                                        "neg2_left": neg2_left,
                                        "neg2_right": neg2_right,
                                        "links": new_train_links}
                    
                    # add temporal edges
                    links_dict = {ele[0]: ele[1] for ele in new_train_links}
                    new_triples = self.transfer_knowledge(self.triple_list, links_dict)
                    self.completion_data_loader = self.completion_data_processing(new_triples)
                    edge_index, edge_type = self.alignment_data_processing(new_triples)
                
        
                self.train_completion_step(step, edge_index, edge_type)

                self.train_alignment_step(step, edge_index, edge_type)  
                
                if step % self.args.eval_freq == 0:
                    self.jmac_model.eval()
                    align_emb, _ = self.jmac_model.get_emb(edge_index, edge_type)

                    self.logger.info("Alignment acc:")
                    flag = self.valid_(self.args.stop_metric, align_emb, in_list=self.prox_acc)

                    self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag, self.logger)
                    if self.early_stop:
                        stop = True 
                        break
            if stop:
                break
        self.jmac_model.eval()
        return edge_index, edge_type
        
    
    def compute_alignment_quality(self, embedding, list1, list2):
        '''
        Revised!
        '''
        SCALE_CONSTANT=20
        embeds1 = embedding[list1]
        embeds2 = embedding[list2]
        
        simi = torch.mm(embeds1, embeds2.t())

        softmax_simi = torch.softmax(simi * SCALE_CONSTANT, dim=1) # align kg1 to kg2
        entropy1 = - torch.log(softmax_simi) * softmax_simi
        entropy1 = torch.mean(torch.sum(entropy1, dim=1))
        
        softmax_simi2 = torch.softmax(simi.t() * SCALE_CONSTANT, dim=1) # align kg2 to kg1
        entropy2 = - torch.log(softmax_simi2) * softmax_simi2
        entropy2 = torch.mean(torch.sum(entropy2, dim=1))

        entropy = entropy1 + entropy2
        return entropy, softmax_simi, softmax_simi2
        


    def test(self, embedding):
        embeds1 = np.array([embedding[e] for e in self.kgs.test_entities1])
        embeds2 = np.array([embedding[e] for e in self.kgs.test_entities2])
        test(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
             metric=self.args.eval_metric, normalize=self.args.eval_norm, csls_k=self.args.csls, accurate=True, logger=self.logger)
        

    def valid_(self, stop_metric, embedding, part=1, in_list=[]):
        if len(self.kgs.valid_entities1) == 0:
            valid_entities1 = self.kgs.test_entities1
            valid_entities2 = self.kgs.test_entities2
        else:
            valid_entities1 = self.kgs.valid_entities1 
            valid_entities2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        embeds1 = np.array([embedding[e] for e in valid_entities1])
        embeds2 = np.array([embedding[e] for e in valid_entities2])
        hits1_12, mrr_12 = valid(embeds1, embeds2, None, self.args.top_k, self.args.test_threads_num,
                                 metric=self.args.eval_metric, logger=self.logger)
        in_list.append(hits1_12)
        if stop_metric == 'hits1':
            return hits1_12
        return mrr_12


    def run(self):
        """
        Run the model!
        """
        t = time.time()
        edge_index, edge_type = self.training()
        self.logger.info("Training finish")
        self.logger.info("Training ends. Total time = {:.3f} s.".format(time.time() - t))
        align_emb, pos_emb = self.jmac_model.get_emb(edge_index, edge_type)
        self.logger.info("Final align acc:")
        self.test(align_emb)



    def load_ent_name_emb(self): 
        """
        Load entity name embeddings; 

        Returns
        -------
        init_emb: Torch FloatTensor
            Initialized embedding of all entities for the alignment-based channel
        """
        self.word_embed = self.args.word_embed # '../../datasets/wiki-news-300d-1M.vec'
        name = "{}_ent_emb.npy".format(self.args.training_data)
        if self.args.no_attr_info:
            name = "{}_ent_emb_no_att.npy".format(self.args.training_data)
        elif self.args.no_name_info:
            name = "{}_ent_emb_no_name.npy".format(self.args.training_data)
            return torch.FloatTensor(np.zeros((30000, 300)) + 0.03).to(self.device)
        if not os.path.exists(name):
            _, _, self.local_name_vectors = self._get_desc_input()
            np.save(name, self.local_name_vectors)
        else:
            self.logger.info("Loading pretrained entity name embedding...")
        self.local_name_vectors = np.load(name)
        init_emb = torch.FloatTensor(self.local_name_vectors).to(self.device)
        return init_emb


    def transfer_knowledge(self, triple_list, links):
        """
        links: dict links[src] = trg
        """
        inverse_links = {v:k for k,v in links.items()}

        additional_triples = []

        for triple in tqdm(triple_list):
            head, rel, tail = triple[0], triple[1], triple[2]
            if links.get(head) and links.get(tail):
                additional_triples.append((links.get(head), rel, links.get(tail)))
            elif inverse_links.get(head) and inverse_links.get(tail):
                additional_triples.append((inverse_links.get(head), rel, inverse_links.get(tail)))

        self.logger.info("Number of transfered facts: {}".format(len(additional_triples)))
        return triple_list + additional_triples


    def _get_local_name_by_only_name_triple(self):
        if 'D_Y' in self.args.training_data:
            return self._get_no_name()
        if 'D_W' in self.args.training_data:
            return self._get_no_name()

        id_ent_dict = {} # ent_id2name
        for e, e_id in self.kgs.kg1.entities_id_dict.items():
            id_ent_dict[e_id] = e
        for e, e_id in self.kgs.kg2.entities_id_dict.items():
            id_ent_dict[e_id] = e

        local_name_dict = {}
        ents = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set # ids
        for e in ents:
            local_name_dict[e] = id_ent_dict[e].split('/')[-1].replace('_', ' ')

        name_triples = list()
        for e, n in local_name_dict.items():
            name_triples.append((e, -1, n))
        return name_triples

    def _get_no_name(self):
        local_name_dict = {}
        ents = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set # ids
        for e in ents:
            local_name_dict[e] = 'none'

        name_triples = list()
        for e, n in local_name_dict.items():
            name_triples.append((e, -1, n))
        return name_triples



    def _get_local_name_by_name_triple(self): 
        if 'D_Y' in self.args.training_data:
            name_attribute_list = {'skos:prefLabel', 'http://dbpedia.org/ontology/birthName'}
        elif 'D_W' in self.args.training_data:
            name_attribute_list = {'http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476'}
        else:
            name_attribute_list = {}

        # alignON two set of ids of triples 
        local_triples = self.kgs.kg1.local_attribute_triples_set | self.kgs.kg2.local_attribute_triples_set
        if len(local_triples) == 0:
            return self._get_local_name_by_only_name_triple()

        triples = list()
        for h, a, v in local_triples:
            v = v.strip('"')
            if v.endswith('"@eng'):
                v = v.rstrip('"@eng')
            triples.append((h, a, v))
        id_ent_dict = {} # ent_id2name
        for e, e_id in self.kgs.kg1.entities_id_dict.items():
            id_ent_dict[e_id] = e
        for e, e_id in self.kgs.kg2.entities_id_dict.items():
            id_ent_dict[e_id] = e

        ############################## name_ids = set() #######################3
        name_ids = set() 
        for a, a_id in self.kgs.kg1.attributes_id_dict.items():
            if a in name_attribute_list:
                name_ids.add(a_id)
        for a, a_id in self.kgs.kg2.attributes_id_dict.items():
            if a in name_attribute_list:
                name_ids.add(a_id)

        for a, a_id in self.kgs.kg1.attributes_id_dict.items():
            if a_id in name_ids:
                self.logger.info(a)
        for a, a_id in self.kgs.kg2.attributes_id_dict.items():
            if a_id in name_ids:
                self.logger.info(a)
        
        ############################## name_ids = set() #######################3

        local_name_dict = {} # local name_dict
        ents = self.kgs.kg1.entities_set | self.kgs.kg2.entities_set # ids
        for (e, a, v) in triples: # this loop does nothing
            if a in name_ids:
                local_name_dict[e] = v
        for e in ents:
            if e not in local_name_dict:
                local_name_dict[e] = id_ent_dict[e].split('/')[-1].replace('_', ' ')

        # now, local_name_dict is a dict of A[entity_id] = preprocessed_entity_name

        name_triples = list()
        for e, n in local_name_dict.items():
            name_triples.append((e, -1, n))
        return name_triples # is a set of (entity_id, -1, preprocessed_entity_name)


    def _get_desc_input(self): 
        # desc graph settings
        start = time.time()
        model = self

        # name_triples is a set of (entity_id, -1, preprocessed_entity_name)
        if self.args.no_name_info:
            name_triples = self._get_no_name()
            return None, None, np.zeros((30000, 300))
        elif self.args.no_attr_info:
            name_triples = self._get_local_name_by_only_name_triple()
        else:
            name_triples = self._get_local_name_by_name_triple()


        # preprocess more names; name[:,0] = entity_ids; name[:, 2] = entity_name
        names = pd.DataFrame(name_triples)
        names.iloc[:, 2] = names.iloc[:, 2].str.replace(r'[{}]+'.format(string.punctuation), '').str.split(' ')
        
        # load word embedding
        with open(self.word_embed, 'r') as f:
            w = f.readlines()
            w = pd.Series(w[1:])
        # w is a dataframe of embeddings of all words (loaded from pretrained embeddings)
        we = w.str.split(' ')
        word = we.apply(lambda x: x[0])
        w_em = we.apply(lambda x: x[1:])
        self.logger.info('concat word embeddings')
        word_em = np.stack(w_em.values, axis=0).astype(np.float)
        word_em = np.append(word_em, np.zeros([1, 300]), axis=0)
        self.logger.info('convert words to ids')
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

        self.logger.info('look up desc embeddings')
        names.iloc[:, 2] = names.iloc[:, 2].apply(lookup_and_padding)

        # entity-desc-embedding dataframe
        e_desc_input = pd.DataFrame(np.repeat([[un_logged_id, ] * 4], model.kgs.entities_num, axis=0),
                                    range(model.kgs.entities_num))

        e_desc_input.iloc[names.iloc[:, 0].values] = np.stack(names.iloc[:, 2].values)

        self.logger.info('generating desc input costs time: {:.4f}s'.format(time.time() - start))
        name_embeds = word_em[e_desc_input.values]
        name_embeds = np.sum(name_embeds, axis=1)

        return word_em, e_desc_input, name_embeds

