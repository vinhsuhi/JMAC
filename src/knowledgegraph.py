import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn



class KnowledgeGraph(nn.Module):
    def __init__(self, lang, kg_train_data, kg_val_data, kg_test_data, num_entity, num_relation, is_supporter_kg,
                 entity_id_base, relation_id_base, device, n_neg_pos=1):
        super(KnowledgeGraph, self).__init__()

        self.lang = lang

        self.train_data = kg_train_data  # training set # train_triples # numpy!
        self.val_data = kg_val_data

        if is_supporter_kg:
            self.train_data = np.concatenate((kg_train_data, kg_val_data), )

        self.test_data = kg_test_data

        self.entity_id_base = entity_id_base
        self.relation_id_base = relation_id_base # TODO: really?
        self.upper_entity_base = 0
        self.upper_relation_base = 0 

        self.num_relation = num_relation # Total number of relations in relation.txt + 1
        self.num_entity = num_entity
        self.is_supporter_kg = is_supporter_kg

        self.n_neg_pos = n_neg_pos
        self.device = device

        self.edge_index = None # subgraph_list_kg
        self.edge_type = None # subgraph_list_align

        self.computed_entity_embedidng_align = None
        self.computed_entity_embedidng_KG = None

        self.transferred_triples = []
        self.triple_keys = set()


        if not is_supporter_kg:
            self.true_tail, self.true_head = self.get_true_tail(np.concatenate((self.train_data, self.val_data, self.test_data), axis=0)) # WHATTT?? # {(h, r): [t1, t2, t3...]}
        
        self.h_train, self.r_train, self.t_train = self.train_data[:, 0], self.train_data[:, 1], self.train_data[:, 2]
        self.h_val, self.r_val, self.t_val = self.val_data[:, 0], self.val_data[:, 1], self.val_data[:, 2]
        self.h_test, self.r_test, self.t_test = self.test_data[:, 0], self.test_data[:, 1], self.test_data[:, 2]

        self.h_train = torch.LongTensor(self.h_train)
        self.h_val = torch.LongTensor(self.h_val)
        self.h_test = torch.LongTensor(self.h_test)
        self.t_train = torch.LongTensor(self.t_train)
        self.t_val = torch.LongTensor(self.t_val)
        self.t_test = torch.LongTensor(self.t_test)
        self.r_train = torch.LongTensor(self.r_train)
        self.r_val = torch.LongTensor(self.r_val)
        self.r_test = torch.LongTensor(self.r_test)

    def get_true_tail(self, triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        triples_np = triples

        true_tail = {}
        true_head = {}

        for head, relation, tail in triples_np:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            if (tail, relation) not in true_head:
                true_head[(tail, relation)] = []
            true_tail[(head, relation)].append(tail)
            true_head[(tail, relation)].append(head)

        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))
        for tail, relation in true_head:
            true_head[(tail, relation)] = np.array(list(set(true_head[(tail, relation)])))

        return true_tail, true_head



    def generate_batch_data(self,h_all,r_all,t_all,batch_size, shuffle = True):
        h_all = torch.unsqueeze(h_all, dim=1)
        r_all = torch.unsqueeze(r_all, dim=1)
        t_all = torch.unsqueeze(t_all, dim=1)
        triple_all = torch.cat([h_all,r_all,t_all],dim=-1) #[B,3]
        triple_dataloader = DataLoader(triple_all,batch_size = batch_size,shuffle = shuffle)

        return triple_dataloader
  
    def get_negative_samples(self,batch_size_each):

       
        rand_negs = torch.randint(high=self.num_entity, size=(batch_size_each,),
                                  device=self.device)  # [b,num_neg = 1]

        rand_negs = rand_negs.view(-1,1)
        
        return rand_negs



