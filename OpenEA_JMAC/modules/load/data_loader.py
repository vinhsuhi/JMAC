from modules.helper.helper import *
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """
    Training Dataset class.

    Parameters
    ----------
    triples:	The triples used for training the model
    params:		Parameters for the experiments
    
    Returns
    -------
    A training Dataset class instance used by DataLoader
    """
    def __init__(self, triples, num_ent, all_src_ent, all_trg_ent, triples_dict, neg_num):
        """
        triples is a list of dicts {'triple': (sub, rel, -1), 'label': sr2o[(sub, rel)], 'sub_samp': 1}
        """
        self.triples	= triples
        self.triples_dict = triples_dict
        self.num_ent = num_ent
        self.all_src_ent = np.array(all_src_ent) # should be np array
        self.all_src_id2idx = {ele: i for i, ele in enumerate(self.all_src_ent)}
        self.all_trg_ent = np.array(all_trg_ent) # should be np array
        self.all_trg_id2idx = {ele: i for i, ele in enumerate(self.all_trg_ent)}

        self.graph_index = {ele: 0 for ele in all_src_ent}
        for ele in self.all_trg_ent:
            self.graph_index[ele] = 1
        self.neg_num = neg_num


    def update_triples(self, triples, triples_dict):
        self.triples = triples
        self.triples_dict = triples_dict
    

    def __len__(self):
        return len(self.triples)

    def get_neg_sample(self, labels, graph_i):
        if graph_i == 0:
            mask = np.ones([len(self.all_src_ent)], dtype=np.bool)
            indices = []
            for ele in labels:
                try:
                    indices.append(self.all_src_id2idx[ele])
                except:
                    continue
            mask[indices] = 0
            neg_sample = np.int32(np.random.choice(self.all_src_ent[mask], self.neg_num, replace=False)).reshape([-1])
        else:
            mask = np.ones([len(self.all_trg_ent)], dtype=np.bool)
            indices = []
            for ele in labels:
                try:
                    indices.append(self.all_trg_id2idx[ele])
                except:
                    continue
            mask[indices] = 0
            neg_sample = np.int32(np.random.choice(self.all_trg_ent[mask], self.neg_num, replace=False)).reshape([-1])
        return neg_sample

    def __getitem__(self, idx):
        ele			= self.triples[idx]
        key = (ele[0], ele[1])
        graph_i = self.graph_index[ele[0]]
        labels = self.triples_dict[key] # should be list
        neg_samples = self.get_neg_sample(labels, graph_i)
        neg_samples = torch.LongTensor(neg_samples)
        triple = torch.LongTensor(ele)
        return triple, neg_samples

    # @staticmethod
    # def collate_fn(data):
    #     triple		= torch.stack([_[0] 	for _ in data], dim=0)
    #     trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
    #     return triple, trp_label
    
    # def get_neg_ent(self, triple, label):
    #     def get(triple, label):
    #         pos_obj		= label
    #         mask		= np.ones([self.num_ent], dtype=np.bool)
    #         mask[label]	= 0
    #         neg_ent		= np.int32(np.random.choice(self.entities[mask], self.neg_num - len(label), replace=False)).reshape([-1])
    #         neg_ent		= np.concatenate((pos_obj.reshape([-1]), neg_ent))

    #         return neg_ent

    #     neg_ent = get(triple, label)
    #     return neg_ent

    # def get_label(self, label):
    #     y = np.zeros([self.num_ent], dtype=np.float32)
    #     for e2 in label: y[e2] = 1.0
    #     return torch.FloatTensor(y)




class TrainDatasetV2(Dataset):
    """
    Training Dataset class.

    Parameters
    ----------
    triples:	The triples used for training the model
    params:		Parameters for the experiments
    
    Returns
    -------
    A training Dataset class instance used by DataLoader
    """
    def __init__(self, triples, num_ent):
        """
        triples is a list of dicts {'triple': (sub, rel, -1), 'label': sr2o[(sub, rel)], 'sub_samp': 1}
        """
        self.triples	= triples
        self.num_ent = num_ent
        # self.neg_num = 100
        # self.entities	= np.arange(num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele			= self.triples[idx]
        triple = ele['key']
        label = ele['label']
        label = self.get_label(label)
        triple = torch.LongTensor(label)
        return triple, label

    # @staticmethod
    # def collate_fn(data):
    #     triple		= torch.stack([_[0] 	for _ in data], dim=0)
    #     trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
    #     return triple, trp_label
    
    # def get_neg_ent(self, triple, label):
    #     def get(triple, label):
    #         pos_obj		= label
    #         mask		= np.ones([self.num_ent], dtype=np.bool)
    #         mask[label]	= 0
    #         neg_ent		= np.int32(np.random.choice(self.entities[mask], self.neg_num - len(label), replace=False)).reshape([-1])
    #         neg_ent		= np.concatenate((pos_obj.reshape([-1]), neg_ent))

    #         return neg_ent

    #     neg_ent = get(triple, label)
    #     return neg_ent

    def get_label(self, label):
        y = np.zeros([self.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)



class TestDataset(Dataset):
    """
    Evaluation Dataset class.

    Parameters
    ----------
    triples:	The triples used for evaluating the model
    params:		Parameters for the experiments
    
    Returns
    -------
    An evaluation Dataset class instance used by DataLoader for model evaluation
    """
    def __init__(self, triples, params):
        self.triples	= triples
        self.p 		= params

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele		= self.triples[idx]
        triple, label	= torch.LongTensor(ele['triple']), np.int32(ele['label'])
        label		= self.get_label(label)

        return triple, label

    @staticmethod
    def collate_fn(data):
        triple		= torch.stack([_[0] 	for _ in data], dim=0)
        label		= torch.stack([_[1] 	for _ in data], dim=0)
        return triple, label
    
    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)