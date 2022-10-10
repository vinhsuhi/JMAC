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
    def __init__(self, triples, num_ent, triples_dict, neg_num):
        """
        triples is a list of dicts {'triple': (sub, rel, -1), 'label': sr2o[(sub, rel)], 'sub_samp': 1}
        """
        self.triples	= triples
        self.triples_dict = triples_dict
        self.num_ent = num_ent
        self.neg_num = neg_num
        self.all_ent = np.arange(num_ent)


    def update_triples(self, triples, triples_dict):
        self.triples = triples
        self.triples_dict = triples_dict
    

    def __len__(self):
        return len(self.triples)

    def get_neg_sample(self, labels):
        # if graph_i == 0:
        mask = np.ones([self.num_ent], dtype=np.bool)
        indices = []
        for ele in labels:
            try:
                indices.append(self.all_src_id2idx[ele])
            except:
                continue
        mask[labels] = 0
        neg_sample = np.int32(np.random.choice(self.all_ent[mask], self.neg_num, replace=False)).reshape([-1])
        return neg_sample

    def __getitem__(self, idx):
        ele			= self.triples[idx]
        key = (ele[0], ele[1])
        labels = self.triples_dict[key] # list of target entities
        neg_samples = self.get_neg_sample(labels)
        neg_samples = torch.LongTensor(neg_samples)
        triple = torch.LongTensor(ele)
        return triple, neg_samples


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