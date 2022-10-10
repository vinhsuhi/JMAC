from __future__ import division
import numpy as np
import torch
import logging


class CompletionEvaluator:
    '''
    Revised!
    '''
    def __init__(self, target_kg, model, device, data_dir):
        self.target_kg = target_kg
        self.device = device
        self.model = model
        self.data_dir = data_dir
        self.edge_index = torch.LongTensor(target_kg.edge_index).to(device)
        self.edge_type = torch.LongTensor(target_kg.edge_type).to(device)
        self.ent_bases = [target_kg.entity_id_base, target_kg.upper_entity_base]
        self.rel_bases = [target_kg.relation_id_base, target_kg.upper_relation_base]


    def test(self,args,is_val=True, edge_index_valid=None, edge_type_valid = None, filterr = False, logger=None):
        '''
        Revised!
        '''
        if edge_index_valid is not None:
            self.edge_index = edge_index_valid
            self.edge_type = edge_type_valid

        er_vocab = self.target_kg.true_tail

        hits = []
        ranks = []
        for _ in range(10):
            hits.append([])

        if is_val:
            kg_batch_generator = self.target_kg.generate_batch_data(self.target_kg.h_val, self.target_kg.r_val, self.target_kg.t_val, batch_size=args.batch_size, shuffle=False)
        else:
            kg_batch_generator = self.target_kg.generate_batch_data(self.target_kg.h_test, self.target_kg.r_test, self.target_kg.t_test, batch_size=args.batch_size, shuffle=False)
            
            
        total_entity_num = self.target_kg.num_entity
        all_index = list(range(total_entity_num))

        for kg_batch_each in kg_batch_generator:
            h_batch = kg_batch_each[:, 0].view(-1).tolist()
            r_batch = kg_batch_each[:, 1].view(-1).tolist()  # global index
            t_batch = kg_batch_each[:, 2].view(-1).tolist()
            predictions = - self.model.forward_linkpred(h_batch, r_batch, self.edge_index, self.edge_type, all_index, self.ent_bases, self.rel_bases)

            if filterr:
                for j in range(kg_batch_each.shape[0]):
                    filt = er_vocab[(kg_batch_each[j][0].item(), kg_batch_each[j][1].item())]
                    target_value = predictions[j, t_batch[j]].item()
                    predictions[j, filt] = -1e6
                    predictions[j, t_batch[j]] = target_value

            _, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(kg_batch_each.shape[0]):
                rank = np.where(sort_idxs[j] == t_batch[j])[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        hits10 = np.mean(hits[9])
        hits1 = np.mean(hits[0])
        mrr = np.mean(1. / np.array(ranks))

        logger.info('Hits @1: {0}'.format(np.mean(hits[0])))
        logger.info('Hits @10: {0}'.format(np.mean(hits[9])))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
        return hits1, hits10, mrr
