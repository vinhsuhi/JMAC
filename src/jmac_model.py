import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.helper.helper import *
from modules.load.data_loader import *
from modules.helper.message_passing import MessagePassing
from torch_scatter import scatter_add


class RelationAwareLayer(MessagePassing):
    '''
    Revised!
    '''
    def __init__(self, in_channels, out_channels, rel_dim, act=lambda x:x, args=None):
        super(self.__class__, self).__init__()

        self.layer_act 		= act
        self.args = args 
        self.rel_transform_weight1 = get_param((rel_dim, in_channels))
        self.rel_transform_weight2 = get_param((in_channels, in_channels))
        self.gcn_weight = get_param((in_channels, out_channels))
        self.loop_rel 	= get_param((1, rel_dim))

        # att weight
        self.w_att      = get_param((2 * in_channels, out_channels))
        self.a_att      = get_param((out_channels, 1))
        self.atv_mlp    = nn.LeakyReLU(args.leaky_relu_w)
        self.bn			= nn.BatchNorm1d(out_channels)

        self.comp_op = args.comp_op


    def forward(self, ent_emb, rel_emb, edge_index, edge_type): 
        '''
        Revised!
        '''
        self.device = edge_index.device
        num_ent = ent_emb.size(0)    
        rel_emb = torch.cat([rel_emb, self.loop_rel], dim=0)
        rel_emb = torch.mm(rel_emb, self.rel_transform_weight1)
        rel_emb = self.atv_mlp(rel_emb)
        rel_emb = torch.mm(rel_emb, self.rel_transform_weight2)
        
        loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
        loop_type   = torch.full((num_ent,), rel_emb.size(0)-1, dtype=torch.long).to(self.device)

        norm = self.compute_norm(edge_index, num_ent)
        
        message_neighbors = self.propagate('add', edge_index, x=ent_emb, edge_type=edge_type, rel_embed=rel_emb, edge_norm=norm, mode='in')
        message_self = self.propagate('add',loop_index, x=ent_emb, edge_type=loop_type, rel_embed=rel_emb, edge_norm=None, mode='in')
        
        out = self.layer_act(self.bn((message_neighbors + message_self)/2))
        return out


    def compute_relation_aware_message(self, ent_embed, rel_embed):
        '''
        Revised!
        Compute message from ent and rel
        '''
        if self.comp_op == 'sub': 	
            trans_embed  = ent_embed - rel_embed
        elif self.comp_op == 'mult': 	
            trans_embed  = ent_embed * rel_embed
        else: 
            raise NotImplementedError

        return trans_embed


    def compute_relation_aware_attention(self, u, v):
        '''
        Revised!
        '''
        combined = torch.cat((u, v), dim=1)
        alpha = torch.mm(self.atv_mlp(torch.mm(combined, self.w_att)), self.a_att)
        return alpha


    def message(self, x_i, x_j, edge_type, rel_embed, edge_norm, mode):
        '''
        Revised!
        '''
        weight = self.gcn_weight
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        rel_aware_message  = self.compute_relation_aware_message(x_j, rel_emb) 
        score = self.compute_relation_aware_attention(x_i, rel_aware_message) 
        out = torch.mm(rel_aware_message, weight) # OK!
        return out, score, edge_norm


    def update(self, aggr_out):
        '''
        Revised!
        '''
        return aggr_out


    def compute_norm(self, edge_index, num_ent):
        '''
        Revised!
        '''
        row, _	= edge_index
        edge_weight 	= torch.ones_like(row).float()
        deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
        deg_inv		= deg.pow(-0.5)							# D^{-0.5}
        deg_inv[deg_inv	== float('inf')] = 0
        norm = deg_inv[row] * edge_weight
        return norm


class BaseModel(nn.Module):
    '''
    Revised!
    '''
    def __init__(self):
        super(BaseModel, self).__init__()
        self.act	= torch.tanh
        self.bceloss	= nn.BCELoss()

    def triple_loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class JMAC(BaseModel):
    def __init__(self, args, entity_name_emb, num_relations, num_entities):
        '''
        Revised!
        '''
        super(JMAC, self).__init__()
        
        self.args = args
        self.ent_info_att = torch.FloatTensor(entity_name_emb)
        assert entity_name_emb.shape[0] == num_entities
        self.entity_dim = args.dim
        self.relation_dim = args.dim
        self.device = args.device

        # 1. Embedding initialization
        self.ent_init_att_completion = get_param((num_entities, self.entity_dim))
        self.rel_init_att_completion = get_param((num_relations, self.entity_dim))
        self.rel_init_att_alignment = get_param((num_relations, self.entity_dim))

        # 2. Defining layers
        self.completion_dropout = nn.Dropout(args.dropout)
        self.atv_mlp    = nn.LeakyReLU(args.leaky_relu_w)
        self.conv1_alignment = RelationAwareLayer(self.entity_dim, self.entity_dim, rel_dim=self.entity_dim, act=self.act, args=args)        
        self.conv2_alignment = RelationAwareLayer(self.entity_dim, self.entity_dim, rel_dim=self.entity_dim, act=self.act, args=args)        
        self.conv1_completion = RelationAwareLayer(self.entity_dim, self.entity_dim, rel_dim=self.entity_dim, act=self.act, args=args)        

        self.name_linear = get_param((self.ent_info_att.shape[1], self.entity_dim))

        self.margin_align = args.margin_align # yeah, the loss gamma
        self.k = args.num_negative
        margin_completion = args.margin_completion
        self.margin_completion = nn.Parameter(torch.Tensor([margin_completion]))
        self.margin_completion.requires_grad = False  

        self.uni_linear1_1, self.uni_linear1_2 = get_param((self.entity_dim * args.num_gcn_layer, self.entity_dim)), get_param((self.entity_dim, self.entity_dim))
        self.uni_linear2_1, self.uni_linear2_2 = get_param((self.entity_dim * args.num_gcn_layer, self.entity_dim)), get_param((self.entity_dim, self.entity_dim))
        self.rel_linear11, self.rel_linear12 = get_param((self.entity_dim, self.entity_dim)), get_param((self.entity_dim, self.entity_dim))
        self.rel_linear21, self.rel_linear22 = get_param((self.entity_dim, self.entity_dim)), get_param((self.entity_dim, self.entity_dim))
        self.rel_linear11_uni, self.rel_linear12_uni = get_param((self.entity_dim, self.entity_dim)), get_param((self.entity_dim, self.entity_dim))
        self.all_linear_completion = get_param((self.entity_dim * (args.num_gcn_layer + 1), self.entity_dim))

        if not self.args.no_name_info:
            self.forward_base = self.forward_name 
        else:
            self.forward_base = self.forward_no_name
        

    def forward_name(self, edge_index, edge_type, ent_bases, rel_bases):
        '''
        Revised!
        '''
        init_ent_completion_att = self.ent_init_att_completion[ent_bases[0]: ent_bases[1]]
        init_ent_align_att = torch.mm(self.ent_info_att[ent_bases[0]: ent_bases[1]].to(self.device), self.name_linear)
        
        init_ent_completion = self.completion_dropout(F.normalize(init_ent_completion_att)) 
        init_ent_align_att = torch.mm(torch.cat((init_ent_completion, init_ent_align_att), dim=1), self.uni_linear1_1)
        #init_ent_align_att = torch.mm(self.atv_mlp(init_ent_align_att), self.uni_linear1_2)

        ent_align_output1 = self.conv1_alignment(init_ent_align_att, self.rel_init_att_alignment[rel_bases[0]: rel_bases[1]], edge_index, edge_type)  
        
        align_layer_embs = [init_ent_align_att, ent_align_output1]
        completion_layer_embs = [init_ent_completion_att]
        completion_rel_layer_embs = [self.rel_init_att_completion[rel_bases[0]: rel_bases[1]]]

        if self.args.num_gcn_layer == 2:
            ent_completion_output1 = self.conv1_completion(init_ent_completion_att, self.rel_init_att_completion[rel_bases[0]: rel_bases[1]], edge_index, edge_type)
            completion_output1_norm = self.completion_dropout(F.normalize(ent_completion_output1)) 
            ent_in_align_emb = torch.mm(torch.cat((completion_output1_norm, ent_align_output1), dim=1), self.uni_linear2_1)
            #ent_in_align_emb = torch.mm(self.atv_mlp(ent_in_align_emb), self.uni_linear2_2)
            
            completion_rel_output1 = torch.mm(self.atv_mlp(torch.mm(self.rel_init_att_completion[rel_bases[0]: rel_bases[1]], self.rel_linear11)), self.rel_linear12)
            align_rel_input = torch.mm(self.atv_mlp(torch.mm(self.rel_init_att_alignment[rel_bases[0]: rel_bases[1]], self.rel_linear11_uni)), self.rel_linear12_uni)
            ent_align_output2 = self.conv2_alignment(ent_in_align_emb, align_rel_input, edge_index, edge_type)

            align_layer_embs.append(ent_align_output2)
            completion_layer_embs.append(ent_completion_output1)
            completion_rel_layer_embs.append(completion_rel_output1)

        ent_align_output = torch.mm(torch.cat(align_layer_embs, dim=1), self.all_linear_completion)        
        return ent_align_output, completion_layer_embs, completion_rel_layer_embs


    def forward_no_name(self, edge_index, edge_type, ent_bases, rel_bases):
        '''
        Revised!
        '''
        init_ent_completion_att = self.ent_init_att_completion[ent_bases[0]: ent_bases[1]]
        completion_layer_embs = [init_ent_completion_att]
        completion_rel_layer_embs = [self.rel_init_att_completion[rel_bases[0]: rel_bases[1]]]
        if self.args.num_gcn_layer == 2:
            ent_completion_output1 = self.conv1_completion(init_ent_completion_att, self.rel_init_att_completion[rel_bases[0]: rel_bases[1]], edge_index, edge_type)
            completion_rel_output1 = torch.mm(self.atv_mlp(torch.mm(self.rel_init_att_completion[rel_bases[0]: rel_bases[1]], self.rel_linear11)), self.rel_linear12)
            completion_layer_embs.append(ent_completion_output1)
            completion_rel_layer_embs.append(completion_rel_output1)
    
        return completion_layer_embs[-1], completion_layer_embs, completion_rel_layer_embs


    def get_emb(self, edge_index, edge_type, ent_bases, rel_bases, pyt=False):
        '''
        Revised!
        '''
        ent_align_output, ent_completion_outputs, _ = self.forward_base(edge_index, edge_type, ent_bases, rel_bases)
        ent_align_output = F.normalize(ent_align_output, 2, -1)
        ent_completion_output = F.normalize(ent_completion_outputs[-1], 2, -1)
        if pyt:
            return ent_align_output.detach().cpu(), ent_completion_output.detach().cpu()
        ent_align_output = ent_align_output.detach().cpu().numpy()
        ent_completion_output = ent_completion_output.detach().cpu().numpy()
        return ent_align_output, ent_completion_output 
    
    
    def alignment_loss_simple(self, links, ent_embeddings1, ent_embeddings2):
        '''
        Revised!
        '''
        if not len(links):
            return 0
        source = links[:, 0].reshape(-1)
        target = links[:, 1].reshape(-1)
        
        source_emb = F.normalize(ent_embeddings1[source], 2, -1)
        target_emb = F.normalize(ent_embeddings2[target], 2, -1)
        A = 1 - torch.sum(source_emb * target_emb, dim=1) # cosine distance
        return A.mean()
    
    
    def alignment_loss(self, feeddict, edge_index1, edge_type1, edge_index2, edge_type2):
        '''
        Revised!
        '''
        links = feeddict["links"]
        if not len(links):
            return 0
        ent_bases1 = feeddict['ent_bases1']
        rel_bases1 = feeddict['rel_bases1']
        ent_bases2 = feeddict['ent_bases2']
        rel_bases2 = feeddict['rel_bases2']
        ent_embeddings1, _, _ = self.forward_base(edge_index1, edge_type1, ent_bases1, rel_bases1)
        ent_embeddings2, _, _ = self.forward_base(edge_index2, edge_type2, ent_bases2, rel_bases2)
        neg_left = feeddict["neg_left"].reshape(-1) # neg_of_target which is in source
        neg_right = feeddict["neg_right"].reshape(-1) # neg_of_source which is in target
        
        source = links[:, 0].reshape(-1)
        target = links[:, 1].reshape(-1)
        
        source_emb = F.normalize(ent_embeddings1[source], 2, -1)
        target_emb = F.normalize(ent_embeddings2[target], 2, -1)
        A = 1 - torch.sum(source_emb * target_emb, dim=1) # cosine distance
        
        neg_l_x = F.normalize(ent_embeddings1[neg_left], 2, -1)
        neg_r_x = F.normalize(ent_embeddings2[neg_right], 2, -1)

        B = 1 - torch.sum(neg_l_x * neg_r_x, dim=1) # cosine distance
        C = -B.view(len(links), -1)
        D = A + self.margin_align
        L1 = F.relu(C + D.view(len(links), 1))
        neg_left = feeddict["neg2_left"].reshape(-1)
        neg_right = feeddict["neg2_right"].reshape(-1)

        neg_l_x = F.normalize(ent_embeddings1[neg_left], 2, -1)
        neg_r_x = F.normalize(ent_embeddings2[neg_right], 2, -1)
        B = 1 - torch.sum(neg_l_x * neg_r_x, dim=1)

        C = -B.view(len(links), -1)
        L2 = F.relu(C + D.view(len(links), 1))
        ent_align_loss = (L1.sum() + L2.sum()) / (2 * self.k * len(links))
        return ent_align_loss * 1


    def forward_linkpred(self, e_index, r_index, edge_index, edge_type, all_index, ent_bases, rel_bases, pred_head=False):
        '''
        Revised!
        '''

        _, pos_layer_embs, pos_rel_layer_embs = self.forward_base(edge_index, edge_type, ent_bases, rel_bases)
        dist = 0
        for layer in range(self.args.num_gcn_layer):
            ent_completion_output = pos_layer_embs[layer]
            all_kg_emb = ent_completion_output[all_index]
            rel_output = pos_rel_layer_embs[layer]
            e = all_kg_emb[e_index]
            r = rel_output[r_index]
            if pred_head:
                er = e - r
            else:
                er = e + r
            dist += torch.cdist(er, all_kg_emb, p=1)
        return dist

    
    def completion_loss(self, data, edge_index1, edge_type1, edge_index2, edge_type2, feeddict, source=True):
        '''
        Revised!
        '''
        links = feeddict['links']
        ent_bases1 = feeddict['ent_bases1']
        rel_bases1 = feeddict['rel_bases1']
        ent_bases2 = feeddict['ent_bases2']
        rel_bases2 = feeddict['rel_bases2']
        _, completion_layer_embs1, completion_rel_layer_embs1 = self.forward_base(edge_index1, edge_type1, ent_bases1, rel_bases1) 
        _, completion_layer_embs2, completion_rel_layer_embs2 = self.forward_base(edge_index2, edge_type2, ent_bases2, rel_bases2) 

        batch_h, batch_t, batch_r = data['batch_h'], data['batch_t'], data['batch_r']

        loss = 0
        for layer in range(self.args.num_gcn_layer):
            ent_completion_output1 = completion_layer_embs1[layer]
            rel_output1 = completion_rel_layer_embs1[layer]

            ent_completion_output2 = completion_layer_embs2[layer]
            rel_output2 = completion_rel_layer_embs2[layer]

            if source:
                ent_completion_output = ent_completion_output1
                rel_output = rel_output1
            else:
                ent_completion_output = ent_completion_output2
                rel_output = rel_output2

            h = ent_completion_output[batch_h]
            t = ent_completion_output[batch_t]
            r = rel_output[batch_r]
        
            score = (h + r) - t
            score = torch.norm(score, 1, -1).flatten()


            def _get_positive_score(score):
                """
                2 - final_loss calls this
                """
                positive_score = score[:self.args.batch_size]
                positive_score = positive_score.view(-1, min(self.args.batch_size, len(positive_score))).permute(1, 0)
                return positive_score


            def _get_negative_score(score):
                """
                2 - final_loss calls this
                """
                negative_score = score[self.args.batch_size:]
                negative_score = negative_score.view(-1, min(self.args.batch_size, len(negative_score))).permute(1, 0)
                return negative_score

            p_score = _get_positive_score(score)
            n_score = _get_negative_score(score)

                
            loss_res = (torch.max(p_score - n_score, -self.margin_completion)).mean() + self.margin_completion
            
            loss_align = self.alignment_loss_simple(links, ent_completion_output1, ent_completion_output2)

            loss += loss_res + loss_align

        return loss
                
