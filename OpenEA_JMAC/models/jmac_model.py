import torch 
import torch.nn as nn
import torch.nn.functional as F
from modules.helper.helper import *
from modules.load.data_loader import *
from modules.helper.message_passing import MessagePassing
from torch_scatter import scatter_add


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.act	= torch.tanh
        self.bceloss	= nn.BCELoss()

    def triple_loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class RelationalAwareLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, rel_dim, act=lambda x:x, args=None):
        super(self.__class__, self).__init__()

        self.layer_act 		= act
        self.args = args
        self.margin = 5.0
                
        self.rel_transform_weight1 = get_param((rel_dim, in_channels))
        self.rel_transform_weight2 = get_param((in_channels, in_channels))
        self.gcn_weight = get_param((in_channels, out_channels))
        self.loop_rel 	= get_param((1, rel_dim))

        # att weight
        self.w_att      = get_param((2 * in_channels, out_channels))
        self.a_att      = get_param((out_channels, 1))
        self.atv_mlp    = nn.LeakyReLU(args.leaky_relu_w)
        self.bn			= nn.BatchNorm1d(out_channels)

        self.opn = args.opn


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
        """
        Revised!
        """
        if self.opn == 'sub': 	
            trans_embed  = ent_embed - rel_embed
        elif self.opn == 'mult': 	
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
        rel_aware_message  = self.compute_relation_aware_message(x_j, rel_emb) # phi(hu, hr)
        score = self.compute_relation_aware_attention(x_i, rel_aware_message) # OK!
        out = torch.mm(rel_aware_message, weight) # OK!
        return out, score, edge_norm


    def update(self, aggr_out):
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


class JMAC_MODEL(BaseModel):
    def __init__(self, num_ent, num_rel, ent_info_att, args=None):
        super(JMAC_MODEL, self).__init__()
        self.args = args
        self.dim = args.emb_dim
        self.completion_dropout = nn.Dropout(args.completion_dropout_rate)
        self.atv_mlp    = nn.LeakyReLU(args.leaky_relu_w)

        num_rel = num_rel * 2

        self.ent_info_att = ent_info_att 
        self.ent_completion_att, self.rel_completion_att, self.rel_info_att = get_param((num_ent, self.dim)), get_param((num_rel, self.dim)), get_param((num_rel, self.dim))                
        self.ent_completion_dim, self.ent_alignment_dim, self.rel_completion_dim = self.ent_completion_att.shape[1], self.ent_info_att.shape[1], self.rel_completion_att.shape[1]
        
        self.margin_align = args.margin_align # yeah, the loss gamma
        self.k = args.num_negative 

        self.conv1_align = RelationalAwareLayer(self.dim, self.ent_alignment_dim, num_rel, rel_dim=self.dim, act=self.act, args=args)
        self.conv2_align = RelationalAwareLayer(self.dim, self.ent_alignment_dim, num_rel, rel_dim=self.dim, act=self.act, args=args)
        self.conv1_completion = RelationalAwareLayer(self.ent_completion_dim, self.ent_completion_dim, num_rel, rel_dim=self.rel_completion_dim, act=self.act, args=args)
        
        margin_completion = args.margin_completion
        self.margin_completion = nn.Parameter(torch.Tensor([margin_completion]))
        self.margin_completion.requires_grad = False        
        
        # MLP for incorporating 
        self.align_linear1_1, self.align_linear1_2 = get_param((self.dim * 2, self.dim)), get_param((self.dim, self.dim))
        self.align_linear2_1, self.align_linear2_2 = get_param((self.dim * 2, self.dim)), get_param((self.dim, self.dim))

        self.rel_linear11, self.rel_linear12 = get_param((self.dim, self.dim)), get_param((self.dim, self.dim))
        self.rel_linear21, self.rel_linear22 = get_param((self.dim, self.dim)), get_param((self.dim, self.dim))
        self.rel_linear11_align, self.rel_linear12_align = get_param((self.dim, self.dim)), get_param((self.dim, self.dim))
        self.all_linear_comp = get_param((self.dim * (args.num_gcn_layer + 1), self.dim))
        
    
    def forward_base(self, edge_index, edge_type):
        '''
        Revising!
        '''
        init_comp = self.completion_dropout(F.normalize(self.ent_completion_att, p=2, dim=-1))
        # incorporate init feature!
        ent_init_alignment_emb = torch.mm(torch.cat((init_comp, self.ent_info_att), dim=1), self.align_linear1_1)
        ent_init_alignment_emb = torch.mm(self.atv_mlp(ent_init_alignment_emb), self.align_linear1_2)

        ent_alignment_output1 = self.conv1_align(ent_init_alignment_emb, self.rel_info_att, edge_index, edge_type)  
        
        comp_layer_embs = [ent_init_alignment_emb, ent_alignment_output1]
        pos_layer_embs = [self.ent_completion_att]
        pos_rel_layer_embs = [self.rel_completion_att]

        if self.args.num_gcn_layer == 2:
            ent_completion_output1 = self.conv1_completion(self.ent_completion_att, self.rel_completion_att, edge_index, edge_type)
            pos_output1_norm = self.completion_dropout(F.normalize(ent_completion_output1))

            # incorporate intermediate feature!
            ent_in_alignment_emb = torch.mm(torch.cat((pos_output1_norm, ent_alignment_output1), dim=1), self.align_linear2_1)
            ent_in_alignment_emb = torch.mm(self.atv_mlp(ent_in_alignment_emb), self.align_linear2_2)

            pos_rel_output1 = torch.mm(self.atv_mlp(torch.mm(self.rel_completion_att, self.rel_linear11)), self.rel_linear12)
            align_rel_input = torch.mm(self.atv_mlp(torch.mm(self.rel_info_att, self.rel_linear11_align)), self.rel_linear12_align)
        
            ent_alignment_output2 = self.conv2_align(ent_in_alignment_emb, align_rel_input, edge_index, edge_type)
            comp_layer_embs.append(ent_alignment_output2)
            pos_layer_embs.append(ent_completion_output1)
            pos_rel_layer_embs.append(pos_rel_output1)

        ent_alignment_output = torch.mm(torch.cat(comp_layer_embs, dim=1), self.all_linear_comp)        
        return ent_alignment_output, pos_layer_embs, pos_rel_layer_embs
        

    def get_emb(self, edge_index, edge_type, pyt=False):
        ent_alignment_output, ent_completion_outputs, _ = self.forward_base(edge_index, edge_type)
        ent_alignment_output = F.normalize(ent_alignment_output, 2, -1)
        ent_completion_output = F.normalize(ent_completion_outputs[-1], 2, -1)
        if pyt:
            return ent_alignment_output.detach().cpu(), ent_completion_output.detach().cpu()
        ent_alignment_output = ent_alignment_output.detach().cpu().numpy()
        ent_completion_output = ent_completion_output.detach().cpu().numpy()
        return ent_alignment_output, ent_completion_output 
    
    
    def alignment_loss_simple(self, links, ent_embeddings):
        source = links[:, 0].reshape(-1)
        target = links[:, 1].reshape(-1)
        
        source_emb = F.normalize(ent_embeddings[source], 2, -1)
        target_emb = F.normalize(ent_embeddings[target], 2, -1)
        A = 1 - torch.sum(source_emb * target_emb, dim=1) # cosine distance
        return A.mean()
    
    
    def alignment_loss(self, feeddict, edge_index, edge_type):
        ent_embeddings, _, _ = self.forward_base(edge_index, edge_type)
        neg_left = feeddict["neg_left"].reshape(-1) 
        neg_right = feeddict["neg_right"].reshape(-1)
        
        links = feeddict["links"]
        source = links[:, 0].reshape(-1)
        target = links[:, 1].reshape(-1)
        
        source_emb = F.normalize(ent_embeddings[source], 2, -1)
        target_emb = F.normalize(ent_embeddings[target], 2, -1)
        A = 1 - torch.sum(source_emb * target_emb, dim=1) # cosine distance
        
        neg_l_x = F.normalize(ent_embeddings[neg_left], 2, -1)
        neg_r_x = F.normalize(ent_embeddings[neg_right], 2, -1)

        B = 1 - torch.sum(neg_l_x * neg_r_x, dim=1) # cosine distance
        C = -B.view(len(links), -1)
        D = A + self.margin_align
        L1 = F.relu(C + D.view(len(links), 1))
        neg_left = feeddict["neg2_left"].reshape(-1)
        neg_right = feeddict["neg2_right"].reshape(-1)

        neg_l_x = F.normalize(ent_embeddings[neg_left], 2, -1)
        neg_r_x = F.normalize(ent_embeddings[neg_right], 2, -1)
        B = 1 - torch.sum(neg_l_x * neg_r_x, dim=1)

        C = -B.view(len(links), -1)
        L2 = F.relu(C + D.view(len(links), 1))
        ent_align_loss = (L1.sum() + L2.sum()) / (2 * self.k * len(links))
        return ent_align_loss


    def forward_linkpred(self, e_index, r_index, edge_index, edge_type, all_index):
        _, pos_layer_embs, pos_rel_layer_embs = self.forward_base(edge_index, edge_type)
        dist = 0
        for layer in range(self.args.num_gcn_layer):
            ent_completion_output = pos_layer_embs[layer]
            all_kg_emb = ent_completion_output[all_index]
            rel_output = pos_rel_layer_embs[layer]
            h = all_kg_emb[e_index]
            r = rel_output[r_index]
            hr = h + r
            dist += torch.cdist(hr, all_kg_emb, p=1)
        return dist

    
    def completion_loss(self, data, edge_index, edge_type, feeddict):
        links = feeddict['links']
        _, pos_layer_embs, pos_rel_layer_embs = self.forward_base(edge_index, edge_type) 
        batch_h, batch_t, batch_r = data['batch_h'], data['batch_t'], data['batch_r']

        loss = 0
        for layer in range(self.args.num_gcn_layer):
            ent_completion_output = pos_layer_embs[layer]
            rel_output = pos_rel_layer_embs[layer]

            h = ent_completion_output[batch_h]
            t = ent_completion_output[batch_t]
            r = rel_output[batch_r]
        
            score = (h + r) - t
            score = torch.norm(score, 1, -1).flatten()

            def _get_positive_score(score):
                """
                2 - final_loss calls this
                """
                positive_score = score[:self.args.completion_batch_size]
                positive_score = positive_score.view(-1, min(self.args.completion_batch_size, len(positive_score))).permute(1, 0)
                return positive_score


            def _get_negative_score(score):
                """
                2 - final_loss calls this
                """
                negative_score = score[self.args.completion_batch_size:]
                negative_score = negative_score.view(-1, min(self.args.completion_batch_size, len(negative_score))).permute(1, 0)
                return negative_score

            p_score = _get_positive_score(score)
            n_score = _get_negative_score(score)
                
            loss_res = (torch.max(p_score - n_score, -self.margin_completion)).mean() + self.margin_completion
            loss_align = self.alignment_loss_simple(links, ent_completion_output)

            loss += loss_res + loss_align

        return loss
                
