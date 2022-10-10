import torch
from src.data_loader import ParseData
from src.validate import CompletionEvaluator
import numpy as np
import argparse
from torch.utils.data import DataLoader
from src.utils import get_language_list
from src.jmac_model import JMAC
from modules.utils.util import get_neg
from tqdm import tqdm
from modules.load.data_loader import *
from modules.helper.helper import *
from modules.finding.evaluation import test
from collections import defaultdict as ddict
import logging
logging.basicConfig(
     level=logging.INFO
 )

import os 

class Logger:
    def __init__(self, name, level, tostdout=True, totxtfile=False):
        self.SAVE_FOLDER = "logging/"
        if not os.path.exists(self.SAVE_FOLDER):
            os.makedirs(self.SAVE_FOLDER)

        self.logger = logging.getLogger(name=name)
        self.logger.propagate = False # not propagate to root logger (print to sdtout)
        if tostdout:
            c_handler = logging.StreamHandler()
            c_format = logging.Formatter('%(asctime)s - %(module)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
            c_handler.setFormatter(c_format)
            c_handler.setLevel(level)
            self.logger.addHandler(c_handler)
        if totxtfile:
            f_handler = logging.FileHandler(os.path.join(self.SAVE_FOLDER, name + ".logs"))
            f_format = logging.Formatter('%(asctime)s - %(module)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
            f_handler.setFormatter(f_format)
            f_handler.setLevel(level)
            self.logger.addHandler(f_handler)
    
    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)
    
    def warn(self, msg, *args, **kwargs):
        self.logger.warn(msg, *args, **kwargs)



def parse_args(args=None):
    '''
    Revised!
    '''
    parser = argparse.ArgumentParser(
        description='Training and Testing JMAC Models',
        usage='run.py [<args>] [-h | --help]'
    )

    # Data loader
    parser.add_argument('--target_language', type=str, default='ja', choices=['ja', 'el', 'es', 'fr', 'en'], help="Target kg for completion")
    parser.add_argument('--data_path', default="datasetdbp5l", type=str, help='Path to the data folder')

    #KG model
    parser.add_argument('--margin_align', default=1, type=float, help="Gamma in alignment margin loss function") 
    parser.add_argument('--margin_completion', default=5, type=float, help="Gamma in completion margin loss function") 
    parser.add_argument('--dim', default=256, type=int, help='Kg embedding dimension for both entities and relations')

    # GNN
    parser.add_argument('--num_gcn_layer', type=int, default=2, choices=[1, 2])
    parser.add_argument('--align_percent', type=float, default=0.5) 

    # Training
    parser.add_argument('--epoch', default=30, type=int,help='How many epochs to train')
    parser.add_argument('--batch_size', default=1000, type=int, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--val_freq', type=int, default=2, help='How frequent to evaluate')
    parser.add_argument('--leaky_relu_w', type=float, default=0.05, help='activate function leakyrelu weight') # TODO: where is this param used
    parser.add_argument('--comp_op', type=str, default='sub', help='The composition operation')
    parser.add_argument('--align_lr', type=float, default=0.0003, help='Alignment component learning rate')
    parser.add_argument('--completion_lr', type=float, default=0.0003, help='Completion component learning rate')

    parser.add_argument('--num_negative', type=int, default=25, help='Number of negative examples')
    parser.add_argument('--pair_sample_weight', type=float, default=0.2, help='Beta vule controlling number of alignment candidates') # TODO: check this

    # Eval alignment 
    parser.add_argument('--top_k', type=list, default=[1, 5, 10], help='Top-k to return related to Hits@k') 
    parser.add_argument('--csls', type=int, default=10, help='CSLS value')
    parser.add_argument('--eval_metric', type=str, default='cosine', help='Metric used to compute embedding similarity when evaluating')
    parser.add_argument('--eval_norm', action='store_true', help='Normalize the embeddings before evaluate')
    parser.add_argument('--noise_rate', type=float, default=0, help='Noise to add to the ititialized node embeddings')

    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Whether to run the model using GPU (cuda) or CPU (cpu)')
    parser.add_argument('--logger', type=str, default="")
    return parser.parse_args(args)


def test_alignment_(embeds11, embeds22, test_ent1, test_ent2, args, logger):
    '''
    Revised!
    '''
    embeds1 = np.array([embeds11[e] for e in test_ent1])
    embeds2 = np.array([embeds22[e] for e in test_ent2])
    top_k, hits, mr, mrr = test(embeds1, embeds2, None, args.top_k, 8,
            metric=args.eval_metric, normalize=args.eval_norm, csls_k=args.csls, accurate=True, logger=logger)
    return top_k, hits, mr, mrr
        

def align_data_processing(triple_list, args):
    """
    Process triples for alignment component input
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
    edge_index = torch.LongTensor(edge_index).t().to(args.device)
    edge_type = torch.LongTensor(edge_type).to(args.device)

    return edge_index, edge_type


def seed_enlargement_triple_transferring(output1, output2, align_test_src, align_test_dst, global_entropies, seed_index, \
                                train_align_pairs, triples1, triples2, global_seeds, ent_bases1, rel_bases1, ent_bases2, rel_bases2, kg1, kg2, args):
    '''
    Revised!
    This is the Alignment seed enlargement and Triple transferring (EnTr) part!
    '''
    
    entropy, simi, _ = compute_alignment_quality(output1, output2, align_test_src, align_test_dst)
    #logger.info("Entropy value: {:.4f}".format(entropy.item()))

    prev_entropy = global_entropies[seed_index]
    additional_pairs = []
    if prev_entropy == -1:
        global_entropies[seed_index] = entropy
    else:
        sample_percent = (global_entropies[seed_index] - entropy) / global_entropies[seed_index] * args.pair_sample_weight
        try:
            num_pairs = int(sample_percent * len(align_test_src))
        except:
            num_pairs = 0

        if num_pairs > 0:
            #logger.info("Number of pairs to sample: {}, with sample percent: {:.4f}".format(num_pairs, sample_percent))
            max_values = simi.max(dim=1)[0]
            src_nodes = max_values.multinomial(num_pairs, replacement=False).tolist()
            rows = simi[src_nodes]
            dst_nodes = rows.max(dim=1)[1].tolist()
            additional_pairs = np.array([src_nodes, dst_nodes]).T

    if len(additional_pairs):
        if not len(train_align_pairs):
            new_align_pairs = additional_pairs
        else:
            new_align_pairs = np.concatenate((train_align_pairs, additional_pairs), axis=0)
        global_seeds[seed_index] = new_align_pairs
        #logger.info("number of new train links: {}".format(len(new_align_pairs)))
    else:
        new_align_pairs = train_align_pairs

    if len(new_align_pairs):

        neg2_left = get_neg(new_align_pairs[:, 1].tolist(), output2, output1, args.num_negative)
        neg_right = get_neg(new_align_pairs[:, 0].tolist(), output1, output2, args.num_negative)

        num_train_pairs = len(new_align_pairs)
        neg_num = args.num_negative
        
        pos = np.ones((num_train_pairs, neg_num)) * (new_align_pairs[:, 0].reshape((num_train_pairs, 1))) # 
        neg_left = pos.reshape((num_train_pairs * neg_num,))
        pos = np.ones((num_train_pairs, neg_num)) * (new_align_pairs[:, 1].reshape((num_train_pairs, 1)))
        neg2_right = pos.reshape((num_train_pairs * neg_num,)) # np x nn
                            
        feeddict = {"neg_left": neg_left,
                            "neg_right": neg_right,
                            "neg2_left": neg2_left,
                            "neg2_right": neg2_right,
                            "links": new_align_pairs, "ent_bases1": ent_bases1,
                            "ent_bases2": ent_bases2, "rel_bases1": rel_bases1,
                            "rel_bases2": rel_bases2}


        links_dict = {ele[0]: ele[1] for ele in new_align_pairs}
    else:
        feeddict = {"neg_left": [], "neg_right": [], "neg2_left": [], "neg2_right": [],
                    "links": [], "ent_bases1": ent_bases1, "ent_bases2": ent_bases2, "rel_bases1": rel_bases1, "rel_bases2": rel_bases2}
        
        links_dict = {}

    new_triples1, new_triples2, new_triple_keys1, new_triple_keys2 = transfer_knowledge(triples1, triples2, links_dict, kg1.triple_keys, kg2.triple_keys, args)
    return new_triples1, new_triples2, new_triple_keys1, new_triple_keys2, feeddict, global_seeds


def process_input_data(kg, args):
    if not len(kg.transferred_triples): 
        triples = kg.train_data.tolist()
        triples_keys = ['{}_{}_{}'.format(el[0], el[1], el[2]) for el in triples]
        kg.triple_keys = set(triples_keys)
        edge_index = torch.LongTensor(kg.edge_index).to(args.device)
        edge_type = torch.LongTensor(kg.edge_type).to(args.device)
    else:
        triples = kg.transferred_triples
        edge_index, edge_type = align_data_processing(triples, args)

    ent_bases = [kg.entity_id_base, kg.upper_entity_base]
    rel_bases = [kg.relation_id_base, kg.upper_relation_base]
    
    return edge_index, edge_type, kg, ent_bases, rel_bases, triples


def compute_alignment_quality(embedding1, embedding2, list1, list2):
    """
    list1 - list2 are valid alignment entities. 
    """
    SCALE_CONSTANT=20
    embeds1 = embedding1[list1]
    embeds2 = embedding2[list2]
    
    simi = torch.mm(embeds1, embeds2.t())

    softmax_simi = torch.softmax(simi * SCALE_CONSTANT, dim=1) # align kg1 to kg2
    entropy1 = - torch.log(softmax_simi) * softmax_simi
    entropy1 = torch.mean(torch.sum(entropy1, dim=1))
    
    softmax_simi2 = torch.softmax(simi.t() * SCALE_CONSTANT, dim=1) # align kg2 to kg1
    entropy2 = - torch.log(softmax_simi2) * softmax_simi2
    entropy2 = torch.mean(torch.sum(entropy2, dim=1))

    entropy = entropy1 + entropy2

    simi = torch.mm(embedding1, embedding2.t())
    donot_sample1 = [ent for ent in range(len(embedding1)) if ent not in list1]
    donot_sample2 = [ent for ent in range(len(embedding2)) if ent not in list2]
    simi[donot_sample1] = -1
    simi[:, donot_sample2] = -1
    softmax_simi = torch.softmax(simi * SCALE_CONSTANT, dim=1)
    softmax_simi2 = torch.softmax(simi.t() * SCALE_CONSTANT, dim=1)

    return entropy, softmax_simi, softmax_simi2
        

def completion_data_processing(triple_list, num_ent, args):
    """
    Prepare a dataloader for the completion-based encoder

    Parameters:
    triple_list: list of KG triples

    Returns:
    -------
    data_loader: dataloader object
    """
    triples = []
    triple_dict = ddict(set)
    for triple in triple_list:
        head, rel, tail = triple[0], triple[1], triple[2]
        triple_dict[(head, rel)].add(tail)
        # triple_dict[(tail , rel + self.num_rel)].add(head) # adding inverse edges
        triples.append([head, rel, tail])
        # triples.append([tail, rel + self.num_rel, head]) # adding inverse edges
        
    triple_dict = {k: list(v) for k, v in triple_dict.items()}
    dataset_class = TrainDataset(triples, num_ent, triple_dict, args.num_negative)
    data_loader = get_data_loader(dataset_class, args.batch_size)
    return data_loader


def get_data_loader(dataset_class, batch_size, shuffle=True):
    return  DataLoader(
            dataset_class,
            batch_size      = batch_size,
            shuffle         = shuffle,
            num_workers     = max(0, 12)
        )    


def transfer_knowledge(triple_list_src, triple_list_dst, links, triple_keys1, triple_keys2, args):
    """
    links: dict links[src] = dst
    """
    inverse_links = {v:k for k,v in links.items()}

    additional_triples_src = []
    additional_triples_dst = []

    for triple in triple_list_src:
        # transfer to target graph!!!
        head, rel, tail = triple[0], triple[1], triple[2]
        if links.get(head) and links.get(tail):
            cur_len = len(triple_keys2)
            triple_keys2.add('{}_{}_{}'.format(links.get(head), rel, links.get(tail)))
            if len(triple_keys2) > cur_len:
                additional_triples_dst.append((links.get(head), rel, links.get(tail)))

    for triple in triple_list_dst:
        head, rel, tail = triple[0], triple[1], triple[2]
        if inverse_links.get(head) and inverse_links.get(tail):
            cur_len = len(triple_keys1)
            triple_keys1.add('{}_{}_{}'.format(inverse_links.get(head), rel, inverse_links.get(tail)))
            if len(triple_keys1) > cur_len:
                additional_triples_src.append((inverse_links.get(head), rel, inverse_links.get(tail)))

    #logger.info("Number of src transfered facts: {}".format(len(additional_triples_src)))
    #logger.info("Number of dst transfered facts: {}".format(len(additional_triples_dst)))
    return triple_list_src + additional_triples_src, triple_list_dst + additional_triples_dst, triple_keys1, triple_keys2


def train_completion_component(step, edge_index1, edge_type1, edge_index2, edge_type2,
            completion_optimizer, JMAC_model, feeddict, completion_data_loader1, completion_data_loader2, args, logger):
    neg_size = args.num_negative
    triple_losses = []
    length = 0
    flag = 0

    for i in range(2):
        completion_data_loader = [completion_data_loader1, completion_data_loader2][i]
        source = [True, False][i]
        for triple, neg in completion_data_loader:
            completion_optimizer.zero_grad()
            this_triple = triple.to(args.device)
            sub, rel, obj = this_triple[:, 0], this_triple[:, 1], this_triple[:, 2]
            if flag == 0:
                length = len(sub)
                flag = 1
            elif len(sub) != length:
                continue
            neg = neg.view(-1)
            neg = torch.LongTensor(neg).to(args.device)
            sub = sub.repeat(neg_size + 1)
            rel = rel.repeat(neg_size + 1)
            tail = torch.cat((obj, neg))
            data = {"batch_h": sub, "batch_r": rel, "batch_t": tail}
            
            loss = JMAC_model.completion_loss(data, edge_index1, edge_type1, edge_index2, edge_type2, feeddict, source)
            
            if loss == 0:
                continue
            loss.backward()
            completion_optimizer.step()
            triple_losses.append(loss.item())   
            del loss
            del data 

        logger.info("Epoch: {}, completion loss : {:.4f}".format(step, np.mean(triple_losses)))
    

def train_alignment_component(step, edge_index1, edge_type1, edge_index2, edge_type2, align_optimizer, JMAC_model, feeddict, logger):
    
    if not len(feeddict["links"]):
        return 0
    
    align_optimizer.zero_grad()
    
    batch_loss = JMAC_model.alignment_loss(feeddict, edge_index1, edge_type1, edge_index2, edge_type2) 
    batch_loss.backward()
    align_optimizer.step()
    logger.info('Epoch: {}, align loss: {:.4f}'.format(step, batch_loss.item()))
    del batch_loss


def main(args, logger):
    '''
    Revised!
    '''
    ########### CPU AND GPU related, Mode related, Dataset Related
    if torch.cuda.is_available() and args.device=='cuda':
        logger.info("Using GPU" + "-" * 80)
        args.device = torch.device("cuda:0")
    else:
        logger.info("Using CPU" + "-" * 80)
        args.device = torch.device("cpu")

    target_lang = args.target_language
    src_langs = get_language_list(args.data_path, logger)
    src_langs.remove(target_lang)

    dataset = ParseData(args, logger)
    kg_object_dict, seeds_train, seeds_test, entity_name_emb = dataset.load_data(noise_rate=args.noise_rate, logger=logger)

    num_relations = dataset.num_relations
    num_entities = dataset.num_entities
    del dataset

    # Build Model
    model = JMAC(args, entity_name_emb, num_relations, num_entities).to(args.device)
    align_optimizer = torch.optim.Adam(model.parameters(), lr=args.align_lr)
    completion_optimizer = torch.optim.Adam(model.parameters(), lr=args.completion_lr)

    logger.info('Model initialization done!')
    completion_evaluator = CompletionEvaluator(kg_object_dict[target_lang], model, args.device, args.data_path)
    
    global_entropies = [-1] * len(seeds_train)
    global_seeds = [None] * len(seeds_train)
    feeddicts = [None] * len(seeds_train)
    completion_dataloader1s = [None] * len(seeds_train)
    completion_dataloader2s = [None] * len(seeds_train)
    edgess1 = [None] * len(seeds_train)
    edgess2 = [None] * len(seeds_train)

    lp_mrr_best = 0
    seen = False
    align_acc = dict()
    for step in tqdm(range(1, args.epoch +1)):
        logger.info(f'Epoch: {step}')
        seed_index = 0
        for (kg1_name, kg2_name) in seeds_train:
            pair_name = '{}_{}'.format(kg1_name, kg2_name)
            if kg1_name == target_lang or kg2_name == target_lang:
                seen = True

            kg1, kg2 = kg_object_dict[kg1_name], kg_object_dict[kg2_name]

            edge_index1, edge_type1, kg1, ent_bases1, rel_bases1, triples1 = process_input_data(kg1, args)
            edge_index2, edge_type2, kg2, ent_bases2, rel_bases2, triples2 = process_input_data(kg2, args)
            
            align_train = seeds_train[(kg1_name, kg2_name)] # Numpy
            align_test = seeds_test[(kg1_name, kg2_name)]
            
            if global_seeds[seed_index] is not None:
                train_align_pairs = global_seeds[seed_index] # Numpy
            else:
                train_align_pairs = align_train
            
            align_test_src = align_test[:, 0].tolist() # Numpy
            align_test_dst = align_test[:, 1].tolist()

            if step == 1 or step % args.val_freq == 1:

                model.eval()                
                output1, _ = model.get_emb(edge_index1, edge_type1, ent_bases1, rel_bases1, pyt=True)
                output2, _ = model.get_emb(edge_index2, edge_type2, ent_bases2, rel_bases2, pyt=True)
                output1_np = output1.cpu().numpy()
                output2_np = output2.cpu().numpy()

                if args.align_percent < 1:
                    logger.info("Alignment acc: {}_{}".format(kg1_name, kg2_name))
                    top_k_al, hits_al, mr_al, mrr_al = test_alignment_(output1_np, output2_np, align_test_src, align_test_dst, args, logger)
                else:
                    logger.info("Full align for training!!!")

                new_triples1, new_triples2, new_triple_keys1, new_triple_keys2, feeddict, global_seeds = seed_enlargement_triple_transferring(output1, output2, align_test_src, align_test_dst, global_entropies, seed_index, \
                            train_align_pairs, triples1, triples2, global_seeds, ent_bases1, rel_bases1, ent_bases2, rel_bases2, kg1, kg2, args)
                
                kg1.triple_keys = new_triple_keys1 
                kg2.triple_keys = new_triple_keys2
                kg1.transferred_triples = new_triples1 # should be list
                kg2.transferred_triples = new_triples2

                completion_dataloader1 = completion_data_processing(new_triples1, kg1.num_entity, args)
                completion_dataloader2 = completion_data_processing(new_triples2, kg2.num_entity, args)
                edge_index1, edge_type1 = align_data_processing(new_triples1, args)
                edge_index2, edge_type2 = align_data_processing(new_triples2, args)
                
                # store for next use
                feeddicts[seed_index] = feeddict
                completion_dataloader1s[seed_index] = completion_dataloader1
                completion_dataloader2s[seed_index] = completion_dataloader2 
                edgess1[seed_index] = [edge_index1, edge_type1]
                edgess2[seed_index] = [edge_index2, edge_type2]

            feeddict = feeddicts[seed_index]
            completion_dataloader1 = completion_dataloader1s[seed_index]
            completion_dataloader2 = completion_dataloader2s[seed_index]
            edge_index1, edge_type1 = edgess1[seed_index]
            edge_index2, edge_type2 = edgess2[seed_index]

            seed_index += 1

            model.train()
            train_completion_component(step, edge_index1, edge_type1, edge_index2, edge_type2, completion_optimizer, 
                    model, feeddict, completion_dataloader1, completion_dataloader2, args, logger)

            train_alignment_component(step, edge_index1, edge_type1, edge_index2, edge_type2, align_optimizer, model, feeddict, logger)
            model.eval()

            if not seen or step % args.val_freq == 0:
                continue

            transferred_triples = kg_object_dict[target_lang].transferred_triples
            edge_index_valid, edge_type_valid = align_data_processing(transferred_triples, args)

            #logger.info(f'=== epoch {step}')
            logger.info(f'[{args.target_language}]')

            model.eval()
            with torch.no_grad():
                logger.info("KG completion on eval set [{}]!!!".format(args.target_language))
                _, _, mrr_val = completion_evaluator.test(args, is_val=True, edge_index_valid=edge_index_valid, edge_type_valid = edge_type_valid, filterr=True, logger=logger)  # Test set
                if mrr_val > lp_mrr_best:
                    lp_mrr_best = mrr_val 
                    logger.info("KG completion on test set [{}]!!!".format(args.target_language))
                    best_hit1, best_hit10, best_mrr = completion_evaluator.test(args, is_val=False, edge_index_valid=edge_index_valid, edge_type_valid = edge_type_valid, filterr=True, logger=logger)  # Test set
                    best_epoch = step

                    align_acc[pair_name] = {'topk': top_k_al, 'hits': hits_al, 'mr': mr_al, 'mrr': mrr_al, 'num_pairs': len(seeds_test[(kg1_name, kg2_name)])}

                    if len(align_acc) == len(seeds_train):
                        num_all_pairs = 0
                        all_hits = 0
                        all_mr = 0
                        all_mrr = 0
                        logger.info("Overall Alignment:")
                        for p in align_acc:
                            this_value = align_acc[p]
                            num_all_pairs += this_value['num_pairs']
                            topk = this_value['topk']
                            all_hits += this_value['hits'] * this_value['num_pairs']
                            all_mr += this_value['mr'] * this_value['num_pairs']
                            all_mrr += this_value['mrr'] * this_value['num_pairs']
                    
                        all_hits = all_hits / num_all_pairs
                        all_mr = all_mr / num_all_pairs
                        all_mrr = all_mrr / num_all_pairs
                    
                        logger.info("Hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}".
                                    format(topk, all_hits, all_mr, all_mrr))

                logger.info("Best epoch: {}, Hits1: {:.4f}, Hits10: {:.4f}, MRR: {:.4f}".format(best_epoch, best_hit1, best_hit10, best_mrr))
        
                

if __name__ == "__main__":
    args = parse_args()
    logger = Logger(args.target_language, logging.INFO, True, True)
    logger.info("hello")
    main(args, logger)
