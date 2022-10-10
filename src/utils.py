import numpy as np
import torch
import pandas as pd
from os.path import join
import os
import logging



def save_model(model, output_dir, filename, args):
    """
    Save the trained knowledge model under output_dir. Filename: 'language.h5'
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save the weights for the whole model
    ckpt_path = os.path.join(output_dir, filename)
    torch.save({
        'state_dict': model.state_dict(),
        'args': args,
    }, ckpt_path)

def get_negative_samples_alignment(batch_size_each, num_entity,num_negative=None):
    '''
    Generate one negative sample
    :param batch_size_each:
    :param num_entity:
    :return:
    '''
    if num_negative == None:
        rand_negs = torch.randint(high=num_entity, size=(batch_size_each,))  # [b,n]
    else:
        rand_negs = torch.randint(high=num_entity, size=(batch_size_each,num_negative))  # [b,n]

    return rand_negs


def load_model(ckpt_path, model, device):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    # Load checkpoint.
    checkpt = torch.load(ckpt_path)
    ckpt_args = checkpt['args']
    state_dict = checkpt['state_dict']
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # 3. load the new state dict
    model.load_state_dict(state_dict)
    model.to(device)


def get_negative_samples_graph(batch_size_each, num_entity):
    '''
    Generate one negative samaple
    :param batch_size_each:
    :param num_entity:
    :return:
    '''
    rand_negs = torch.randint(high=num_entity, size=(batch_size_each,))  # [b,1]

    return rand_negs



def Ranking_all_batch(predicted_t, embedding_matrix, k = None):
    '''
    Compute k-nearest neighbors in a batch
    If k== None, return ranked all candidatees
    otherwise, return top_k candidates
    :param predicted_t:
    :param embedding_matrix:
    :param k:
    :return:
    '''

    total_entity = embedding_matrix.shape[0]
    predicted_t = torch.unsqueeze(predicted_t, dim=1)  # [b,1,d]
    predicted_t = predicted_t.repeat(1, total_entity, 1)  # [b,n,d]

    distance = torch.norm(predicted_t - embedding_matrix, dim=2)  # [b,n]

    if k==None:
        k = total_entity

    top_k_scores, top_k_t = torch.topk(-distance, k=k)
    return top_k_t, top_k_scores




def get_language_list(data_dir, logger):
    entity_dir = data_dir + "/entity"
    entity_files = list(os.listdir(entity_dir))
    entity_files = list(filter(lambda x: x[-3:] == "tsv", entity_files))
    entity_files = sorted(entity_files)
    logger.info("Number of KGs is %d" % len(entity_files))

    kg_names = []
    for each_entity_file in entity_files:
        kg_name_each = each_entity_file[:2]
        kg_names.append(kg_name_each)

    return kg_names




def get_kg_edges_for_each(data_dir, language, is_target_KG=False):
    '''
    TODO: whether include directional edges (1. do not incorporate. 2. adding the numbe of relation embeddings)
    :param data_dir:
    :param language:
    :param is_target_KG:
    :return:
    '''
    train_df = pd.read_csv(join(data_dir, language + '-train.tsv'), sep='\t', header=None,
                           names=['v1', 'relation', 'v2'])

    val_df = pd.read_csv(join(data_dir, language + '-val.tsv'), sep='\t', header=None,
                         names=['v1', 'relation', 'v2'])

    # Training data graph construction
    sender_node_list = train_df['v1'].values.astype(np.int).tolist()
    sender_node_list += train_df['v2'].values.astype(np.int).tolist()

    receiver_node_list = train_df['v2'].values.astype(np.int).tolist()
    receiver_node_list += train_df['v1'].values.astype(np.int).tolist()

    edge_weight_list = train_df['relation'].values.astype(np.int).tolist() + train_df['relation'].values.astype(
        np.int).tolist()

    # unified: Adding validation edges from supporter KG as well
    if not is_target_KG:
        sender_node_list += val_df['v1'].values.astype(np.int).tolist()
        sender_node_list += val_df['v2'].values.astype(np.int).tolist()

        receiver_node_list += val_df['v2'].values.astype(np.int).tolist()
        receiver_node_list += val_df['v1'].values.astype(np.int).tolist()

        edge_weight_list += val_df['relation'].values.astype(np.int).tolist()
        edge_weight_list += val_df['relation'].values.astype(np.int).tolist()

    edge_index = np.vstack((sender_node_list, receiver_node_list))
    edge_weight = np.asarray(edge_weight_list)
    return edge_index, edge_weight


def get_graph(data_dir, language, is_target_KG):
    edge_index, edge_type = get_kg_edges_for_each(data_dir + "/kg", language, is_target_KG=is_target_KG)
    return edge_index, edge_type

