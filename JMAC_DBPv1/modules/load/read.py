import os
import numpy as np


def load_embeddings(file_name):
    if os.path.exists(file_name):
        return np.load(file_name)
    return None


def sort_elements(triples, elements_set):
    dic = dict()
    for s, p, o in triples:
        if s in elements_set:
            dic[s] = dic.get(s, 0) + 1
        if p in elements_set:
            dic[p] = dic.get(p, 0) + 1
        if o in elements_set:
            dic[o] = dic.get(o, 0) + 1
    # set the frequency of other entities that have no relation triples to zero
    for e in elements_set:
        if e not in dic:
            dic[e] = 0
    # firstly sort by values (i.e., frequencies), if equal, by keys (i.e, URIs)
    sorted_list = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ordered_elements = [x[0] for x in sorted_list]
    assert len(dic) == len(elements_set)
    return ordered_elements, dic


def generate_sharing_id(train_links, kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
    """
    TODO: what is this: 
    mark id to entities names. But linked entities have same ids which are even
    unlinked entites still have ids but it will be odd
    OUTPUT: ids1 (dict: name2id); ids2 (dict: name2id)
    """
    ids1, ids2 = dict(), dict()
    if ordered:
        linked_dic = dict()
        for x, y in train_links:
            linked_dic[y] = x # linked_dic[target] = source
        kg2_linked_elements = [x[1] for x in train_links] # target entities appear in gt
        kg2_unlinked_elements = set(kg2_elements) - set(kg2_linked_elements) # redundent entities
        ids1, ids2 = generate_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_unlinked_elements, ordered=ordered)
        for ele in kg2_linked_elements:
            ids2[ele] = ids1[linked_dic[ele]] # same id for linked entities; which is even
    else:
        index = 0
        for e1, e2 in train_links:
            assert e1 in kg1_elements
            assert e2 in kg2_elements
            ids1[e1] = index
            ids2[e2] = index
            index += 1
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                index += 1
        for ele in kg2_elements:
            if ele not in ids2:
                ids2[ele] = index
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    assert len(ids2) == len(set(kg2_elements))
    return ids1, ids2


def generate_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True, id2idx=False):
    """
    ids1: even
    ids2: odd
    return ids1; ids2 (dict: name2id)
    """
    ids1, ids2 = dict(), dict()
    source_id2idx = {}
    target_id2idx = {}
    if ordered:
        kg1_ordered_elements, _ = sort_elements(kg1_triples, kg1_elements)
        kg2_ordered_elements, _ = sort_elements(kg2_triples, kg2_elements)
        n1 = len(kg1_ordered_elements)
        n2 = len(kg2_ordered_elements)
        n = max(n1, n2)

        index_source = 0
        index_target = 0

        for i in range(n):
            if i < n1 and i < n2:
                ids1[kg1_ordered_elements[i]] = i * 2
                source_id2idx[i*2] = index_source
                ids2[kg2_ordered_elements[i]] = i * 2 + 1
                target_id2idx[i*2 + 1] = index_target
                index_source += 1
                index_target += 1
            elif i >= n1:
                ids2[kg2_ordered_elements[i]] = n1 * 2 + (i - n1)
                target_id2idx[n1 * 2 + (i - n1)] = index_target
                index_target += 1
            else:
                ids1[kg1_ordered_elements[i]] = n2 * 2 + (i - n2)
                source_id2idx[n2 * 2 + (i - n2)] = index_source
                index_source += 1
    else:
        index = 0
        for ele in kg1_elements:
            if ele not in ids1:
                ids1[ele] = index
                source_id2idx[index] = index
                index += 1
        maxx = index
        for ele in kg2_elements:
            if ele not in ids2:
                ids2[ele] = index
                target_id2idx[index] = index - maxx
                index += 1
    assert len(ids1) == len(set(kg1_elements))
    assert len(ids2) == len(set(kg2_elements))
    if id2idx:
        return ids1, ids2, source_id2idx, target_id2idx
    return ids1, ids2


def uris_list_2ids(uris, ids):
    id_uris = list()
    for u in uris:
        assert u in ids
        id_uris.append(ids[u])
    assert len(id_uris) == len(set(uris))
    return id_uris


def uris_pair_2ids(uris, ids1, ids2):
    """
    name_gt --> id_gt
    """
    id_uris = list()
    for u1, u2 in uris:
        # assert u1 in ids1
        # assert u2 in ids2
        if u1 in ids1 and u2 in ids2:
            id_uris.append((ids1[u1], ids2[u2]))
    # assert len(id_uris) == len(set(uris))
    return id_uris


def uris_relation_triple_2ids(uris, ent_ids, rel_ids):
    """
    what does this function do?
    Return triple ids
    """
    id_uris = list()
    for u1, u2, u3 in uris:
        assert u1 in ent_ids
        assert u2 in rel_ids
        assert u3 in ent_ids
        id_uris.append((ent_ids[u1], rel_ids[u2], ent_ids[u3]))
    assert len(id_uris) == len(set(uris))
    return id_uris


def uris_attribute_triple_2ids(uris, ent_ids, attr_ids):
    id_uris = list()
    for u1, u2, u3 in uris:
        assert u1 in ent_ids
        assert u2 in attr_ids
        id_uris.append((ent_ids[u1], attr_ids[u2], u3))
    assert len(id_uris) == len(set(uris))
    return id_uris
    
    

def read_relation_triples(file_path, logger):
    logger.info("read relation triples: {}".format(file_path))
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, relations = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 3
        h = params[0].strip()
        r = params[1].strip()
        t = params[2].strip()
        triples.add((h, r, t))
        entities.add(h)
        entities.add(t)
        relations.add(r)
    return triples, entities, relations


def read_links(file_path, logger):
    logger.info("read links: {}".format(file_path))
    links = list()
    refs = list()
    reft = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        e1 = params[0].strip()
        e2 = params[1].strip()
        refs.append(e1)
        reft.append(e2)
        links.append((e1, e2))
    assert len(refs) == len(reft)
    return links


def read_dict(file_path):
    file = open(file_path, 'r', encoding='utf8')
    ids = dict()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        ids[params[0]] = int(params[1])
    file.close()
    return ids


def read_pair_ids(file_path):
    file = open(file_path, 'r', encoding='utf8')
    pairs = list()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        pairs.append((int(params[0]), int(params[1])))
    file.close()
    return pairs


def pair2file(file, pairs):
    if pairs is None:
        return
    with open(file, 'w', encoding='utf8') as f:
        for i, j in pairs:
            f.write(str(i) + '\t' + str(j) + '\n')
        f.close()


def radio_2file(radio, folder):
    path = folder + str(radio).replace('.', '_')
    if not os.path.exists(path):
        os.makedirs(path)
    return path + '/'


def embed2file(results_folder, file_name, embedding, kg1_id_dict, kg2_id_dict, seperate=True):
    if embedding is None or kg1_id_dict is None or kg2_id_dict is None:
        return
    if seperate:
        with open(results_folder + 'kg1_' + file_name, 'w', encoding='utf8') as f:
            for entity_uri, entity_index in kg1_id_dict.items():
                f.write(str(entity_uri) + ' ' + ' '.join(map(str, embedding[entity_index])) + '\n')
        with open(results_folder + 'kg2_' + file_name, 'w', encoding='utf8') as f:
            for entity_uri, entity_index in kg2_id_dict.items():
                f.write(str(entity_uri) + ' ' + ' '.join(map(str, embedding[entity_index])) + '\n')
    else:
        with open(results_folder + 'combined_' + file_name, 'w', encoding='utf8') as f:
            for entity_uri, entity_index in kg1_id_dict.items():
                f.write(str(entity_uri) + ' ' + ' '.join(map(str, embedding[entity_index])) + '\n')
            for entity_uri, entity_index in kg2_id_dict.items():
                f.write(str(entity_uri) + ' ' + ' '.join(map(str, embedding[entity_index])) + '\n')

def read_attribute_triples(file_path, logger):
    if file_path is None:
        return set(), set(), set()
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, attributes = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip().strip('\n').split('\t')
        if len(params) < 3:
            continue
        head = params[0].strip()
        attr = params[1].strip()
        value = params[2].strip()
        if len(params) > 3:
            for p in params[3:]:
                value = value + ' ' + p.strip()
        value = value.strip().rstrip('.').strip()
        entities.add(head)
        attributes.add(attr)
        triples.add((head, attr, value))
    return triples, entities, attributes


if __name__ == '__main__':
    mydict = {'b': 10, 'c': 10, 'a': 10, 'd': 20}
    sorted_dic = sorted(mydict.items(), key=lambda x: (x[1], x[0]), reverse=True)
