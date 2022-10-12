from modules.load.kg import KG
from modules.load.read import *


class KGs:
    def __init__(self, kg1: KG, kg2: KG, train_links, test_links, valid_links=None, mode='mapping', ordered=True, linkpred=False):
        ent_ids1, ent_ids2, self.source_id2idx, self.target_id2idx = generate_mapping_id(kg1.relation_triples_set, kg1.entities_set,
                                                    kg2.relation_triples_set, kg2.entities_set, ordered=ordered, id2idx=True)
        rel_ids1, rel_ids2 = generate_mapping_id(kg1.relation_triples_set, kg1.relations_set,
                                                    kg2.relation_triples_set, kg2.relations_set, ordered=ordered)
        attr_ids1, attr_ids2 = generate_mapping_id(kg1.attribute_triples_set, kg1.attributes_set,
                                                    kg2.attribute_triples_set, kg2.attributes_set, ordered=ordered)

        id_relation_triples1 = uris_relation_triple_2ids(kg1.relation_triples_set, ent_ids1, rel_ids1)
        id_relation_triples2 = uris_relation_triple_2ids(kg2.relation_triples_set, ent_ids2, rel_ids2)

        id_attribute_triples1 = uris_attribute_triple_2ids(kg1.attribute_triples_set, ent_ids1, attr_ids1)
        id_attribute_triples2 = uris_attribute_triple_2ids(kg2.attribute_triples_set, ent_ids2, attr_ids2)

        self.uri_kg1 = kg1
        self.uri_kg2 = kg2

        kg1 = KG(id_relation_triples1, id_attribute_triples1)
        kg2 = KG(id_relation_triples2, id_attribute_triples2)
        kg1.set_id_dict(ent_ids1, rel_ids1, attr_ids1) # dict
        kg2.set_id_dict(ent_ids2, rel_ids2, attr_ids2) # dict

        self.uri_train_links = train_links
        self.uri_test_links = test_links
        self.train_links = uris_pair_2ids(self.uri_train_links, ent_ids1, ent_ids2)
        self.test_links = uris_pair_2ids(self.uri_test_links, ent_ids1, ent_ids2)

        self.train_entities1 = [link[0] for link in self.train_links]
        self.train_entities2 = [link[1] for link in self.train_links]
        self.test_entities1 = [link[0] for link in self.test_links]
        self.test_entities2 = [link[1] for link in self.test_links]

        self.kg1 = kg1
        self.kg2 = kg2

        self.valid_links = list()
        self.valid_entities1 = list()
        self.valid_entities2 = list()
        if valid_links is not None:
            self.uri_valid_links = valid_links
            self.valid_links = uris_pair_2ids(self.uri_valid_links, ent_ids1, ent_ids2)
            self.valid_entities1 = [link[0] for link in self.valid_links]
            self.valid_entities2 = [link[1] for link in self.valid_links]

        self.useful_entities_list1 = self.kg1.entities_list
        self.useful_entities_list2 = self.kg2.entities_list

        self.entities_num = len(self.kg1.entities_set | self.kg2.entities_set)
        self.relations_num = len(self.kg1.relations_set | self.kg2.relations_set)
        self.attributes_num = len(self.kg1.attributes_set | self.kg2.attributes_set)

def read_kgs_from_folder(training_data_folder, division, mode, ordered, linkpred=False, logger=None):
    '''
    Revised!
    '''
    kg1_path_train = training_data_folder + 'rel_triples_1'
    kg2_path_train = training_data_folder + 'rel_triples_2'

    kg1_relation_triples, _, _ = read_relation_triples(kg1_path_train, logger) # name_relation_triples
    kg2_relation_triples, _, _ = read_relation_triples(kg2_path_train, logger) # name_relation_triples
    kg1_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_1', logger) # name_attribute_triples
    kg2_attribute_triples, _, _ = read_attribute_triples(training_data_folder + 'attr_triples_2', logger) # name_attribute_triples

    # read groundtruth
    train_links = read_links(training_data_folder + division + 'train_links', logger)
    valid_links = read_links(training_data_folder + division + 'valid_links', logger)
    test_links = read_links(training_data_folder + division + 'test_links', logger)

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)

    kgs = KGs(kg1, kg2, train_links, test_links, valid_links=valid_links, mode=mode, ordered=ordered, linkpred=linkpred)
    return kgs
