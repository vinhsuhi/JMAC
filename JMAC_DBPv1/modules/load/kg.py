import numpy as np 
from collections import defaultdict as ddict


def parse_triples(triples):
    subjects, predicates, objects = set(), set(), set()
    for s, p, o in triples:
        subjects.add(s)
        predicates.add(p)
        objects.add(o)
    return subjects, predicates, objects


class KG:
    def __init__(self, relation_triples):
        """
        This class's objects will be defined in KGs class
        relation_triples: ids_version
        """

        self.entities_set, self.entities_list = None, None
        self.relations_set, self.relations_list = None, None # set and list of relations
        self.attributes_set, self.attributes_list = None, None
        self.entities_num, self.relations_num, self.attributes_num = None, None, None
        self.relation_triples_num, self.attribute_triples_num = None, None
        self.local_relation_triples_num, self.local_attribute_triples_num = None, None # local num and num are equal

        self.entities_id_dict = None # name2id
        self.relations_id_dict = None # name2id
        self.attributes_id_dict = None # name2id

        self.rt_dict, self.hr_dict = None, None # self.rt_dict; show all neighbors of head; (r1, t1), (r2, t2)
        self.entity_relations_dict = None; # self.entity_relations_dict: contains set of relations related to head; A[head]= {r1, r2,...}
        self.entity_attributes_dict = None # self.entity_attribute_dict: contains set of attribute related to head; A[head]= {a1, a2,...}
        self.av_dict = None # show all neighbor of entity; (a1, v1), (a2, v2)

        self.sup_relation_triples_set, self.sup_relation_triples_list = None, None
        self.sup_attribute_triples_set, self.sup_attribute_triples_list = None, None

        self.relation_triples_set = None
        self.attribute_triples_set = None
        self.relation_triples_list = None
        self.attribute_triples_list = None

        self.local_relation_triples_set = None # to store relation_triples ids without any changes (set)
        self.local_relation_triples_list = None # to store relation_triples ids without any changes (list)
        self.local_attribute_triples_set = None # 
        self.local_attribute_triples_list = None

        self.lp_valid = None
        self.lp_test = None
        self.lp_valid_all = None
        self.lp_test_all = None
    
        self.set_relations(relation_triples)
        #self.set_attributes(attribute_triples)

        

    def update_linkpred_info(self, valid, test):
        self.lp_valid = set(valid)
        self.lp_test = set(test)

        
        

    def create_er_vocab(self):
        er_vocab = ddict(set)
        for triple in self.relation_triples_set:
            h, r, t = triple
            er_vocab[(h, r)].add(t)
        for triple in self.lp_valid:
            h, r, t = triple 
            er_vocab[(h, r)].add(t)
        for triple in self.lp_test:
            h, r, t = triple 
            er_vocab[(h, r)].add(t)
        self.er_vocab = er_vocab


    def set_relations(self, relation_triples):
        """
        relation_triples: list of relation_triple ids
        """

        self.relation_triples_set = set(relation_triples)
        self.relation_triples_list = list(self.relation_triples_set)
        self.local_relation_triples_set = self.relation_triples_set
        self.local_relation_triples_list = self.relation_triples_list

        # extract the set of heads, relations, and tails
        heads, relations, tails = parse_triples(self.relation_triples_set)
        self.entities_set = heads | tails

        self.relations_set = relations
        self.entities_list = list(self.entities_set)
        self.relations_list = list(self.relations_set)
        self.entities_num = len(self.entities_set)
        self.relations_num = len(self.relations_set)
        self.relation_triples_num = len(self.relation_triples_set)
        self.local_relation_triples_num = len(self.local_relation_triples_set)
        self.generate_relation_triple_dict()
        self.parse_relations()

    def set_attributes(self, attribute_triples):
        """
        at
        """
        self.attribute_triples_set = set(attribute_triples)
        self.attribute_triples_list = list(self.attribute_triples_set)
        self.local_attribute_triples_set = self.attribute_triples_set
        self.local_attribute_triples_list = self.attribute_triples_list

        entities, attributes, values = parse_triples(self.attribute_triples_set)
        self.attributes_set = attributes
        self.attributes_list = list(self.attributes_set)
        self.attributes_num = len(self.attributes_set)

        # add the new entities from attribute triples
        self.entities_set |= entities
        self.entities_list = list(self.entities_set)
        self.entities_num = len(self.entities_set)

        self.attribute_triples_num = len(self.attribute_triples_set)
        self.local_attribute_triples_num = len(self.local_attribute_triples_set)
        self.generate_attribute_triple_dict() # return self.av_dict: contains set of (a, v) of each entity
        self.parse_attributes() # return self.entity_relations_dict: contains set of relations related to head; A[head]= {r1, r2,...}

    def generate_relation_triple_dict(self):
        """
        self.rt_dict = {head: {(r1, t1), (r2, t2)}}
        self.hr_dict = {tail: {(h1, r1), (h2, r2)}}
        """
        self.rt_dict, self.hr_dict = dict(), dict()
        for h, r, t in self.local_relation_triples_list:
            rt_set = self.rt_dict.get(h, set())
            rt_set.add((r, t))
            self.rt_dict[h] = rt_set
            hr_set = self.hr_dict.get(t, set())
            hr_set.add((h, r))
            self.hr_dict[t] = hr_set

    def generate_attribute_triple_dict(self):
        """
        self.av_dict
        """
        self.av_dict = dict()
        for h, a, v in self.local_attribute_triples_list:
            av_set = self.av_dict.get(h, set())
            av_set.add((a, v))
            self.av_dict[h] = av_set

    def parse_relations(self):
        """
        Return: A[ent] = {r1, r2...} (relation set)
        """
        self.entity_relations_dict = dict()
        for ent, attr, _ in self.local_relation_triples_set:
            attrs = self.entity_relations_dict.get(ent, set())
            attrs.add(attr)
            self.entity_relations_dict[ent] = attrs

    def parse_attributes(self):
        """
        Return: A[ent] = {a1, a2,...} (attribute set)
        """
        self.entity_attributes_dict = dict()
        for ent, attr, _ in self.local_attribute_triples_set:
            attrs = self.entity_attributes_dict.get(ent, set())
            attrs.add(attr)
            self.entity_attributes_dict[ent] = attrs

    def set_id_dict(self, entities_id_dict, relations_id_dict, attributes_id_dict):
        """
        is called in KGs.__init__(); just simple attach name2ids
        """
        self.entities_id_dict = entities_id_dict
        self.relations_id_dict = relations_id_dict
        self.attributes_id_dict = attributes_id_dict
