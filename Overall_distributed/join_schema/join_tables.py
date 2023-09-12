import copy
import logging
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

def get_equivalent_key_group(join_keys, equivalent_keys):
    equivalent_group = dict()
    table_equivalent_group = dict()
    table_key_equivalent_group = dict()
    table_key_group_map = dict()
    for table in join_keys:
        for key in join_keys[table]:
            seen = False
            for indicator in equivalent_keys:
                if key in equivalent_keys[indicator]:
                    if seen:
                        assert False, f"{key} appears in multiple equivalent groups."
                    if indicator not in equivalent_group:
                        equivalent_group[indicator] = [key]
                    else:
                        equivalent_group[indicator].append(key)
                    if table not in table_key_equivalent_group:
                        table_key_equivalent_group[table] = dict()
                        table_equivalent_group[table] = set([indicator])
                        table_key_group_map[table] = dict()
                        table_key_group_map[table][key] = indicator
                    else:
                        table_equivalent_group[table].add(indicator)
                        table_key_group_map[table][key] = indicator
                    if indicator not in table_key_equivalent_group[table]:
                        table_key_equivalent_group[table][indicator] = [key]
                    else:
                        table_key_equivalent_group[table][indicator].append(key)

                    seen = True
            if not seen:
                assert False, f"no equivalent groups found for {key}."
    return equivalent_group, table_equivalent_group, table_key_equivalent_group, table_key_group_map

'''
def process_condition_join(cond, tables_all):
    start = None
    join = False
    join_keys = {}
    for i in range(len(cond)):
        s = cond[i]
        if s == "=":
            start = i
            if cond[i + 1] == "=":
                end = i + 2
            else:
                end = i + 1
            break

    if start is None:
        return None, None, False, None

    left = cond[:start].strip()
    ops = cond[start:end].strip()
    right = cond[end:].strip()
    table1 = left.split(".")[0].strip().lower()
    if table1 in tables_all:
        left = tables_all[table1] + "." + left.split(".")[-1].strip()
    else:
        return None, None, False, None
    if "." in right:
        table2 = right.split(".")[0].strip().lower()
        if table2 in tables_all:
            right = tables_all[table2] + "." + right.split(".")[-1].strip()
            join = True
            join_keys[table1] = left
            join_keys[table2] = right
            return table1 + " " + table2, cond, join, join_keys
    return None, None, False, None
'''

def build_join_graph(schema):
    '''
    L = [(0,1), (1,2)]
    all_pairs_shortest_path = {0: {0: [0], 1: [0, 1], 2: [0, 1, 2]}, 
                                1: {0: [1, 0], 1: [1], 2: [1, 2]}, 
                                2: {0: [2, 1, 0], 1: [2, 1], 2: [2]}}
    all_pair_list = [(0,1,[0, 1]), (1,2,[1, 2]), (0,2,[0, 1, 2])]
    join_graph = [(0,1,[(0, 1)]), (1,2,[(1, 2)]), (0,2,[(0, 1),(1, 2)])]
    '''
    # build graph from schema
    table_index_dict = {table.table_name: i for i, table in enumerate(schema.tables)}
    inverse_table_index_dict = {table_index_dict[k]: k for k in table_index_dict.keys()}
    G = nx.Graph()
    for relationship in schema.relationships:
        start_idx = table_index_dict[relationship.start]
        end_idx = table_index_dict[relationship.end]
        G.add_edge(start_idx, end_idx, relationship=relationship)

    # iterate over pairs
    all_pairs = dict(nx.all_pairs_shortest_path(G))
    all_pair_list = []
    for left_idx, right_idx_dict in all_pairs.items():
        for right_idx, shortest_path_list in right_idx_dict.items():
            if left_idx >= right_idx:
                continue
            all_pair_list.append((left_idx, right_idx, shortest_path_list,))
            
    # sort by length of path
    join_graph = []
    all_pair_list.sort(key=lambda x: len(x[2]))
    for left_idx, right_idx, shortest_path_list in all_pair_list:
        left_table = inverse_table_index_dict[left_idx]
        right_table = inverse_table_index_dict[right_idx]
        logger.debug(f"Evaluating {left_table} and {right_table}")
        path_list = [G[shortest_path_list[i]][shortest_path_list[i + 1]]['relationship'].identifier
                             for i in range(len(shortest_path_list) - 1)]
        join_graph.append((left_table, right_table, path_list,))
    
    return join_graph

class JoinTree:
    def __init__(self, values=None, relationship=None) -> None:
        self.values = values
        self.relationship = relationship
        self.left = None
        self.right = None
        
    def insert_left(self, branch):
        if self.left == None:
            self.left = JoinTree(branch)
        else:
            t = JoinTree(branch)
            t.left = self.left
            self.left = t
    
    def insert_right(self, branch):
        if self.right == None:
            self.right = JoinTree(branch)
        else:
            t = JoinTree(branch)
            t.right = self.rightt
            self.right = t
            
    def add_branch_left(self, branch):
        if self.left == None:
            self.left = copy.copy(branch)
        else:
            raise ValueError("something wrong when insert left!")
            self.left.add_branch_left(branch)
            
    def add_branch_right(self, branch):
        if self.right == None:
            self.right = copy.copy(branch)
        else:
            raise ValueError("something wrong when insert right!")
            self.right.add_branch_right(branch)
            
    def get_left(self):
        return self.left
    
    def get_right(self):
        return self.right
    
    def set_values(self, obj):
        self.values = obj
    
    def get_values(self):
        return self.values   
    
    def get_relationship(self):
        return self.relationship