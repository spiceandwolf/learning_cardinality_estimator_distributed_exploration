from collections import deque
import copy
import logging
import math

import numpy as np
from join_schema.join_tables import JoinTree

from myutils.my_utils import parse_conditions, parse_query

logger = logging.getLogger(__name__)

class Cauchy_estimator:
    '''
    Assumption is that is built over acyclic graph for now.
    
    为了适用于分布式环境, 最小化传输开销
    利用cauchy不等式估计join size的一个上界
    利用factorjoin的idea
    '''
    
    def __init__(self, schema_graph, join_graph):
        self.schema_graph = schema_graph
        # networkX
        self.join_graph = join_graph
    
    def get_join_size_estimation(self, query, base_estimators, column_min_max_vals, bins, alias2table = {'cast_info': 'ci', 'movie_companies': 'mc', 'movie_info':'mi', 'movie_keyword': 'mk',
                   'movie_info_idx': 'mi_idx', 'title': 't'}):
        '''
        bins 每个属性的取值间隔
        1.解析query, 获取tables, join keys和predicates
        2.对每个table的predicates编码
        3.根据predicates获取有关partition的base_estimator
        4.根据所有base_estimator的估计结果计算各级join的size的上界
        '''
        # analysis query
        query_items = parse_query(query, self.schema_graph)
        # print('query:\n')  #重要
        # print(query_items)  #重要
        table_set = query_items.table_set
        relationship_set = query_items.relationship_set
        table_condition_dict = query_items.table_where_condition_dict
        cached_sub_queries = dict()
        
        # encode
        domain_info = {}
        for table in table_set:
            left_bounds = {}
            right_bounds = {}
            
            table_obj = self.schema_graph.table_dictionary[table]
            # drop irrelevant attributes
            encoding_attrs = [attr for attr in table_obj.attributes if attr not in table_obj.irrelevant_attributes]
            # 初始化每个属性的上下界
            for attr in encoding_attrs:
                table_attr = f'{alias2table[table]}.{attr}' 
                left_bounds[attr] = column_min_max_vals[table_attr][0]
                right_bounds[attr] = column_min_max_vals[table_attr][1]
            # 每个表中每个属性对应的条件 
            conditions = table_condition_dict[table]
            # 对conditions进行正则化等处理
            # 根据conditions调整每个属性的积分上下界
            domain_info[table] = parse_conditions(conditions, left_bounds, right_bounds, bins)
             
        # get corresponding partion estimators
        logger.info(f'domain_info:{domain_info}')
        estimators = self.get_join_estimators(table_set, domain_info, base_estimators)
        
        # 根据query的join condition建立连接树(左深树形式)
        join_trees = self.get_join_tree(table_set, relationship_set)
        
        # compute upper bound
        res = []
        
        for join_tree in join_trees:
            
            res.append(self.compute_upper_bound(join_tree, domain_info, bins, estimators))
            
        return res
            
    def get_join_tree(self, table_set, relationship_set):
        '''
        identifier的两个表都没见过会产生新的item左序遍历(leaf, branch, leaf)
        其中一个见过产生, 和被包含的branch产生新的item左序遍历(branch, branch', leaf)
        两个都见过说明图中有环, 暂不考虑
        每有一个identifier的两表是都没见过时, 就应该新添加一个seen_tables存储这两个表
        当有一个identifier的两表分别在两个不同的seen_tables中时
        将seen_tables多的join tree作为左子树, seen_tables少的join tree作为右子树
        生成一个新的join tree并合并seen_tables
        '''
        
        join_trees = []
        seen_list = []
    
        for identifier in relationship_set:
            relationship = self.schema_graph.relationship_dictionary[identifier]
            left_table = relationship.start
            start_attr = relationship.start_attr
            right_table = relationship.end
            end_attr = relationship.end_attr
            
            cur_tables = {left_table, right_table}
            repeated_Num = []
            
            # 通常是两个不同的表
            if len(cur_tables) == 2:
                # 比较每个join tree的seen_tables
                for i, seen_tables in enumerate(seen_list):
                    res = cur_tables.intersection(seen_tables)
                    
                    # 交集有2个说明这两个表在同一个join tree, 说明成环
                    if len(res) == 2:
                        repeated_Num.append((res[0], i))
                        repeated_Num.append((res[1], i))
                        break
                    
                    # 交集有1个，记录对应join tree的位置，用于稍后的合并
                    elif len(res) == 1:
                        repeated_Num.append((res, i))
                        
                    # 交集有0个, 说明不在这个join tree中
                    elif len(res) == 0:
                        continue
                        
            # 说明identifier的两个表在2个不同的join tree中或一个join tree中有环
            if len(repeated_Num) == 2:
                _, first_index = repeated_Num[0]
                _, second_index = repeated_Num[1]
                left_index = first_index if len(seen_list[first_index]) > len(seen_list[second_index]) else second_index
                right_index = first_index if left_index == second_index else second_index
                
                # 合并这两个join tree的seen_tables
                seen_list[left_index].update(seen_list[right_index])
                seen_list[left_index].update(cur_tables)
                
                # 除去已合并的seen_tables
                seen_list.pop(right_index)
                
                # 生成新节点并将这2个join tree合并为左右子树
                branch = JoinTree(seen_list[left_index], relationship)
                branch.add_branch_left(join_trees[left_index])
                branch.add_branch_right(join_trees[right_index])
                
                # 除去join_trees中已合并的join_tree, 并更新join_trees
                join_trees.pop(right_index)
                join_trees[left_index] = branch
                
            # 说明identifier的某个表已在某个join tree中，另一个表是新的
            elif len(repeated_Num) == 1:
                seen_table, seen_index = repeated_Num
                new_table = left_table if seen_table == right_table else right_table
                
                # 合并被选中的join tree的seen_tables和新的表
                seen_list[seen_index].add(new_table)
                
                # 生成新节点并将原join tree和新表添加为左右子树
                branch = JoinTree(seen_list[seen_index], relationship)
                branch.add_branch_left(join_trees[seen_index])
                leaf = JoinTree({new_table})
                branch.add_branch_right(leaf)
                
                # 更新join_trees
                join_trees[seen_index] = branch
                
            # 说明identifier的两个表都是新表
            elif len(repeated_Num) == 0:
                # 生成新节点
                branch = JoinTree(cur_tables, relationship)
                left_leaf = JoinTree({left_table})
                right_leaf = JoinTree({right_table})
                
                # 添加对应的左右叶子节点
                branch.add_branch_left(left_leaf)
                branch.add_branch_right(right_leaf)
                
                # 更新join_trees和seen_list
                join_trees.append(branch)
                seen_list.append(cur_tables)       
        
        # 如果join tree的数量大于1，说明这个查询的关系之间不是联通的
        logger.info(f"查询生成{len(join_trees)}个join tree(s)")
                
        return join_trees 

    def get_join_estimators(self, table_set, domain_info, base_estimators, alias2table = {'cast_info': 'ci', 'movie_companies': 'mc', 'movie_info':'mi', 'movie_keyword': 'mk',
                   'movie_info_idx': 'mi_idx', 'title': 't'}):
        '''
        根据attrs的范围选择会被使用的partition
        '''
        selected_estimators = {}
        for table in table_set:
            left_bounds, right_bounds = domain_info[table]
            for estimator in base_estimators[table]:
                table_obj = self.schema_graph.table_dictionary[table]
                # drop irrelevant attributes
                encoding_attrs = [attr for attr in table_obj.attributes if attr not in table_obj.irrelevant_attributes]
                
                selected = True
                for attr in encoding_attrs:
                    table_attr = f'{alias2table[table]}.{attr}'
                    attr_min, attr_max = estimator.attributes[table_attr]
                    if attr_min >= float(right_bounds[attr]) or attr_max <= float(left_bounds[attr]):
                        selected = False

                if selected:
                    if selected_estimators.get(table) is None:
                        selected_estimators[table] = [estimator]
                    else:
                        selected_estimators[table].append(estimator)
                        
        return selected_estimators
    
    def compute_mergeable_relationships(self, query, start_table):
        """
        Compute which relationships are merged starting from a certain table 
        """

        relationships = []
        queue = deque()
        queue.append(start_table)

        while queue:
            # BFS
            table = queue.popleft()

            # list neighbours
            table_obj = self.schema_graph.table_dictionary[table]

            for relationship in table_obj.incoming_relationships:
                if relationship.identifier in query.relationship_set and \
                        relationship.identifier not in relationships:
                    relationships.append(relationship.identifier)
                    queue.append(relationship.start)

            for relationship in table_obj.outgoing_relationships:
                if relationship.identifier in query.relationship_set and \
                        relationship.identifier not in relationships:
                    relationships.append(relationship.identifier)
                    queue.append(relationship.end)

        return relationships  
    
    def compute_upper_bound(self, join_tree, domain_info, bins, estimators, alias2table = {'cast_info': 'ci', 'movie_companies': 'mc', 'movie_info':'mi', 'movie_keyword': 'mk',
                   'movie_info_idx': 'mi_idx', 'title': 't'}):
        '''
        根据左右子树计算该节点的upper_bound
        若是叶节点则求基数
        '''
        # 左右子树为None的是叶节点
        if join_tree.get_left() is None and join_tree.get_right() is None:
            # 返回这个节点对应的表的基数
            table = join_tree.get_values().pop()
            # 结果用字典的形式保存
            res = dict()
            for estimator in estimators[table]:
                rows = estimator.rows
                # key是对应的partition id的集合
                estimator.get_model_size()
                prob = estimator.gaussian_prob(domain_info[table]).item()
                res[(estimator.id,)] = max(prob * rows, 1.0)
            logger.info(f'{table} res:{res}')    
            return res
        
        # 非空说明是分支节点
        else:
            # 首先计算左右两节点每个NDV值的选择率
            relationship = join_tree.get_relationship()
            left_table = relationship.start
            start_attr = relationship.start_attr
            right_table = relationship.end
            end_attr = relationship.end_attr
            left_table_attr = f'{alias2table[left_table]}.{start_attr}'
            right_table_attr = f'{alias2table[right_table]}.{end_attr}'
            
            left_table_lower, left_table_upper = domain_info[left_table]
            right_table_lower, right_table_upper = domain_info[right_table]
            # join key的积分范围统一
            # inner join 其他类型需要后续添加类型条件判断
            left_table_lower[start_attr] = max(left_table_lower[start_attr], right_table_lower[end_attr])
            right_table_lower[end_attr] = left_table_lower[start_attr]
            
            left_table_upper[start_attr] = min(left_table_upper[start_attr], right_table_upper[end_attr])
            right_table_upper[end_attr] = left_table_upper[start_attr]       
            
            # 根据范围和间隔确定涉及key的数量
            left_key_num = (left_table_upper[start_attr] - left_table_lower[start_attr]) / bins[left_table_attr]
            right_key_num = (right_table_upper[end_attr] - right_table_lower[end_attr]) / bins[right_table_attr]
            
            assert left_key_num == right_key_num, "not inner join!"
            
            # 初始化
            left_probs = {}
            right_probs = {}
            left_card_bound = left_table_lower[start_attr]
            right_card_bound = left_card_bound
            left_key_num = math.ceil(left_key_num)
            rng = np.random.RandomState(1234)
            noise = rng.rand(left_key_num)
            
            # 结果数组的划分只与在父节点中连接的表的分区数量有关
            # above_table不一定在当前节点的relationship中
            left_bounds = self.compute_upper_bound(join_tree.get_left(), domain_info, bins, estimators)
            right_bounds = self.compute_upper_bound(join_tree.get_right(), domain_info, bins, estimators)
            
            # 按包含的partition将上一个节点的结果分类
            left_groups = {x: [] for x in estimators[left_table]}
            for est_set in left_bounds.keys():
                for est in estimators[left_table]:
                    if est.id in est_set:
                        left_groups[est].append(est_set)
                        break
                    
            right_groups = {x: [] for x in estimators[right_table]}
            for est_set in right_bounds.keys():
                for est in estimators[right_table]:
                    if est.id in est_set:
                        right_groups[est].append(est_set)
                        break
                  
            # 结果用字典的形式保存
            res = dict()
            
            left_table_obj = self.schema_graph.table_dictionary[left_table]
            right_table_obj = self.schema_graph.table_dictionary[right_table]
            
            for estimator in estimators[left_table]:
                left_res = estimator.sqr_gaussian_prob(domain_info[left_table]).item()
                left_probs[estimator] = math.sqrt(left_res)
            for estimator in estimators[right_table]:
                right_res = estimator.sqr_gaussian_prob(domain_info[right_table]).item()
                right_probs[estimator] = math.sqrt(right_res)
                
            logger.info(f'{left_table}:{left_probs}')
            logger.info(f'{right_table}:{right_probs}')
            
            for left_est in estimators[left_table]:
                for right_est in estimators[right_table]:
                    
                    join_res = left_probs[left_est] * right_probs[right_est]
                    
                    for left_bound_set in left_groups[left_est]:
                        for right_bound_set in right_groups[right_est]:
                            join_res = join_res * left_bounds[left_bound_set] * right_bounds[right_bound_set]
                            # key是对应的partition id的集合
                            join_set =  set(left_bound_set) | set(right_bound_set) 
                            logger.info(f'join_set:{join_set}')
                            res[tuple(join_set)] = join_res
            logger.info(f'res:{res}')
            # 返回值是card的数组
            return res    
        
            # 参与join的左表的attr中有一个是主键，join size相当于直接求外键的基数
            if start_attr == left_table_obj.primary_key or start_attr in left_table_obj.primary_key:
                
                for left_est in estimators[left_table]:
                    for right_est in estimators[right_table]:
                        
                        join_res = right_est.gaussian_prob(domain_info[right_table]).item()
                        
                        for left_bound_set in left_groups[left_est]:
                            for right_bound_set in right_groups[right_est]:
                                join_res = join_res * right_bounds[right_bound_set]
                                # key是对应的partition id的集合
                                join_set = set(left_bound_set) | set(right_bound_set) 
                                res[tuple(join_set)] = join_res
                print(f'res:{res}')
                return res
            
            # 参与join的右表的attr中有一个是主键，join size相当于直接求外键的基数
            elif end_attr == right_table_obj.primary_key or end_attr in right_table_obj.primary_key:
                
                for left_est in estimators[left_table]:
                    for right_est in estimators[right_table]:
                        
                        join_res = left_est.gaussian_prob(domain_info[left_table]).item()
                        
                        for left_bound_set in left_groups[left_est]:
                            for right_bound_set in right_groups[right_est]:
                                join_res = join_res * left_bounds[left_bound_set]
                                # key是对应的partition id的集合
                                join_set =  set(left_bound_set) | set(right_bound_set) 
                                res[tuple(join_set)] = join_res
                print(f'res:{res}')
                return res
            
            # 计算每个NDV的基数
            else:
                print(f'left_key_num:{left_key_num}')
                # for i in range(left_key_num):
                #     domain_info[left_table][0][start_attr] = left_card_bound
                #     left_card_bound = left_card_bound + bins[left_table_attr] * noise[i]
                #     domain_info[left_table][1][start_attr] = left_card_bound
                #     for estimator in estimators[left_table]:
                #         left_probs[estimator].append(estimator.gaussian_prob(domain_info[left_table]).item())
                    
                #     domain_info[right_table][0][end_attr] = right_card_bound
                #     right_card_bound = right_card_bound + bins[right_table_attr] * noise[i]
                #     domain_info[right_table][1][end_attr] = right_card_bound
                #     for estimator in estimators[right_table]:
                #         right_probs[estimator].append(estimator.gaussian_prob(domain_info[right_table]).item())
                        
                # 用cauchy不等式估算
                for estimator in estimators[left_table]:
                    res = estimator.sqr_gaussian_prob(domain_info[left_table]).item()
                    left_probs[estimator].append(math.sqrt(res))
                for estimator in estimators[right_table]:
                    res = estimator.sqr_gaussian_prob(domain_info[right_table]).item()
                    right_probs[estimator].append(math.sqrt(res))
                
                for left_est in estimators[left_table]:
                    for right_est in estimators[right_table]:
                        
                        join_res = left_probs[left_est] * right_probs[right_est]
                        
                        for left_bound_set in left_groups[left_est]:
                            for right_bound_set in right_groups[right_est]:
                                join_res = join_res * left_bounds[left_bound_set] * right_bounds[right_bound_set]
                                # key是对应的partition id的集合
                                join_set =  set(left_bound_set) | set(right_bound_set) 
                                res[tuple(join_set)] = join_res
                print(f'res:{res}')
                # 返回值是card的数组
                return res        
