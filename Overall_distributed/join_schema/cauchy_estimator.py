import logging

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
    
    def get_join_size_estimation(self, query, base_estimators, column_min_max_vals, alias2table = {'cast_info': 'ci', 'movie_companies': 'mc', 'movie_info':'mi', 'movie_keyword': 'mk',
                   'movie_info_idx': 'mi_idx', 'title': 't'}):
        '''
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
        table_condition_dict = query_items.table_where_condition_dict
        cached_sub_queries = dict()
        
        # encode
        domain_info = {}
        for table in table_set:
            left_bounds = {}
            right_bounds = {}
            bins = {} #
            table_obj = self.schema_graph.table_dictionary[table]
            # drop irrelevant attributes
            encoding_attrs = [attr for attr in table_obj.attributes if attr not in table_obj.irrelevant_attributes]
            # 初始化每个属性的上下界
            for attr in encoding_attrs:
                attr = f'{alias2table[table]}.{attr}' 
                left_bounds[attr] = column_min_max_vals[attr][0]
                right_bounds[attr] = column_min_max_vals[attr][1]
            # 每个表中每个属性对应的条件 
            conditions = table_condition_dict[table]
            domain_info[table] = parse_conditions(conditions, left_bounds, right_bounds, bins)
             
        # get corresponding partion estimators
        estimators = self.get_join_estimators(table_set, domain_info, base_estimators)
        
        # compute upper bound
        for start_table, end_table, path_list in self.join_graph:
            # analysis join case
            for identifier in path_list:
                relationship = self.schema_graph.relationship_dictionary[identifier]
                left_table = relationship.start
                start_attr = relationship.start_attr
                right_table = relationship.end
                end_attr = relationship.end_attr
                
                left_table_lower, left_table_upper = domain_info[left_table]
                right_table_lower, right_table_upper = domain_info[right_table]
                # join key的积分范围统一
                left_table_lower[start_attr] = max(left_table_lower[start_attr], right_table_lower[start_attr])
                right_table_lower[start_attr] = left_table_lower[start_attr]
                left_table_upper[start_attr] = max(left_table_upper[start_attr], right_table_upper[start_attr])
                right_table_upper[start_attr] = left_table_upper[start_attr]
                
                
            if len(path_list) == 1:
                # Joining two tables or Self join of one table
                relationship = self.schema_graph.relationship_dictionary[identifier]
            elif len(path_list) > 1:
                # more tables
                for identifier in path_list:
                    relationship = self.schema_graph.relationship_dictionary[identifier]
                    
                    upper_bound = 0
                    cached_sub_queries[identifier] = upper_bound
            else:
                raise ValueError("Unknown operator")
                
        total_bound = 0
            
        return total_bound

    def get_join_estimators(self, table_set, domain_info, base_estimators):
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
                for attr in encoding_attrs:
                    attr_min = estimator.attributes[attr].min
                    attr_max = estimator.attributes[attr].max
                    if attr_min >= left_bounds[attr] or attr_max <= right_bounds[attr]:
                        if selected_estimators.get(table) is None:
                            selected_estimators[table] = [estimator]
                        else:
                            selected_estimators[table].append(estimator)
        
        return selected_estimators
                   