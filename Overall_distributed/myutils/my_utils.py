import csv
import logging
import math
import numpy as np
from scipy import stats
import sqlparse
from sqlparse.tokens import Token
import torch
from myutils.graph_representation import Query, QueryType

logger = logging.getLogger(__name__)

def _extract_identifiers(tokens, enforce_single=True):
    identifiers = [token for token in tokens if isinstance(token, sqlparse.sql.IdentifierList)]
    if len(identifiers) >= 1:
        if enforce_single:
            assert len(identifiers) == 1
        identifiers = identifiers[0]
    else:
        identifiers = [token for token in tokens if isinstance(token, sqlparse.sql.Identifier)]
    return identifiers

# Find corresponding table of attribute
def _find_matching_table(attribute, schema, alias_dict):
    table_name = None
    for table_obj in schema.tables:
        if table_obj.table_name not in alias_dict.keys():
            continue
        if attribute in table_obj.attributes:
            table_name = table_obj.table_name

    assert table_name is not None, f"No table found for attribute {attribute}."
    return table_name

def _fully_qualified_attribute_name(identifier, schema, alias_dict, return_split=False):
    if len(identifier.tokens) == 1:
        attribute = identifier.tokens[0].value
        table_name = _find_matching_table(attribute, schema, alias_dict)
        if not return_split:
            return table_name + '.' + attribute
        else:
            return table_name, attribute

    # Replace alias by full table names
    assert identifier.tokens[1].value == '.', "Invalid Identifier"
    if not return_split:
        return alias_dict[identifier.tokens[0].value] + '.' + identifier.tokens[2].value
    else:
        return alias_dict[identifier.tokens[0].value], identifier.tokens[2].value

def parse_query(query_str, schema):
    """
    Parses simple SQL queries and returns cardinality query object.
    :param query_str:
    :param schema:
    :return:
    """
    query = Query(schema)

    # split query into part before from
    parsed_tokens = sqlparse.parse(query_str)[0]
    from_idxs = [i for i, token in enumerate(parsed_tokens) if token.normalized == 'FROM']
    assert len(from_idxs) == 1, "Nested queries are currently not supported."
    from_idx = from_idxs[0]
    tokens_before_from = parsed_tokens[:from_idx]

    # split query into part after from and before group by
    group_by_idxs = [i for i, token in enumerate(parsed_tokens) if token.normalized == 'GROUP BY']
    assert len(group_by_idxs) == 0 or len(group_by_idxs) == 1, "Nested queries are currently not supported."
    group_by_attributes = None
    if len(group_by_idxs) == 1:
        tokens_from_from = parsed_tokens[from_idx:group_by_idxs[0]]
        order_by_idxs = [i for i, token in enumerate(parsed_tokens) if token.normalized == 'ORDER BY']
        if len(order_by_idxs) > 0:
            group_by_end = order_by_idxs[0]
            tokens_group_by = parsed_tokens[group_by_idxs[0]:group_by_end]
        else:
            tokens_group_by = parsed_tokens[group_by_idxs[0]:]
        # Do not enforce single because there could be order by statement. Will be ignored.
        group_by_attributes = _extract_identifiers(tokens_group_by, enforce_single=False)
    else:
        tokens_from_from = parsed_tokens[from_idx:]

    # Get identifier to obtain relevant tables
    identifiers = _extract_identifiers(tokens_from_from)
    identifier_token_length = \
        [len(token.tokens) for token in identifiers if isinstance(token, sqlparse.sql.Identifier)][0]

    if identifier_token_length == 3:
        # (title, t)
        tables = [(token[0].value, token[2].value) for token in identifiers if
                  isinstance(token, sqlparse.sql.Identifier)]
    else:
        # (title, title), no alias
        tables = [(token[0].value, token[0].value) for token in identifiers if
                  isinstance(token, sqlparse.sql.Identifier)]
    alias_dict = dict()
    for table, alias in tables:
        query.table_set.add(table)
        alias_dict[alias] = table

    # If there is a group by clause, parse it
    if group_by_attributes is not None:

        identifier_token_length = \
            [len(token.tokens) for token in _extract_identifiers(group_by_attributes)][0]

        if identifier_token_length == 3:
            # lo.d_year
            group_by_attributes = [(alias_dict[token[0].value], token[2].value) for token in
                                   _extract_identifiers(group_by_attributes)]
            for table, attribute in group_by_attributes:
                query.add_group_by(table, attribute)
        else:
            # d_year
            for group_by_token in _extract_identifiers(group_by_attributes):
                attribute = group_by_token.value
                table = _find_matching_table(attribute, schema, alias_dict)
                query.add_group_by(table, attribute)

    # 暂且还不研究AQP
    # Obtain projection/ aggregation attributes
    count_statements = [token for token in tokens_before_from if
                        token.normalized == 'COUNT(*)' or token.normalized == 'count(*)']
    assert len(count_statements) <= 1, "Several count statements are currently not supported."
    if len(count_statements) == 1:
        query.query_type = QueryType.CARDINALITY
    # else:
    #     query.query_type = QueryType.AQP
    #     identifiers = _extract_identifiers(tokens_before_from)

    #     # Only aggregation attribute, e.g. sum(lo_extendedprice*lo_discount)
    #     if not isinstance(identifiers, sqlparse.sql.IdentifierList):
    #         handle_aggregation(alias_dict, query, schema, tokens_before_from)
    #     # group by attributes and aggregation attribute
    #     else:
    #         handle_aggregation(alias_dict, query, schema, identifiers.tokens)

    # Obtain where statements
    where_statements = [token for token in tokens_from_from if isinstance(token, sqlparse.sql.Where)]
    assert len(where_statements) <= 1
    if len(where_statements) == 0:
        return query
    
    where_statements = where_statements[0]
    assert len(
        [token for token in where_statements if token.normalized == 'OR']) == 0, "OR statements currently unsupported."

    # Parse where statements
    '''
    # 暂且还不研究in语法
    # parse multiple values differently because sqlparse does not parse as comparison
    in_statements = [idx for idx, token in enumerate(where_statements) if token.normalized == 'IN']
    for in_idx in in_statements:
        assert where_statements.tokens[in_idx - 1].value == ' '
        assert where_statements.tokens[in_idx + 1].value == ' '
        # ('bananas', 'apples')
        possible_values = where_statements.tokens[in_idx + 2]
        assert isinstance(possible_values, sqlparse.sql.Parenthesis)
        # fruits
        identifier = where_statements.tokens[in_idx - 2]
        assert isinstance(identifier, sqlparse.sql.Identifier)

        if len(identifier.tokens) == 1:

            left_table_name, left_attribute = _fully_qualified_attribute_name(identifier, schema, alias_dict,
                                                                              return_split=True)
            query.add_where_condition(left_table_name, left_attribute + ' IN ' + possible_values.value)

        else:
            assert identifier.tokens[1].value == '.', "Invalid identifier."
            # Replace alias by full table names
            query.add_where_condition(alias_dict[identifier.tokens[0].value],
                                      identifier.tokens[2].value + ' IN ' + possible_values.value)
    '''
    # normal comparisons
    comparisons = [token for token in where_statements if isinstance(token, sqlparse.sql.Comparison)]
    
    for comparison in comparisons:
        left = comparison.left
        assert isinstance(left, sqlparse.sql.Identifier), "Invalid where condition"
        comparison_tokens = [token for token in comparison.tokens if token.ttype == Token.Operator.Comparison]
        assert len(comparison_tokens) == 1, "Invalid comparison"
        operator_idx = comparison.tokens.index(comparison_tokens[0])
       
        if len(left.tokens) == 1:

            left_table_name, left_attribute = _fully_qualified_attribute_name(left, schema, alias_dict,
                                                                              return_split=True)
            left_part = left_table_name + '.' + left_attribute
            right = comparison.right

            # Join relationship       解析1
            if isinstance(right, sqlparse.sql.Identifier):
                assert len(right.tokens) == 1, "Invalid Identifier"

                right_attribute = right.tokens[0].value
                right_table_name = _find_matching_table(right_attribute, schema, alias_dict)
                right_part = right_table_name + '.' + right_attribute

                assert comparison.tokens[operator_idx].value == '=', "Invalid join condition"
                assert left_part + ' = ' + right_part in schema.relationship_dictionary.keys() or \
                       right_part + ' = ' + left_part in schema.relationship_dictionary.keys(), "Relationship unknown"
                if left_part + ' = ' + right_part in schema.relationship_dictionary.keys():
                    query.add_join_condition(left_part + ' = ' + right_part)
                elif right_part + ' = ' + left_part in schema.relationship_dictionary.keys():
                    query.add_join_condition(right_part + ' = ' + left_part)

            # Where condition      解析2
            else:
                where_condition = left_attribute + "".join(
                    [token.value.strip() for token in comparison.tokens[operator_idx:]])
                query.add_where_condition(left_table_name, where_condition)

        else:
            # Replace alias by full table names
            left_part = _fully_qualified_attribute_name(left, schema, alias_dict)

            right = comparison.right
            # Join relationship
            if isinstance(right, sqlparse.sql.Identifier):
                assert right.tokens[1].value == '.', "Invalid Identifier"
                right_part = alias_dict[right.tokens[0].value] + '.' + right.tokens[2].value
                assert comparison.tokens[operator_idx].value == '=', "Invalid join condition"
                assert left_part + ' = ' + right_part in schema.relationship_dictionary.keys() or \
                       right_part + ' = ' + left_part in schema.relationship_dictionary.keys(), "Relationship unknown"
                if left_part + ' = ' + right_part in schema.relationship_dictionary.keys():
                    query.add_join_condition(left_part + ' = ' + right_part)
                elif right_part + ' = ' + left_part in schema.relationship_dictionary.keys():
                    query.add_join_condition(right_part + ' = ' + left_part)

            # Where condition
            else:
                query.add_where_condition(alias_dict[left.tokens[0].value],
                                          left.tokens[2].value + comparison.tokens[operator_idx].value + right.value)
    logger.debug('query:')  # 重要
    logger.debug(query.query_type, query.table_set, query.relationship_set, query.table_where_condition_dict, query.conditions,
          query.aggregation_operations, query.group_bys)  # 重要
    return query

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load_data(file_name):
    joins = []
    predicates = []
    tables = []
    label = []

    # Load queries
    with open(file_name, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
            label.append(row[3])
    print("Loaded queries")
    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]
    return joins, predicates, tables, label

def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(l)) for l in labels])
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val

def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    return np.array(np.round(np.exp(labels)), dtype=np.int64)

fmetric = open('./metric_result/result.txt', 'a')

def print_model_info(model_path):
    fmetric.write("model_path: {}".format(model_path) + '\n')

def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    fmetric.write("Median: {}".format(np.median(qerror)) + '\n' + "90th percentile: {}".format(
        np.percentile(qerror, 90)) + '\n' + "95th percentile: {}".format(np.percentile(qerror, 95)) + \
                  '\n' + "99th percentile: {}".format(np.percentile(qerror, 99)) + '\n' + "99th percentile: {}".format(
        np.percentile(qerror, 99)) + '\n' + "Max: {}".format(np.max(qerror)) + '\n' + \
                  "Mean: {}".format(np.mean(qerror)) + '\n')

    logger.info(f"Median: {np.median(qerror)}")
    logger.info(f"90th percentile: {np.percentile(qerror, 90)}")
    logger.info(f"95th percentile: {np.percentile(qerror, 95)}")
    logger.info(f"99th percentile: {np.percentile(qerror, 99)}")
    logger.info(f"Max: {np.max(qerror)}")
    logger.info(f"Mean: {np.mean(qerror)}")

def print_mse(preds_unnorm, labels_unnorm):
    fmetric.write("MSE: {}".format(((preds_unnorm - labels_unnorm) ** 2).mean()) + '\n')
    logger.info(f"MSE: {((preds_unnorm - labels_unnorm) ** 2).mean()}")
    
def print_mape(preds_unnorm, labels_unnorm):
    fmetric.write("MAPE: {}".format(((np.abs(preds_unnorm - labels_unnorm) / labels_unnorm)).mean() * 100) + '\n')
    logger.info(f"MAPE: {((np.abs(preds_unnorm - labels_unnorm) / labels_unnorm)).mean() * 100}")

def print_pearson_correlation(x, y):
    PCCs = stats.pearsonr(x, y)
    fmetric.write("Pearson Correlation: {}".format(PCCs) + '\n\n')
    logger.info(f"Pearson Correlation: {PCCs}")
    
def print_training_time(train_start, train_end):
    fmetric.write("Training Time: {}s".format(train_end - train_start) + '\n')  # Write training time
    
def make_points(attrs, predicates, statistics, bias):
    left_bounds = {}
    right_bounds = {}
    mus = {}
    stds = {}
    for attr in attrs:
        if(len(attr.split('.')) == 2):
            # 只适用于imdb
            alias2table = {'cast_info': 'ci', 'movie_companies': 'mc', 'movie_info':'mi', 'movie_keyword': 'mk',
                   'movie_info_idx': 'mi_idx', 'title': 't'}
            col_name = alias2table[attr.split('.')[0]] + f".{attr.split('.')[1]}"
        # normalize化
        # left_bounds[col_name] = (statistics[col_name][0] - statistics[col_name][-2]) / statistics[col_name][-1]
        # right_bounds[col_name] = (statistics[col_name][1] - statistics[col_name][-2]) / statistics[col_name][-1]
        left_bounds[col_name] = statistics[col_name][0]
        right_bounds[col_name] = statistics[col_name][1]
        mus[col_name] = statistics[col_name][-2]
        stds[col_name] = statistics[col_name][-1]
    
    for predicate in predicates:
        if len(predicate) == 3:
            # Proper predicate
            # column = (predicate[0].split('.'))[-1]
            column = predicate[0] # 适用于imdb的
            operator = predicate[1]
            val = float(predicate[2])
                
            if operator == '=':
                # normalize化
                # left_bounds[column] = (val - bias - mus[col_name]) / stds[col_name]
                # right_bounds[column] = (val + bias - mus[col_name]) / stds[col_name]
                left_bounds[column] = val - bias
                right_bounds[column] = val + bias
                
            elif operator == '<':
                # normalize化
                # right_bounds[column] = (val - mus[col_name]) / stds[col_name]
                right_bounds[column] = val
            elif operator  == ">":
                # normalize化
                # left_bounds[column] = (val - mus[col_name]) / stds[col_name]
                left_bounds[column] = val
                
    return list(left_bounds.values()), list(right_bounds.values())

def get_column_min_max_vals(file_name_column_min_max_vals):
    # Get min and max values for each column
    with open(file_name_column_min_max_vals, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            col = row[0]    # 注意imdb数据集和人工数据集间表名的差异
            column_min_max_vals[col] = [float(row[1]), float(row[2])]
            
    return column_min_max_vals

def load_statistics(statistics_file):
    # Get statistics for each column
    with open(statistics_file, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        statistics = {}
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            statistics[row[0]] = list(map(float, row[1:]))
    return statistics        

def parse_conditions(conditions, left_bounds, right_bounds, bins):
    print('conditions:\n', conditions)
    rng = np.random.RandomState(1234)
    noise = rng.rand(1)
    for condition in conditions:
        if '=' in condition:
            attr, literal = condition.split('=', 1)
            literal = float(literal.strip())
            if literal >= left_bounds[attr] and literal <= right_bounds[attr]:
                left_bounds[attr] = literal - bins[attr] * noise
                right_bounds[attr] = literal + bins[attr] * noise
            else:
                left_bounds[attr] = literal
                right_bounds[attr] = literal
                
        elif '<=' in condition:
            attr, literal = condition.split('<=', 1)
            literal = float(literal.strip())
            right_bounds[attr] = literal + bins[attr] * noise
        
        elif '<' in condition:
            attr, literal = condition.split('<', 1)
            right_bounds[attr] = literal
            
        elif '>=' in condition:
            attr, literal = condition.split('>=', 1)
            literal = float(literal.strip())
            left_bounds[attr] = literal - bins[attr] * noise
            
        elif '>' in condition:
            attr, literal = condition.split('>', 1)
            literal = float(literal.strip())
            left_bounds[attr] = literal
        
        else:
            raise ValueError("Unknown operator")

    return left_bounds, right_bounds