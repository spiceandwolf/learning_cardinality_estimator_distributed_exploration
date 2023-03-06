import argparse
import logging
import csv
import os
import time
import copy

import pandas as pd
from Synthetic_distributed.graph_representation import QueryType
from Synthetic_distributed.utils import parse_query

import schema

def read_table_csv(table, csv_seperator=','):
    """
    从csv文件中读取数据还原table
    """
    df_rows = pd.read_csv(table.csv_file_location, header=None, escapechar='\\', encoding='utf-8', quotechar='"', sep=csv_seperator)
    df_rows.columns = [table.table_name + '.' + attr for attr in table.attributes]

    return df_rows.apply(pd.to_numeric, errors="ignore")

def write_table_csv(table_path, df_rows, csv_seperator=','):
    """
    根据table生成csv
    """
    df_rows.to_csv(table_path, index=False, sep=csv_seperator)

def partition_dataset(schema):
    """
        每个table:
        	A1	                A2	                A3	                A4
        P1	✓(0~3w3k333)	    ✓(3w3k333~6w6k666)	✓(6w6k666~9w9k999)	
        P2	                    ✓(0~3w3k333)	    ✓(3w3k333~6w6k666)	✓(6w6k666~9w9k999)
        P3	✓(6w6k666~9w9k999)	                    ✓(0~3w3k333)	    ✓(3w3k333~6w6k666)
        P4	✓(3w3k333~6w6k666)	✓(6w6k666~9w9k999)	                    ✓(0~3w3k333)
        ...
        999
    """
    distributed_status = [] # 每个节点中涉及的各个表及其属性

    table_path = './nodes/db'
    os.makedirs('nodes/db', exist_ok=True)

    for i in range(node_nums):
        logger.info(f"准备生成子节点node_{i}中的模拟分布式数据库")
        node_status = {}

        for table in schema.tables:
            logger.info(f"进入表{table.table_name}")
            attributes = table.attributes
            df_rows = read_table_csv(table)
            row_num = df_rows.shape[0]
            node_rows = int(row_num / partition_nums)

            attrs = [] # 在table在节点i中存储的属性
            partitioned_table = [] # 被切分后的table
            for j in range(partition_nums):
                attrs.append(attributes[(i + j) % node_nums])
                partition_rows = df_rows.iloc[j * node_rows : (j + 1) * node_rows, 
                                              (i + j) % len(attributes) : (i + j) % len(attributes) + 1]
                partitioned_table.append(partition_rows)
            node_table = pd.concat(partitioned_table, axis=1)
            
            table_path = table_path + f"/{table.table_name}_node_{i}.csv"
            write_table_csv(table_path, node_table)
            logger.info(f"生成子节点node_{i}中表{table.table_name}对应的部分")
            
            node_status[table.table_name] = attrs
        
        distributed_status.append(node_status)
    
    return distributed_status # distributed_status : [..., ({table_a:[..., Ai, Aj, ...]}, {table_b:[..., Aj, Ak, ...]}, ...), ...]
            
def partition_sql(schema, table_sql_path, distributed_status, version, node_nums):
    """
    因为在各分布式节点中存在的属性并不是所属Table中的全部属性
    为了在某一分布式节点中执行某一条query/Q
    需要参考这一节点没存储的属性
    把原本的query/Q中对应属性的谓词删除(或改写成永真的或是其他masking方法)
    """
    with open(table_sql_path) as f:
        queries = f.readlines()

        # 每个查询依次按节点处理
        for query_no, query_str in enumerate(queries):
            query_str = query_str.strip()
            logger.debug(f"处理 query {query_no}: {query_str}")
            query = parse_query(query_str.strip(), schema)  # 解析
            
            assert query.query_type == QueryType.CARDINALITY

            # only operate on copy so that query object is not changed
            # optimized version of:
            original_query = copy.deepcopy(query)

            for i in range(node_nums):
                query = copy.deepcopy(query)
                node_status = distributed_status[i]
                # 获取每一个表在当前节点中存储的属性
                for table in query.table_set:
                    # e.g. involved_table_status : {..., "node_i":attrs[], "node_j":attrs[], ...}
                    involved_attrs = list(set(table.attributes) - set(node_status[table])) # 没在这个节点中出现的属性
                    query.remove_attributes_for_masking(involved_attrs)
                    
                


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='csv_sql_partitioning')
    parser.add_argument('--dataset', type=str, help='Which dataset to be used', default='Synthetic_distributed')
    parser.add_argument('--version', type=str, help='datasets_dir', default='cols_4_distinct_1000_corr_5_skew_5')
    parser.add_argument('--alias', type=str, help='alias', default='cdcs')
    parser.add_argument('--node_nums', type=int, help='node nums', default=4)
    parser.add_argument('--partition_nums', type=int, help='partition nums per node', default=3)

    args = parser.parse_args()
    version = args.version
    alias = args.alias
    node_nums = args.node_nums
    partition_nums = args.partition_nums

    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
        # [%(threadName)-12.12s]
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("logs/{}_{}.log".format(args.dataset, time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    total_data_dir_path = '../Synthetic/csvdata_sql' 
    table_csv_path = total_data_dir_path + '/{}.csv'
    table_sql_path = total_data_dir_path + '/{}.sql'

    # 生成schema
    schema = schema.gen_synthetic_schema(table_csv_path, version)
    logger.info(f"生成关系{version}对应的schema")

    # 用多个csv文件模拟分布式情况下各节点中的数据
    # 原始sql在 {version}.sql中, 插入数据库的原始数据在 {version}.csv 或 {version}_nohead.csv
    # distributed_status : [..., {table_a:[..., Ai, Aj, ...]}, {table_b:[..., Aj, Ak, ...]}, ...]
    distributed_status = partition_dataset(schema)
    # 生成子节点对应的subquery/subQ/distributed_query/distributed_Q
    partition_sql(schema, table_sql_path, distributed_status, version, node_nums)
