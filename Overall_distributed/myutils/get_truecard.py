import argparse
from physical_db import DBConnection, TrueCardinalityEstimator
from tqdm import tqdm
from schema import gen_imdb_cols2_schema

parser = argparse.ArgumentParser(description='get_truecard')

parser.add_argument('--dataset', type=str, help='datasets_dir', default='imdb')

args = parser.parse_args()
dataset = args.dataset

db_connection = DBConnection(db='imdb_num', db_password="linux123", db_user='user1', db_host="localhost")
table_csv_path = '../Overall/train-test-data/imdbdata-num/no_head/' + '/{}.csv'
test_file = '../../Overall/train-test-data/imdb-cols-sql/2/test-2-num.sql'
corrected_test_file = '../../Overall/train-test-data/imdb-cols-sql/2/test-2-num_corrected.sql'
schema = gen_imdb_cols2_schema(table_csv_path)

true_estimator = TrueCardinalityEstimator(schema, db_connection)
query_list = []

with open(test_file) as f:
    queries = f.readlines()

    for query in tqdm(queries):

        try:
            query = query.split(',')[0]
            cardinality_true = true_estimator.true_cardinality(query)
            if cardinality_true == 0:
                continue
            query_list.append(query + ',' + str(cardinality_true))
        except Exception as e:
            pass
        continue
    
with open(corrected_test_file, 'a') as f:
    for query_item in query_list:
        f.write(query_item + '\n')