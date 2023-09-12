import gc
import os
import time
import numpy as np
import pandas as pd
import torch
from join_schema.cauchy_estimator import Cauchy_estimator
from join_schema.join_tables import build_join_graph
from join_schema.prepare_single_tables import Base_estimators
from myutils.prepare_single_table import get_col_statistics, prepare_single_table
from myutils.schema import gen_imdb_cols4_schema
import myutils.my_utils as my_utils
from t_vae_master.models import GaussianVAE

table_csv_path = '../Overall/train-test-data/imdbdata-num/no_head/' + '/{}.csv'
table_path = '../Overall/train-test-data/imdb-cols-sql/4/train-4-num.sql' + '/{}.csv'
test_file = '../Overall/train-test-data/imdb-cols-sql/4/test-4-num.sql.csv'
file_name_column_min_max_vals = './data/col4_min_max_vals.csv'
bias = 0.1
n_latent = 8
n_epoch = 20

schema = gen_imdb_cols4_schema(table_csv_path)
base_estimators = {}
id = 0

bins = {'t.id':0.5, 't.production_year':0.5, 't.phonetic_code':0.5, 't.series_years':0.5}

if os.path.exists(file_name_column_min_max_vals):
    os.remove(file_name_column_min_max_vals)

# 训练模型
for table in schema.tables:
    
    dataset = 'imdb'
    decoder = 'normal'
    learning_rate = 1e-4
    print(f"Dataset: {dataset} / {table.table_name} / Decoder: {decoder} / Learning Rate: {learning_rate}")
    t = time.time()
    
    load_model = True
    save_dir = f"save/{dataset}/{decoder}/"
    save_path = f"save/{dataset}/{decoder}/model_{learning_rate}"
    model_path = f"save/{dataset}/col4/model_{table.table_name}_{learning_rate}_{n_latent}_{n_epoch}"
        
    os.makedirs(save_dir, mode=0o777, exist_ok=True)
    
    hdf_path = './data/' + table.table_name
    df_rows, df_rows_meta = prepare_single_table(schema, table.table_name, hdf_path)
    
    print(df_rows.shape)
    split_index = int(df_rows.to_numpy().shape[0] * 0.1)
    X_valid = df_rows.to_numpy()[:split_index]
    X_train = df_rows.to_numpy()[split_index:]
    # print("X_train[0]: ", X_train[0])
    
    col_statistics = get_col_statistics(df_rows, df_rows_meta, file_name_column_min_max_vals)
    
    if os.path.exists(model_path) and load_model:
        # predict
        model = GaussianVAE(n_in=X_train.shape[1], n_latent=n_latent, n_h=500)
        model.network.load_state_dict(torch.load(model_path))
        model.network.eval()
    else:
        # Train
        gc.collect()
        torch.cuda.empty_cache()
        
        model = GaussianVAE(n_in=X_train.shape[1], n_latent=n_latent, n_h=500)
        print(f"Model: {type(model)}")
        model.fit(X_train, k=1, batch_size=50,
              learning_rate=learning_rate, n_epoch=n_epoch,
              warm_up=False, is_stoppable=True,
              X_valid=X_valid, path=save_path)
    
        torch.save(model.network.state_dict(), model_path)
    
    min_max_val = {}
    for name, min, max in zip(col_statistics['name'], col_statistics['min'], col_statistics['max']):
           min_max_val[name] = [min, max] 
      
    estimator = Base_estimators(table.table_name, id, df_rows, model)
    estimator.set_attr_max_min(df_rows.columns, min_max_val)
    print(estimator.attributes)
    est_list = []
    est_list.append(estimator)
    base_estimators[table.table_name] = est_list
    
# 测试
# load data
joins, predicates, tables, label = my_utils.load_data(test_file)
queries = zip(joins, predicates, tables, label)
column_min_max_vals = my_utils.get_column_min_max_vals(file_name_column_min_max_vals)
preds_test = []
components = []

train_time = time.time() - t
print(train_time)
count = 0
preds_test = []
# Test
for query in queries:
    
    join_graph = build_join_graph(schema)
    estimator = Cauchy_estimator(schema, join_graph)
    estimator.get_join_size_estimation(query, base_estimators, column_min_max_vals, bins)
        
# Print metrics
my_utils.print_model_info(model_path)
print("\nQ-Error " + test_file + ":")
my_utils.print_qerror(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
print("\nMSE validation set:")
my_utils.print_mse(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
print("\nMAPE validation set:")
my_utils.print_mape(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
print("\nPearson Correlation validation set:")
my_utils.print_pearson_correlation(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
