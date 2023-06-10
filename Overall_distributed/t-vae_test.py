import os
import time
import numpy as np
import pandas as pd
import torch
from myutils.prepare_single_table import get_col_statistics, prepare_single_table
from myutils.csv_utils import read_table_csv
from myutils.schema import gen_imdb_cols2_schema
import myutils.my_utils as my_utils
from t_vae_master.models import GaussianVAE

table_csv_path = '../Overall/train-test-data/imdbdata-num/no_head/' + '/{}.csv'
table_path = '../Overall/train-test-data/imdb-cols-sql/2/train-2-num.sql' + '/{}.csv'
test_file = '../Overall/train-test-data/imdb-cols-sql/2/test-2-num.sql.csv'
file_name_column_min_max_vals = './data/col2_min_max_vals.csv'
bias = 0.1
n_latent = 16
n_epoch = 40

schema = gen_imdb_cols2_schema(table_csv_path)
for table in schema.tables:
    
    dataset = 'imdb'
    decoder = 'normal'
    learning_rate = 1e-4
    print(f"Dataset: {dataset} / Decoder: {decoder} / Learning Rate: {learning_rate}")
    t = time.time()
    
    load_model = True
    save_dir = f"save/{dataset}/{decoder}/"
    save_path = f"save/{dataset}/{decoder}/model_{learning_rate}"
    model_path = f"save/{dataset}/model_{table.table_name}_{learning_rate}_{n_latent}_{n_epoch}"
        
    os.makedirs(save_dir, mode=0o777, exist_ok=True)
    
    hdf_path = './data/' + table.table_name
    df_rows, df_rows_meta = prepare_single_table(schema, table.table_name, hdf_path)
    
    print(df_rows.shape)
    split_index = int(df_rows.to_numpy().shape[0] * 0.1)
    X_valid = df_rows.to_numpy()[:split_index]
    X_train = df_rows.to_numpy()[split_index:]
    # print("X_train[0]: ", X_train[0])
    
    if os.path.exists(model_path) and load_model:
        # predict
        model = GaussianVAE(n_in=X_train.shape[1], n_latent=n_latent, n_h=500)
        model.network.load_state_dict(torch.load(model_path))
        model.network.eval()
    else:
        # Train
        get_col_statistics(df_rows, df_rows_meta, file_name_column_min_max_vals)
        model = GaussianVAE(n_in=X_train.shape[1], n_latent=n_latent, n_h=500)
        print(f"Model: {type(model)}")
        model.fit(X_train, k=1, batch_size=100,
              learning_rate=learning_rate, n_epoch=n_epoch,
              warm_up=False, is_stoppable=True,
              X_valid=X_valid, path=save_path)
    
        torch.save(model.network.state_dict(), model_path)
    
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
        # if count < 40:
        #     count = 1 + count
        #     continue
        # 不考虑join, 单表, 每种属性只查询一次
        _, predicate, _, _ = query
        est = 0
        left_bounds, right_bounds = my_utils.make_points(df_rows.columns, predicate, column_min_max_vals, bias)
        # 通过cdf计算概率
        # print("left_bounds : ", left_bounds)
        # print("right_bounds : ", right_bounds)
        # 通过importance_sampling估计概率
        prob = model.gaussian_prob(np.array(left_bounds), np.array(right_bounds))
        # print("prob : ", prob)
        est_card = prob * df_rows.shape[0] + 1
        # print("est_card : ", est_card)
        preds_test.append(est_card.cpu().numpy())
        # count = 1 + count
        # print("count : ", count)
        # if count == 7:
        #     break
        
        
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
