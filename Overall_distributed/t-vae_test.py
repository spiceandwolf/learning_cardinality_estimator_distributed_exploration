import argparse
import gc
import logging
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

seed = 1234
bias = 0.1

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='imdb', help='Which dataset to be used')  # imdb-light
    # file path
    parser.add_argument('--table_csv_path', default='../Overall/train-test-data/imdbdata-num/no_head/')
    parser.add_argument('--test_file', default='../Overall/train-test-data/imdb-cols-sql/2/test-2-num.sql.csv')
    parser.add_argument('--statistics_file', default='./data/col2_min_max_vals.csv')
    # hyperparameters
    parser.add_argument('--n_latent', default='8', type=int)
    parser.add_argument('--n_h', default='250', type=int)
    parser.add_argument('--tau', default='0.5', type=float)
    parser.add_argument('--n_epoch', default='50', type=int)
    parser.add_argument('--learning_rate', default='1e-3', type=float)
    
    args = parser.parse_args()
    dataset = args.dataset
    # file path
    table_csv_path = args.table_csv_path + '/{}.csv'
    test_file = args.test_file
    statistics_file = args.statistics_file
    # hyperparameters
    n_latent = args.n_latent
    n_h = args.n_h
    tau = args.tau
    n_epoch = args.n_epoch
    learning_rate = args.learning_rate
    
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        # [%(threadName)-12.12s]
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("logs/{}_{}.log".format('imdb', time.strftime("%Y%m%d-%H%M%S"))),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    if os.path.exists(statistics_file):
        os.remove(statistics_file)

    schema = gen_imdb_cols2_schema(table_csv_path)
    for table in schema.tables:
        
        decoder = 'normal'
        logger.info(f"Dataset: {dataset} / Decoder: {decoder} / Learning Rate: {learning_rate}")
        logger.info(f"n_latent: {n_latent} / n_h: {n_h} / tau: {tau}")
        t = time.time()
        
        load_model = True
        save_dir = f"save/{dataset}/{decoder}/"
        model_path = f"save/{dataset}/single/model_{table.table_name}_{learning_rate}_{n_latent}_{n_h}_{tau}_{n_epoch}"
        
        logger.info(f'model : {model_path}')
            
        os.makedirs(save_dir, mode=0o777, exist_ok=True)
        
        hdf_path = './data/' + table.table_name
        df_rows, df_rows_meta = prepare_single_table(schema, table.table_name, hdf_path)
        col_statistics = get_col_statistics(df_rows, df_rows_meta, statistics_file)
        
        print(df_rows.shape)
        df_rows.sample(frac=1, random_state=seed)
        split_index = int(df_rows.to_numpy().shape[0] * 0.1)
        X_valid = df_rows.to_numpy()[:split_index]
        X_train = df_rows.to_numpy()[split_index:]
        # print("X_train[0]: ", X_train[0])
        
        if os.path.exists(model_path) and load_model:
            # predict
            model = GaussianVAE(n_in=X_train.shape[1], n_latent=n_latent, n_h=n_h, tau=tau)
            model.network.load_state_dict(torch.load(model_path))
            model.network.eval()
        else:
            # Train
            model = GaussianVAE(n_in=X_train.shape[1], n_latent=n_latent, n_h=n_h, tau=tau)
           
            model.fit(X_train, k=1, batch_size=100,
                learning_rate=learning_rate, n_epoch=n_epoch,
                warm_up=False, is_stoppable=True,
                X_valid=X_valid, path=model_path)
        
        # load data
        joins, predicates, tables, label = my_utils.load_data(test_file)
        queries = zip(joins, predicates, tables, label)
        column_min_max_vals = my_utils.get_column_min_max_vals(statistics_file)
        preds_test = []
        components = []
        statistics = my_utils.load_statistics(statistics_file)
        
        train_time = time.time() - t
        print(train_time)
        count = 0
        preds_test = []
        # Test
        for query in queries:
            
            _, predicate, _, _ = query
            est = 0
            left_bounds, right_bounds = my_utils.make_points(df_rows.columns, predicate, statistics, bias)
            # print("left_bounds : ", left_bounds)
            # print("right_bounds : ", right_bounds)
            # 通过importance_sampling估计概率
            # debug = model.importance_sampling(np.array(left_bounds))
            # print(f'debug : {debug}')
            prob = model.gaussian_prob(np.array(left_bounds), np.array(right_bounds))
            est_card = max(prob.item() * df_rows.shape[0], 1)
            # print("est_card : ", prob.item())
            preds_test.append(est_card)
            
        model.get_model_size()    
        # Print metrics
        my_utils.print_model_info(model_path)
        logger.info("Q-Error " + test_file + ":")
        my_utils.print_qerror(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
        logger.info("MSE validation set:")
        my_utils.print_mse(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
        logger.info("MAPE validation set:")
        my_utils.print_mape(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
        logger.info("Pearson Correlation validation set:")
        my_utils.print_pearson_correlation(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
