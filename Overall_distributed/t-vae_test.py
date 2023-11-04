import argparse
import gc
import logging
import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
from myutils.prepare_single_table import get_col_statistics, prepare_single_table
from myutils.csv_utils import read_table_csv
from myutils.schema import gen_power_schema
import myutils.my_utils as my_utils
from t_vae_master.models import GaussianVAE

seed = 1234
# bias = 0.1

def train(schema, dataset, statistics_file, learning_rate, n_latent, n_h, tau, n_epoch):
    models = dict()
    meta_datas = dict()
    for table in schema.tables:
        
        decoder = 'normal'
        logger.info(f"Dataset: {dataset} / Decoder: {decoder} / Learning Rate: {learning_rate}")
        logger.info(f"n_latent: {n_latent} / n_h: {n_h} / tau: {tau}")
        t = time.time()
        
        save_dir = f"save/{dataset}/{decoder}/"
        model_path = f"{save_dir}/model_{table.table_name}_{learning_rate}_{n_latent}_{n_h}_{tau}_{n_epoch}"
        
        logger.info(f'model : {model_path}')
            
        os.makedirs(save_dir, mode=0o777, exist_ok=True)
        
        hdf_path = './data/' + table.table_name
        df_rows, df_rows_meta = prepare_single_table(schema, table.table_name, hdf_path, header=0, csv_seperator=';')
        get_col_statistics(df_rows, df_rows_meta, statistics_file)
        
        print(df_rows.shape)
        df_rows.sample(frac=1, random_state=seed)
        split_index = int(df_rows.to_numpy().shape[0] * 0.1)
        X_valid = df_rows.to_numpy()[:split_index]
        X_train = df_rows.to_numpy()[split_index:]
        
        model = GaussianVAE(n_in=X_train.shape[1], n_latent=n_latent, n_h=n_h, tau=tau)
           
        model.fit(X_train, k=1, batch_size=100,
            learning_rate=learning_rate, n_epoch=n_epoch,
            warm_up=False, is_stoppable=True,
            X_valid=X_valid, path=model_path)
        
        train_time = time.time() - t
        print(train_time)
        
        models[table.table_name] = model
        meta_datas[table.table_name] = df_rows_meta
        
    with open(f"./data/{dataset}/meta_data.pkl", 'wb') as f:
        pickle.dump(meta_datas, f, pickle.HIGHEST_PROTOCOL)
        
    return models

def evaluate(model, test_file, columns, statistics, bias, n_rows):
    joins, predicates, tables, label = my_utils.load_data(test_file)
    queries = zip(joins, predicates, tables, label)
    preds_test = []
    
    preds_test = []
    # Test
    for query in queries:
        
        _, predicate, _, _ = query
        
        left_bounds, right_bounds = my_utils.make_points(columns, predicate, statistics, bias)
        # print("left_bounds : ", left_bounds)
        # print("right_bounds : ", right_bounds)
        prob = model.gaussian_prob(np.array(left_bounds), np.array(right_bounds))
        est_card = max(prob.item() * n_rows, 1)
        # print("est_card : ", prob.item())
        preds_test.append(est_card)
        
    model.get_model_size()    
    # Print metrics
    logger.info("Q-Error " + test_file + ":")
    my_utils.print_qerror(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
    logger.info("MSE validation set:")
    my_utils.print_mse(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
    logger.info("MAPE validation set:")
    my_utils.print_mape(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
    logger.info("Pearson Correlation validation set:")
    my_utils.print_pearson_correlation(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))    
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='power', help='Which dataset to be used')  # power
    # action
    parser.add_argument('--train', help="Trains the model", action='store_true')
    parser.add_argument('--evaluate', help="Evaluates the model's qerror", action='store_true')
    # file path
    parser.add_argument('--table_csv_path', default='../../data/power/')
    parser.add_argument('--test_file', default='../workload/power/powertest.sql.csv')
    parser.add_argument('--statistics_file', default='./data/power_statistics.csv')
    # settings
    parser.add_argument('--bias', default={'power.Global_active_power': 0.0005, 'power.Global_reactive_power': 0.0005, 'power.Voltage': 0.005, 'power.Global_intensity': 0.05, 'power.Sub_metering_1': 0.5, 'power.Sub_metering_2': 0.5, 'power.Sub_metering_3': 0.5}
                        , type=dict())
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
    # settings
    bias = args.bias
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

    schema = gen_power_schema(table_csv_path)
    
    models = dict()

    if args.train:
        models = train(schema, dataset, statistics_file, learning_rate=learning_rate, n_latent=n_latent, n_h=n_h, tau=tau, n_epoch=n_epoch)
        
        logger.info("training finished")
        
    if args.evaluate:
        with open(f'./data/{dataset}/meta_data.pkl', 'rb') as f:
            meta_datas = pickle.load(f)
        
        for table in schema.tables:
            if models is not None:
                model = models[table.table_name]
            else:
                decoder = 'normal'
                save_dir = f"save/{dataset}/{decoder}/"
                model_path = f"{save_dir}/model_{table.table_name}_{learning_rate}_{n_latent}_{n_h}_{tau}_{n_epoch}"
                
                assert os.path.exists(model_path), f"model not found : {model_path}"
                
                n_in = meta_datas[table.table_name]['input_size']
                model = GaussianVAE(n_in=n_in, n_latent=n_latent, n_h=n_h, tau=tau)
                model.network.load_state_dict(torch.load(model_path))
                model.network.eval()
            columns = meta_datas[table.table_name]['relevant_attributes_full']
            statistics = my_utils.load_statistics(statistics_file)
            n_rows = meta_datas[table.table_name]['length']
            evaluate(model, test_file, columns, statistics, bias, n_rows)