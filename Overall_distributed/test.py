import os
from tqdm import tqdm

n_latent_list = [8, 16, 32, 64]
n_h_list = [125, 250, 500, 750]
tau_list = [0.125, 0.25, 0.5, 0.75]
n_epoch = 50
learning_rate_list = [1e-3, 5e-4, 1e-4]

count = len(n_latent_list)*len(n_h_list)*len(tau_list)*len(learning_rate_list)

with tqdm(total=count) as pbar:
    for n_latent in n_latent_list:
        for n_h in n_h_list:
            for tau in tau_list:
                for learning_rate in learning_rate_list:
                    os.system(f'python t-vae_test.py --dataset imdb \
                                --table_csv_path ../Overall/train-test-data/imdbdata-num/no_head/ \
                                --test_file ../Overall/train-test-data/imdb-cols-sql/2/test-2-num.sql.csv \
                                --statistics_file ./data/col2_min_max_vals.csv \
                                --n_latent {n_latent} \
                                --n_h {n_h} \
                                --tau {tau} \
                                --n_epoch {n_epoch} \
                                --learning_rate {learning_rate} ')
                    pbar.update(1)

# python t-vae_test.py --dataset power --train --table_csv_path ../../data/power/ --statistics_file ./data/power_statistics.csv --bias {'power.Global_active_power': 0.0005, 'power.Global_reactive_power': 0.0005, 'power.Voltage': 0.005, 'power.Global_intensity': 0.05, 'power.Sub_metering_1': 0.5, 'power.Sub_metering_2': 0.5, 'power.Sub_metering_3': 0.5} --n_latent 8 --n_h 250 --tau 0.5 --n_epoch 20 --learning_rate 1e-3