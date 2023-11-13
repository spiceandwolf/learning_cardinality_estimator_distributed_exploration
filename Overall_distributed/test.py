import os
from tqdm import tqdm

n_latent_list = [8, 16, 32, 64]
n_h_list = [125, 250, 500, 750]
tau_list = [0.125, 0.25, 0.5, 0.75]
n_epoch = 10
learning_rate_list = [1e-3, 5e-4, 1e-4]

count = len(n_latent_list)*len(n_h_list)*len(tau_list)*len(learning_rate_list)

with tqdm(total=count) as pbar:
    for n_latent in n_latent_list:
        for n_h in n_h_list:
            for tau in tau_list:
                for learning_rate in learning_rate_list:
                    os.system(f'python t-vae_test.py --dataset power '
                              f'--log_path ./logs/power '
                              f'--train '
                              f'--table_csv_path ../../data/power/ '
                              f'--statistics_file ./data/power_statistics.csv '
                              f'--bias \'{{"power.Global_active_power": 0.0005, "power.Global_reactive_power": 0.0005, "power.Voltage": 0.005, "power.Global_intensity": 0.05, "power.Sub_metering_1": 0.5, "power.Sub_metering_2": 0.5, "power.Sub_metering_3": 0.5}}\' '
                              f'--n_latent {n_latent} '
                              f'--n_h {n_h} '
                              f'--tau {tau} '
                              f'--n_epoch {n_epoch} '
                              f'--learning_rate {learning_rate} '
                              f'--evaluate '
                              f'--test_file ../../workload/power/powertest.sql.csv ')
                    pbar.update(1)
    
                    
# python t-vae_test.py --dataset power --train --table_csv_path ../../data/power/ --statistics_file ./data/power_statistics.csv --bias '{"power.Global_active_power": 0.0005, "power.Global_reactive_power": 0.0005, "power.Voltage": 0.005, "power.Global_intensity": 0.05, "power.Sub_metering_1": 0.5, "power.Sub_metering_2": 0.5, "power.Sub_metering_3": 0.5}' --n_latent 8 --n_h 250 --tau 0.5 --n_epoch 20 --learning_rate 1e-3
# python t-vae_test.py --dataset power --evaluate --statistics_file ./data/power_statistics.csv --model_path ./save/power/normal/model_power_0.001_8_250_0.5_20 --bias '{"power.Global_active_power": 0.0005, "power.Global_reactive_power": 0.0005, "power.Voltage": 0.005, "power.Global_intensity": 0.05, "power.Sub_metering_1": 0.5, "power.Sub_metering_2": 0.5, "power.Sub_metering_3": 0.5}' --n_latent 8 --n_h 250 --tau 0.5 --test_file ../../workload/power/powertest.sql.csv
# python t-vae_test.py --dataset power --log_path ./logs/power_fine_tuning/ --train --table_csv_path ../../data/power/ --statistics_file ./data/power_statistics.csv --bias '{"power.Global_active_power": 0.0005, "power.Global_reactive_power": 0.0005, "power.Voltage": 0.005, "power.Global_intensity": 0.05, "power.Sub_metering_1": 0.5, "power.Sub_metering_2": 0.5, "power.Sub_metering_3": 0.5}' --evaluate --test_file ../../workload/power/powertest.sql.csv --n_latent 16 --n_h 200 --tau 0.5 --n_epoch 20 --learning_rate 5e-4