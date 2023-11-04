import os
import re
from matplotlib import pyplot as plt, ticker
import numpy as np
import tqdm 

if __name__ == '__main__':
    log_dir_path = '../logs'
    log_paths = os.listdir(log_dir_path)
    
    logs = {}
    
    for log_path in tqdm.tqdm(log_paths, desc='logs'):
        full_log_path = os.path.join(log_dir_path, log_path)
        
        data = {}
        with open(full_log_path, 'r') as log_data:
            log_info_list = log_data.readlines()
            print(log_path)
            
            valids = []
            
            for log_info in tqdm.tqdm(log_info_list, desc=log_path):
                
                if re.search('model : (.*)', log_info, re.IGNORECASE):
                    model_path = re.search('model : (.*)', log_info, re.IGNORECASE).group(1)
                elif re.search('epoch: (.*)', log_info, re.IGNORECASE):
                    if re.search('/ Valid: (.*)/ Save', log_info, re.IGNORECASE):
                        valid = re.search('/ Valid: (.*)/ Save', log_info, re.IGNORECASE).group(1)
                        valids.append(valid)
                elif re.search('Parameters total size is (.*)', log_info, re.IGNORECASE):
                    data['model_size'] = float(re.search('Parameters total size is (.*) MB', log_info, re.IGNORECASE).group(1))
                elif re.search('Learning Rate: (.*)', log_info, re.IGNORECASE):
                    data['learning_rate'] = re.search('Learning Rate: (.*)', log_info, re.IGNORECASE).group(1)
                elif re.search('n_latent: (.*) / n_h: (.*) / tau: (.*)', log_info, re.IGNORECASE):
                    resObj = re.search('n_latent: (.*) / n_h: (.*) / tau: (.*)', log_info, re.IGNORECASE)
                    data['n_latent'] = resObj.group(1)
                    data['n_h'] = resObj.group(2)
                    data['tau'] = resObj.group(3)
                elif re.search('Median: (.*)', log_info, re.IGNORECASE):
                    qerror_median = float(re.search('Median: (.*)', log_info, re.IGNORECASE).group(1))
                    if qerror_median != 'nan':
                        data['qerror_median'] = qerror_median
                    else:
                        data['qerror_median'] = 0
                elif re.search('90th percentile: (.*)', log_info, re.IGNORECASE):
                    qerror_90th = float(re.search('90th percentile: (.*)', log_info, re.IGNORECASE).group(1))
                    if qerror_90th != 'nan':
                        data['qerror_90th'] = qerror_90th
                    else:
                        data['qerror_90th'] = 0
            if len(valids) == 0:
                data['valid'] = 0
                data['qerror_median'] = 0
                data['qerror_90th'] = 0
                data['model_size'] = 0
            else:
                data['valid'] = min(valids)
            # print(data)        
            
        logs[model_path] = data
        
    n_latent_list = [8, 16, 32, 64]
    n_h_list = [125, 250, 500, 750]
    tau_list = [0.125, 0.25, 0.5, 0.75]
    n_epoch = 50
    learning_rate_list = [1e-3, 5e-4, 1e-4]
    
    x_list = ['0'] * len(log_paths)
    
    for i, n_h in enumerate(n_h_list):
        for j, learning_rate in enumerate(learning_rate_list):
            for k, n_latent in enumerate(n_latent_list):
                for l, tau in enumerate(tau_list):
                    for logs_key, logs_value in logs.items():
                        if logs_value['n_h'] == str(n_h) and logs_value['learning_rate'] == str(learning_rate) and logs_value['n_latent'] == str(n_latent) and logs_value['tau'] == str(tau):
                            x_list[l + k*len(tau_list) + j*len(tau_list)*len(n_latent_list) + i*len(tau_list)*len(n_latent_list)*len(learning_rate_list)] = logs_key 
    y_qerror_medians = []
    y_qerror_90ths = []
    y_model_size = []
    
    for key in x_list:
        data = logs[key]
        # print(data)
        y_qerror_medians.append(data['qerror_median'])
        y_qerror_90ths.append(data['qerror_90th'])
        y_model_size.append(data['model_size'])
    
    # 作图
    # x轴先根据隐藏节点数量n_h划分，然后是学习率learning_rate，再然后是隐变量n_latent，最后是tau
    # qerror
    plt.figure(figsize=(21.6, 10.8), dpi=100)
    qerror_fig, axs = plt.subplots(4, 1)
    plt.xlim((0, len(log_paths)))
                     
    for i in range(len(axs)):
        offset = len(tau_list)*len(n_latent_list)*len(learning_rate_list)
        x = x_list[i*offset : (i + 1)*offset]
        
        x1_labels = y_qerror_medians[i*offset : (i + 1)*offset]
        rects1 = axs[i].bar(x, x1_labels, label='qerror_median', zorder=2)
        x2_labels = y_qerror_90ths[i*offset : (i + 1)*offset]
        rects2 = axs[i].bar(x, x2_labels, label='qerror_90th', zorder=1, bottom=0)
    
        # plt.axvline(x=16, color='red', ls='--')
        # plt.axvline(x=32, color='red', ls='--')
    
        axs[i].set_xticks([])
        axs[i].set_xlim(-1, 48)
        axs[i].set_yticks([1, 10, 100, 1000])
        axs[i].set_yticklabels([1, 10, 100, 1000])
        axs[i].set_ylim(1, 1000)
        axs[i].set_yscale('log')
        axs[i].vlines([15.5, 31.5], 1, 1000, color='red', ls='--')
        axs[i].vlines([3.5, 7.5, 11.5, 19.5, 23.5, 27.5, 35.5, 39.5, 43.5], 1, 10000, color='green', ls='--')
        axs[i].set_ylabel(n_h_list[i])
    
    axs[-1].set_xticks([1.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5, 29.5, 33.5, 37.5, 41.5, 45.5])
    axs[-1].set_xticklabels(['8', '16', '32', '64', '8', '16', '32', '64', '8', '16', '32', '64'])
    
    qerror_fig.text(0.5, 0, 'model', ha='center')
    qerror_fig.text(0, 0.5, 'log(qerror)', va='center', rotation='vertical')
    
    plt.savefig(r'qerror_fig.png', dpi=100, bbox_inches = 'tight')
    
    # model_size
    plt.figure(figsize=(21.6, 10.8), dpi=100)
    model_size_fig, axs = plt.subplots(4, 1)
    plt.xlim((0, len(log_paths)))
                     
    for i in range(len(axs)):
        offset = len(tau_list)*len(n_latent_list)*len(learning_rate_list)
        x = x_list[i*offset : (i + 1)*offset]
        
        x1_labels = y_model_size[i*offset : (i + 1)*offset]
        rects1 = axs[i].bar(x, x1_labels, label='model_size')
    
        axs[i].set_xticks([])
        axs[i].set_xlim(-1, 48)
        axs[i].set_yticks(np.linspace(0, 10, 5))
        axs[i].set_yticklabels(np.linspace(0, 10, 5))
        axs[i].set_ylim(0, 10)
        axs[i].vlines([15.5, 31.5], 1, 1000, color='red', ls='--')
        axs[i].vlines([3.5, 7.5, 11.5, 19.5, 23.5, 27.5, 35.5, 39.5, 43.5], 1, 10000, color='green', ls='--')
        axs[i].set_ylabel(n_h_list[i])
    
    axs[-1].set_xticks([1.5, 5.5, 9.5, 13.5, 17.5, 21.5, 25.5, 29.5, 33.5, 37.5, 41.5, 45.5])
    axs[-1].set_xticklabels(['8', '16', '32', '64', '8', '16', '32', '64', '8', '16', '32', '64'])
    
    model_size_fig.text(0.5, 0, 'model', ha='center')
    model_size_fig.text(0, 0.5, 'model_size / MB', va='center', rotation='vertical')
    
    plt.savefig(r'model_size_fig.png', dpi=100, bbox_inches = 'tight')
    
    plt.show()
    