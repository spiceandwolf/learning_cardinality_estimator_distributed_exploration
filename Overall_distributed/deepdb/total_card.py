from datetime import datetime
import scipy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

met = []
mee = []
q_errors = []

def get_card(csv_path):
    results = pd.read_csv(csv_path, header=0)
    met = []
    mee = []
    latency_ms = []
    for _, result in results.iterrows():
        mee.append(result['cardinality_predict'])
        met.append(result['cardinality_true'])
        latency_ms.append(result['latency_ms'])
    return mee, met, latency_ms

mee_1, met_1, latency_1 = get_card('./power_1.deepdb.results.csv')
mee_2, met_2, latency_2 = get_card('./power_2.deepdb.results.csv')
mee_3, met_3, latency_3 = get_card('./power_3.deepdb.results.csv')

fmetric = open('./power.deepdb.txt', 'a')

mee = np.array(mee_1) + np.array(mee_2) + np.array(mee_3)
met = np.array(met_1)
mse = mean_squared_error(mee, met)
latencies = (np.array(latency_1) + np.array(latency_2) + np.array(latency_3)) / 3
for cardinality_predict, cardinality_true in zip(mee, met):
    q_error = max(cardinality_predict / cardinality_true, cardinality_true / cardinality_predict)
    if cardinality_predict == 0 and cardinality_true == 0:
        q_error = 1.0
    q_errors.append(q_error)
PCCs = sc.stats.pearsonr(mee, met)  # 皮尔逊相关系数
fmetric.write('time:' + str(datetime.now()) + '\n')
fmetric.write('PCCs:'+str(PCCs[0])+'\n')
print('PCCs:', PCCs[0])
# mse = sum(np.square(met - mee))/len(met)
mape = sum(np.abs((met - mee) / met)) / len(met) * 100
# fmetric.write('MSE: '+ str(mse)+'\n')
# fmetric.write('MAPE: '+ str(mape)+'\n')
print('MSE: ', mse)
print('MAPE: ', mape)
# print percentiles of published JOB-light
q_errors = np.array(q_errors)
q_errors.sort()

for i, percentile in enumerate([50, 90, 95, 99, 100]):
    fmetric.write(f"Q-Error {percentile}%-Percentile: {np.percentile(q_errors, percentile)}\n")
    
fmetric.write(f"Q-Mean wo inf {np.mean(q_errors[np.isfinite(q_errors)])}\n")

fmetric.write(f"Latency avg: {np.mean(latencies):.2f}ms\n")

fmetric.close()