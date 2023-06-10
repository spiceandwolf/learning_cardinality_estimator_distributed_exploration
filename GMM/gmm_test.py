import operator
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import BayesianGaussianMixture as BGMM
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import multivariate_normal
import torch

from csv_sql_partitioning import read_table_csv
from schema import gen_synthetic_schema
from gmm_gpu import GaussianMixture as GMM_gpu
from gmm_gpu import kmeans_fun_gpu
import my_utils

# class GMM(object):
#     def __init__(self, k: int, d: int):
#         '''
#         k: K值
#         d: 样本属性的数量
#         '''
#         self.K = k
#         # 初始化参数
#         self.p = np.random.rand(k)
#         self.p = self.p / self.p.sum()      # 保证所有p_k的和为1
#         self.means = np.random.rand(k, d)
#         self.covs = np.empty((k, d, d))
#         for i in range(k):                  # 随机生成协方差矩阵，必须是半正定矩阵
#             self.covs[i] = np.eye(d) * np.random.rand(1) * k

#     def fit(self, data: np.ndarray):
#         '''
#         data: 数据矩阵，每一行是一个样本，shape = (N, d)
#         '''
#         for _ in range(100):
#             density = np.empty((len(data), self.K))
#             for i in range(self.K):
#                 # 生成K个概率密度函数并计算对于所有样本的概率密度
#                 norm = stats.multivariate_normal(self.means[i], self.covs[i])
#                 density[:,i] = norm.pdf(data)
#             # 计算所有样本属于每一类别的后验
#             posterior = density * self.p
#             posterior = posterior / posterior.sum(axis=1, keepdims=True)
#             # 计算下一时刻的参数值
#             p_hat = posterior.sum(axis=0)
#             mean_hat = np.tensordot(posterior, data, axes=[0, 0])
#             # 计算协方差
#             cov_hat = np.empty(self.covs.shape)
#             for i in range(self.K):
#                 tmp = data - self.means[i]
#                 cov_hat[i] = np.dot(tmp.T*posterior[:,i], tmp) / p_hat[i]
#             # 更新参数
#             self.covs = cov_hat
#             self.means = mean_hat / p_hat.reshape(-1,1)
#             self.p = p_hat / len(data)

#         print(self.p)
#         print(self.means)
#         print(self.covs)

table_csv_path = '../Synthetic/csvdata_sql' + '/{}.csv'
version = 'cols_2_distinct_10000_corr_2_skew_2'
test_file = '../Synthetic/sql_truecard/cols_2_distinct_10000_corr_2_skew_2test.sql.csv'
file_name_column_min_max_vals = '../Synthetic/learnedcardinalities-master/data/cols_2_distinct_10000_corr_2_skew_2_min_max_vals.csv'
bias = 0.1

schema = gen_synthetic_schema(table_csv_path, version)
for table in schema.tables:
    df_rows = read_table_csv(table)
    print(df_rows.shape)
    
    # 找出最有gmm的组件个数
    # n_components = np.arange(50, 100)
    # models = [GMM(n, covariance_type='full', random_state=0).fit(df_rows)
    #       for n in n_components]
    # components = [m.bic(df_rows) for m in models]
    # min_index, _ = min(enumerate(components), key=operator.itemgetter(1))
    # print(min_index)
    # gmm = models[min_index]
    
    # sklearn's bayesiangmm
    t = time.time()
    n_components = 1000
    df_rows_gpu = torch.from_numpy(np.asarray(df_rows)).float().cuda()
    print(df_rows_gpu.shape[0])
    k_mean, choice_cluster = kmeans_fun_gpu(df_rows_gpu)
    print(len(choice_cluster))
    bgmm = GMM_gpu(n_components=n_components, n_features=len(choice_cluster)).fit(df_rows)
    # bgmm_t = time.time() - t
    # print(bgmm_t)
    my_utils.print_training_time(t, time.time())
    # predict
    # load data
    joins, predicates, tables, label = my_utils.load_data(test_file)
    queries = zip(joins, predicates, tables, label)
    column_min_max_vals = my_utils.get_column_min_max_vals(file_name_column_min_max_vals)
    preds_test = []
    components = []
    for idx in range(n_components):
        if bgmm.weights_[idx] > 0.001:
            components.append(idx)
    print(len(components))  
    for query in queries:
        # 不考虑join, 单表, 每种属性只查询一次
        _, predicate, _, _ = query
        est = 0
        left_bounds, right_bounds = my_utils.make_points(table.attributes, predicate, column_min_max_vals, bias)
        bgmm_results = bgmm.predict_proba(np.array([left_bounds, right_bounds]))
        probs = []
        for component in components:
            mean = bgmm.means_[component]
            cov = bgmm.covariances_[component]
            weight = bgmm.weights_[component]
            m_normal = multivariate_normal(mean=mean, cov=cov)
            est = (m_normal.cdf(np.array(right_bounds)) - m_normal.cdf(np.array(left_bounds))) * weight + est
        est_rows = est * df_rows.shape[0] 
        preds_test.append(est_rows)
        
    # Print metrics
    print("\nQ-Error " + test_file + ":")
    my_utils.print_qerror(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
    print("\nMSE validation set:")
    my_utils.print_mse(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
    print("\nMAPE validation set:")
    my_utils.print_mape(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))
    print("\nPearson Correlation validation set:")
    my_utils.print_pearson_correlation(np.array(preds_test, dtype=np.float64), np.array(label, dtype=np.float64))

    
    # sklearn's gmm
    # t = time.time()
    # gmm = GMM(n_components=1000, covariance_type='spherical').fit(df_rows)
    # gmm_t = time.time() - t
    # print(gmm_t)
    # gmm_results = gmm.predict_proba(np.array([[4, 4]]))
    # est = 0
    # for result in gmm_results :
    #     for idx, prob in enumerate(result) :
            
    #         mean = gmm.means_[idx]
    #         cov = gmm.covariances_[idx]
    #         m_normal = multivariate_normal(mean=mean, cov=cov)
    #         est = (m_normal.cdf(np.array([4.1, 4.1])) - m_normal.cdf(np.array([3.9, 3.9]))) * prob + est
    #     est_rows = est * df_rows.shape[0] 
    #     print('gmm_results : ', est_rows) 
           
    # samples, _ = gmm.sample(100000)
    # count = 0
    # for sample in samples:
    #     if sample[0] > 101-0.5 and sample[0] < 101+0.5 :
    #         if sample[1] > 101-0.5 and sample[1] < 101+0.5 :
    #             count = count + 1

    # print(count)


