import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from t_vae_master.models.utils import from_numpy, to_numpy, make_data_loader
from myutils.csv_sql_partitioning import read_table_csv
from myutils.schema import gen_synthetic_schema
import myutils.my_utils as my_utils
import numpy as np
import time

# 定义变分自编码器VAE
class Variable_AutoEncoder(nn.Module):
    def __init__(self, n_in, n_latent, n_h):
        super(Variable_AutoEncoder, self).__init__()
        self.n_in = n_in
        self.n_latent = n_latent
        self.n_h = n_h
        
        # 定义编码器
        self.Encoder = nn.Sequential(
            nn.Linear(n_in, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU()
        )
        self.fc_m = nn.Linear(n_h, n_latent)
        self.fc_sigma = nn.Linear(n_h, n_latent)
        # 定义解码器
        self.Decoder = nn.Sequential(
            nn.Linear(n_latent, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_in),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        code = input.view(input.size(0), -1)
        code = self.Encoder(input)
        # m, sigma = code.chunk(2, dim=1)
        m = self.fc_m(code)
        sigma = self.fc_sigma(code)
        e = torch.randn_like(sigma)
        c = torch.exp(sigma) * e + m
        # c = sigma * e + m
        output = self.Decoder(c)
        # output = output.view(input.size(0), 1, 28, 28)
        return output, m, sigma

 # 定义超参数
learning_rate = 1e-3
batch_size = 100
epochsize = 30   
device = torch.device("cuda")
table_csv_path = '../Synthetic/csvdata_sql' + '/{}.csv'
version = 'cols_4_distinct_10000_corr_2_skew_2'
test_file = '../Synthetic/sql_truecard/cols_4_distinct_10000_corr_2_skew_2test.sql.csv'
file_name_column_min_max_vals = '../Synthetic/learnedcardinalities-master/data/cols_4_distinct_10000_corr_2_skew_2_min_max_vals.csv'
bias = 0.1
n_latent = 10000
# 定义并导入网络结构

schema = gen_synthetic_schema(table_csv_path, version)
for table in schema.tables:
    df_rows = read_table_csv(table)
    print(df_rows.shape)
    # predict
    # load data
    joins, predicates, tables, label = my_utils.load_data(test_file)
    queries = zip(joins, predicates, tables, label)
    column_min_max_vals = my_utils.get_column_min_max_vals(file_name_column_min_max_vals)
    preds_test = []
    components = []
    
    # dataset = version
    # decoder = 'normal'
    # learning_rate = 1e-4
    # print(f"Dataset: {dataset} / Decoder: {decoder} / Learning Rate: {learning_rate}")
    
    split_index = int(df_rows.to_numpy().shape[0] * 0.1)
    X_valid = df_rows.to_numpy()[:split_index]
    X_train = df_rows.to_numpy()[split_index:]
    data_loader = make_data_loader(X_train, device=device, batch_size=batch_size)
    # # Train
    # save_dir = f"save/{dataset}/{decoder}/"
    # save_path = f"save/{dataset}/{decoder}/model_{learning_rate}"
    # model_path = f"save/{dataset}/model_{learning_rate}_{n_latent}"
    
    # os.makedirs(save_dir, mode=0o777, exist_ok=True)
    VAE = Variable_AutoEncoder(n_in=X_train.shape[1], n_latent=n_latent, n_h=500)
    VAE = VAE.to(device)
    # VAE.load_state_dict(torch.load('VAE.ckpt'))
    criteon = nn.MSELoss()
    optimizer = optim.Adam(VAE.parameters(), lr=learning_rate)
    print("start train...")
    
    
    t = time.time()
    
    for epoch in range(epochsize):
        # 训练网络
        for batchidx, realdata in enumerate(data_loader):
            # 生成假图像
            fakedata, m, sigma = VAE(realdata[0])
            KLD = 0.5 * torch.sum(
                torch.pow(m, 2) +
                torch.pow(sigma, 2) -
                torch.log(1e-8 + torch.pow(sigma, 2)) - 1
            ) / (realdata[0].size(0)*28*28)
            # 计算均方差损失
            # MSE = criteon(fakeimage, realimage)
            MSE = torch.sum(torch.pow(fakedata - realdata[0], 2)) / (realdata[0].size(0))
            # 总的损失函数
            loss = MSE + KLD
            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batchidx%300 == 0:
                print("epoch:{}/{}, batchidx:{}/{}, loss:{}, MSE:{}, KLD:{}"
                    .format(epoch, epochsize, batchidx, len(X_train), loss, MSE, KLD))
    
    train_time = time.time() - t
    print(train_time)
    
    # Test
    for query in queries:
        # 不考虑join, 单表, 每种属性只查询一次
        _, predicate, _, _ = query
        est = 0
        left_bounds, right_bounds = my_utils.make_points(table.attributes, predicate, column_min_max_vals, bias)
        prob, _, _ = VAE(left_bounds)
        print(prob)
        break   
