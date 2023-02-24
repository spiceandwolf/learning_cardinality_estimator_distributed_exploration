from math import log
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

def example():
    # 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
    plt.figure(figsize=(8, 6), dpi=80)

    # 再创建一个规格为 1 x 1 的子图
    plt.subplot(1,1,1)

    x = np.linspace(-2, 6, 50)
    y1 = x + 3      # 曲线 y1
    y2 = 3 - x      # 曲线 y2

    # 绘制颜色为蓝色、宽度为 1 像素的连续曲线 y1
    plt.plot(x, y1, color="blue", linewidth=1.0, linestyle="-", label="y1")
    # 绘制散点(3, 6)
    plt.scatter([3], [6], s=30, color="blue")      # s 为点的 size
    # 对(3, 6)做标注
    plt.annotate("(3, 6)",
                xy=(3.3, 5.5),       # 在(3.3, 5.5)上做标注
                fontsize=16,         # 设置字体大小为 16
                xycoords='data')  # xycoords='data' 是说基于数据的值来选位置
    # 绘制颜色为紫色、宽度为 2 像素的不连续曲线 y2
    plt.plot(x, y2, color="#800080", linewidth=2.0, linestyle="--", label="y2")
    # 绘制散点(3, 0)
    plt.scatter([3], [0], s=50, color="#800080")
    # 对(3, 0)做标注
    plt.annotate("(3, 0)",
                xy=(3.3, 0),            # 在(3.3, 0)上做标注
                fontsize=16,          # 设置字体大小为 16
                xycoords='data')    # xycoords='data' 是说基于数据的值来选位置
    plt.text(4, -0.5, "this point very important",
         fontdict={'size': 12, 'color': 'green'})  # fontdict设置文本字体

    # 显示图例
    plt.legend(loc="upper left")

    x = np.linspace(0,10,100)
    # 绘制散点图
    plt.scatter(x, np.sin(x), marker='o')
    plt.grid()

    # 绘制更复杂的散点图
    rng = np.random.RandomState(0)
    x = rng.randn(100)
    y = rng.randn(100)
    colors = rng.rand(100)
    sizes = 1000 * rng.rand(100)
    plt.scatter(x,y,c=colors,s=sizes,alpha=0.3)
    plt.colorbar()
    plt.grid(True)

    # 设置横轴的上下限
    plt.xlim(-1, 6)
    # 设置纵轴的上下限
    plt.ylim(-2, 10)

    # 设置横轴标签
    plt.xlabel("X")
    # 设置纵轴标签
    plt.ylabel("Y")

    # 设置横轴精准刻度
    plt.xticks([-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
    # 设置纵轴精准刻度
    plt.yticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # 带标签
    # 设置横轴精准刻度
    # plt.xticks([-1, 0, 1, 2, 3, 4, 5, 6],
        #    ["-1m", "0m", "1m", "2m", "3m", "4m", "5m", "6m"])
    # 设置纵轴精准刻度
    # plt.yticks([-2, 0, 2, 4, 6, 8, 10],
        #    ["-2m", "0m", "2m", "4m", "6m", "8m", "10m"])

    # 保存图片 直接输入文件路径，新建文件名字 
    # dpi图片分辨率 越高越清晰。bbox_inches = 'tight',去掉空白部分
    plt.savefig(r'test.png', dpi=400, bbox_inches = 'tight')

    plt.show()

# 为了便于画图，对qerror求对数
def qerror_ln(qerrors):
    qerror_lns = []
    for qerror in qerrors:
        qerror = log(qerror)
        qerror_lns.append(qerror)
    return qerror_lns

def qerror(preds, targets):
    qerrors = []
    for i in range(len(targets)):
        if (preds[i] > targets[i]):
            # if (preds[i] / targets[i]) > 10:
            #     qerrors.append(10.)
            # else:
            #     qerrors.append(preds[i] / targets[i])
            qerrors.append(preds[i] / targets[i])
        else:
            # if targets[i] / preds[i] > 10:
            #     qerrors.append(10.)
            # else:
            #     qerrors.append(targets[i] / preds[i])
            qerrors.append(targets[i] / preds[i])
    return qerrors

def drawn_scattergram(path, scatter_infos, fig_size):
    
    plt.figure(figsize=fig_size, dpi=1000)
    plt.subplot(1,1,1)
    plt.xlim((0, 101))
    plt.ylim((0, 12))
    plt.xticks(np.linspace(1, 100, 100))
    plt.yticks(np.linspace(0, 12, 13))
    plt.xlabel("sqls")
    plt.ylabel("card")
    plt.grid(axis="x", linestyle='-')

    for list_x, list_y, color, label in scatter_infos:
        plt.scatter(list_x[:100], list_y[:100], c=color, label=label)

    plt.legend(loc="upper left")

    plt.savefig(path, dpi=1000, bbox_inches = 'tight')   

def draw_cards_from_sql(file_path):
    cards = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            cardinality = line.split(',')[-1].strip('\n')
            cards.append((float(cardinality)))
    return cards

def draw_learning_card(file_path, count):
    cards = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            cardinality = line.split(',')[0]
            cards.append(float(cardinality))
    subsets = []
    if count == 1:
        return cards
    for i in range(count):
        subset = cards[0 + i:len(cards):count]
        subsets.append(subset)
    return subsets

# example()

# mscn
# cols_2_distinct_10000_corr_2_skew_2
# version = 'cols_2_distinct_10000_corr_2_skew_2'
# file_path = './sql_truecard' + '/'f"{version}test.sql"
# distributed_pg_card_path_1 = './sql_truecard' + '/'f"{version}test_cdcs.col0.sql"
# distributed_pg_card_path_2 = './sql_truecard' + '/'f"{version}test_cdcs.col1.sql"
# distributed_mscn_card_path = './sql_truecard/' + '/'f"{version}test.sql.mscn.result.distributed.csv"
# mscn_card_path = './sql_truecard' + '/'f"{version}test.sql.mscn.result.csv"
# fig_path = version + '.mscn.png'

# distributed_pg_cards_1 = draw_cards_from_sql(distributed_pg_card_path_1)
# distributed_pg_cards_2 = draw_cards_from_sql(distributed_pg_card_path_2)
# true_cards = draw_cards_from_sql(file_path)
# distributed_mscn_cards_1, distributed_mscn_cards_2 = draw_learning_card(distributed_mscn_card_path, 2)
# mscn_cards = draw_learning_card(mscn_card_path, 1)

# distributed_mscn_cards = []
# for card_1, card_2 in zip(distributed_mscn_cards_1, distributed_mscn_cards_2):
#     distributed_mscn_cards.append(card_1 * card_2 / 100000)

# distributed_true_cards = []
# for card_1, card_2 in zip(distributed_pg_cards_1, distributed_pg_cards_2):
#     distributed_true_cards.append(card_1 * card_2 / 100000)

# distributed_true_qerrors = qerror(distributed_true_cards, true_cards)
# distributed_true_qerrors = qerror_ln(distributed_true_qerrors)
# distributed_mscn_qerrors = qerror(distributed_mscn_cards, true_cards)
# distributed_mscn_qerrors = qerror_ln(distributed_mscn_qerrors)


# counts = np.linspace(1, len(true_cards), len(true_cards))
# scatter_infos = [
#     [counts, distributed_true_qerrors, 'dodgerblue', 'distributed_true_qerrors'],
#     [counts, distributed_mscn_qerrors, 'red', 'distributed_mscn_qerrors']
# ]
# scatter_infos = [
#     [counts, distributed_pg_cards_1, 'peru', 'distributed_true_cards_1'],
#     [counts, distributed_pg_cards_2, 'brown', 'distributed_true_cards_2'],
#     [counts, true_cards, 'black', 'true_cards'],
#     [counts, distributed_mscn_cards_1, 'dodgerblue', 'distributed_mscn_cards_1'],
#     [counts, distributed_mscn_cards_2, 'green', 'distributed_mscn_cards_2'],
#     [counts, mscn_cards, 'red', 'mscn_cards']
# ]
# cols_2_distinct_10000_corr_2_skew_2

# cols_2_distinct_10000_corr_8_skew_2
version = 'cols_2_distinct_10000_corr_8_skew_2'
file_path = './sql_truecard' + '/'f"{version}test.sql"
distributed_pg_card_path_1 = './sql_truecard' + '/'f"{version}test_cdcs.col0.sql"
distributed_pg_card_path_2 = './sql_truecard' + '/'f"{version}test_cdcs.col1.sql"
distributed_mscn_card_path = './sql_truecard' + '/'f"{version}test.sql.mscn.result.distributed.csv"
mscn_card_path = './sql_truecard' + '/'f"{version}test.sql.mscn.result.csv"
fig_path = version + '.mscn.png'

distributed_pg_cards_1 = draw_cards_from_sql(distributed_pg_card_path_1)
distributed_pg_cards_2 = draw_cards_from_sql(distributed_pg_card_path_2)
true_cards = draw_cards_from_sql(file_path)
distributed_mscn_cards_1, distributed_mscn_cards_2 = draw_learning_card(distributed_mscn_card_path, 2)
mscn_cards = draw_learning_card(mscn_card_path, 1)

distributed_mscn_cards = []
for card_1, card_2 in zip(distributed_mscn_cards_1, distributed_mscn_cards_2):
    distributed_mscn_cards.append(card_1 * card_2 / 100000)

distributed_true_cards = []
for card_1, card_2 in zip(distributed_pg_cards_1, distributed_pg_cards_2):
    distributed_true_cards.append(card_1 * card_2 / 100000)

distributed_true_qerrors = qerror(distributed_true_cards, true_cards)
distributed_true_qerrors = qerror_ln(distributed_true_qerrors)
distributed_mscn_qerrors = qerror(distributed_mscn_cards, true_cards)
distributed_mscn_qerrors = qerror_ln(distributed_mscn_qerrors)

counts = np.linspace(1, len(true_cards), len(true_cards))
scatter_infos = [
    [counts, distributed_true_qerrors, 'dodgerblue', 'distributed_true_qerrors'],
    [counts, distributed_mscn_qerrors, 'red', 'distributed_mscn_qerrors']
]
# cols_2_distinct_10000_corr_8_skew_2
# mscn

# deepdb
# cols_2_distinct_10000_corr_2_skew_2
# version = 'cols_2_distinct_10000_corr_2_skew_2'
# file_path = './sql_truecard' + '/'f"{version}test.sql"
# distributed_pg_card_path_1 = './sql_truecard' + '/'f"{version}test_cdcs.col0.sql"
# distributed_pg_card_path_2 = './sql_truecard' + '/'f"{version}test_cdcs.col1.sql"
# distributed_deepdb_card_path = './sql_truecard/cols_2_distinct_10000_corr_2_skew_2test.sql.deepdb.result.distributed.csv'
# deepdb_card_path = './sql_truecard/cols_2_distinct_10000_corr_2_skew_2test.sql.deepdb.results.csv'
# fig_path = version + '.deepdb.png'

# distributed_pg_cards_1 = draw_cards_from_sql(distributed_pg_card_path_1)
# distributed_pg_cards_2 = draw_cards_from_sql(distributed_pg_card_path_2)
# true_cards = draw_cards_from_sql(file_path)
# distributed_deepdb_cards_1, distributed_deepdb_cards_2 = draw_learning_card(distributed_deepdb_card_path, 2)

# distributed_deepdb_cards = []
# for card_1, card_2 in zip(distributed_deepdb_cards_1, distributed_deepdb_cards_2):
#     distributed_deepdb_cards.append(card_1 * card_2 / 100000)

# distributed_true_cards = []
# for card_1, card_2 in zip(distributed_pg_cards_1, distributed_pg_cards_2):
#     distributed_true_cards.append(card_1 * card_2 / 100000)

# distributed_true_qerrors = qerror(distributed_true_cards, true_cards)
# distributed_true_qerrors = qerror_ln(distributed_true_qerrors)
# distributed_deepdb_qerrors = qerror(distributed_deepdb_cards, true_cards)
# distributed_deepdb_qerrors = qerror_ln(distributed_deepdb_qerrors)

# counts = np.linspace(1, len(true_cards), len(true_cards))
# scatter_infos = [
#     [counts, distributed_true_qerrors, 'dodgerblue', 'distributed_true_qerrors'],
#     [counts, distributed_deepdb_qerrors, 'red', 'distributed_deepdb_qerrors']
# ]
# cols_2_distinct_10000_corr_2_skew_2

# cols_2_distinct_10000_corr_8_skew_2
# version = 'cols_2_distinct_10000_corr_8_skew_2'
# file_path = './sql_truecard' + '/'f"{version}test.sql"
# distributed_pg_card_path_1 = './sql_truecard' + '/'f"{version}test_cdcs.col0.sql"
# distributed_pg_card_path_2 = './sql_truecard' + '/'f"{version}test_cdcs.col1.sql"
# distributed_deepdb_card_path = './sql_truecard' + '/'f"{version}test.sql.deepdb.result.distributed.csv"
# deepdb_card_path = './sql_truecard' + '/'f"{version}test.sql.deepdb.results.csv"
# fig_path = version + '.deepdb.png'

# distributed_pg_cards_1 = draw_cards_from_sql(distributed_pg_card_path_1)
# distributed_pg_cards_2 = draw_cards_from_sql(distributed_pg_card_path_2)
# true_cards = draw_cards_from_sql(file_path)
# distributed_deepdb_cards_1, distributed_deepdb_cards_2 = draw_learning_card(distributed_deepdb_card_path, 2)

# distributed_deepdb_cards = []
# for card_1, card_2 in zip(distributed_deepdb_cards_1, distributed_deepdb_cards_2):
#     distributed_deepdb_cards.append(card_1 * card_2 / 100000)

# distributed_true_cards = []
# for card_1, card_2 in zip(distributed_pg_cards_1, distributed_pg_cards_2):
#     distributed_true_cards.append(card_1 * card_2 / 100000)

# distributed_true_qerrors = qerror(distributed_true_cards, true_cards)
# distributed_true_qerrors = qerror_ln(distributed_true_qerrors)
# distributed_deepdb_qerrors = qerror(distributed_deepdb_cards, true_cards)
# distributed_deepdb_qerrors = qerror_ln(distributed_deepdb_qerrors)
# counts = np.linspace(1, len(true_cards), len(true_cards))
# scatter_infos = [
#     [counts, distributed_true_qerrors, 'dodgerblue', 'distributed_true_qerrors'],
#     [counts, distributed_deepdb_qerrors, 'red', 'distributed_deepdb_qerrors']
# ]
# cols_2_distinct_10000_corr_8_skew_2
# deepdb

# naru
# cols_2_distinct_10000_corr_2_skew_2
# version = 'cols_2_distinct_10000_corr_2_skew_2'
# file_path = './sql_truecard' + '/'f"{version}test.sql"
# distributed_pg_card_path_1 = './sql_truecard' + '/'f"{version}test_cdcs.col0.sql"
# distributed_pg_card_path_2 = './sql_truecard' + '/'f"{version}test_cdcs.col1.sql"
# distributed_naru_card_path = './sql_truecard' + '/'f"{version}test.sql.naru.result.distributed.csv"
# naru_card_path = './sql_truecard' + '/'f"{version}test.sql.naru.results.csv"
# fig_path = version + '.naru.png'

# distributed_pg_cards_1 = draw_cards_from_sql(distributed_pg_card_path_1)
# distributed_pg_cards_2 = draw_cards_from_sql(distributed_pg_card_path_2)
# true_cards = draw_cards_from_sql(file_path)
# distributed_naru_cards_1, distributed_naru_cards_2 = draw_learning_card(distributed_naru_card_path, 2)

# distributed_naru_cards = []
# for card_1, card_2 in zip(distributed_naru_cards_1, distributed_naru_cards_2):
#     distributed_naru_cards.append(card_1 * card_2 / 100000)

# distributed_true_cards = []
# for card_1, card_2 in zip(distributed_pg_cards_1, distributed_pg_cards_2):
#     distributed_true_cards.append(card_1 * card_2 / 100000)

# distributed_true_qerrors = qerror(distributed_true_cards, true_cards)
# distributed_true_qerrors = qerror_ln(distributed_true_qerrors)
# distributed_naru_qerrors = qerror(distributed_naru_cards, true_cards)
# distributed_naru_qerrors = qerror_ln(distributed_naru_qerrors)
# counts = np.linspace(1, len(true_cards), len(true_cards))
# scatter_infos = [
#     [counts, distributed_true_qerrors, 'dodgerblue', 'distributed_true_qerrors'],
#     [counts, distributed_naru_qerrors, 'red', 'distributed_naru_qerrors']
# ]
# cols_2_distinct_10000_corr_2_skew_2

# cols_2_distinct_10000_corr_8_skew_2
# version = 'cols_2_distinct_10000_corr_8_skew_2'
# file_path = './sql_truecard' + '/'f"{version}test.sql"
# distributed_pg_card_path_1 = './sql_truecard' + '/'f"{version}test_cdcs.col0.sql"
# distributed_pg_card_path_2 = './sql_truecard' + '/'f"{version}test_cdcs.col1.sql"
# distributed_naru_card_path = './sql_truecard' + '/'f"{version}test.sql.naru.result.distributed.csv"
# naru_card_path = './sql_truecard' + '/'f"{version}test.sql.naru.results.csv"
# fig_path = version + '.naru.png'

# distributed_pg_cards_1 = draw_cards_from_sql(distributed_pg_card_path_1)
# distributed_pg_cards_2 = draw_cards_from_sql(distributed_pg_card_path_2)
# true_cards = draw_cards_from_sql(file_path)
# distributed_naru_cards_1, distributed_naru_cards_2 = draw_learning_card(distributed_naru_card_path, 2)

# distributed_naru_cards = []
# for card_1, card_2 in zip(distributed_naru_cards_1, distributed_naru_cards_2):
#     distributed_naru_cards.append(card_1 * card_2 / 100000)

# distributed_true_cards = []
# for card_1, card_2 in zip(distributed_pg_cards_1, distributed_pg_cards_2):
#     distributed_true_cards.append(card_1 * card_2 / 100000)

# distributed_true_qerrors = qerror(distributed_true_cards, true_cards)
# distributed_true_qerrors = qerror_ln(distributed_true_qerrors)
# distributed_naru_qerrors = qerror(distributed_naru_cards, true_cards)
# distributed_naru_qerrors = qerror_ln(distributed_naru_qerrors)
# counts = np.linspace(1, len(true_cards), len(true_cards))
# scatter_infos = [
#     [counts, distributed_true_qerrors, 'dodgerblue', 'distributed_true_qerrors'],
#     [counts, distributed_naru_qerrors, 'red', 'distributed_naru_qerrors']
# ]
# cols_2_distinct_10000_corr_8_skew_2
# naru

drawn_scattergram(fig_path, scatter_infos, (25, 24))
