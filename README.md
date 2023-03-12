# learning_cardinality_estimator_distributed_exploration

在分布式条件下评估已有的学习型基数估计器性能，研究探索适用于分布式数据库的学习型基数估计器应具有的性质和特点

## 模拟简单的分布式查询环境

### 0.实验准备

 0.1 绘图

  python脚本 /utils/dawning.py。  

  使用qerror表现AI基数估计器的准确程度。  

  为了便于显示qerror都被ln函数处理过了。  

  越接近x轴的表示越精确的，所以在同列的点中，越下方的点表示越好的性能，两点越接近表示性能越接近。  

 0.2 实验设计  

  mscn和deepdb对比 : query-driven vs data-driven，两种简单模型。缺陷在于两模型结构不同，无法控制变量。

  deepdb和Naru/Neurocard对比 : data-driven的简单模型vs data-driven的复杂模型。  

  Naru和UAE-Q对比 : data-driven的复杂模型 vs query-driven的复杂模型(都是深度自回归模型)。  
  
  mscn和UAE-Q对比 : query-driven的简单模型 vs query-driven的复杂模型。  

 0.3 问题描述

  query/Q : 未经分布式系统划分，全局的、完整的、原始的查询语句。  

  subquery/subQ/distributed_query/distributed_Q : 经分布式系统划分后各分布式节点中局部的、片段的、新生成的查询语句。$\color{red}{这种查询准确得说，是因为在各分布式节点中存在的属性并不是所属Table中的全部属性，为了在某一分布式节点中执行某一条query/Q，需要参考这一节点没存储的属性，把原本的query/Q中对应属性的谓词删除(改写成永真的，或是其他masking方法)，而生成/改写出的一组新的查询。}$  

  Table/T/Relation/R : 未经过分布式系统划分，全局的、完整的、原始的关系表，简称为原始表。  
  
  subTable/Ts/subRelation/Rs : 经分布式系统划分后各分布式节点中局部的、片段的、新生成的关系表，简称为子表。  

  分布式基数估计 : 在分布式数据库中到来的一条sql语句，估计query/Q 的基数制定查询计划(是必须的吗)，估计全部的subquery/subQ/distributed_query/distributed_Q的基数制定对应的分布式查询计划。  

  适用操作 : 每个属性有且只有一个的=,<, >(≤,≥也可以支持吗)。采用特定的编码方式支持每个属性多个以及多种操作。

 ***$\color{Azure}{基于学习的ai分布式基数估计器实现目标：满足在各结点上对应子查询的基数估计准确同时，也要满足该完整查询语句的准确基数估计！}$***

### 1.模拟最简单的垂直划分，多少属性多少子表，query只在单表上进行

 总共生成10W条数据。  

 在get_truecard.py中添加划分子表的代码。存在一定问题，每次运行已有version前要记得删除对应的分布式子查询sql ！！！，有时间的话记得优化！！！  

 在各学习型基数估计的执行python文件中添加测试子表的代码。  

 目前分布式条件下的基数估计 : 基于独立性假设，相互独立的(相对的)不同属性之间的选择率乘积 *rows。  

 1.1 mscn  

  代表类型 : query-driven中的一般神经网络模型。

  只需在编码时修改，不需要读子表对应的sql文件。  

  修改为每个编码只对一个predict编码，其他不变。  

  初步结论 :  

   mscn在对分布式查询的估计效果目前来看和真实基数有很大差距。  

   mscn的结果没有能体现出数据间的相关性，基于独立性假设的真实基数估计可以看出在corr=2时的准确度要显著优于corr=8的情况。  

   1.1.1 cols_2_distinct_10000_corr_2_skew_2  
   ![cols_2_distinct_10000_corr_2_skew_2](https://github.com/spiceandwolf/learning_cardinality_estimator_distributed_exploration/blob/main/Synthetic/cols_2_distinct_10000_corr_2_skew_2.mscn.png)  

   1.1.2 cols_2_distinct_10000_corr_8_skew_2  
   ![cols_2_distinct_10000_corr_8_skew_2](https://github.com/spiceandwolf/learning_cardinality_estimator_distributed_exploration/blob/main/Synthetic/cols_2_distinct_10000_corr_8_skew_2.mscn.png)  

 1.2 deepdb  

  代表类型 : data-driven中的SPN模型。  

  根据Synthetic/deepdb/deepdb_job_ranges/evaluation/cardinality_evaluation.py修改出测试分布式各节点上子表的代码。  

  初步结论：  

   在这两个原始数据集上，deepdb的结构使其结果基本不受从集中式到分布式转变的影响，表现非常好，几乎可以cover对应分布式查询的真实基数。  

   但也反映了对于高相关性的情况下，deepdb在完整查询语句下的估计效果因为当前采用的分布式条件下基于独立性假设的计算方法(相互独立的(相对的)不同属性之间的选择率乘积* rows)而打折扣，还是无法同时满足所需目标。  

   在现在这种数据集上的效果最好，需要更复杂的模拟环境做进一步研究。  

   1.2.1 cols_2_distinct_10000_corr_2_skew_2  
   ![cols_2_distinct_10000_corr_2_skew_2](https://github.com/spiceandwolf/learning_cardinality_estimator_distributed_exploration/blob/main/Synthetic/cols_2_distinct_10000_corr_2_skew_2.deepdb.png)  

   1.2.2 cols_2_distinct_10000_corr_8_skew_2  
   ![cols_2_distinct_10000_corr_8_skew_2](https://github.com/spiceandwolf/learning_cardinality_estimator_distributed_exploration/blob/main/Synthetic/cols_2_distinct_10000_corr_8_skew_2.deepdb.png)

 1.3 Naru/NerouCard  

  代表类型 : data-driven中的自回归模型。  

  Naru在Synthetic/naru/eval_model.py的RunN()函数里读取已转成csv格式的sql，所以要先生成各节点上子表对应的csv格式sql，然后去用RunN()去查询。  

  Naru使用Synthetic/naru/estimators.py中的ProgressiveSampling类构建估计器。
  存在一个小bug，如果使用—column-masking优化会导致无法仅对一列进行基数估计。  

  初步结论 :
   在低相关性的数据集上，分布式结构下的Naru表现出非常不错的效果，和子查询的真实基数估计的结果都很接近，同时完整查询的log(qerror)也不高。  

   在高相关性的数据集上，分布式结构下的Naru的结果和子查询的真实基数估计的结果对于完整查询的log(qerror)都有所升高，不过大部分例子中两者性能都很接近，有时分布式结构下的Naru对完整查询的log(qerror)还是更低的一个，说明可能具有更好的估计效果。  

   1.3.1 cols_2_distinct_10000_corr_2_skew_2  
   ![cols_2_distinct_10000_corr_2_skew_2](https://github.com/spiceandwolf/learning_cardinality_estimator_distributed_exploration/blob/main/Synthetic/cols_2_distinct_10000_corr_2_skew_2.naru.png)

   1.3.2 cols_2_distinct_10000_corr_8_skew_2  
   ![cols_2_distinct_10000_corr_2_skew_2](https://github.com/spiceandwolf/learning_cardinality_estimator_distributed_exploration/blob/main/Synthetic/cols_2_distinct_10000_corr_2_skew_2.naru.png)

 1.4 UAE/UAE-Q
  初步结论 :  
  1.4.1  
  1.4.2  

### 2.模拟一般分布式数据库的数据划分，query只在单表上进行

 总共生成10w条数据。  

 4种属性A1, A2, A3, A4，值域1w，相关性和偏斜度是变量，其中A1和A2, A3, A4存在相关性，这两组之间相互独立。数据分布在4个节点P1, P2, P3, P4上，A1作为全局主键。  

    |  |A1                |A2                |A3                |A4                |  
    |--|------------------|------------------|------------------|------------------|
    |P1|✓(0~5w)           |✓(0~5w)          |                  |                  |
    |P2|✓(0~5w)           |                 |✓(0~5w)           |✓(0~5w)           |
    |P3|✓(5w~10w)         |✓(5w~10w)        |                  |                  |
    |P4|✓(5w~10w)         |                  |✓(5w~10w)         |✓(5w~10w)        |

  对于使用学习型基数估计器的情况，先将原本的query/Q按照给定的划分模式改写成subquery/subQ后用join连接起来，再进行查询。但还是无法处理水平划分带来的影响？  
  对于使用postgresql数据库进行分布式基数估计的情况，先各分布式节点中用不同名的表来模拟，再将原本的query/Q按照给定的划分模式和对应的表名改写成subquery/subQ后用join连接起来，直接交给postgresql做查询处理，从而获取真实基数。  
  但传统方法的分布式基数估计怎么模拟？用subquery/subQ直接在模拟分布式节点的不同名表上查询然后怎么估算？还是不去考虑这个问题？ 

 ***现有学习型基数估计模型如何应对分布式环境下对原本表结构的改变(表被分区后，额外添加的分区键及垂直化分导致属性数量的改变，水平划分导致数据分布的变化，更多子表的产生导致需要更多的join操作)？***

 在postgresql数据库中建立多个db模拟分布式节点，每个db实例中都有表{table.table_name}

  2.1 mscn  
  2.2 deepdb  
  2.3 Naru/NerouCard  
