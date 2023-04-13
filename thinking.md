# 随笔乱涂

2.0  
分析 : 分布式查询中涉及基数估计的操作  
结论 :   
查询优化中的直接连接和半连接等。3月更新 : 见问题2.20。   
2.1  
分析 : AI基数估计器具有更高的准确性及泛化能力(?)，无法频繁更新(缺点？但意味着稳定？或许可以依靠这一特性实现将大多数查询在本地副本进行查询优化？即查询本地主副本(负责写入的)+本地从副本(只是备份的)) ，要求AI基数估计器还是存在于各个节点上？3月更新 : ***所有节点都应该知晓数据库的schema?如果是全局统一的data-driven模型，需要先在各个节点中抽取足够的数据传输到某一个用于训练的节点上，再将训练好的模型分发，还可以直接解决多表连接情况下各节点中数据间的关系？问题在于数据量及传输代价；如果是局部的data-driven模型，直接在本地训练模型时不涉及数据传输，但解决各节点中数据间的关系，还需要传输数据到某一个用于训练的节点上用机器学习方法找出？这时为了计算多表连接下各表间的关系还是需要从各节点抽取数据再传输到某一结点上进行计算。但这一过程涉及到的数据量能否有所减小？从各节点本地训练得到的模型是各节点中数据的一种信息压缩，或许可以通过使用这些模型，而不是原始数据的样本来计算多表连接下对应的总体模型？这一idea出自论文[ Unsupervised Selectivity Estimation by Integrating Gaussian Mixture Models and an Autoregressive Model](https://openproceedings.org/2022/conf/edbt/paper-65.pdf), 这篇文章就先对一个relation中的属性用GMM建模然后再使用DAR模型给属性间的相关性建模。如果采用的是Gaussian Processes生产的模型所需的传输代价可能会相对于直接传数据来说更小。***   
结论 :  
优先选择局部混合模型？每一节点负责更新主副本的ai基数估计器，然后将其扩散给其他节点的从副本(有一致性问题吗，分布式机器学习是什么样的？)。3月更新 : 在分布式数据库中最后可能会会出现的情况是各节点依据相关性在逻辑上被划分成不同的集合，各集合负责维护的学习型基数估计器不同，集合内可能相同。   
2.2  
分析 : 如果要给所有节点同步模型，传输模型过程中的开销怎么计算？  
结论 :  
相比大模型更倾向于小模型？  
2.3  
分析 : AI基数估计器只在一个主节点上，直接根据这个AI基数估计器估计分布到各个节点的子查询对应的基数，然后确定查询策略？(优势是更少的通讯代价？)  
结论 :  
似乎可以使用基于查询驱动的AI基数估计器实现？3月更新 : 在各节点上，query-driven模型下的学习型基数估计器如何解决各种划分带来的问题？data-driven的模型似乎更适合分布式数据库？***当前有query-driven模型是基于query plan的，或许可以训练这种的query-driven模型？***    
2.4  
分析 : 如何处理站点依赖？  
2.5  
分析 : deepdb源码中的log和debug模式很友好，日后开发需要参考。  
2.6  
分析 : 估计前连接与估计后连接。估计前连接，对多表连接的形式建模。估计后连接，分布式条件下各节点估计值的连接。  
2.7  
分析 : Naru中为了计算范围查询采用采样的方式，为什么不用概率密度函数+积分? Naru的渐进式采样是在生成的Naru模型基础上根据Naru模型给出的条件概率分布进行采样的。渐进式采样只能串行执行？  
结论 :  
采样时使用多维直方图来代替会有多大影响？能否融合负采样？3月更新 : ***Naru使用自回归模型作为基础，自回归模型根据输入，输出一个由条件概率组成的序列，可以直接计算出点查询的概率(所以不能使用概率密度函数+积分的方法计算概率)。对于范围查询，如果是小的范围查询，查询范围内的所有点查询的概率之和就是这个范围查询的概率，但如果是较大的范围查询，因为无法继续这样容易的计算出所有范围内的点查询，因此一般使用蒙特卡罗法计算范围查询的概率。一般的蒙特卡洛法直接使用均匀采样，但存在误差，为了无偏采样，Naru使用渐进式采样是为了计算范围查询的概率。渐进式采样是一种使用S次采样估计查询范围R1 × · · · × Rn的密度的方法。*** Progressive sampling bears connections to sampling algorithms in graphical models. Notice that the autoregressive factorization corresponds to a complex graphical model where each node i has all nodes with indices < i as its parents. In this interpretation, progressive sampling extends the forward sampling with likelihood weighting algorithm [23] to allow variables taking on ranges of values (the former, in its default form, allows equality predicates only).原文中这一段还需要继续理解理解。  

***  
↑  2月  ↑  
***  

2.8  
~~分析 : 一条sql语句进入分布式数据库后到被进行基数估计都会经历哪些步骤？以OB为例 : 1.进行语法和词法解析，生成查询语法树。 ➔ 2.对查询语法树进行语义分析，生成对应的查询对象。 ➔ 3.依据关系代数对查询对象进行等价改写。 ➔ 4.为查询对象生成执行代价最优的逻辑计划。 ➔ 5……. 在第4步的时候一般就会使用到基数，在这一步或更前面的过程中就应该涉及根据查询对分布式数据库中各节点中有关的数据进行基数估计，第3步后获得的sql是要被基数估计前的最终形态？参考OB中的Multi-Part DML的样子，什么样的？~~感觉是最没用的分析。  
2.9  
分析 : 概率图与图卷积神经网络GNN  
2.10  
~~分析 : 黎曼积分与勒贝格积分，定义域和值域，数据和分布，查询和基数。~~并没什么意义。  
2.11  
分析 : 不同属性之间的数据可能不互相独立也不同分布，按行采样和按列采样有不同的意义。基于数据的方法都是按列采样的吗。不同节点间数据采样时算如何做到同分布？迁移学习？ 
结论 :   
Naru在训练的过程中是按行采样的(保持了属性之间的相关性？但对每个属性独自的数据分布会有影响吗？***尝试按块采样？有相关论文吗***)，但进行推理时似乎是按列采样？(因为是渐进式采样，需要补充论文原文作为依据。依据 : Intuitively, a sample of the first dimension $x_1^{(i)}$ would allow us to “zoom in” into the more meaningful region of the second dimension.)。***在某一个数据库确定的一个表(无论该表是原始完整的还是部分的还是join得到的)中采样时，每次采样都是独立的，样本都是来自于表中的数据，服从同一种联合分布，所以可以说样本是iid的；但如果是在不同的表上采样，能说明样本是同分布的吗？这种情况只能应用到切比雪夫大数定律的程度，此时中心极限定理只能应用林德伯格中心极限定理及李雅普诺夫Lyapunov中心极限定理，其中后者的应用场景最常用。参考(zhihu:概率论——大数定律与中心极限定理)[https://zhuanlan.zhihu.com/p/259280292?utm_source=wechat_session]。***  
2.12  
分析 : 现有学习型基数估计器能否和数据分布到某一节点的函数相结合？分布式的学习型基数估计器在构造时应该考虑这一点。  
2.13  
分析 : 幂律分布。“DeepWalk中如果图符合幂律分布的话，就可以用NLP的方法做了，论文是用的Word2vec”，数据库中数据的分布会符合幂律分布吗。  
2.14   
分析 : 在[OB4.0的分布式查询优化相关资料](https://zhuanlan.zhihu.com/p/586113453)中有提到 : ***分布式查询优化一定要使用一阶段的方法，即要同时枚举本地算法和分布式算法并且使用分布式代价模型来计算代价，而不是通过分阶段的方式来枚举本地算法和分布式算法。  在各节点中的传统基数估计器都是“全局相同结构的”，因为在连接时从其他节点上拉取数据的schema很有可能和本地的schema不同，所以各节点应知晓数据库中所有schema的全貌。因此，与之相对应的，现有的学习型基数估计方法要是想应用于分布式数据库，各个节点中持有的学习型基数估计模型在训练时也必须是涵盖所有schema。***  ~~半连接优化 : 只需要各节点自己的基数估计器。 直接连接优化 : ~~  
结论 :   
能否说现有的学习型基数估计器无法适配分布式数据库？query/data-driven的基数估计器都会因为水平划分造成的不同节点上复数个同构子schema而不能正常工作，因为它们可能在这些场地上产生相似的结果(受困于当前学习型基数估计器的实现，有些学习型基数估计器在估计同一查询时的结果可能不唯一)，另外在训练模型时，特别是data-driven的，受到水平划分的影响，只能得到所属节点自身的数据模型，没法直接用于正常的基数估计？；应在模型中引入分区信息，降低受到水平划分的影响。能否实现不依赖于schema全貌的学习型基数估计，每个节点的学习型基数估计只需专注自己的schema，但有来自其他场地连接操作时只需要把其他场地的模型拼在一起就能继续正常估计？  
2.15  
分析 : 直方图法在的join条件下的估计过程。  
结论 :   
在[3.6.1.2 Refinements: Relative Effectiveness of Histograms](https://dsf.berkeley.edu/cs286/papers/synopses-fntdb2012.pdf)  
2.16  
分析 : [贝叶斯优化不需要求导数。](https://zhuanlan.zhihu.com/p/76269142)训练过程中求导数的重要性可以参照UAE模型的论文。主要问题是在一个节点集合内(已经不适用独立性假设)的学习型基数估计器的损失函数可能不可导(毕竟各节点的种类不同，有些属性相同，有些不同。需要数学证明？)，此时使用这种优化方法？    
2.17  
分析 : 使用模型融合的方式来得到最后的模型？Stacking可以与无监督学习方法结合，案例可参考Kaggle的“Otto Group Product Classification Challenge”中，Mike Kim提出的方法 [6]。(高斯过程和meta-learning结合)[https://zhuanlan.zhihu.com/p/146818995], 各表间分布十分近似的就可以认为是同分布的，可以一起处理。  
结论 :   
根据new bing的回答贝叶斯深度学习和meta learning相结合的例子有bayesian meta-learning for the few-shot setting via deep kernels, bayesian model-agnostic meta-learning, pac-bayesian meta-learning: from theory to practice. 参考这些方法构造模型？  
2.18  
分析 : FLAT在处理多表连接时没有使用全外连接的方式，而是局部连接的一种树结构，可以参考。  
2.19  
分析 : localnn模型的效果和MSCN模型的效果差不多([见Learned Cardinality Estimation : A Design Space Exploration and a Comparative Evaluation](http://dbgroup.cs.tsinghua.edu.cn/ligl/papers/vldb22-card-exp.pdf))，但为什么基本没有后续研究。  
2.20  
分析 : 分布式查询下学习型基数估计器对半连接算法的影响？半连接(semi-join)是对全连接结果属性列的一种缩减操作,它由投影和连接操作导出,投影操作实现连接属性基数的缩减,连接操作实现左连接关系元组数的缩减。  
结论 :   
***在分布式数据库中的查询优化，需要估算查询造成的多表连接的基数。更准确的基数估计主要是影响半连接算法的准确度，不影响算法本身？***解决了问题2.0？基数估计在物理操作符(hash join/index scan)被选中前执行。  
2.21  
分析 : PRMs, probabilitistic relational models, 概率关系模型。一个PRM包含了schema全部内容的模型 : NeuroCard; 多个PRMs包含了schema全部内容的模型 : FLAT, BayesCard, Deepdb
结论 :    
根据综述[Cardinality Estimation in DBMS: A Comprehensive Benchmark Evaluation](https://arxiv.org/pdf/2109.05877.pdf), 多个PRMs构成的模型的泛化能力更强，尤其是在涉及属性的数量越多，数据偏斜越严重的复杂数据集上(真实世界中的数据也有这样的特点)。在FACE模型的论文中比较了FACE模型分别采用这两种方式的q-error和模型尺寸，结果显示还是多个PRMs更站优势。  
2.22  
分析 : 根据综述[Cardinality Estimation in DBMS: A Comprehensive Benchmark Evaluation](https://arxiv.org/pdf/2109.05877.pdf)，q-error不能完全刻画基数估计器在正确率上的特点，越大的基数在查询优化方面会造成的影响越大，所以准确估计大基数的重要性比准确估计小基数大重要性要高。***因此损失函数应该能反映这个特点！***  
2.23  
分析 : ***在分布式上做文章，不要太在意单个节点的估计准确率！***   
2.24  
分析 : deepdb默认在2kw条数据下建模，如果超出了直接采样2kw。  
2.25  
分析 : face的可调整重要度采样似乎是根据分布进行采样的，在推理阶段使用，用于蒙特卡洛过程。在训练阶段，face为了处理离散变量用于训练将其进行去量化处理，其过程是使用三次样条插值将其累计分布函数CDF构造成连续可导的，然后在离散值对应的整数内抽样求概率。概率带入CDF的反函数得到的值就是去量化后的值。推理时不用去量化分布进行基数估计的原因是所用的去量化分布都是边缘分布，估计时所需要的是联合分布。***去量化的目的是因为离散变量的概率密度函数用正则流拟合得到的函数特点很差，概率基本集中在离散点取值的周围，并且难以求积分。*** ***直接把这个方法应用在直方图上也行？*** 在提取单表中每列之间的相关性时，FACE模型采用的是一个叫coupling层的结构，但这个结构是正则流模型中需要训练的部分。  
2.26  
分析 : 关于VAE模型，在原始空间中的概率是否也等于在隐空间的概率，能不能通过在隐空间求概率来得到原始空间的概率。  
2.27  
分析 : 可以在自回归模型中引入分布增强，提升模型能力，缩小模型的大小。  

***  
↑  3月  ↑  
***  
  
2.28  
分析 : Skimmed Sketches. The skimmed sketch technique [102] observes that much of the error in join size estimation using sketches arises from collisions with high frequencies. Instead, Ganguly et al. propose “skimming” off the high frequency items from each relation by extracting the (approximate) heavy hitters, so each relation is broken into a “low” and a “high” relation. The join can now be broken into four pieces, each of which can be estimated from either the estimated heavy hitters, or from sketches after the contributions of the sketches have been subtracted off. These four pieces can be thought of as (a) highhigh (product of frequencies of items which are heavy hitters in both relations) (b) low-low (inner product estimation from the skimmed sketches) and (c) high-low and low-high (product of heavy hitter items with corresponding items from the other sketch). This is shown to be an improvement over the original scheme based on averaging multiple estimates together (Section 5.3.3.1). However, it is unclear whether there is a significant gain over the hashing version of AMS sketches where the hashing randomly separates the heavy hitters with high probability.   
Using the Count-Min sketch to estimate inner products.  
Using the AMS Sketch to estimate inner products.   
以上大部分属于[Synopses for Massive Data: Samples, Histograms, Wavelets, Sketches](https://dsf.berkeley.edu/cs286/papers/synopses-fntdb2012.pdf)5.3.4.2章的内容。使用各种sketck进行join size的估计都是用各种sketch对查询涉及的数据频率建模，相同数据间的相乘后再把查询范围内的对应乘积相加。factorJoin也是差不多的思路，使用分桶的方式进行建模。  
对join size的估计方法可以分成2大类，分别是基于直方图(指多维直方图，sketches和wavelets)的和基于采样的，用机器学习来解决join size估计问题可能更应该从基于直方图的方向去考虑。