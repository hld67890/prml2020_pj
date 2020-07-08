# prml2020_pj

运行main.py就可以用啦。

getans()用来算没有答案的test.csv，getfold()用来算10折验证。



### 文件

##### fingerprint+sigmoid线性层

train_cv,test.csv,train.csv都是自带的数据。submit.csv是预测test.csv的结果。

main.py,getdata.py,model.py都是模型的代码，运行main.py就可以了。

pred文件夹里面是十折验证时候输出的预测结果，myans.out是预测test.csv时输出的预测结果。

##### fingerprint+random forest

train_cv,test.csv,train.csv都是自带的数据。submit_prob.csv,submit_class.csv是预测test.csv的结果，分别是直接输出概率和输出分类（按0.5截断）的结果。

main.py,getdata.py,model.py都是模型的代码，运行main.py就可以了。

pred文件夹里面是十折验证时候输出的预测结果，myans.out是预测test.csv时输出的预测结果。

调参.csv记录了调参的过程。



### 三个模型

#####fingerprint+sigmoid线性层

最先实现的模型，照着分享的ppt里面做的，结果和ppt里也差不多。

先用rdkit算出Morgan Fingerprint，然后直接丢进sigmoid的线性层进行二分类。效果看起来不错，很多都能预测正确。但是也有一些根本算不对的。

除了调Morgan Fingerprint的参数外也测试了调网络的结构和参数的方式，比如多加几个线性层之类的，但是都没有显著优化。基本直接线性层就能完全饱和了。还测试了toy2里面引入其他数据的办法，只产生了负面效果。

##### toy2

在上面的基础上调参了一下，没什么效果。还试了一下迁移学习用一个AID1706_binarized_sars的文档，从里面挑了10000个样本先跑一遍，然后再对数据进行训练，产生了负优化。不用管这个模型。

##### fingerprint+random forest

对着排行榜第三的人写的，调了一个scikit的库，训练代码只有10行。效果非常好，调参之后也有显著的优化。调参之后看起来和那个人的十折验证结果差不多。

调参是直接对着test集调的，但是调完之后dev集比test集就稍差一些，超参数没有非常过拟合。

调参的过程在/fingerprint+random_forest/调参.xlsx里面，可以考虑画个图出来。里面也写了选择过程。关键字是prc的均值。



### Morgan Fingerprint的参数

Morgan Fingerprint有两个参数，一个是半径(=2/3)，一个是hash出的序列长度(=512/1024/2048)，选哪个效果都差不多。就选半径=2，长度=1024了。

序列长度也就是丢进网络里的输入的维度。





### 为什么选择Morgan Fingerprint

直观上来说分子性质是局部性质，应该和结构的关系比较大，用Fingerprint算出来有哪些结构，然后学习看起来比较好。



### 可以参考的资料

random forest调参是按这个来的https://zhuanlan.zhihu.com/p/139510947

pretarined gnn模型https://github.com/snap-stanford/pretrain-gnns

pretarined gnn模型的论文https://arxiv.org/abs/1905.12265

pretarined gnn模型的中文讨论https://zhuanlan.zhihu.com/p/109768720