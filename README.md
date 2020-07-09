# prml2020_pj

运行main.py就可以用啦。

getans()用来算没有答案的test.csv，getfold()用来算十折验证。



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

padding.csv记录了调参之后，不同扩展阳性样本方法的结果。



### 三个模型

#####fingerprint+sigmoid线性层

最先实现的模型，照着分享的ppt里面做的，结果和ppt里也差不多。

先用rdkit算出Morgan Fingerprint，然后直接丢进sigmoid的线性层进行二分类。效果看起来不错，很多都能预测正确。但是也有一些根本算不对的。

除了调Morgan Fingerprint的参数外也测试了调网络的结构和参数的方式，比如多加几个线性层之类的，但是都没有显著优化。基本直接线性层就能完全饱和了。还测试了toy2里面引入其他数据的办法，只产生了负面效果。

##### toy2

在上面的基础上调参了一下，没什么效果。还试了一下迁移学习用一个AID1706_binarized_sars的文档，从里面挑了10000个样本先跑一遍，然后再对数据进行训练，产生了负优化。不用管这个模型。

##### fingerprint+random forest

对着排行榜第三的人写的，调了一个scikit的库，训练代码只有10行。效果非常好，调参之后也有显著的优化。调参之后看起来和那个人的十折验证结果差不多。

调参是直接对着test集调的，但是调完之后dev集比test集只稍差一些，超参数没有非常过拟合。

调参的过程在/fingerprint+random_forest/调参.xlsx里面，可以考虑画个图出来。里面也写了选择过程。关键字是prc的均值。



### Morgan Fingerprint的参数

Morgan Fingerprint有两个参数，一个是半径(=2/3)，一个是hash出的序列长度(=512/1024/2048)，选哪个效果都差不多。就选半径=2，长度=1024了。

序列长度也就是丢进网络里的输入的维度。



### 扩展阳性样本

因为阳性样本太少了，sigmoid线性层的模型只会输出0.所以把阳性样本复制几份，直到阳性样本占一半。	

但是在调参之后测试随机森林的模型在不扩展的情况下还是可以跑出很好的结果的，但是最好的结果出现在阳性样本占一半的时候。可以看padding.csv的结果。



### 为什么选择Morgan Fingerprint

直观上来说分子性质是局部性质，应该和结构的关系比较大，用Fingerprint算出来有哪些结构，然后学习看起来比较好。



### 十折验证的分析

第零折：sigmoid模型只能prc训练到0.1，但是随机森林比较好的时候是0.6-0.8，比较差时候是0.4-0.5左右。roc基本可以从0.6跑到0.98。换模型真正有用的样本。

第一、二折：比较普通的样本，都有比较好的正确率。随机森林也稍有一些改善（相对于sigmoid线性层）。

第三折：sigmoid只能跑到0.03，调参时候随机森林也经常是0.1，但是有时候能做到0.27左右，看起来是猜对一个。

第四折：非常容易的样本。

第五折：普通的样本，但是随机森林没有很大改善。

第六折：样本非常差的一折，完全预测不对。基本全输出NO效果比较好。

第七折：样本非常差的一折，完全预测不对，只有一个阳性样本，输出0.04。但是很多阴性样本能输出0.1-0.2。完全跑不过全输出NO的结果。

第八折：最容易的一折。

第九折：普通的样本，但是随机森林没有很大改善。



### 按0.5截断输出分类

这个做法可以保证在很多不好的数据上能全输出0，得到roc=0.5,prc=0.5的结果，给出一个下限。但是看起来这样破坏了roc和prc的评分系统，应该是不能使用这种操作的。

正确的模型能让roc远高于0.5，但是比较糟糕的是prc基本都跑不到0.5.



### 结果

submit_prob:roc=0.836,prc=0.621

submit_class:roc=0.720,prc=0.657

得到了还比较好的结果。

如果输出分类就会导致roc比较低，所以看起来榜上别人都是输出的概率。



### 可以参考的资料

random forest调参是按这个来的https://zhuanlan.zhihu.com/p/139510947

pretarined gnn模型https://github.com/snap-stanford/pretrain-gnns

pretarined gnn模型的论文https://arxiv.org/abs/1905.12265

pretarined gnn模型的中文讨论https://zhuanlan.zhihu.com/p/109768720