from getdata import *
from model import *
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import numpy as np
from sklearn import metrics

def process ( rawdata ):
    #计算Morgan Fingerprint
    n = len ( rawdata[0] )
    feature = np.zeros ( (n,1024) )
    label = np.zeros ( (n) )
    for i in range (n):
        mol = Chem.MolFromSmiles(rawdata[0][i])
        fp = AllChem.GetMorganFingerprintAsBitVect ( mol , 2 , 1024 )
        vec = np.array ( [fp[j] for j in range (1024)] )
        feature[i] = ( vec )
        label[i] = ( rawdata[1][i] )
    return n,feature,label

def padding ( n , feature , label ):
    #补充阳性数据
    cnt = 0
    gf = np.zeros ( (n,1024) )
    gl = np.zeros ( (n) )
    for f,l in zip(feature,label):
        if l == 1:
            gf[cnt] = f
            gl[cnt] = l
            cnt += 1
    pfeature = np.zeros ( (n+n, 1024) )
    plabel = np.zeros ( (n+n) )
    for i in range(n):
        pfeature[i],plabel[i] = (feature[i],label[i])
    now = n
    while now < n+n:
        for f,l in zip(gf,gl):
            if now < n+n and l == 1:
                pfeature[now] = f
                plabel[now] = l
                now += 1
    return pfeature,plabel

def getscore ( pred , label ):
    #计算分数
    npred = pred[:,1]
    #print (npred)
    roc = metrics.roc_auc_score (label,npred)
    print ( "roc_auc=" , roc )
    p,r,thr = metrics.precision_recall_curve(label,npred)
    prc = metrics.auc ( r , p )
    print ( "prc_auc=" , prc )

def getans ():
    #预测无标签数据
    rawdata = getdata("train.csv")
    n,feature,label = process ( rawdata )
    pfeature,plabel = padding ( n , feature , label )
    model = trainmodel ( (pfeature,plabel) )
    testdata = getdata_nolabel("test.csv")
    tn,tfeature,tlabel = process ( testdata )
    predict = evaluate ( model , (tfeature,tlabel) )
    fle = open ( "myans.out" , "w" )
    for i in predict:
        fle.write (str(i[1])+'\n')
    fle.close ()

def getfold ():
    #预测十折验证
    for i in range ( 10 ):

        #输入数据
        rawdata = getdata("./train_cv/fold_"+str(i)+"/train.csv")

        #处理数据
        n,feature,label = process ( rawdata )
        pfeature,plabel = padding ( n , feature , label )

        #训练模型
        model = trainmodel ( (pfeature,plabel) )

        #预测测试集
        testdata = getdata("./train_cv/fold_"+str(i)+"/test.csv")
        tn,tfeature,tlabel = process ( testdata )
        predict = evaluate ( model , (tfeature,tlabel) )
        
        #输出预测结果
        fle = open ( "./pred/pred"+str(i)+".out" , "w" )
        for i in range (tn):
            fle.write(str(predict[i][1])+'\n')
        fle.close ()

        #计算分数
        getscore ( predict , tlabel )

getans()
#getfold()