from getdata import *
from model import *
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import numpy as np
from sklearn import metrics

morganlengeth = 1024

def process ( rawdata ):
    #算Morgan Fingerprint
    n = len ( rawdata[0] )
    feature = np.zeros ( (n,morganlengeth) )
    label = np.zeros ( (n) )
    for i in range (n):
        mol = Chem.MolFromSmiles(rawdata[0][i])
        fp = AllChem.GetMorganFingerprintAsBitVect ( mol , 2 , morganlengeth )
        vec = np.array ( [fp[j] for j in range (morganlengeth)] )
        feature[i] = ( vec )
        label[i] = ( rawdata[1][i] )
    return n,feature,label

def padding ( n , feature , label ):
    #把阳性样本的数量扩展到一半
    cnt = 0
    gf = np.zeros ( (n,morganlengeth) )
    gl = np.zeros ( (n) )
    for f,l in zip(feature,label):
        if l == 1:
            gf[cnt] = f
            gl[cnt] = l
            cnt += 1
    paddinglength = n
    pfeature = np.zeros ( (n+paddinglength, morganlengeth) )
    plabel = np.zeros ( (n+paddinglength) )
    for i in range(n):
        pfeature[i],plabel[i] = (feature[i],label[i])
    now = n
    while now < n+paddinglength:
        for f,l in zip(feature,label):
            if now < n+paddinglength and l == 1:
                pfeature[now] = f
                plabel[now] = l
                now += 1
    return pfeature,plabel

def getscore ( pred , label ):
    #给概率预测算分
    npred = pred[:,1]
    roc = metrics.roc_auc_score (label,npred)
    p,r,thr = metrics.precision_recall_curve(label,npred)
    prc = metrics.auc ( r , p )
    print ( "roc_auc=" , roc , " ||| prc_auc=" , prc )
    return roc , prc

def getscore_class ( pred , label ):
    #给分类预测算分
    npred = pred
    roc = metrics.roc_auc_score (label,npred)
    p,r,thr = metrics.precision_recall_curve(label,npred)
    prc = metrics.auc ( r , p )
    print ( "class:roc_auc=" , roc , " ||| prc_auc=" , prc )
    return roc , prc

def getans ():
    #预测无标签的数据
    rawdata = getdata("train.csv")
    n,feature,label = process ( rawdata )
    pfeature,plabel = padding ( n , feature , label )
    model = trainmodel ( (pfeature,plabel) )
    testdata = getdata_nolabel("test.csv")
    tn,tfeature,tlabel = process ( testdata )

    #概率预测
    predict = evaluate_prob ( model , (tfeature,tlabel) )
    fle = open ( "myans.out" , "w" )
    for i in predict:
        fle.write (str(i[1])+'\n')
    fle.close ()

    #分类预测
    #predict = evaluate_class ( model , (tfeature,tlabel) )
    #fle = open ( "myans.out" , "w" )
    #for i in predict:
    #    fle.write (str(i)+'\n')
    #fle.close ()


def getfold ():
    #预测十折验证
    for i in range ( 0 , 10 ):

        #输入数据
        rawdata = getdata("./train_cv/fold_"+str(i)+"/train.csv")

        #处理数据
        n,feature,label = process ( rawdata )
        pfeature,plabel = padding ( n , feature , label )

        #训练
        model = trainmodel ( (pfeature,plabel)  )
        
        #预测dev集
        #testdata = getdata("./train_cv/fold_"+str(i)+"/dev.csv")
        #预测测试集
        testdata = getdata("./train_cv/fold_"+str(i)+"/test.csv")
        tn,tfeature,tlabel = process ( testdata )
            
        #概率预测prob
        predict = evaluate_prob ( model , (tfeature,tlabel) )

        #输出到文件
        fle = open ( "./pred/pred"+str(i)+".out" , "w" )
        for i in range (tn):
            fle.write(str(tlabel[i])+' '+str(predict[i][1])+'\n')
        fle.close ()

        #输出概率预测prob
        nr,np = getscore ( predict , tlabel )
            
        #分类预测class
        #predict1 = evaluate_class ( model , (tfeature,tlabel) )
        #输出分类预测class
        #getscore_class (predict1,tlabel)

getans()
#getfold()