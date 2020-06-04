from getdata import *
from model import *
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import numpy as np
from sklearn import metrics

def process ( rawdata ):
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
    npred = pred[:,1]
    #print (npred)
    roc = metrics.roc_auc_score (label,npred)
    print ( "roc_auc=" , roc )
    p,r,thr = metrics.precision_recall_curve(label,npred)
    prc = metrics.auc ( r , p )
    print ( "prc_auc=" , prc )

def getans ():
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
    for i in range ( 10 ):

        rawdata = getdata("./train_cv/fold_"+str(i)+"/train.csv")

        n,feature,label = process ( rawdata )

        pfeature,plabel = padding ( n , feature , label )

        # prt = pfeature.astype(np.int32)
        # fle = open("pred.out","w")
        # for i in prt:
        #     fle.write(str(i)+'\n')
        # fle.close ()
        # print ( "finish" )

        model = trainmodel ( (pfeature,plabel) )

        #预测训练集
        #predict = evaluate ( model , (feature,label) )
        #print ( predict )
        #fle = open ( "pred.out" , "w" )
        #for i in range (n):
        #    fle.write(str(label[i])+' '+str(predict[i][1])+'\n')
        #fle.close ()

        #预测测试集
        testdata = getdata("./train_cv/fold_"+str(i)+"/test.csv")
        tn,tfeature,tlabel = process ( testdata )
        predict = evaluate ( model , (tfeature,tlabel) )
        #print ( predict )
        fle = open ( "./pred/pred"+str(i)+".out" , "w" )
        #for i in range (tn):
        #    fle.write(str(tlabel[i])+' '+str(predict[i][1])+'\n')
        for i in range (tn):
            fle.write(str(predict[i][1])+'\n')
        fle.close ()

        getscore ( predict , tlabel )

#getans()
getfold()