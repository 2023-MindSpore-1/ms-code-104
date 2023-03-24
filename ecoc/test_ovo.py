import numpy as numpy
import CodeMatrix.CodeMatrix
from DataComplexity.datacomplexity import get_data_complexity
from FeatureSelection.FeatureSelector import BSSWSS
import DataLoader as dl
from Classifiers.ECOCClassifier import SimpleECOCClassifier
from Classifiers.BaseClassifier import get_base_clf
import sklearn.metrics as metrics
from CodeMatrix.SFFS import sffs
from matplotlib import pyplot as plt
import os
plt.style.use('ggplot')

def getAllData():
    dataList=[]
    for item in list(os.walk("data_uci"))[0][2]:
        if os.path.splitext(item)[1]=='.data':
            dataList.append(str(item).split("_")[0])
    return numpy.unique(dataList)

def getScoreOfCol(CodeMatrixCol,trainX,trainY,validationX,validationY,estimator):
    listPositive=[]
    listNegative=[]
    index=-1
    tempTrainX=[]
    tempTrainY=[]
    tempValidationX=[]
    tempValidationY=[]
    for item in CodeMatrixCol:
        index+=1
        if item==1:
            listPositive.append(chr(ord('A')+index))
        elif item==-1:
            listNegative.append(chr(ord('A')+index))
    index=-1
    for item in trainY:
        index+=1
        if item in listPositive:
            tempTrainX.append(trainX[index])
            tempTrainY.append(1)
        elif item in listNegative:
            tempTrainX.append(trainX[index])
            tempTrainY.append(-1)
    index=-1
    for item in validationY:
        index+=1
        if item in listPositive:
            tempValidationX.append(validationX[index])
            tempValidationY.append(1)
        elif item in listNegative:
            tempValidationX.append(validationX[index])
            tempValidationY.append(-1)
    est = estimator.fit(tempTrainX, tempTrainY)
    return est.score(tempValidationX,tempValidationY)

def getAllScale(trainY):
    scaleList=[]
    for item in numpy.unique(trainY):
        scaleList.append(numpy.sum(trainY==item))
    return scaleList

def getScaleOfCol(CodeMatrixCol,trainY):
    positive=0
    negative=0
    listPositive=[]
    listNegative=[]
    index=-1
    for item in CodeMatrixCol:
        index+=1
        if item==1:
            listPositive.append(chr(ord('A')+index))
        elif item==-1:
            listNegative.append(chr(ord('A')+index))
    for item in listPositive:
        positive+=numpy.sum(trainY==item)
    for item in listNegative:
        negative+=numpy.sum(trainY==item)
    return min(positive,negative),max(negative,positive)

def getDataComplexitybyCol(row,trainX,trainY):
    result=[]
    listPositive=[]
    listNegative=[]
    X=[]
    y=[]
    index=-1
    for item in row:
        index+=1
        if item==1:
            listPositive.append(chr(ord('A')+index))
        elif item==-1:
            listNegative.append(chr(ord('A')+index))
    tempY=trainY.tolist()
    index=-1
    for item in tempY:
        index+=1
        if item in listPositive:
            y.append(1)
            X.append(trainX[index])
        elif item in listNegative:
            y.append(-1)
            X.append(trainX[index])
    X=numpy.array(X)
    y=numpy.array(y)
    temp=[]
    # dc = get_data_complexity('F1')
    # temp.append(round(dc.score(X, y),4))
    # dc = get_data_complexity('F2')
    # temp.append(round(dc.score(X, y),4))
    # dc = get_data_complexity('F3')
    # temp.append(round(dc.score(X, y),4))
    # dc = get_data_complexity('N1')
    # temp.append(round(dc.score(X, y),4))
    dc = get_data_complexity('N2')
    temp.append(round(dc.score(X, y),4))
    dc = get_data_complexity('N3')
    temp.append(round(dc.score(X, y),4))
    # result.append(temp)
    return temp

def getDataComplexity(CodeMatrix,trainX,trainY):
    # print(CodeMatrix)
    CodeMatrix=numpy.transpose(CodeMatrix)
    result=[]
    for row in CodeMatrix:
        listPositive=[]
        listNegative=[]
        X=[]
        y=[]
        index=-1
        for item in row:
            index+=1
            if item==1:
                listPositive.append(chr(ord('A')+index))
            elif item==-1:
                listNegative.append(chr(ord('A')+index))
        tempY=trainY.tolist()
        index=-1
        for item in tempY:
            index+=1
            if item in listPositive:
                y.append(1)
                X.append(trainX[index])
            elif item in listNegative:
                y.append(-1)
                X.append(trainX[index])
        X=numpy.array(X)
        y=numpy.array(y)
        temp=[]
        # dc = get_data_complexity('F1')
        # temp.append(round(dc.score(X, y),4))
        # dc = get_data_complexity('F2')
        # temp.append(round(dc.score(X, y),4))
        # dc = get_data_complexity('F3')
        # temp.append(round(dc.score(X, y),4))
        # dc = get_data_complexity('N1')
        # temp.append(round(dc.score(X, y),4))
        # dc = get_data_complexity('N2')
        # temp.append(round(dc.score(X, y),4))
        dc = get_data_complexity('N3')
        temp.append(round(dc.score(X, y),4))
        result.append(temp)
    return (numpy.array(result)).transpose()

def getAccuracyByEncoding(dataName, encoding, trainX, trainY, testX, testY):
    if(encoding != 'ecoc_one'):
        codeMatrix = getattr(CodeMatrix.CodeMatrix,
                             encoding)(trainX, trainY)[0]
    y=[]
    x=[]
    index=-1
    for row in codeMatrix.transpose():
        index+=1
        tempCodeMatrix=codeMatrix.transpose()
        tempCodeMatrix=tempCodeMatrix.tolist()
        tempCodeMatrix.pop(index)
        tempCodeMatrix=numpy.array(tempCodeMatrix).transpose()
        # y.append(getScoreOfCol(row,trainX,trainY,validationX,validationY,get_base_clf('SVM')))
        estimator = get_base_clf('SVM')  # get SVM classifier object.
        sec = SimpleECOCClassifier(estimator, tempCodeMatrix)
        sec.fit(trainX, trainY)
        pred = sec.predict(testX)
        x.append(round(metrics.accuracy_score(testY, pred),4))
    validationX, validationY, X, Y, instanceNum = dl.loadDataset(
            'data_uci/'+dataName+'_validation.data', 'data_uci/'+dataName+'_validation.data')
    print(len(codeMatrix.transpose()))
    print(x.index(max(x)))
    print(x[x.index(max(x))])
    # print(getScoreOfCol(codeMatrix.transpose()[x.index(max(x))],trainX,trainY,validationX,validationY,get_base_clf('SVM')))
    print(getDataComplexitybyCol(codeMatrix.transpose()[x.index(max(x))],trainX,trainY))
    print(getDataComplexitybyCol(codeMatrix.transpose()[23],trainX,trainY))
    # print(getDataComplexitybyCol(codeMatrix.transpose()[24],trainX,trainY).append)
    index=-1
    y=[]
    for item in x:
        index+=1
        temp=getDataComplexitybyCol(codeMatrix.transpose()[index],trainX,trainY)
        temp.append(index)
        y.append(temp)
        print(temp)
    # print(getDataComplexitybyCol(codeMatrix.transpose()[42],trainX,trainY))
    # print(getDataComplexitybyCol(codeMatrix.transpose()[44],trainX,trainY))
    # print(getDataComplexitybyCol(codeMatrix.transpose()[40],trainX,trainY))
    # print(getDataComplexitybyCol(codeMatrix.transpose()[39],trainX,trainY))

    # y.append(getScoreOfCol(row,trainX,trainY,validationX,validationY,get_base_clf('SVM')))
    # tempCodeMatrix=codeMatrix.transpose()
    # tempCodeMatrix=[tempCodeMatrix.tolist()]
    temp=codeMatrix.transpose().tolist()
    tempCodeMatrix=[]
    index=-1
    for item in y:
        index+=1
        if(item[0]<14 and item[0]>0.5):
            tempCodeMatrix.append(temp[index])
    # tempCodeMatrix.pop(41)
    print(len(tempCodeMatrix))
    tempCodeMatrix=numpy.array(tempCodeMatrix).transpose()
    estimator = get_base_clf('SVM')  # get SVM classifier object.
    sec = SimpleECOCClassifier(estimator, tempCodeMatrix)
    sec.fit(trainX, trainY)
    pred = sec.predict(testX)
    return round(metrics.accuracy_score(testY, pred),4), codeMatrix


def getAllAccuracy(dataName):
    scoreList=[]
    # encodingList = ("ovo", "ova", "dense_rand",
    #                 "sparse_rand", "decoc", "agg_ecoc", "ecoc_one")
    encodingList = ("ovo",)                
    trainX, trainY, testX, testY, instanceNum = dl.loadDataset(
        'data_uci/'+dataName+'_train.data', 'data_uci/'+dataName+'_test.data')
    validationX, validationY, X, Y, instanceNum = dl.loadDataset(
            'data_uci/'+dataName+'_validation.data', 'data_uci/'+dataName+'_validation.data')
    index=0
    
    for item in encodingList:
        index+=1
        score, codeMatrix = getAccuracyByEncoding(
            dataName, item, trainX, trainY, testX, testY)
        scoreList.append(score)
        print(item+":"+str(score))
        # result=getDataComplexity(codeMatrix,trainX,trainY)
        # # print(codeMatrix)
        # x=[]
        # y=[]
        # for row in codeMatrix.transpose():
        #     y.append(getScoreOfCol(row,trainX,trainY,validationX,validationY,get_base_clf('SVM')))
        # print(y)
        # plt.figure()
        # plt.plot(range(0,len(y),1),result[0],"--",label="N3")
        # # plt.plot(range(0,len(y),1),result[1],"--",label="N3")
        # # plt.plot(range(0,len(y),1),result[3],"--",label="N1")
        # # plt.plot(range(0,len(y),1),result[4],"--",label="N2")
        # # plt.plot(range(0,len(y),1),result[5],"--",label="N3")
        # plt.plot(range(0,len(y),1),y,"+-",label="Score")
        # plt.xticks(fontsize=15)
        # plt.yticks(fontsize=15)
        # plt.xlabel("Col", fontsize=20)
        # plt.ylabel("Score", fontsize=20)
        # plt.ylim(0, 1)
        # plt.title(dataName+"_"+item+" score:"+str(score),fontsize=25)
        # plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
        # plt.savefig("pictures/"+dataName+"_"+item+"_N.png")
        # plt.show()
    
    # for a, b in zip(encodingList, scoreList):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=15)
    # plt.savefig("pictures/"+"size-version.png")

# dataList=getAllData()
# dataList=("dermatology","abalone","iris")
# dataList=("mfeatzer",)
dataList=("mfeatmor",)
# dataList=("glass",)
index=0
for item in dataList:
    index+=1
    # plt.subplot((len(dataList)*2-1)*100+10+index*2-1)
    getAllAccuracy(item)


# a=numpy.array([1,1,-1,-1])
# b=numpy.array(['A','A','B','B','C','D','D'])
# print(getAllScale(b))
