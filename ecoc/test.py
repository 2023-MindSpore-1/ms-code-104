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

def getDataComplexity(CodeMatrix,trainX,trainY):
    print(CodeMatrix)
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
        dc = get_data_complexity('F1')
        temp.append(round(dc.score(X, y),4))
        dc = get_data_complexity('F2')
        temp.append(round(dc.score(X, y),4))
        dc = get_data_complexity('F3')
        temp.append(round(dc.score(X, y),4))
        dc = get_data_complexity('N1')
        temp.append(round(dc.score(X, y),4))
        dc = get_data_complexity('N2')
        temp.append(round(dc.score(X, y),4))
        dc = get_data_complexity('N3')
        temp.append(round(dc.score(X, y),4))
        result.append(temp)
    print((numpy.array(result)).transpose())

def getAccuracyByEncoding(dataName, encoding, feature, trainX, trainY, testX, testY):
    fs = BSSWSS(k=feature)  # remain 2 features.o
    fs.fit(trainX, trainY)
    trainX, testX = fs.transform(trainX), fs.transform(testX)
    estimator = get_base_clf('SVM')  # get SVM classifier object.
    if(encoding != 'ecoc_one'):
        codeMatrix = getattr(CodeMatrix.CodeMatrix,
                             encoding)(trainX, trainY)[0]
    else:
        validationX, validationY, X, Y, instanceNum = dl.loadDataset(
            'data_uci/'+dataName+'_validation.data', 'data_uci/'+dataName+'_validation.data')
        codeMatrix = getattr(CodeMatrix.CodeMatrix, encoding)(
            trainX, trainY, validationX, validationY, estimator)[0]

    sec = SimpleECOCClassifier(estimator, codeMatrix)
    sec.fit(trainX, trainY)
    pred = sec.predict(testX)
    return round(metrics.accuracy_score(testY, pred),4), codeMatrix


def getAllAccuracy(dataName):
    scoreList=[]
    encodingList = ("ovo", "ova", "dense_rand",
                    "sparse_rand", "decoc", "agg_ecoc", "ecoc_one")
    # encodingList = ("ovo","ova")                
    trainX, trainY, testX, testY, instanceNum = dl.loadDataset(
        'data_uci/'+dataName+'_train.data', 'data_uci/'+dataName+'_test.data')
    validationX, validationY, X, Y, instanceNum = dl.loadDataset(
            'data_uci/'+dataName+'_validation.data', 'data_uci/'+dataName+'_validation.data')
    for item in encodingList:
        score, codeMatrix = getAccuracyByEncoding(
            dataName, item, len(trainX[0]), trainX, trainY, testX, testY)
        scoreList.append(score)
        print(item+":"+str(score))
        # getDataComplexity(codeMatrix,trainX,trainY)
        print(codeMatrix)
        # for row in codeMatrix.transpose():
        #     print(getScaleOfCol(row,trainY))
        #     print(getScoreOfCol(row,trainX,trainY,validationX,validationY,get_base_clf('SVM')))
    # plt.bar(encodingList,scoreList)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.xlabel("encoding", fontsize=20)
    # plt.ylabel("score", fontsize=20)
    # plt.ylim(0, 1)
    # plt.title("Scores of Different Matrix about "+dataName,fontsize=25)
    # for a, b in zip(encodingList, scoreList):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=15)
    # plt.savefig("pictures/"+"size-version.png")

# dataList=getAllData()
# dataList=("dermatology","abalone","iris")
dataList=("iris",)
# plt.figure()
index=0
for item in dataList:
    index+=1
    plt.subplot((len(dataList)*2-1)*100+10+index*2-1)
    getAllAccuracy(item)
# plt.show()

# a=numpy.array([1,1,-1,-1])
# b=numpy.array(['A','A','B','B','C','D','D'])
# print(getAllScale(b))
