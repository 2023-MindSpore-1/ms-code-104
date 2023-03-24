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
import heapq
plt.style.use('ggplot')


def getAllData():
    dataList = []
    for item in list(os.walk("data_uci"))[0][2]:
        if os.path.splitext(item)[1] == '.data':
            dataList.append(str(item).split("_")[0])
    return numpy.unique(dataList)


def getScoreOfCol(CodeMatrixCol, trainX, trainY, validationX, validationY, estimator):
    listPositive = []
    listNegative = []
    index = -1
    tempTrainX = []
    tempTrainY = []
    tempValidationX = []
    tempValidationY = []
    for item in CodeMatrixCol:
        index += 1
        if item == 1:
            listPositive.append(chr(ord('A')+index))
        elif item == -1:
            listNegative.append(chr(ord('A')+index))
    index = -1
    for item in trainY:
        index += 1
        if item in listPositive:
            tempTrainX.append(trainX[index])
            tempTrainY.append(1)
        elif item in listNegative:
            tempTrainX.append(trainX[index])
            tempTrainY.append(-1)
    index = -1
    for item in validationY:
        index += 1
        if item in listPositive:
            tempValidationX.append(validationX[index])
            tempValidationY.append(1)
        elif item in listNegative:
            tempValidationX.append(validationX[index])
            tempValidationY.append(-1)
    est = estimator.fit(tempTrainX, tempTrainY)
    return est.score(tempValidationX, tempValidationY)


def getAllScale(trainY):
    scaleList = []
    for item in numpy.unique(trainY):
        scaleList.append(numpy.sum(trainY == item))
    return scaleList


def getScaleOfCol(CodeMatrixCol, trainY):
    positive = 0
    negative = 0
    listPositive = []
    listNegative = []
    index = -1
    for item in CodeMatrixCol:
        index += 1
        if item == 1:
            listPositive.append(chr(ord('A')+index))
        elif item == -1:
            listNegative.append(chr(ord('A')+index))
    for item in listPositive:
        positive += numpy.sum(trainY == item)
    for item in listNegative:
        negative += numpy.sum(trainY == item)
    return min(positive, negative), max(negative, positive)


def getDataComplexitybyCol(row, trainX, trainY):
    result = []
    listPositive = []
    listNegative = []
    X = []
    y = []
    if -1 in row:
        index = -1
        for item in row:
            index += 1
            if item == 1:
                listPositive.append(chr(ord('A')+index))
            elif item == -1:
                listNegative.append(chr(ord('A')+index))
    else:
        index = -1
        for item in row:
            index += 1
            if item == 1:
                listPositive.append(chr(ord('A')+index))
            elif item == 0:
                listNegative.append(chr(ord('A')+index))
    # print(row)
    tempY = trainY.tolist()
    index = -1
    for item in tempY:
        index += 1
        if item in listPositive:
            y.append(1)
            X.append(trainX[index])
        elif item in listNegative:
            y.append(-1)
            X.append(trainX[index])
    X = numpy.array(X)
    y = numpy.array(y)
    temp = []
    dc = get_data_complexity('F1')
    temp.append(round(dc.score(X, y), 4))
    # dc = get_data_complexity('F2')
    # temp.append(round(dc.score(X, y), 4))
    # dc = get_data_complexity('F3')
    # temp.append(round(dc.score(X, y), 4))
    # dc = get_data_complexity('N1')
    # temp.append(round(dc.score(X, y),4))
    dc = get_data_complexity('N2')
    temp.append(round(dc.score(X, y), 4))
    dc = get_data_complexity('N3')
    temp.append(round(dc.score(X, y), 4))
    # result.append(temp)
    return temp


def getDCArray(DCResult, length):
    DCResult = DCResult.tolist()
    result = []
    index = -1
    for i in range(0, length):
        rowScore = []
        for j in range(0, length):
            if(i == j):
                rowScore.append(numpy.zeros(len(DCResult[0])).tolist())
            elif(j < i):
                rowScore.append(result[j][i])
            else:
                index += 1
                rowScore.append(DCResult[index])
        result.append(rowScore)
    return result


def getColScore(codeMatrix, trainX, trainY, testX, testY):
    result = []
    estimator = get_base_clf('SVM')  # get SVM classifier object.
    sec = SimpleECOCClassifier(estimator, codeMatrix)
    sec.fit(trainX, trainY)
    pred = sec.predict(testX)
    stdScore = round(metrics.accuracy_score(testY, pred), 4)
    index = -1
    for row in codeMatrix.transpose():
        index += 1
        tempCodeMatrix = codeMatrix.transpose()
        tempCodeMatrix = tempCodeMatrix.tolist()
        tempCodeMatrix.pop(index)
        tempCodeMatrix = numpy.array(tempCodeMatrix).transpose()
        # y.append(getScoreOfCol(row,trainX,trainY,validationX,validationY,get_base_clf('SVM')))
        estimator = get_base_clf('SVM')  # get SVM classifier object.
        sec = SimpleECOCClassifier(estimator, tempCodeMatrix)
        sec.fit(trainX, trainY)
        pred = sec.predict(testX)
        k = (stdScore-round(metrics.accuracy_score(testY, pred), 4))*100
        result.append((index, row, k))
    return result, stdScore


def getDataComplexity(CodeMatrix, trainX, trainY):
    # print(CodeMatrix)
    CodeMatrix = numpy.transpose(CodeMatrix)
    result = []
    for row in CodeMatrix:
        listPositive = []
        listNegative = []
        X = []
        y = []
        index = -1
        for item in row:
            index += 1
            if item == 1:
                listPositive.append(chr(ord('A')+index))
            elif item == -1:
                listNegative.append(chr(ord('A')+index))
        tempY = trainY.tolist()
        index = -1
        for item in tempY:
            index += 1
            if item in listPositive:
                y.append(1)
                X.append(trainX[index])
            elif item in listNegative:
                y.append(-1)
                X.append(trainX[index])
        X = numpy.array(X)
        y = numpy.array(y)
        temp = []
        dc = get_data_complexity('F1')
        temp.append(round(dc.score(X, y), 4))
        # dc = get_data_complexity('F2')
        # temp.append(round(dc.score(X, y),4))
        dc = get_data_complexity('F3')
        temp.append(round(dc.score(X, y),4))
        # dc = get_data_complexity('N1')
        # temp.append(round(dc.score(X, y),4))
        # dc = get_data_complexity('N2')
        # temp.append(round(dc.score(X, y), 4))
        dc = get_data_complexity('N3')
        temp.append(round(dc.score(X, y), 4))
        result.append(temp)
    return numpy.array(result)


def getMedian(data):
    # data=numpy.array(data).transpose()[index]
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2


def updateCodeMatrix(codeMatrix, DCArray, trainX, trainY):
    codeMatrix = codeMatrix.transpose()
    resultCodeMatrix = []
    for col in codeMatrix:
        resultCol = col.tolist()
        listPositive = []
        listNegative = []
        listZero = []
        index = -1
        for item in col:
            index += 1
            if item == 1:
                listPositive.append(index)
            elif item == -1:
                listNegative.append(index)
            else:
                listZero.append(index)
        fPositiveList = []
        fNegativeList = []
        for item in listZero:
            fPositive = 0
            fNegative = 0
            for i in listPositive:
                fPositive += DCArray[item][i][0]
            fPositive /= len(listPositive)
            for i in listNegative:
                fNegative += DCArray[item][i][0]
            fNegative /= len(listNegative)
            fPositiveList.append(fPositive)
            fNegativeList.append(fNegative)
        positiveMedian = getMedian(fPositiveList)
        negativeMedian = getMedian(fNegativeList)
        index = -1
        for item in listZero:
            index += 1
            if (fPositiveList[index] < positiveMedian) or (fNegativeList[index] < negativeMedian):
                if fPositiveList[index] < fNegativeList[index]:
                    resultCol[item] = 1
                else:
                    resultCol[item] = -1
        resultCodeMatrix.append(resultCol)
        # print(resultCol)
    return numpy.array(resultCodeMatrix).transpose()


def getAccuracyByEncoding(dataName, encoding, trainX, trainY, testX, testY,DCArray,validationX,validationY):
    print(dataName+"_"+encoding+": ")
    #下降阈值
    delta=0.01
    length=len(numpy.unique(trainY))
    if(encoding != 'ecoc_one'):
        codeMatrix = getattr(CodeMatrix.CodeMatrix,
                             encoding)(trainX, trainY)[0]
    
    y = []
    estimator = get_base_clf('SVM')  # get SVM classifier object.
    sec = SimpleECOCClassifier(estimator, codeMatrix)
    sec.fit(trainX, trainY)
    pred = sec.predict(testX)
    stdScore = round(metrics.accuracy_score(testY, pred), 4)
    
    # 获得原矩阵F3测度
    resultF3=[]
    resultN3=[]
    result=getDataComplexity(codeMatrix, trainX, trainY)
    # 0:F1 1:F2 2:F3 3:N2 4:N3
    resultF3 = result.transpose()[1].tolist()
    resultN3 = result.transpose()[2].tolist()

    deleteList=[]
    currentScore=stdScore
    tempScore=currentScore

    #N3纠错
    temp=resultN3
    k=max(temp)
    tempScore=currentScore
    print("shrinking by "+"N3")
    while ((currentScore-tempScore)/currentScore)<delta and min(temp)<=0.5:
        currentScore=tempScore
        index=temp.index(min(temp))
        # print(index)
        temp[index]=k
        deleteList.append(index)
        tempCodeMatrix=numpy.delete(codeMatrix,deleteList,axis=1)
        estimator = get_base_clf('SVM')  # get SVM classifier object.
        sec = SimpleECOCClassifier(estimator, tempCodeMatrix)
        sec.fit(trainX, trainY)
        pred = sec.predict(validationX)
        tempScore = round(metrics.accuracy_score(validationY, pred), 4)
        # print(tempScore)
    if ((currentScore-tempScore)/currentScore)>=delta :
        deleteList.pop()
    # print(deleteList)

    #F3裁剪
    temp=resultF3
    k=max(temp)
    print("shrinking by "+"F3")
    while ((currentScore-tempScore)/currentScore)<delta:
        currentScore=tempScore
        index=temp.index(min(temp))
        # print(index)
        temp[index]=k
        deleteList.append(index)
        tempCodeMatrix=numpy.delete(codeMatrix,deleteList,axis=1)
        estimator = get_base_clf('SVM')  # get SVM classifier object.
        sec = SimpleECOCClassifier(estimator, tempCodeMatrix)
        sec.fit(trainX, trainY)
        pred = sec.predict(validationX)
        tempScore = round(metrics.accuracy_score(validationY, pred), 4)
        # print(tempScore)

    deleteList.pop()
    print(len(deleteList))
    print(deleteList)
    tempCodeMatrix=numpy.delete(codeMatrix,deleteList,axis=1).transpose().tolist()
    
    estimator = get_base_clf('SVM')  # get SVM classifier object.
    sec = SimpleECOCClassifier(estimator, numpy.delete(codeMatrix,deleteList,axis=1))
    sec.fit(trainX, trainY)
    pred = sec.predict(testX)
    tempScore = round(metrics.accuracy_score(testY, pred), 4)
    print("thinned："+str(tempScore))

    #F1扩充
    DClist=[]
    for item in DCArray:
        DClist.append(numpy.array(item).transpose()[0].tolist())
    index=-1
    for item in DClist:
        index+=1
        item[index]=(max(item)+min(item))/2
        newCol=[]
        targetList=[]
        target=item.index(min(item))
        targetList.append(target)
        targetList.append(DClist[target].index(max(DClist[target])))
        # print(item)
        # print(target)
        # print(DClist[target])
        for i in range(0,length):
            if i==index:
                newCol.append(1)
            elif i in targetList:
                newCol.append(-1)
            else:
                newCol.append(0)
        tempCodeMatrix.append(newCol)
    #生成新矩阵
    newCodeMatrix=numpy.array(tempCodeMatrix).transpose()
    
    #结果预测
    estimator = get_base_clf('SVM')  # get SVM classifier object.
    sec = SimpleECOCClassifier(estimator, newCodeMatrix)
    sec.fit(trainX, trainY)
    pred = sec.predict(testX)
    tempScore = round(metrics.accuracy_score(testY, pred), 4)
    print("new："+str(tempScore))

    estimator = get_base_clf('SVM')  # get SVM classifier object.
    sec = SimpleECOCClassifier(estimator, codeMatrix)
    sec.fit(trainX, trainY)
    pred = sec.predict(testX)
    tempScore = round(metrics.accuracy_score(testY, pred), 4)
    print("old："+str(tempScore))
    
    # tempCodeMatrix=numpy.delete(codeMatrix,deleteList,axis=1)
    
    # codeMatrixResult = updateCodeMatrix(codeMatrix, DCArray, trainX, trainY)
    # temp = codeMatrixResult.transpose().tolist()
    # codeMatrixResult = numpy.array(temp).transpose()
    # y, stdScore2 = getColScore(
    #     codeMatrixResult, trainX, trainY, validationX, validationY)
    # # for item in y:
    # #     print(item)
    # print(stdScore-stdScore2)
    return stdScore, codeMatrix


def getAllAccuracy(dataName):
    scoreList = []
    encodingList = ("ovo", "ova", "dense_rand",
                    "sparse_rand", "decoc", "agg_ecoc")
    # encodingList = ("ovo",)
    trainX, trainY, testX, testY, instanceNum = dl.loadDataset(
        'data_uci/'+dataName+'_train.data', 'data_uci/'+dataName+'_test.data')
    validationX, validationY, X, Y, instanceNum = dl.loadDataset(
        'data_uci/'+dataName+'_validation.data', 'data_uci/'+dataName+'_validation.data')
    # 通过ovo矩阵获得类的一对一测度
    ovoCodeMatrix = getattr(CodeMatrix.CodeMatrix,
                            "ovo")(trainX, trainY)[0]
    result = getDataComplexity(ovoCodeMatrix, trainX, trainY)
    length=len(numpy.unique(trainY))
    DCArray = getDCArray(result, length)

    index = 0
    for item in encodingList:
        index += 1
        score, codeMatrix = getAccuracyByEncoding(
            dataName, item, trainX, trainY, testX, testY,DCArray,validationX,validationY)
        scoreList.append(score)
        # print(item+":"+str(score))
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
# dataList=("glass",)
# dataList=("mfeatmor",)
# dataList = ("glass","dermatology", "mfeatmor",
#             "mfeatpix", "mfeatzer", "optdigits", "sat", "yeast")
dataList=("glass",)
index = 0
for item in dataList:
    # print(item)
    index += 1
    # plt.subplot((len(dataList)*2-1)*100+10+index*2-1)
    # print(numpy.zeros(5).tolist())
    getAllAccuracy(item)


# a=numpy.array([1,1,-1,-1])
# b=numpy.array(['A','A','B','B','C','D','D'])
# print(getAllScale(b))
