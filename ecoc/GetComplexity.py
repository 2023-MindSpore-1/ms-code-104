import numpy as numpy
import ecoc.CodeMatrix.CodeMatrix
from ecoc.DataComplexity.datacomplexity import get_data_complexity
from ecoc.FeatureSelection.FeatureSelector import BSSWSS
import ecoc.DataLoader as dl
from ecoc.Classifiers.ECOCClassifier import SimpleECOCClassifier
from ecoc.Classifiers.BaseClassifier import get_base_clf
import sklearn.metrics as metrics
from ecoc.CodeMatrix.SFFS import sffs
from matplotlib import pyplot as plt
from ecoc.DataComplexity.Get_Complexity import *

def getDataComplexitybyCol(trainX,trainY):
    #数据分割
    group1_data=trainX[numpy.where(trainY==1)]
    group1_label=numpy.ones(len(group1_data))
    group2_data=trainX[numpy.where(trainY==-1)]
    group2_label=-numpy.ones(len(group2_data))
    temp=[]
    dc = get_data_complexity('F1')
    temp.append(round(dc.score(trainX,trainY),4))
    dc = get_data_complexity('F3')
    temp.append(round(dc.score(trainX,trainY),4))
    dc = get_data_complexity('N3')
    temp.append(round(dc.score(trainX,trainY),4))
    # temp.append(round(get_complexity_N4(group1_data,group1_label,group2_data,group2_label),4))
    # temp.append(round(get_complexity_(group1_data,group1_label,group2_data,group2_label),4))
    # temp.append(round(get_complexity_L3(group1_data,group1_label,group2_data,group2_label),4))
    return temp