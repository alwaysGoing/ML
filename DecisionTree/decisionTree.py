# -*- coding: utf-8 -*-
# @Time    : 2019/6/27 20:33
# @Author  : Wang
# @FileName: decisionTree.py
# @Software: PyCharm

from math import log
import operator
import copy

# 计算熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in labelCounts:
        prob = labelCounts[key] / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
myDat, labels = createDataSet()
labelsBackup = copy.copy(labels)
print(myDat)
print(labels)
print(calcShannonEnt(myDat))

# 划分数据集，如果属性axis等于value，则把其加入到retDataSet中，同时将属性axis去掉
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

print(splitDataSet(myDat, 0, 0))

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 # 最后一维是label
    baseEntropy = calcShannonEnt(dataSet)
    bestinfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] # 统计每个样本的第i个特征，组成list
        uniquevals = set(featList)
        newEntropy = 0
        for value in uniquevals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestinfoGain):
            bestinfoGain = infoGain
            bestFeature = i
    return bestFeature


print(chooseBestFeatureToSplit(myDat))

# 多数表决
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 创建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat]) # 删除labels中的bestFeat属性
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# print(createTree(myDat, labels))

def classify(inputTree, featLabels, testVect):
    print(featLabels)
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVect[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVect)
            else:
                classLabel = secondDict[key]
    return classLabel

print(labels)
myTree = createTree(myDat, labels)
print(myTree)
print('backup: ', labelsBackup)
print(classify(myTree, labelsBackup, [1, 1]))