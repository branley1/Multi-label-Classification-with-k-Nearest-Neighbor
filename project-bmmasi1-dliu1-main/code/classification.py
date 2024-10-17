import pandas as pd
import numpy as np
from math import sqrt
from scipy.spatial import distance as d
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss
from sklearn.linear_model import LinearRegression
from operator import itemgetter

def loadFile(filename):
    try:
        df = pd.read_csv(filename)
    except:
        print("invalid file")

    #normalize data?????
    if filename == "enzyme_data.csv":
        df.pop("CIDs")
    
    return df




#helper functions
def kNearestneighbor(train, query, k):
    """
    gets K nearest neighbors based on distance
    """
    indexes = []
    current = 0
    for j in range(k):
        leastDistance = 2**64
        for i in range(len(train)):
            d = distance(train.iloc[i],query)
            if(d < leastDistance) and (i not in indexes):
                current = i
                leastDistance = d
        indexes.append(current)
    return indexes

def distance(a,b):
    if len(a) != len(b):
        print("wrong length or data type, distance not possible")
        return None
    sum = 0
    for i in range(len(a)):
        sum += (a[i] - b[i])**2
    return sqrt(sum)

def trainData(train,trainLabels,k):
    """
    Training process - rank k nearest neighbors based on trustworthiness

    q is list of all training points x indexed by their index
    q[i] is a list, containing k neighborLists of all its neighbors j
    neighborList is a dictionary of 4 items with properties relating to x and x's neighbor j
        featureDist is a list with length featureLength, containing j[feature] - x[feature]
        dist is float of Euclidean distance between j and x
        cosDist is float of cosine distance between j and x
        labelSet is label set of j

    then, apply hamming loss on label set between j and x to determine quality of each neighborList
    """
    rankingModel = []
    q=[]
    hm=[]
    print(train.shape)
    for i in range(len(train)):

        neighborRank = []
        #get index of neighbors for each training point
        neighbors = kNearestneighbor(train,train.iloc[i],k)

        for j in neighbors:

            if j!=i:

                #we create a list of items, [feature value difference list,Euclidean distance, Cosine distance]
                #then append that list onto q
                neighborList = []
            
                #difference between neighbor's feature value with self
                for feature in train.columns:
                    neighborList.append(train.iloc[j][feature] - train.iloc[i][feature])          

                #euclidean distance
                dist = distance(train.iloc[j],train.iloc[i])
                neighborList.append(dist)

                #cosine distance
                cosDist = d.cosine(train.iloc[j],train.iloc[i])
                neighborList.append(cosDist)

                q.append(neighborList)
            
                hm.append(hamming_loss(trainLabels[i],trainLabels[j]))

    #regression

    df = pd.DataFrame(q)
    df = df.fillna(0)
    reg = LinearRegression().fit(df,hm)
    
    return reg

def buildRankModel():
    pass

def vote():
    """
    weighted voting strategy to produce final prediction - use Hamming loss to determine weights
    """
    pass


def rankingKNN(k):
    k = k+1
    weights = []
    
    #load data
    data = pd.DataFrame()
    data = loadFile("enzyme_data.csv")
    #data = data.head(50)
    labels = data.pop(data.columns[-1])

   
    for column in data:
        if data[column].max() - data[column].min() != 0:
            data[column] = (data[column]-data[column].min())/(data[column].max()-data[column].min())
        else:
            data.drop(column,axis=1)
    newLabels = []
    for i in labels:
        sing = []

        i = i.replace('_','')
        for char in range(len(i)):
            sing.append(i[char])
        newLabels.append(sing)
    labels =  newLabels

    Xtrain,Xtest,Ytrain,Ytest = train_test_split(data,labels,test_size=0.4,random_state=0)



    #train dataset
    model = trainData(Xtrain,Ytrain,k)
    
    
    #get nearest neighbors of query point

    totalLoss = 0
    knnLoss = 0
    for index in range(len(Xtest)):
        neighbors = kNearestneighbor(Xtrain,Xtest.iloc[index],k-1)
        for i in neighbors:
            q=[]
            neighborList = []

            c = 0
            
            
            #difference between neighbor's feature value with self
            for feature in range(len(Xtrain.columns)):
                neighborList.append(Xtrain.iloc[i][feature] - Xtest.iloc[index][feature])          

                #euclidean distance
            dist = distance(Xtrain.iloc[i],Xtest.iloc[index])
            neighborList.append(dist)

                #cosine distance
            cosDist = d.cosine(Xtrain.iloc[i],Xtest.iloc[index])
            neighborList.append(cosDist)

            q.append(neighborList)
            predict = model.predict(q)
            weights.append(float(1 - predict))
            
                #predict new labels
        queryLabel = []
        knnLabel = []
        denominator = 0
        
        for weight in weights:
            
            denominator+=weight
        

        for label in range(len(labels[0])):
            counter = 0
            numerator = 0
            knnNum = 0
            
            for neighbor in neighbors:

                if labels[neighbor][label] == '0':
                    numerator+= weights[counter] * (-1)
                    knnNum += -1
                else:
                    numerator+=weights[counter]
                    knnNum+=1
                counter+=1
            total = numerator/denominator
            
            if total > 0:
                queryLabel.append('1')
            else:
                queryLabel.append('0')
            if knnNum > 0:
                knnLabel.append('1')
            else:
                knnLabel.append('0')
        #print("knntest: ",knnLabel)
        #print("guess: ",queryLabel)
        #print(Ytest[index])
        
        totalLoss += hamming_loss(queryLabel,Ytest[index])

        knnLoss += hamming_loss(knnLabel,Ytest[index])

    print("k = ", k-1)
    print("KNN Loss Final: ",knnLoss/len(Ytest))
    print("Hamming Loss Final: ",totalLoss/len(Ytest))
    
rankingKNN(3)
rankingKNN(5)
rankingKNN(7)
rankingKNN(11)
rankingKNN(15)
