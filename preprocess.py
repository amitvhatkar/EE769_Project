import pandas as pd
import numpy as np

def oneHotencodeData(file_name):   # function to convert an input in one-hot encoding
    data=pd.read_csv(file_name)
    data=data.values
    file=open("featured_data.csv","a")
    featured_data=[]
    first=True
    print('Encoded file not present')
    for single_data in data:
        first=True
        for length in range(0,len(single_data[0:10]),2):
            a=np.zeros(13)
            b=np.zeros(4)
            a[single_data[length+1]-1]=1
            b[single_data[length]-1]=1
            c=np.concatenate((b,a))
            if first:
                d=c
                first=False
            else:
                d=np.concatenate((d,c))
        op=np.zeros(1)
        op[0]=single_data[10]
        d=np.concatenate((d,op))
        d.tofile(file,sep=",",format="%d")
        file.write("\n")
    return pd.read_csv('featured_data.csv')


def kFoldCrossVaildation(k, data):   # function to create a set of five equal datas used for training and tesing 
    train_test_data=[]
    hopps = int((len(data) + 1)/k)
    #print(len(data), hopps, k)
    for i in range(0,len(data),hopps):
        train_test_data.append(data[i:i+(hopps - 1)])
    return train_test_data