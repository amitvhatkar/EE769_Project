from pathlib import Path
import pandas as pd
import numpy as np
import sys
import pickle as p
import argparse

def predict_output(a, weights, biases):
    for w,b in zip(weights, biases):
        a=sigmoid_calc(w.dot(a)+b)
    return a


def sigmoid_calc(inp):   #activation function
    return 1.0/(1.0+np.exp(-inp))


def sigmoid_calc(inp):   #activation function
	return 1.0/(1.0+np.exp(-inp))


def predict_accuracy(a, b):
	count = 0;
	for index, row in b.iterrows():
		label = row['class']
		if label == a[index]:
			count += 1;
	return count/len(a)


if __name__ == "__main__":
	print("Reading test file")
	test_data1=pd.read_csv('test_with_label.csv', index_col = None)  #loading test data
	test_data=pd.read_csv('test_with_label.csv')  #loading test data
	test_data=test_data.values
	prediction=[]

	print("Reading weights and biases from trained model (pickle)")
	with open("weight_file.p", 'rb') as pickle_file:
		weights  = p.load(pickle_file)

	with open("biases_file.p", 'rb') as pickle_file:
		biases = p.load(pickle_file)

	print("Embedding of test data: Converting test data to one hot encoding")
	print("Predicting output for test data using forward propogation...")
	for single_data in test_data:      #converting test data in encoded format to feed to our model for prediction
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
		predicted_value = np.argmax(predict_output(d.reshape(85,1), weights, biases))
		prediction.append(predicted_value)  #predicting for particular row in test data

	print("accuracy", predict_accuracy(prediction, test_data1))

	prediction=np.array(prediction)   #saving file after prediction on all the data elements of test data
	prediction=pd.DataFrame(prediction)
	prediction.index = np.arange(0, len(prediction))
	headers=['predicted_class']
	prediction.index.name='id'
	prediction.columns=headers
	prediction.to_csv("output.csv", index=True, encoding='utf-8', header=True)
	print("Predicted values stored to 'output.csv'")


