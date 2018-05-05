from pathlib import Path
import pandas as pd
import numpy as np
import sys
import pickle as p
import argparse
import nn
import preprocess as pp


def predict_accuracy(a, b):
	count = 0;
	for index, row in b.iterrows():
		label = row['class']
		if label == a[index]:
			count += 1;
	return count/len(a)

'''
This function converts the given data to one hot encoding
10 features are converted into 85 features for 5 cards
'''
def preprocess_data(file):
	encoded_data_file = Path("featured_data.csv")  #for checking if the encoded file is present or not
	if encoded_data_file.is_file():
		feature_data=pd.read_csv('featured_data.csv')  # if present then loading encoded file using pandas.csv
		feature_data=feature_data.values
		print('encoded_file_present',len(feature_data))
	else:
		# if not present then generate an encoding file to be used in our model training
		feature_data=pp.oneHotencodeData(file)
		feature_data=feature_data.values
	return feature_data


if __name__ == "__main__":

	# command line handling
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', '-f', action='store', dest='file', default = "train.csv" , help="Provide a file name")
	parser.add_argument('-r', '--learning_rate', action='store', type=float, dest='learning_rate', default = 0.1, help="learning rate")
	parser.add_argument('-l', '--lambda', action='store', type=float, dest='lambdaa', default = 1, help="lambda")
	parser.add_argument('-k', '--k-fold', action='store', type=int, dest='k_fold', default = 5, help="k-fold cross validation")
	parser.add_argument('-n', '--layered_parameter', action='store', dest='layer_info', nargs='*', type=int, default=[85, 100, 20, 10], help='list of neural network parameters')
	parser.add_argument('-e', '--epoch', action='store', type=int, dest='epoch', default = 1, help="epoch")

	args = parser.parse_args()
	
	train_file_name = args.file
	lambda_reg = args.lambdaa
	k_fold = args.k_fold
	layerInfo = args.layer_info
	learning_rate = args.learning_rate
	epoch = args.epoch


	# convert to one hot encoding
	print("Read train file and convert to one hot encoding")
	feature_data = preprocess_data("train.csv")
	

	# k-fold cross validation
	k_fold_data=pp.kFoldCrossVaildation(k_fold , feature_data)  #getting k-fold data for training
	print("perform k-fold cross validation, here k is: ", k_fold)
	# create nn class
	neural_net=nn.NeuralNetwork(layerInfo)    #creating object of neural network with 2 hidden layers and one input and output layer
	print(layerInfo, learning_rate, lambda_reg)
	
	print('Initialise neural network')


	# forward and backword propogation loop
	total = 0
	first=True
	for i in range(0,len(k_fold_data)):
		print("validation fold", i+1	)
		first=True
		for j in range(0,len(k_fold_data)):
			if i != j:
				if first==True:
					train_data=k_fold_data[j]
					first=False
				else:
					train_data=np.append(train_data,k_fold_data[j],axis=0)  #combining all training data into one set to train our model and excluding one test data
		neural_net.gradient_descent(train_data,epoch,len(train_data),learning_rate,lambda_reg) #calling gradient descent for every set of train data
		accuracy = neural_net.accuracy(k_fold_data[i])		
		print("Accuracy", accuracy)
		total = total + accuracy #prediction on test data

	print("average accuracy on train data", total/k_fold)
	#print(neural_net.accuracy(feature_data))                        # calculating accuracy for entire train data
	
	#p.dump(neural_net.weights, open("weight_file.p", "wb"))		#storing best weight and biases in a pickle file
	#p.dump(neural_net.biases, open("baises_file.p", "wb"))


	print("predicting for test data")
	print("One hot encoding of test data")
	test_data1=pd.read_csv('test_with_label.csv', index_col = None)  #loading test data
	test_data=pd.read_csv('test_with_label.csv')  #loading test data
	test_data=test_data.values
	prediction=[]
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
		predicted_value = np.argmax(neural_net.predict_output(d.reshape(85,1)))
		prediction.append(predicted_value)  #predicting for particular row in test data

	print("accuracy on test data", predict_accuracy(prediction, test_data1))


	prediction=np.array(prediction)   #saving file after prediction on all the data elements of test data
	prediction=pd.DataFrame(prediction)
	prediction.index = np.arange(0, len(prediction))
	headers=['predicted_class']
	prediction.index.name='id'
	prediction.columns=headers
	prediction.to_csv("output.csv", index=True, encoding='utf-8', header=True)