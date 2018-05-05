import numpy as np
import pandas as pd


class NeuralNetwork(object):  #Neural Network class is created  which is comptaible with increasing number of layers
    
    def __init__(self, conf):
        self.config=conf
        self.num_layers=len(conf)
        self.biases=[np.random.rand(y,1) for y in self.config[1:]]
        self.weights=[np.random.rand(y,x) for x,y in zip(self.config[:-1],self.config[1:])]
    

    def predict_output(self,a):
        for w,b in zip(self.weights,self.biases):
            a=self.sigmoid_calc(w.dot(a)+b)
        return a

    def sigmoid_calc(self,inp):   #activation function
        return 1.0/(1.0+np.exp(-inp))
    
    def cross_entropy_cost_derivative(self,predicted_op,actual_op):
        return predicted_op-actual_op
    
    def prediction_result(self,data,actual_op):
        op_predicted=np.argmax(self.predict_output(data))
        if(op_predicted==actual_op):
            return 1
        else:
            return 0

    def accuracy(self,full_data):
        total_predictions=0
        for data in full_data:
            total_predictions+=self.prediction_result(data[0:85].reshape(85,1),data[85])
        return total_predictions/len(full_data)
        

    def gradient_descent(self,train_data,no_of_times,train_data_size,learning_rate,lamda): #gradient descent function
        data_size=len(train_data)
        first=True
        featured_inputs=[]
        for i in range(no_of_times):
            #random.shuffle(train_data)
            for j in range(0,train_data_size):
                temp_biases,temp_weights=self.back_prop(train_data[j,0:85].reshape(85,1),train_data[j,85])
                self.weights=[(1-learning_rate*(lamda/train_data_size))*w-(learning_rate)*t_w for w, t_w in zip(self.weights, temp_weights)]
                self.biases = [b-(learning_rate)*nb for b, nb in zip(self.biases, temp_biases)]
            print("Completed epoch",i+1)
            

    def back_prop(self,data,output):  #back propogation algorithm
        op=np.zeros(10)
        op[output]=1
        temp_weights=[np.zeros(w.shape) for w in self.weights]
        temp_biases=[np.zeros(b.shape) for b in self.biases]
        activation=data
        activations=[data]       
        calc_weights=[]
        for b,w in zip(self.biases,self.weights):
            cal=w.dot(activation)+b
            calc_weights.append(cal)           #storing weights for each layer so as to be used later for calculating errors
            activation=self.sigmoid_calc(cal)   
            activations.append(activation)    #storing all the activations in a list
        #print(activations,calc_weights)
        #print(max_index,max_val,activations[-1],"\n")
        error=self.cross_entropy_cost_derivative(activations[-1],op.reshape(10,1))   #calculating cost derivative of last layer
        #print(op_achieved,op,output)                                                #
        temp_weights[-1]=np.dot(error,activations[-2].transpose())
        temp_biases[-1]=error
        for l in range(2,self.num_layers):
            cal=calc_weights[-l]
            sigma_der=self.sigmoidPrime(cal)
            error=np.dot(self.weights[-l+1].transpose(),error)*(sigma_der)  #calculating back propogation error
            temp_weights[-l]=np.dot(error,activations[-l-1].transpose())    #calculating weights and biases after backpropogation
            temp_biases[-l]=error
            #print(temp_weights)
        return temp_biases,temp_weights


    def cross_entropy_cost(self,pred_opt,opt):   #cross entropy cost function
    	return np.sum(-opt*np.log(pred_opt)-(1-opt)*np.log(1-pred_opt))


    def sigmoidPrime(self,x):   #sigmoid derivative used in backpropogating error to layers
        return self.sigmoid_calc(x)*(1-self.sigmoid_calc(x))


    
    '''def cal_Cost(self,data,lamda):  # calculating total cost after each epoches
        total_cost=0.0
        for d in data:
            pred_op=self.predicted_output(d[0:85].reshape(85,1))
            op=np.zeros(10)
            op[d[85]]=1
            total_cost+=self.cross_entropy_cost(pred_op,op)/len(data)
        total_cost+=(lamda/2*len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
        return total_cost
    '''