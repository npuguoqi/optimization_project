# -*- coding: utf-8 -*-
"""
@author: Qi Guo
"""
'''
an univariate nonlinear function is utilized and its expression is sin(x)/x, -10<x<10
the training set consists of seventy training points with Gaussian noise of zero mean and 0.01 variance added to the training points
the training data is in the path '/xls_file/for_test/fun_result_matrix.csv'
please notice that the network may not converge for Bayesian regularization because the initial weights and biases may be unreasonable
another reason for the problem is that a large alpha at the initial stage of the training may lead to unconvergence, this can be validated by setting a large alpha (for example alpha=0.5) in Levenberg-Marquardt
if the problem exists, you need to retrain the network
the problem does not exist for Levenberg Marquardt without regularization
we are finding ways to solve the problem, like pre-training the network to obtain reasonable network weights and biases, but the work has not been completed
'''
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt
from scipy.linalg import norm, pinv, eig
from math import exp, pow
import math
import xlrd
import os
import csv
import random

#define neural network layer
class Layer:
    def __init__(self, w, b, neuron_number, transfer_function, layer_index):
        self.w = w                                  #weights
        self.b = b                                  #biases
        self.neuron_number = neuron_number          #number of neurons
        self.transfer_function = transfer_function  #transfer function
        self.layer_index = layer_index              #layer index

#define neural network structure
class NetStruct:
    '''
    Constructor
    the weight initial method can be divided into random_generate and nguyen_widrow
    the network initial method can be divided into random_generate and read_from_file.
    when the network initial method is random_generate, file_name is []. 
    when the network initial method is read_from_file, file_name represents file name. 
    '''
    def __init__(self, input_val, output_val, hidden_layers, activ_fun_list,weight_initial_method,net_initial_method,file_name):
        #network initial method is random_generate
        if net_initial_method=='random_generate':
            if len(hidden_layers) == len(activ_fun_list):
                activ_fun_list.append('line')
            self.activ_fun_list = activ_fun_list
            input_val = np.array(input_val)
            output_val = np.array(output_val)
            if (input_val.shape[1] != output_val.shape[1]):
                print('dimension are not same')
                sys.exit()
            self.input_val = input_val
            self.output_val = output_val
            #the number of input layer neurons is the number of input data rows
            #the number of output layer neurons is 1
            inputlayer_neuron_number = self.input_val.shape[0]
            outputlayer_neuron_number =1
            tmp = []
            tmp.append(inputlayer_neuron_number)
            tmp.extend(hidden_layers)
            tmp.append(outputlayer_neuron_number)
            self.hidden_layers = np.array(tmp)
            self.layer_num = len(self.hidden_layers)
            #initialize weights and biases
            #random_generate
            if weight_initial_method == 'random_generate':
                self.layers = []
                for i in range(0, self.layer_num):
                    if i == 0:
                        self.layers.append(Layer([], [], self.hidden_layers[i], 'none', i))
                        continue
                    first = self.hidden_layers[i - 1]
                    second = self.hidden_layers[i]
                    self.layers.append(
                        Layer(np.random.randn(second, first), np.random.randn(second, 1), self.hidden_layers[i],self.activ_fun_list[i - 1],i))
            #nguyen_widrow
            else:
                self.layers = []
                for i in range(0, self.layer_num):
                    if i == 0:
                        self.layers.append(Layer([], [], self.hidden_layers[i], 'none', i))
                        continue
                    if i == self.layer_num - 1:
                        first = self.hidden_layers[i - 1]
                        second = self.hidden_layers[i]
                        self.layers.append(
                            Layer(np.random.uniform(-1, 1, [second, first]), np.random.uniform(-1, 1, [second, 1]),self.hidden_layers[i], self.activ_fun_list[i - 1], i))
                        continue
                    first = self.hidden_layers[i - 1]
                    second = self.hidden_layers[i]
                    weight_matrix = np.random.uniform(-1, 1, [second, first])
                    bias_matrix = np.random.uniform(-1, 1, [second, 1])
                    for nrow in range(0, weight_matrix.shape[0]):
                        sum = 0
                        for ncol in range(0, weight_matrix.shape[1]):
                            sum = sum + math.fabs(weight_matrix[nrow][ncol])
                        for ncol in range(0, weight_matrix.shape[1]):
                            weight_matrix[nrow][ncol] = weight_matrix[nrow][ncol] / sum
                    H = first
                    K = second
                    G = 0.7 * math.pow(H, 1 / (K - 1))
                    for nrow in range(0, bias_matrix.shape[0]):
                        bias_matrix[nrow][0] = (2 * nrow / K - 1) * bias_matrix[nrow][0] * G
                    self.layers.append(
                        Layer(weight_matrix, bias_matrix, self.hidden_layers[i], self.activ_fun_list[i - 1], i))
        #network initial method is read_from_file
        else:
            self.input_val = input_val
            self.output_val = output_val
            relative_path='net_parameter'                        #network parameters path(/net_parameter)
            relative_path=os.path.join(relative_path,file_name)
            file_object = open(relative_path, 'r')
            #read the layer of the network
            line = file_object.readline()
            line=line.strip('\n')
            self.layer_num=int(float(line))
            tmp=[]
            line=file_object.readline()
            for i in range(0,int(self.layer_num)):
                line=line.strip('\n')
                tmp.append(int(float(line)))
                line=file_object.readline()
            self.hidden_layers = np.array(tmp)
            #read transfer functions
            self.activ_fun_list=[]
            line=line.strip('\n')
            activ_fun_list_num=int(float(line))
            line=file_object.readline()
            for i in range(0,activ_fun_list_num):
                line=line.strip('\n')
                self.activ_fun_list.append(line)
                line=file_object.readline()
            #read weights and biases
            net_parameter=[]
            while line:
                line=line.strip('\n')
                net_parameter.append(float(line))
                line=file_object.readline()
            file_object.close()

            self.layers = []
            index = 0
            for i in range(0, self.layer_num):
                if i == 0:
                    self.layers.append(Layer([], [], self.hidden_layers[i], 'none', i))  # 初始化神经网络，该行初始化第一层权值和输入层神经元数目
                    continue
                curr_layer_w=np.array(net_parameter[index:index+self.hidden_layers[i]*self.hidden_layers[i-1]])
                index=index+self.hidden_layers[i]*self.hidden_layers[i-1]
                curr_layer_b=np.array(net_parameter[index:index+self.hidden_layers[i]])
                index=index+self.hidden_layers[i]
                self.layers.append(Layer(curr_layer_w.reshape(self.hidden_layers[i],self.hidden_layers[i-1],order='C'),curr_layer_b.reshape(self.hidden_layers[i],1,order='C'),self.hidden_layers[i],self.activ_fun_list[i-1],i))

class NetUtilize:
    '''
    Constructor
    the initialized parameters include: network structure, mu, decrement of mu, increment of mu, maximum of mu, tolerence, goal of gradient, goal of gamma, training method
    '''
    def __init__(self, net_struct, mu, mu_dec, mu_inc, mu_max, iteration, tol, gradient, gamma_criterion,training_method):
        self.net_struct = copy.deepcopy(net_struct)
        self.mu = mu
        self.mu_dec = mu_dec
        self.mu_inc = mu_inc
        self.mu_max = mu_max
        self.iteration = iteration
        self.tol = tol
        self.gradient = gradient
        self.gamma_criterion = gamma_criterion
        self.training_method = training_method

    #train
    def train(self):
        #training method can be divided into Levenberg-Marquardt and Bayesian-Regularization
        if self.training_method == 'Levenberg-Marquardt':        #if training method is Levenberg-Marquardt, return training error
            #method_1 is according to:Hagan MT, Menhaj MB (1994) Training feedforward networks with the Marquardt algorithm. IEEE Trans Neural Networks 5:989–993. doi: 10.1109/72.329697
            #method_2 is according to:Madsen K, Nielsen HB, Tingleff O (2004) METHODS FOR NON-LINEAR LEAST SQUARES PROBLEMS. Informatics and Mathematical Modelling, Technical University of Denmark
            #please use method_1 as method_2 has not been completely finished
            train_error = self.lm('method_1')
            return train_error
        elif self.training_method == 'Bayesian-Regularization':  #if training method is Levenberg-Marquardt，return eigenvalues alpha,beta,gamma,ED,EW
            [eigenvalues, alpha_list, beta_list, gamma_list, E_D_list, E_W_list] = self.br('method_1')
            return [eigenvalues, alpha_list, beta_list, gamma_list, E_D_list, E_W_list]
    #predict
    def sim(self, input_val):
        self.net_struct.input_val = input_val
        self.forward()
        layer_num = self.net_struct.layer_num
        predict = self.net_struct.layers[layer_num - 1].output_val
        return predict

    #transfer function
    def activ_fun(self, input_val, activ_type):
        if activ_type == 'sigm':
            output_val = 1.0 / (1.0 + np.exp(-input_val))
        elif activ_type == 'tanh':
            output_val = (np.exp(input_val) - np.exp(-input_val)) / (np.exp(input_val) + np.exp(-input_val))
        elif activ_type == 'line':
            output_val = input_val
        return output_val

    #derivatives of transfer function
    def activ_fun_grad(self, input_val, activ_type):
        if activ_type == 'sigm':
            grad = self.activ_fun(input_val, activ_type) * (1.0 - self.activ_fun(input_val, activ_type))
        elif activ_type == 'tanh':
            grad = 1.0 - self.activ_fun(input_val, activ_type) * self.activ_fun(input_val, activ_type)
        elif activ_type == 'line':
            m = input_val.shape[0]
            n = input_val.shape[1]
            grad = np.ones((m, n))
        return grad

    #forward
    def forward(self):  # 前向
        for i in range(0, self.net_struct.layer_num):
            if i == 0:
                curr_layer = self.net_struct.layers[i]
                curr_layer.input_val = self.net_struct.input_val
                curr_layer.output_val = self.net_struct.input_val
                continue
            before_layer = self.net_struct.layers[i - 1]
            curr_layer = self.net_struct.layers[i]
            curr_layer.input_val = curr_layer.w.dot(before_layer.output_val) + curr_layer.b
            curr_layer.output_val = self.activ_fun(curr_layer.input_val, self.net_struct.activ_fun_list[i - 1])

    #backward
    def backward(self):
        layer_num = self.net_struct.layer_num
        last_layer = self.net_struct.layers[layer_num - 1]
        last_layer.error = self.activ_fun_grad(last_layer.input_val, self.net_struct.activ_fun_list[layer_num - 2])

        layer_index = list(range(1, layer_num - 1))
        layer_index.reverse()
        for i in layer_index:
            curr_layer = self.net_struct.layers[i]
            curr_layer.error = (last_layer.w.transpose().dot(last_layer.error)) * self.activ_fun_grad(curr_layer.input_val, self.net_struct.activ_fun_list[i - 1])
            last_layer = curr_layer

    #calculate partial derivative
    def parderiv(self):
        layer_num = self.net_struct.layer_num
        for i in range(1, layer_num):
            before_layer = self.net_struct.layers[i - 1]
            before_input_val = before_layer.output_val.transpose()
            curr_layer = self.net_struct.layers[i]
            curr_error = curr_layer.error
            curr_error = curr_error.reshape(curr_error.shape[0] * curr_error.shape[1], 1, order='F')
            row = curr_error.shape[0]
            col = before_input_val.shape[1]
            tmp_arr = np.zeros((row, col))
            num = before_input_val.shape[0]
            neuron_number = curr_layer.neuron_number
            for i in range(0, num):
                tmp_arr[neuron_number * i:neuron_number * i + neuron_number, :] = np.repeat([before_input_val[i, :]],
                                                                                            neuron_number,
                                                                                            axis=0)
            tmp_w_par_deriv = curr_error * tmp_arr
            curr_layer.w_par_deriv = np.zeros(
                (num, before_layer.neuron_number * curr_layer.neuron_number))
            for i in range(0, num):
                tmp = tmp_w_par_deriv[neuron_number * i:neuron_number * i + neuron_number, :]
                tmp = tmp.reshape(tmp.shape[0] * tmp.shape[1], order='C')
                curr_layer.w_par_deriv[i, :] = tmp
            curr_layer.b_par_deriv = curr_layer.error.transpose()

    #calculate Jacobian matrix
    def caljacobian(self):
        layer_list = self.net_struct.hidden_layers
        row = self.net_struct.input_val.shape[1]
        col = 0
        for i in range(0, len(layer_list) - 1):
            col = col + layer_list[i] * layer_list[i + 1] + layer_list[i + 1]
        j = np.zeros((row, col))
        layer_num = self.net_struct.layer_num
        index = 0
        for i in range(1, layer_num):
            curr_layer = self.net_struct.layers[i]
            w_col = curr_layer.w_par_deriv.shape[1]
            b_col = curr_layer.b_par_deriv.shape[1]
            j[:, index:index + w_col] = curr_layer.w_par_deriv
            index = index + w_col
            j[:, index:index + b_col] = curr_layer.b_par_deriv
            index = index + b_col
        return j

    #calculate JJ and Je
    def caljjje(self):
        layer_num = self.net_struct.layer_num
        e = self.net_struct.layers[layer_num - 1].output_val - self.net_struct.output_val
        e = e.transpose()
        j = self.caljacobian()
        jj = j.transpose().dot(j)
        je = -j.transpose().dot(e)
        return [jj, je]

    #calculate weights and biases
    def cal_w(self, net_struct):
        layer_num = net_struct.layer_num
        layer_list = net_struct.hidden_layers
        col = 0
        for i in range(0, len(layer_list) - 1):
            col = col + layer_list[i] * layer_list[i + 1] + layer_list[i + 1]
        w = np.zeros((col, 1))
        index = 0
        for i in range(1, layer_num):
            tmp_w = net_struct.layers[i].w
            tmp_w = tmp_w.reshape(layer_list[i] * layer_list[i - 1], 1, order='C')
            w[index:index + layer_list[i] * layer_list[i - 1], :] = tmp_w
            index = index + layer_list[i] * layer_list[i - 1]
            w[index:index + layer_list[i], :] = net_struct.layers[i].b
            index = index + layer_list[i]
        N = w.shape[0]
        return [w, N]
    '''
    Levenberg-Marquardt
    you can define regularization parameters here
    alpha=0.5 will result in unconvergence
    alpha=0.05 will result in underfitting
    alpha=0.0005 will produce good prediction function
    alpha=0.000005 will result in overfitting
    '''
    def lm(self, method='method_1'):
        beta = 1
        alpha = 0.0005
        train_error = []
        mu = self.mu
        mu_dec = self.mu_dec
        mu_inc = self.mu_inc
        mu_max = self.mu_max
        iteration = self.iteration
        tol = self.tol
        gradient = self.gradient
        output_val = self.net_struct.output_val
        self.forward()
        predict = self.net_struct.layers[self.net_struct.layer_num - 1].output_val
        [w, N] = self.cal_w(self.net_struct)
        performance = beta * self.performance(output_val, predict) + alpha * norm(w) * norm(w)
        if method == 'method_1':
            for i in range(0, iteration):
                sum_squared_error = self.performance(output_val,self.net_struct.layers[self.net_struct.layer_num - 1].output_val)
                print(i, ' ', sum_squared_error)
                train_error.append(sum_squared_error)
                if (sum_squared_error < tol):
                    print('sum of squared error convergence!')
                    break
                self.backward()
                self.parderiv()
                [jj, je] = self.caljjje()
                while (1):
                    H = beta * jj + alpha * np.eye(jj.shape[0]) + mu * np.eye(jj.shape[0])
                    delta_w_b = pinv(H).dot(beta * je - alpha * w)
                    old_net_struct = copy.deepcopy(self.net_struct)
                    self.update_net_struct(delta_w_b)
                    [w, N] = self.cal_w(self.net_struct)
                    self.forward()
                    predict_new = self.net_struct.layers[self.net_struct.layer_num - 1].output_val
                    performance_new = beta * self.performance(output_val, predict_new) + alpha * norm(w) * norm(w)
                    if (performance_new < performance):
                        mu = mu * mu_dec
                        performance_old = performance
                        performance = performance_new
                        break
                    else:
                        mu = mu * mu_inc
                        self.net_struct = copy.deepcopy(old_net_struct)
                if (math.fabs(performance - performance_old)) < gradient:
                    print('performance function gradient convergence!')
                    break
                if (mu > mu_max):
                    print('mu exceed mu_max!')
                    break
        elif method == 'method_2':
            u = 1
            v = 2
            for i in range(0, iteration):
                print(i, ' ', performance)
                train_error.append(performance)
                if (performance < tol):
                    break
                self.backward()
                self.parderiv()
                [jj, je] = self.caljjje()
                H = jj + u * np.eye(jj.shape[0])
                delta_w_b = pinv(H).dot(je)
                old_net_struct = copy.deepcopy(self.net_struct)
                self.update_net_struct(delta_w_b)
                self.forward()
                predict_new = self.net_struct.layers[self.net_struct.layer_num - 1].output_val
                performance_new = self.performance(output_val, predict_new)
                q = (performance - performance_new) / (
                    delta_w_b.transpose().dot(2 * je) - 0.5 * (delta_w_b.transpose().dot(2 * jj)).dot(delta_w_b))
                if q > 0:
                    s = 1.0 / 3.0
                    v = 2
                    performance = performance_new
                    tmp = 1 - pow(2 * q - 1, 3)
                    if s > tmp:
                        u = u * s
                    else:
                        u = u * tmp
                else:
                    u = u * v
                    v = 2 * v
                    self.net_struct = copy.deepcopy(old_net_struct)
        return train_error
    '''
    Bayesian-Regularization
    the regularization parameters calculated by Bayesian regularization is: alpha/beta=0.00054
    '''
    def br(self, method='method_1'):
        alpha = 0
        beta = 1
        gamma = 0
        E_W = 0
        E_D = 0
        train_error = []
        alpha_list = []
        beta_list = []
        gamma_list = []
        E_D_list = []
        E_W_list = []
        mu = self.mu
        mu_dec = self.mu_dec
        mu_inc = self.mu_inc
        mu_max = self.mu_max
        iteration = self.iteration
        tol = self.tol
        gradient = self.gradient
        gamma_criterion = self.gamma_criterion
        output_val = self.net_struct.output_val
        if method == 'method_1':
            for i in range(0, iteration):
                self.forward()
                predict = self.net_struct.layers[self.net_struct.layer_num - 1].output_val
                [w, N] = self.cal_w(self.net_struct)
                performance = beta * self.performance(output_val, predict) + alpha * norm(w) * norm(w)
                E_D_performance = self.performance(output_val, predict)
                E_D = self.performance(output_val, predict)
                print(i, ' ', E_D)
                train_error.append(E_D)
                if (E_D < tol):
                    print('sum of squared error convergence!')
                    break
                self.backward()
                self.parderiv()
                [jj, je] = self.caljjje()
                while (1):
                    H = beta * jj + alpha * np.eye(jj.shape[0])
                    A = H + mu * np.eye(jj.shape[0])
                    delta_w_b = pinv(A).dot(beta * je - alpha * w)
                    old_net_struct = copy.deepcopy(self.net_struct)
                    self.update_net_struct(delta_w_b)
                    [w_new, N] = self.cal_w(self.net_struct)
                    self.forward()
                    predict_new = self.net_struct.layers[self.net_struct.layer_num - 1].output_val
                    performance_new = beta * self.performance(output_val, predict_new) + alpha * norm(w_new) * norm(
                        w_new)
                    E_D_performance_new = self.performance(output_val, predict_new)
                    if (performance_new < performance):
                        mu = mu * mu_dec
                        break
                    else:
                        mu = mu * mu_inc
                        self.net_struct = copy.deepcopy(old_net_struct)
                if math.fabs(performance_new - performance) < 1e-10:
                    print('performance function gradient convergence!')
                    break
                if math.fabs(E_D_performance - E_D_performance_new) < gradient:
                    print('error gradient convergence!')
                    break
                if mu > mu_max:
                    print('mu exceed mu_max!')
                    break
                gamma = N - 2 * alpha * np.trace(pinv(2 * beta * jj + 2 * alpha * np.eye(jj.shape[0])))
                E_W = norm(w_new) * norm(w_new)
                E_D = self.performance(output_val, predict_new)
                alpha = gamma / (2 * E_W)
                beta = math.fabs((self.net_struct.input_val.shape[1] - gamma) / (2 * E_D))
                alpha_list.append(alpha)
                beta_list.append(beta)
                gamma_list.append(gamma)
                E_D_list.append(E_D)
                E_W_list.append(E_W)
                if i >= 1:
                    if math.fabs(gamma_list[-1] - gamma_list[-2]) < gamma_criterion:
                        print('gama convergence')
                        break
            eigenvalues, eigenvectors = eig(2 * beta * jj)
            tmp_eigenvalues = np.real(eigenvalues)
        elif method == 'method_2':
            u = 1
            v = 2
            for i in range(0, iteration):
                print(i, ' ', performance)
                train_error.append(performance)
                if (performance < tol):
                    break
                self.backward()
                self.parderiv()
                [jj, je] = self.caljjje()
                H = jj + u * np.eye(jj.shape[0])
                delta_w_b = pinv(H).dot(je)
                old_net_struct = copy.deepcopy(self.net_struct)
                self.update_net_struct(delta_w_b)
                self.forward()
                predict_new = self.net_struct.layers[self.net_struct.layer_num - 1].output_val
                performance_new = self.performance(output_val, predict_new)
                q = (performance - performance_new) / (
                    delta_w_b.transpose().dot(je) - 0.5 * (delta_w_b.transpose().dot(jj)).dot(delta_w_b))
                if q > 0:
                    s = 1.0 / 3.0
                    v = 2
                    performance = performance_new
                    tmp = 1 - pow(2 * q - 1, 3)
                    if s > tmp:
                        u = u * s
                    else:
                        u = u * tmp
                else:
                    u = u * v
                    v = 2 * v
                    self.net_struct = copy.deepcopy(old_net_struct)
        return [tmp_eigenvalues, alpha_list, beta_list, gamma_list, E_D_list, E_W_list]

    #calculate sum squared error
    def performance(self, actual_val, predict_val):  # 计算sum squared error
        tmp = norm(actual_val - predict_val)
        return tmp * tmp

    #update network
    def update_net_struct(self, delta_w_b):
        layer_num = self.net_struct.layer_num
        index = 0
        for i in range(1, layer_num):
            before_layer = self.net_struct.layers[i - 1]
            curr_layer = self.net_struct.layers[i]
            w_num = before_layer.neuron_number * curr_layer.neuron_number
            b_num = curr_layer.neuron_number
            delta_w = delta_w_b[index:index + w_num]
            delta_w = delta_w.reshape(curr_layer.neuron_number, before_layer.neuron_number, order='C')
            index = index + w_num
            delta_b = delta_w_b[index:index + b_num]
            index = index + b_num
            curr_layer.w = curr_layer.w + delta_w
            curr_layer.b = curr_layer.b + delta_b

    #save network
    def save_net(self,filename):
        relative_path=''
        path_list=['net_parameter',filename]
        for path in path_list:
            relative_path=os.path.join(relative_path,path)
        file_object = open(relative_path, 'w')
        file_object.write(str(len(self.net_struct.hidden_layers))+'\n')
        for i in range(0,len(self.net_struct.hidden_layers)):
            file_object.write(str(self.net_struct.hidden_layers[i])+'\n')
        file_object.write(str(len(self.net_struct.activ_fun_list))+'\n')
        for i in range(0,len(self.net_struct.activ_fun_list)):
            file_object.write(self.net_struct.activ_fun_list[i]+'\n')
        [w,N]=self.cal_w(self.net_struct)
        for i in range(0,len(w)):
            file_object.write(str(w[i][0])+'\n')
        file_object.close()

#normalization
def mapminmax(input_array):
    tmp_array = copy.deepcopy(input_array)
    row = input_array.shape[0]
    col = input_array.shape[1]
    min_list = []
    max_list = []
    for i in range(0, row):
        min_val = min(input_array[i, :])
        max_val = max(input_array[i, :])
        for j in range(0, col):
            tmp_array[i][j] = 2 * (tmp_array[i][j] - min_val) / (max_val - min_val) - 1
        min_list.append(min_val)
        max_list.append(max_val)
    return tmp_array, min_list, max_list

#read excel file
def read_excel(filename):
    if os.path.exists(filename):
        pass
    else:
        print('no file')
    rb = xlrd.open_workbook(filename)
    table = rb.sheet_by_name('Sheet1')
    nrows = table.nrows
    ncols = table.ncols
    tmp_matrix = np.empty([nrows, ncols])
    for i in range(nrows):
        for j in range(ncols):
            tmp_matrix[i][j] = table.cell(i, j).value
    return tmp_matrix

#read csv file
def read_csv(filename):
    if os.path.exists(filename):
        pass
    else:
        print('no file')
    csvFile = open(filename, 'r')
    reader = csv.reader(csvFile)
    tmp_data = []
    for item in reader:
        tmp_data.append(item)
    nrows = len(tmp_data)
    ncols = len(tmp_data[0])
    tmp_data_matrix = np.empty([nrows, ncols])
    for i in range(nrows):
        for j in range(ncols):
            tmp_data_matrix[i][j] = float(tmp_data[i][j])
    return tmp_data_matrix

#calculate R2
def cal_R2(actual_data, predict_data):
    mean = np.mean(actual_data)
    SST = 0
    SSE = 0
    for i in range(0, actual_data.shape[0]):
        SST = SST + (actual_data[i][0] - mean) ** 2
        SSE = SSE + (actual_data[i][0] - predict_data[i][0]) ** 2
    return 1 - SSE / SST

#calculate coefficients of linear least square
def linear_least_square(A, b):
    return (pinv((A.transpose()).dot(A))).dot((A.transpose()).dot(b))

#save normalization parameter for the network
def save_normalization_parameter(filename_normalization,input_min_list,input_max_list,output_min_list,output_max_list):
    relative_path = ''
    path_list = ['net_parameter', filename_normalization]
    for path in path_list:
        relative_path = os.path.join(relative_path, path)
    csvFile = open(relative_path, 'w', newline='')
    writer = csv.writer(csvFile)
    writer.writerow(input_min_list)
    writer.writerow(input_max_list)
    writer.writerow(output_min_list)
    writer.writerow(output_max_list)
    csvFile.close()

#read normalization parameter for the network
def read_normalization_parameter(filename_normalization):
    relative_path = ''
    path_list = ['net_parameter', filename_normalization]
    for path in path_list:
        relative_path = os.path.join(relative_path, path)
    if os.path.exists(relative_path):
        pass
    else:
        print('no file')
    csvFile = open(relative_path, 'r')
    reader = csv.reader(csvFile)
    tmp_data = []
    for item in reader:
        tmp_data.append(item)
    csvFile.close()
    input_min_list=[]
    input_max_list=[]
    output_min_list=[]
    output_max_list=[]
    ncols = len(tmp_data[0])
    for j in range(ncols):
        input_min_list.append(float(tmp_data[0][j]))

    ncols = len(tmp_data[1])
    for j in range(ncols):
        input_max_list.append(float(tmp_data[1][j]))

    ncols = len(tmp_data[2])
    for j in range(ncols):
        output_min_list.append(float(tmp_data[2][j]))

    ncols = len(tmp_data[3])
    for j in range(ncols):
        output_max_list.append(float(tmp_data[3][j]))
    return input_min_list,input_max_list,output_min_list,output_max_list

if __name__ == '__main__':

    input_traindata = np.array([np.linspace(-10, 10, 70)])
    output_traindata = read_csv(r'xls_file\for_test\fun_result_matrix.csv')
    input_testdata = np.array([np.linspace(-10, 10, 400)])
    output_testdata = np.array([np.linspace(-10, 10, 400)])
    input_testdata_train = np.array([np.linspace(-10, 10, 70)])
    output_testdata_train = np.array([np.linspace(-10, 10, 70)])

    for i in range(0, input_testdata.shape[1]):
        output_testdata[0][i] = math.sin(input_testdata[0][i]) / input_testdata[0][i]
    for i in range(0, input_testdata_train.shape[1]):
        output_testdata_train[0][i] = output_traindata[0][i]

    #normalization
    [input_traindata_std, input_min_list, input_max_list] = mapminmax(input_traindata)
    [output_traindata_std, output_min_list, output_max_list] = mapminmax(output_traindata)

    # initialized neural network
    hidden_layers = [15]
    activ_fun_list = ['sigm']
    net = NetStruct(input_traindata_std, output_traindata_std, hidden_layers, activ_fun_list,'nguyen_widrow','random_generate','')
    netuse = NetUtilize(net, mu=1e-3, mu_dec=0.1, mu_inc=10, mu_max=1e10, iteration=1000, tol=1e-30, gradient=1e-10,gamma_criterion=1e-6, training_method='Levenberg-Marquardt')
    if netuse.training_method == 'Levenberg-Marquardt':
        error = netuse.train()
        netuse.save_net('net_for_test.txt')
        save_normalization_parameter('net_for_test_normalization.csv', input_min_list, input_max_list, output_min_list,output_max_list)
    elif netuse.training_method == 'Bayesian-Regularization':
        [eigenvalues, alpha, beta, gamma, D, W] = netuse.train()
        netuse.save_net('net_for_test.txt')
        save_normalization_parameter('net_for_test_normalization.csv',input_min_list,input_max_list,output_min_list,output_max_list)

    if netuse.training_method == 'Bayesian-Regularization':

        #calculate mean square error and maximum relative error for training data
        print('----------training data----------')
        for i in range(input_testdata_train.shape[0]):
            for j in range(input_testdata_train.shape[1]):
                input_testdata_train[i][j] = 2 * (input_testdata_train[i][j] - input_min_list[i]) / (input_max_list[i] - input_min_list[i]) - 1
        z1_train = netuse.sim(input_testdata_train)
        #anti-normalization
        for i in range(0, z1_train.shape[0]):
            for j in range(0, z1_train.shape[1]):
                z1_train[i][j] = (z1_train[i][j] + 1) / 2 * (output_max_list[i] - output_min_list[i]) + output_min_list[i]
        mse_train=norm(z1_train-output_testdata_train)*norm(z1_train-output_testdata_train)/z1_train.shape[1]
        #calculate maximum relative error
        max_error_train = math.fabs((z1_train[0][0] - output_testdata_train[0][0]) / output_testdata_train[0][0])
        for j in range(0, output_testdata_train.shape[1]):
            relative_error_train = math.fabs((z1_train[0][j] - output_testdata_train[0][j]) / (output_testdata_train[0][j]))
            print(j,':',relative_error_train)
            if relative_error_train > max_error_train:
                max_error_train = relative_error_train
        print('mse:', mse_train)
        print('max_error:', max_error_train)

        #calculate mean square error and maximum relative error for test data
        print('----------test data----------')
        for i in range(input_testdata.shape[0]):
            for j in range(input_testdata.shape[1]):
                input_testdata[i][j] = 2 * (input_testdata[i][j] - input_min_list[i]) / (input_max_list[i] - input_min_list[i]) - 1
        z1 = netuse.sim(input_testdata)
        #anti-normalization
        for i in range(0, z1.shape[0]):
            for j in range(0, z1.shape[1]):
                z1[i][j] = (z1[i][j] + 1) / 2 * (output_max_list[i] - output_min_list[i]) + output_min_list[i]
        mse_test=norm(z1-output_testdata)*norm(z1-output_testdata)/z1.shape[1]
        #calculate maximum relative error
        max_error = math.fabs((z1[0][0] - output_testdata[0][0]) / output_testdata[0][0])
        for j in range(0, output_testdata.shape[1]):
            relative_error = math.fabs((z1[0][j] - output_testdata[0][j]) / (output_testdata[0][j]))
            print(j,':',relative_error)
            if relative_error > max_error:
                max_error = relative_error
        print('mse:',mse_test)
        print('max_error:', max_error)

        #plot
        fig = plt.figure()

        ax2 = fig.add_subplot(232)
        ax2.semilogy(alpha)
        #ax2.loglog(alpha)
        ax2.set_title('alpha')

        ax3 = fig.add_subplot(233)
        ax3.semilogy(beta)
        #ax3.loglog(beta)
        ax3.set_title('beta')

        [w, N] = netuse.cal_w(netuse.net_struct)
        N_list = [N for i in range(0, len(gamma))]
        #write weights and biases
        file_object = open(r'data_for_plot\for_regularization\w.txt', 'w')
        for i in range(0, len(w)):
            file_object.write(str(w[i][0]) + '\n')
        file_object.close()

        ax4 = fig.add_subplot(234)
        ax4.semilogy(gamma)
        ax4.semilogy(N_list)
        #ax4.loglog(gama)
        #ax4.loglog(N_list)
        ax4.set_title('gamma')

        ax5 = fig.add_subplot(235)
        ax5.semilogy(W)
        #ax5.loglog(W)
        ax5.set_title('E_W')

        ax6 = fig.add_subplot(236)
        ax6.semilogy(D)
        #ax6.loglog(D)
        ax6.set_title('E_D')
        # plt.show()
        print('alpha/beta:', alpha[-1] / beta[-1])

        #write to file
        #wirte E_D
        file_object = open(r'data_for_plot\for_regularization\E_D.txt', 'w')
        for i in range(0, len(D)):
            file_object.write(str(D[i]) + '\n')
        file_object.close()
        #write E_W
        file_object = open(r'data_for_plot\for_regularization\E_W.txt', 'w')
        for i in range(0, len(W)):
            file_object.write(str(W[i]) + '\n')
        file_object.close()
        #write alpha
        file_object = open(r'data_for_plot\for_regularization\alpha.txt', 'w')
        for i in range(0, len(alpha)):
            file_object.write(str(alpha[i]) + '\n')
        file_object.close()
        #write beta
        file_object = open(r'data_for_plot\for_regularization\beta.txt', 'w')
        for i in range(0, len(beta)):
            file_object.write(str(beta[i]) + '\n')
        file_object.close()
        #write gamma
        file_object = open(r'data_for_plot\for_regularization\gama.txt', 'w')
        for i in range(0, len(gamma)):
            file_object.write(str(gamma[i]) + '\n')
        file_object.close()
        #write eigenvalues of Hessian matrix
        file_object = open(r'data_for_plot\for_regularization\eigenvalues.txt', 'w')
        for i in range(0, eigenvalues.shape[0]):
            file_object.write(str(float(eigenvalues[i])) + '\n')
        file_object.close()

    else:
        #calculate mean square error and maximum relative error for training data
        print('----------training data----------')
        for i in range(input_testdata_train.shape[0]):
            for j in range(input_testdata_train.shape[1]):
                input_testdata_train[i][j] = 2 * (input_testdata_train[i][j] - input_min_list[i]) / (
                input_max_list[i] - input_min_list[i]) - 1
        z1_train = netuse.sim(input_testdata_train)
        #anti-normalization
        for i in range(0, z1_train.shape[0]):
            for j in range(0, z1_train.shape[1]):
                z1_train[i][j] = (z1_train[i][j] + 1) / 2 * (output_max_list[i] - output_min_list[i]) + output_min_list[i]
        mse_train = norm(z1_train - output_testdata_train) * norm(z1_train - output_testdata_train) / z1_train.shape[1]
        #calculate maximum relative error
        max_error_train = math.fabs((z1_train[0][0] - output_testdata_train[0][0]) / output_testdata_train[0][0])
        for j in range(0, output_testdata_train.shape[1]):
            relative_error_train = math.fabs(
                (z1_train[0][j] - output_testdata_train[0][j]) / (output_testdata_train[0][j]))
            print(j, ':', relative_error_train)
            if relative_error_train > max_error_train:
                max_error_train = relative_error_train
        print('mse:', mse_train)
        print('max_error:', max_error_train)

        # calculate mean square error and maximum relative error for training data
        print('----------test data----------')
        for i in range(input_testdata.shape[0]):
            for j in range(input_testdata.shape[1]):
                input_testdata[i][j] = 2 * (input_testdata[i][j] - input_min_list[i]) / (
                input_max_list[i] - input_min_list[i]) - 1
        z1 = netuse.sim(input_testdata)
        #anti-normalization
        for i in range(0, z1.shape[0]):
            for j in range(0, z1.shape[1]):
                z1[i][j] = (z1[i][j] + 1) / 2 * (output_max_list[i] - output_min_list[i]) + output_min_list[i]
        mse_test = norm(z1 - output_testdata) * norm(z1 - output_testdata) / z1.shape[1]
        #calculate maximum relative error
        max_error = math.fabs((z1[0][0] - output_testdata[0][0]) / output_testdata[0][0])
        for j in range(0, output_testdata.shape[1]):
            relative_error = math.fabs((z1[0][j] - output_testdata[0][j]) / (output_testdata[0][j]))
            print(j, ':', relative_error)
            if relative_error > max_error:
                max_error = relative_error
        print('mse:', mse_test)
        print('max_error:', max_error)
        #plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.semilogy(error)
        ax1.set_title('training error')
        [w, N] = netuse.cal_w(netuse.net_struct)
        #write w
        file_object = open(r'data_for_plot\for_regularization\w.txt', 'w')
        for i in range(0, len(w)):
            file_object.write(str(w[i][0]) + '\n')
        file_object.close()

    #linear least square
    A = np.ones([output_testdata.shape[1], 2])
    for i in range(0, output_testdata.shape[1]):
        A[i][0] = output_testdata[0][i]
    b = z1.transpose()
    X = linear_least_square(A, b)
    print('X:', X)

    R2_X = np.linspace(output_min_list[0] - 1, output_max_list[0] + 1, 100)
    R2_Y = np.linspace(output_min_list[0] - 1, output_min_list[0] + 1, 100)
    for i in range(0, R2_X.shape[0]):
        R2_Y[i] = 1.0 * R2_X[i] + 0.0

    R2_train = cal_R2(output_testdata_train.transpose(), z1_train.transpose())
    print('R2_train:', R2_train)
    R2 = cal_R2(output_testdata.transpose(), z1.transpose())
    print('R2:', R2)
    fig_fit = plt.figure()
    ax7 = fig_fit.add_subplot(121)
    ax7.plot(R2_X, R2_Y)
    ax7.set_title(label='training data')
    ax7.set_xlabel(xlabel='actual value')
    ax7.set_ylabel(ylabel='predicted value')
    ax7.scatter(output_testdata_train[0], z1_train[0], color='', marker='o', edgecolors='k')
    ax7.set_aspect(1)

    ax8 = fig_fit.add_subplot(122)
    ax8.plot(R2_X, R2_Y)
    ax8.set_title(label='test data')
    ax8.set_xlabel(xlabel='actual value')
    ax8.set_ylabel(ylabel='predicted value')
    ax8.scatter(output_testdata[0], z1[0], color='', marker='o', edgecolors='k')
    ax8.set_aspect(1)

    fig_fit_=plt.figure()
    x1=np.array([np.linspace(-10, 10, 70)])
    x2=np.array([np.linspace(-10, 10, 400)])
    ax9=fig_fit_.add_subplot(111)
    type1 = ax9.scatter(x1[0],output_traindata[0], color='b', marker='+', edgecolors='k')
    type2 = ax9.plot(x2[0],z1[0], 'b')
    type3 = ax9.plot(x2[0],output_testdata[0], 'r')

    plt.show()