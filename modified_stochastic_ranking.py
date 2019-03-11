# -*- coding: utf-8 -*-
"""
@author: Qi Guo
"""
'''
Constrained genetic algorithm by modified stochastic ranking
The test function is taken from:Barbosa HJC, Lemonge ACC (2003) A new adaptive penalty scheme for genetic algorithms. Inf Sci (Ny) 156:215–251. doi: 10.1016/S0020-0255(03)00177-4
The test function is TP1 and the formulation of the test problem is (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
The optimization problem is a two dimensional minimization problem subject to two nonlinear inequality constraints
The search space is bounded by [0,6]*[0,6], and the optimum solution is (x1, x2; f)=(2.246826; 2.381865; 13.59085)
The constraints and object are substituted by surrogate models obtained by Bayesian regularized neural network
The optimum obtained by the code is (2.246340, 2.374102; 13.60592) (One of the optimum, the values are almost the same)
For the sake of the error caused by the prediction and the randomness of the optimization algorithm , we consider that the solution is accurate and can be relied upon.
'''
#the stochastic ranking is modified by multiple constraint ranking
#please refer to:Garcia R de P, de Lima BSLP, Lemonge AC de C, Jacob BP (2017) A rank-based constraint handling technique for engineering design optimization problems solved by genetic algorithms. Comput Struct 187:77–87. doi: 10.1016/j.compstruc.2017.03.023
import pynet
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import os

def objective_function(x):
    # TP1
    # You can test other benchmark function by modifying code
    return ((x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2)

def net_function(tmp_input_data,netuse,input_traindata_min_list, input_traindata_max_list, output_traindata_min_list, output_traindata_max_list):

    input_data=np.zeros([len(tmp_input_data),1])
    for i in range(0,len(tmp_input_data)):
        input_data[i][0]=tmp_input_data[i]
    #normalization
    for i in range(input_data.shape[0]):
        for j in range(input_data.shape[1]):
            input_data[i][j] = 2 * (input_data[i][j] - input_traindata_min_list[i]) / (input_traindata_max_list[i] - input_traindata_min_list[i]) - 1
    z1 = netuse.sim(input_data)
    # anti-normalization
    for i in range(0, z1.shape[0]):
        for j in range(0, z1.shape[1]):
            z1[i][j] = (z1[i][j] + 1) / 2 * (output_traindata_max_list[i] - output_traindata_min_list[i]) + output_traindata_min_list[i]
    return z1[0][0]
#write to csv file
def save_csv(filename,data):
    if os.path.exists(filename):
        os.remove(filename)
    csvFile = open(filename, 'w', newline='')  #set newline, otherwise there will a line between two lines
    writer = csv.writer(csvFile)
    for i in range(0,data.shape[0]):
        writer.writerow(data[i])
    csvFile.close()
#for details of how to rank each individual, please refer to:Garcia R de P, de Lima BSLP, Lemonge AC de C, Jacob BP (2017) A rank-based constraint handling technique for engineering design optimization problems solved by genetic algorithms. Comput Struct 187:77–87. doi: 10.1016/j.compstruc.2017.03.023
#the ranking has been tested based on the example from the literature
#redefine the ranking according to the literature
def sort_modify(arr):
    index=np.argsort(arr)
    index_modify=np.ones([1,len(arr)])
    ncount=1
    tmp=1
    for i in range(0,len(arr)):
        if i==0:
            index_modify[0,index[i]]=1
        else:
            if arr[index[i]]==arr[index[i-1]]:
                ncount=ncount+1
            else:
                ncount=ncount+1
                tmp=ncount
            index_modify[0,index[i]]=tmp
    return index_modify
#calculate the constraint ranking
def cal_multiple_constraints_ranking(tmp_matrix):
    nrows=tmp_matrix.shape[0]
    ncols=tmp_matrix.shape[1]
    sort_matrix=np.ones([nrows,ncols])
    for j in range(0,ncols):
        index_modify=sort_modify(tmp_matrix[:,j])
        for i in range(0,nrows):
            sort_matrix[i][j]=index_modify[0,i]
    number_of_infeasible=[]
    for i in range(0,nrows):
        ncount=0
        for j in range(0,ncols):
            if tmp_matrix[i][j]!=0:
                ncount=ncount+1
        number_of_infeasible.append(ncount)
    index_of_infeasible=sort_modify(number_of_infeasible)
    tmp_list=[]
    for i in range(0,nrows):
        tmp_list.append(index_of_infeasible[0,i]+np.sum(sort_matrix[i,:]))
    return tmp_list
class GA(object):
    t_Count = 0
    flag = False
    '''
    Constructor
    '''
    def __init__(self, m, t, pc, pm,varnum,chromlength,violationnum):  # 初始化
        self.pc = pc                               #crossover probability
        self.pm = pm                               #mutation  probability
        self.m = m                                 #population size
        self.t = t                                 #iteration
        self.varnum=varnum                         #number of design variables
        self.chromlength=chromlength               #chromlength
        self.violationnum=violationnum             #number of constraints
        self.pop=np.zeros([0,varnum*chromlength])  #population
        self.net_parameter=[]

    '''
    initialize
    '''
    def initialize(self):
        for i in range(0, self.m):                                      # for each individual in the population
            individual = np.zeros([1, self.varnum * self.chromlength])  # initialize individual
            for j in range(0, self.varnum * self.chromlength):
                individual[0, j] = random.randint(0, 1)                 # initialize chromosome
            self.pop = np.row_stack([self.pop, individual])             # generate population
    def read_net_parameter(self,netuse,input_traindata_min_list, input_traindata_max_list, output_traindata_min_list, output_traindata_max_list):
        tmp=[netuse,input_traindata_min_list, input_traindata_max_list, output_traindata_min_list, output_traindata_max_list]
        self.net_parameter.append(tmp)
    #xor
    def xor(self, a, b):
        if a == b:
            return 0
        else:
            return 1
    #flip
    @staticmethod
    def notvalue(value):
        if value == 1:
            return 0
        else:
            return 1
    '''
    calculate fitness for other benchmark functions
    '''
    def calfitness(self, res):
        fitness = objective_function(res)
        return fitness
    '''
    decode
    '''
    def decode(self, individual):  #we adopt gray code, first turn the gray code to binary code, and then turn the binary code to float
        res = []
        tmpindividual = []         #temp list
        for i in range(0, len(individual)):
            tmpindividual.append(individual[i])
        for i in range(0, self.varnum):                                                #obtain binary code
            tmpindividual[i * self.chromlength] = individual[i * self.chromlength]
            for j in range(0, self.chromlength - 1):
                tmpindividual[i * self.chromlength + 1 + j] = self.xor(tmpindividual[i * self.chromlength + 1 + j - 1],individual[i * self.chromlength + 1 + j])
        for i in range(0, self.varnum):                                                #turn to float
            value = 0
            for j in range(0, self.chromlength):
                value = value + tmpindividual[(i + 1) * self.chromlength - j - 1] * (2 ** (j))
            #TP1
            #set design variable ranges
            if i==0:
                value = 6 * value / (2 ** self.chromlength - 1) - 0
            else:
                value = 6 * value / (2 ** self.chromlength - 1) - 0

            res.append(value)
        return np.array(res)  #return a matrix of float number
    '''
    modified stochastic ranking
    please refer to:Runarsson TP, Yao X (2005) Search Biases in Constrained Evolutionary Optimization. IEEE Trans Syst Man Cybern Part C (Applications Rev 35:233–243. doi: 10.1109/TSMCC.2004.841906
    please refer to:Garcia R de P, de Lima BSLP, Lemonge AC de C, Jacob BP (2017) A rank-based constraint handling technique for engineering design optimization problems solved by genetic algorithms. Comput Struct 187:77–87. doi: 10.1016/j.compstruc.2017.03.023
    '''
    @staticmethod
    def bubblesort(arr, popviolation,index_modify_violation):
        tmparr = []
        tmppopviolation = []
        tmpindex_modify_violation=[]
        for i in range(0, len(arr)):
            tmparr.append(arr[i])
            tmppopviolation.append(popviolation[i])
            tmpindex_modify_violation.append(index_modify_violation[i])
        index = [i for i in range(0, len(arr))]       #to avoid exchanging the individual directly, the matrix is designed to reduce computation
        arrLen = len(arr)                             #length of array
        for i in range(0, arrLen):
            flag=0
            for j in range(0, arrLen - 1):
                randfloat = random.uniform(0, 1)
                if (tmppopviolation[j] == 0 and tmppopviolation[j + 1] == 0) or randfloat < 0.45:
                    if tmparr[j] < tmparr[j + 1]:
                        temp = tmparr[j]
                        tmparr[j] = tmparr[j + 1]
                        tmparr[j + 1] = temp
                        tempindex = index[j]
                        index[j] = index[j + 1]
                        index[j + 1] = tempindex
                        temp_tmppopviolation = tmppopviolation[j]
                        tmppopviolation[j] = tmppopviolation[j + 1]
                        tmppopviolation[j + 1] = temp_tmppopviolation
                        temp=tmpindex_modify_violation[j]
                        tmpindex_modify_violation[j]=tmpindex_modify_violation[j+1]
                        tmpindex_modify_violation[j + 1]=temp
                        flag=1
                else:
                    if tmpindex_modify_violation[j] < tmpindex_modify_violation[j + 1]:
                        temp = tmparr[j]
                        tmparr[j] = tmparr[j + 1]
                        tmparr[j + 1] = temp
                        tempindex = index[j]
                        index[j] = index[j + 1]
                        index[j + 1] = tempindex
                        temp_tmppopviolation = tmppopviolation[j]
                        tmppopviolation[j] = tmppopviolation[j + 1]
                        tmppopviolation[j + 1] = temp_tmppopviolation
                        temp=tmpindex_modify_violation[j]
                        tmpindex_modify_violation[j]=tmpindex_modify_violation[j+1]
                        tmpindex_modify_violation[j + 1]=temp
                        flag=1
            if flag==0:
                break

        return index
    '''
    obtain the maximum value
    '''
    def maxab(self, a, b):
        if a >= b:
            return a
        else:
            return b
    '''
    define constraints
    '''
    def violation(self, individual):
        violationlist = []
        # TP1
        violation_value_1=net_function(individual,self.net_parameter[1][0],self.net_parameter[1][1],self.net_parameter[1][2],self.net_parameter[1][3],self.net_parameter[1][4])
        violation_value_2 = net_function(individual, self.net_parameter[2][0], self.net_parameter[2][1],self.net_parameter[2][2], self.net_parameter[2][3], self.net_parameter[2][4])
        gp_1=self.maxab(violation_value_1,0)
        violationlist.append(gp_1)
        gp_2=self.maxab(violation_value_2,0)
        violationlist.append(gp_2)
        return violationlist
    def show_violation(self,individual):
        violation_value_1=net_function(individual,self.net_parameter[1][0],self.net_parameter[1][1],self.net_parameter[1][2],self.net_parameter[1][3],self.net_parameter[1][4])
        violation_value_2 = net_function(individual, self.net_parameter[2][0], self.net_parameter[2][1],self.net_parameter[2][2], self.net_parameter[2][3], self.net_parameter[2][4])
        print('violation:',violation_value_1,' ',violation_value_2,' ',)
    def show_object_violation(self,individual):
        object=net_function(individual,self.net_parameter[0][0],self.net_parameter[0][1],self.net_parameter[0][2],self.net_parameter[0][3],self.net_parameter[0][4])
        violation_value_1=net_function(individual,self.net_parameter[1][0],self.net_parameter[1][1],self.net_parameter[1][2],self.net_parameter[1][3],self.net_parameter[1][4])
        violation_value_2 = net_function(individual, self.net_parameter[2][0], self.net_parameter[2][1],self.net_parameter[2][2], self.net_parameter[2][3], self.net_parameter[2][4])
        print('object:',object)
        print('violation:',violation_value_1,' ',violation_value_2,' ',)
    '''
    show results,including the optimal value, mean value and number of infeasible individuals, etc.
    '''
    def showres(self, pop):
        popfitness = []
        popviolation = []
        countviolation = 0
        totalfitness = 0
        total_val=np.zeros([1,self.varnum])
        violation_matrix = np.ones([self.m, self.violationnum])
        for i in range(0, self.m):
            val = self.decode(pop[i, :])
            total_val=total_val+val
            fitness=net_function(val,self.net_parameter[0][0],self.net_parameter[0][1],self.net_parameter[0][2],self.net_parameter[0][3],self.net_parameter[0][4])
            #fitness = self.calfitness(val)
            totalfitness = totalfitness + fitness
            violationlist = self.violation(val)
            popfitness.append(fitness)
            popviolation.append(sum(violationlist))
            for j in range(0,self.violationnum):
                violation_matrix[i][j]=violationlist[j]
            if sum(violationlist) > 0.1:                #
                countviolation = countviolation + 1     #

        index_modify_violation = cal_multiple_constraints_ranking(violation_matrix)
        index = GA.bubblesort(popfitness, popviolation,index_modify_violation)
        meanfitness = totalfitness / self.m
        bestval = self.decode(pop[index[self.m - 1], :])
        #bestfitness = self.calfitness(bestval)
        bestfitness=net_function(bestval,self.net_parameter[0][0],self.net_parameter[0][1],self.net_parameter[0][2],self.net_parameter[0][3],self.net_parameter[0][4])
        bestvalviolationlist = self.violation(bestval)
        mean_val=total_val/self.m
        print(meanfitness, bestfitness, countviolation)
        print(bestval)
        print(bestvalviolationlist, '   ', sum(bestvalviolationlist))
        print(mean_val)
        self.show_violation(bestval)
        return [bestfitness, meanfitness, bestval, countviolation,mean_val]
    '''
    selection
    rank based selection
    elitism strategy(the best element is always copied into the next generation)
    '''
    def select_new(self, pop):

        tmppop = np.zeros([0, self.varnum * self.chromlength])  #temp population
        popfitness = []                                         #list of population fitness
        popviolation = []
        violation_matrix = np.ones([self.m, self.violationnum])
        for i in range(0, self.m):
            val = self.decode(pop[i, :])
            #fitness = self.calfitness(val)                     #test for other benchmark functions
            fitness = net_function(val, self.net_parameter[0][0], self.net_parameter[0][1], self.net_parameter[0][2],self.net_parameter[0][3], self.net_parameter[0][4])
            violationlist = self.violation(val)
            for j in range(0,self.violationnum):
                violation_matrix[i][j]=violationlist[j]
            popfitness.append(fitness)                          #add individual fitness
            popviolation.append(sum(violationlist))             #add individual violation


        index_modify_violation=cal_multiple_constraints_ranking(violation_matrix)  #multiple constraint ranking
        index = GA.bubblesort(popfitness, popviolation,index_modify_violation)     #modified stochastic ranking

        #rank based selection
        #please refer to:Shukla A, Pandey HM, Mehrotra D (2015) Comparative review of selection techniques in genetic algorithm. In: 2015 International Conference on Futuristic Trends on Computational Analysis and Knowledge Management (ABLAZE). IEEE, pp 515–519
        nmax = 2.0
        nmin = 0.0

        newprobality = [nmin / self.m]

        for i in range(1, self.m):
            newprobality.append(newprobality[i - 1] + 1 / self.m * (nmin + (nmax - nmin) * (i) / (self.m - 1)))
        #the best individual is copied to the next generation
        #please refer to:Barbosa HJC, Lemonge ACC (2003) A new adaptive penalty scheme for genetic algorithms. Inf Sci (Ny) 156:215–251. doi: 10.1016/S0020-0255(03)00177-4
        copynum = 1 + 0

        tmppop = np.row_stack([tmppop, pop[index[self.m - 1], :]])
        copypop = pop[index[self.m - 1]]
        for i in range(0, copynum - 1):
            randint = random.randint(0, self.varnum * self.chromlength - 1)
            copypop[randint] = GA.notvalue(copypop[randint])
            tmppop = np.row_stack([tmppop, copypop])

        for i in range(copynum, self.m):
            randfloat = random.uniform(0, 1)
            count = 0
            while (newprobality[count] < randfloat and count < self.m - 1):
                count = count + 1
            tmppop = np.row_stack([tmppop, pop[index[count], :]])

        return tmppop

    '''
    crossover
    please refer to:Barbosa HJC, Lemonge ACC (2003) A new adaptive penalty scheme for genetic algorithms. Inf Sci (Ny) 156:215–251. doi: 10.1016/S0020-0255(03)00177-4
    The standard one-point, two-point and uniform crossover operators are applied
    The probability of one-point crossover:0.2
    The probability of two-point crossover:0.4
    The probability of uniform crossover  :0.4
    '''
    def crossover_new(self, pop):
        seq = list(range(0, self.m))
        random.shuffle(seq)

        for i in range(0, self.m, 2):
            if i + 1 > self.m - 1:
                break
            randfloat = random.uniform(0, 1)
            if (randfloat < self.pc):
                randselect = random.uniform(0, 1)
                #one-point crossover
                if randselect < 0.2:
                    for k in range(0, self.varnum):
                        l = k * self.chromlength
                        u = (k + 1) * self.chromlength - 1
                        randint = random.randint(l, u)
                        for j in range(randint, u):
                            tmp = pop[seq[i], j]
                            pop[seq[i], j] = pop[seq[i + 1], j]
                            pop[seq[i + 1], j] = tmp
                #two-point crossover
                if randselect >= 0.2 and randselect <= 0.6:
                    for k in range(0, self.varnum):
                        l = k * self.chromlength
                        u = (k + 1) * self.chromlength - 1
                        randint1 = random.randint(l, u - 1)
                        randint2 = random.randint(randint1, u)
                        for j in range(randint1, randint2):
                            tmp = pop[seq[i], j]
                            pop[seq[i], j] = pop[seq[i + 1], j]
                            pop[seq[i + 1], j] = tmp
                #uniform crossover
                if randselect > 0.6:
                    for k in range(0, self.varnum):
                        l = k * self.chromlength
                        u = (k + 1) * self.chromlength - 1
                        randintselect = random.randint(0, 1)
                        for j in range(l, u):
                            if randintselect == 1:
                                tmp = pop[seq[i], j]
                                pop[seq[i], j] = pop[seq[i + 1], j]
                                pop[seq[i + 1], j] = tmp
        return pop
    '''
    mutation
    '''
    def mutation(self, pop):
        for i in range(0, self.m):
            randfloat = random.uniform(0, 1)
            if (randfloat < self.pm):
                for k in range(0, self.varnum):
                    l = k * self.chromlength
                    u = (k + 1) * self.chromlength - 1
                    randint = random.randint(l, u)
                    if pop[i, randint] == 0:
                        pop[i, randint] = 1
                    else:
                        pop[i, randint] = 0
        return pop

    '''
    run
    '''

    def run(self):
        self.initialize()
        ncount = 0
        allbestfitness = []
        allmeanfitness = []
        allbestval = []
        allcountviolation = []
        all_meanval=np.zeros([self.t,self.varnum])
        all_bestval=np.zeros([self.t,self.varnum])
        while (ncount < self.t):
            ncount = ncount + 1
            self.t_Count = ncount
            [bestfitness, meanfitness, bestval, countviolation,mean_val] = self.showres(self.pop)
            for j in range(0,self.varnum):
                all_meanval[ncount-1][j]=mean_val[0][j]
                all_bestval[ncount-1][j]=bestval[j]
            allbestfitness.append(bestfitness)
            allmeanfitness.append(meanfitness)
            allbestval.append(bestval)
            allcountviolation.append(countviolation)
            self.pop = self.select_new(self.pop)
            self.pop = self.crossover_new(self.pop)
            self.pop = self.mutation(self.pop)
            print(ncount, '\n')
        save_csv(r'data_for_plot\for_optimization\mean_val.csv',all_meanval)
        save_csv(r'data_for_plot\for_optimization\best_val.csv', all_bestval)
        return [allbestfitness, allmeanfitness, allbestval, allcountviolation,all_meanval,all_bestval]

if (__name__ == '__main__'):
    '''
    Gray code, rank based selection and elitism (the best individual is always copied into the next generation)
    '''
    #read normalization data
    input_traindata_min_list, input_traindata_max_list, output_traindata_min_list, output_traindata_max_list = pynet.read_normalization_parameter('net_for_object_normalization.csv')
    input_traindata_violation_min_list_1, input_traindata_violation_max_list_1, output_traindata_violation_min_list_1, output_traindata_violation_max_list_1 = pynet.read_normalization_parameter('net_for_violation_1_normalization.csv')
    input_traindata_violation_min_list_2, input_traindata_violation_max_list_2, output_traindata_violation_min_list_2, output_traindata_violation_max_list_2 = pynet.read_normalization_parameter('net_for_violation_2_normalization.csv')
    #read network parameters
    net_for_object = pynet.NetStruct([], [], [], [], '', 'read_from_file', 'net_for_object.txt')
    netuse_for_object = pynet.NetUtilize(net_for_object, mu=1e-3, mu_dec=0.1, mu_inc=10, mu_max=1e10, iteration=2500, tol=1e-30,gradient=1e-10, gamma_criterion=1e-6, training_method='Bayesian-Regularization')

    net_for_violation_1 = pynet.NetStruct([], [], [], [], '', 'read_from_file', 'net_for_violation_1.txt')
    netuse_for_violation_1 = pynet.NetUtilize(net_for_violation_1, mu=1e-3, mu_dec=0.1, mu_inc=10, mu_max=1e10, iteration=2500, tol=1e-30,gradient=1e-10, gamma_criterion=1e-6, training_method='Bayesian-Regularization')

    net_for_violation_2 = pynet.NetStruct([], [], [], [], '', 'read_from_file', 'net_for_violation_2.txt')
    netuse_for_violation_2 = pynet.NetUtilize(net_for_violation_2, mu=1e-3, mu_dec=0.1, mu_inc=10, mu_max=1e10, iteration=2500, tol=1e-30,gradient=1e-10, gamma_criterion=1e-6, training_method='Bayesian-Regularization')

    #initialize GA solver
    GAobj = GA(m=300, t=151, pc=0.8, pm=0.08,varnum=2,chromlength=20,violationnum=2)
    GAobj.read_net_parameter(netuse_for_object,input_traindata_min_list, input_traindata_max_list, output_traindata_min_list, output_traindata_max_list)
    GAobj.read_net_parameter(netuse_for_violation_1,input_traindata_violation_min_list_1, input_traindata_violation_max_list_1, output_traindata_violation_min_list_1, output_traindata_violation_max_list_1)
    GAobj.read_net_parameter(netuse_for_violation_2, input_traindata_violation_min_list_2,input_traindata_violation_max_list_2, output_traindata_violation_min_list_2,output_traindata_violation_max_list_2)

    #run
    [allbestfitness, allmeanfitness, allbestval,  allcountviolation,all_meanval,all_bestval] = GAobj.run()

    #write to file
    file_object = open(r'data_for_plot\for_optimization\optimum.txt', 'w')
    for i in range(0, len(allbestfitness)):
        file_object.write(str(allbestfitness[i]) + '\n')
    file_object.close()

    file_object = open(r'data_for_plot\for_optimization\mean.txt', 'w')
    for i in range(0, len(allmeanfitness)):
        file_object.write(str(allmeanfitness[i]) + '\n')
    file_object.close()

    file_object = open(r'data_for_plot\for_optimization\number_of_infeasible.txt', 'w')
    for i in range(0, len(allcountviolation)):
        file_object.write(str(allcountviolation[i]) + '\n')
    file_object.close()

    #plot
    fig_for_convergence=plt.figure()
    ax_for_convergece=fig_for_convergence.add_subplot(111)
    xaxis = list(range(0, GAobj.t))
    ax_for_convergece.plot(xaxis[:], allbestfitness[:])
    ax_for_convergece.plot(xaxis[:], allmeanfitness[:])

    fig_for_violation=plt.figure()
    ax_for_violation=fig_for_violation.add_subplot(111)
    xaxis = list(range(0, GAobj.t))
    ax_for_violation.plot(xaxis[:],allcountviolation[:])

    fig_for_val=plt.figure()
    for i in range(0,GAobj.varnum):
        ax=fig_for_val.add_subplot(1,2,i+1)
        xaxis = list(range(0, GAobj.t))
        ax.plot(xaxis[:],all_meanval[:,i])
        ax.plot(xaxis[:],all_bestval[:,i])
    plt.show()