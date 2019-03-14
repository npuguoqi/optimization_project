# -*- coding: utf-8 -*-
"""
@author: Qi Guo
"""
'''
The problem is taken from:Barbosa HJC, Lemonge ACC (2003) A new adaptive penalty scheme for genetic algorithms. Inf Sci (Ny) 156:215–251. doi: 10.1016/S0020-0255(03)00177-4
The test function is TP1 and the formulation of the test problem is (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
The optimization problem is a two dimensional minimization problem subject to two nonlinear inequality constraints
you can use this file to generate training data and test data for neural networks
you need to comment out some codes for certain function
when you generate the data, you need to use pynet.py to train the network
'''
import numpy as np
import copy
import xlrd
import os
import xlwt
import random
from xlutils.copy import copy as excelcopy
import math
import csv
from optimal_lhd import generate_sysmetrix_latin_matrix_
from optimal_lhd import generate_sysmetrix_latin_matrix
from optimal_lhd import simulated_annealing_multiobj
from optimal_lhd import generate_data
from optimal_lhd import cal_J_d
from optimal_lhd import cal_correlation


def generate_test_data():

    n=40
    k=2
    #TP1
    bounds = np.array([[0, 6], [0, 6]])
    latin_test=generate_sysmetrix_latin_matrix(n,k)                                 #test data
    latin_result=generate_sysmetrix_latin_matrix_(n,k,q=30,p=1,ncount=int(n*k*k))

    latin_best, latin_best_phi_q_correlation_list, latin_phi_q_correlation_list, latin_phi_q_list, rho_list=simulated_annealing_multiobj(latin_result,T0=1,T_MIN=1e-18,I_MAX=100,FAC_T=0.95,q=30,p=1)

    data_test = generate_data(latin_test, bounds)     #get test data
    data_result=generate_data(latin_best,bounds)      #get training data
    data_test_transpose = data_test.transpose()       #transpose
    data_result_transpose=data_result.transpose()     #transpose

    fun_result=[]
    fun_test=[]
    for j in range(0,data_result_transpose.shape[1]):
        #The problem is taken from:Barbosa HJC, Lemonge ACC (2003) A new adaptive penalty scheme for genetic algorithms. Inf Sci (Ny) 156:215–251. doi: 10.1016/S0020-0255(03)00177-4
        #The readers can refer to the literature
        #TP1
        #object
        fun_result.append((data_result_transpose[0][j]**2+data_result_transpose[1][j]-11)**2+(data_result_transpose[0][j]+data_result_transpose[1][j]**2-7)**2)
        fun_test.append((data_test_transpose[0][j]**2+data_test_transpose[1][j]-11)**2+(data_test_transpose[0][j]+data_test_transpose[1][j]**2-7)**2)
        #constraint 1
        #fun_result.append((data_result_transpose[0][j]-0.05)**2+(data_result_transpose[1][j]-2.5)**2-4.84)
        #fun_test.append((data_test_transpose[0][j]-0.05)**2+(data_test_transpose[1][j]-2.5)**2-4.84)
        #constraint 2
        #fun_result.append(-(data_result_transpose[0][j])**2-(data_result_transpose[1][j]-2.5)**2+4.84)
        #fun_test.append(-(data_test_transpose[0][j])**2-(data_test_transpose[1][j]-2.5)**2+4.84)

    fun_result_matrix = np.array([fun_result])
    fun_test_matrix = np.array([fun_test])
    return data_result_transpose, fun_result_matrix, data_test_transpose, fun_test_matrix

def save_excel(filename, data):
    if os.path.exists(filename):
        os.remove(filename)
    file = xlwt.Workbook()
    file.add_sheet('Sheet1', cell_overwrite_ok=False)
    file.save(filename)
    rb = xlrd.open_workbook(filename)
    table = rb.sheet_by_name('Sheet1')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            table.put_cell(i, j, 2, data[i][j], 0)
    wb = excelcopy(rb)
    wb.save(filename)


def save_csv(filename, data):
    if os.path.exists(filename):
        os.remove(filename)
    csvFile = open(filename, 'w', newline='')
    writer = csv.writer(csvFile)
    for i in range(0, data.shape[0]):
        writer.writerow(data[i])
    csvFile.close()

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

if __name__ == '__main__':
    data_result_transpose, fun_result_matrix, data_test_transpose, fun_test_matrix = generate_test_data()

    save_csv(r'xls_file\data_result_transpose.csv', data_result_transpose)
    save_csv(r'xls_file\fun_result_matrix.csv', fun_result_matrix)

    save_csv(r'xls_file\data_test_transpose.csv', data_test_transpose)
    save_csv(r'xls_file\fun_test_matrix.csv', fun_test_matrix)

    save_csv(r'xls_file\data_test_transpose_train.csv', data_result_transpose)
    save_csv(r'xls_file\fun_test_matrix_train.csv', fun_result_matrix)