# optimization_project
## 1. file description
There are six source code files and three folders in the repository.  
The folder **net_parameter** is used to store neural network parameters.  
The folder **xls_file**      is used to store training data and test data for neural network.  
The folder **data_for_plot** is used to store plotting data.  
Theere are three subfolders **for_LHD**, **for_regularization** and **for_optimization** in the folder **data_for_plot**.  
The file **optimal_lhd.py**            is used to generate an optimal Latin hypercube design.  
The file **pynet_for_test.py**         is used to illustrate the predicted functions under different regularization parameters.  
The file **generate_sampling_data.py** is used to generate training data and test data for neural network.  
The file **pynet.py**                  is used to train and save neural network.  
The file **stochastic_ranking.py**          is used to optimize objective function subjected to constraints by stochastic ranking.  
The file **modified_stochastic_ranking.py** is used to optimize objective function subjected to constraints by modified stochastic ranking.  
## 2. instructions for the use of the source code
### 2.1 optimal_lhd.py
When running **optimal_lhd.py**, you can obtain an optimal Latin hypercube design (LHD) with 13 levels and 2 factors. The number of levels and factors can be setted in the code. The code will illustrate three LHDs including: a random generated LHD, a starting LHD and an optimal LHD. The convergence curves of the simulated annealing algorithm are also presented.
### 2.2 pynet_for_test.py
When running **pynet_for_test.py**, you can obtain the predicted function under circumstances of alpha/beta=0.0005. The regularization parameters can be setted in the code to obtain different predicted functions. You may also select the method of Bayesian regularization to train the neural network. The optimal regularization parameters calculated by Bayesian regularization is 0.00054.
### 2.3 stochastic_ranking.py
When running **stochastic_ranking.py**, you can obtain an optimal value for a benchmark function taken from literature. The literature and benchmark function are presented in code comments. The object and constraints are substituted by surrogate models. The code reloads the surrogate models by the following method
```
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
```
### 2.4 modified_stochastic_ranking.py
When running **modified_stochastic_ranking.py**, you can obtain an optimal value for a benchmark function taken from literature. The object and constraints are substituted by surrogate models. The literature and benchmark function are presented in code comments. The code **stochastic_ranking.py** and **modified_stochastic_ranking** yield the same results for the benchmark function.
### 2.5 generate_sampling_data.py
As mentioned before, when running the optimization algorithm, the object and constraints are substitutude by surrogate models. Of course, you can obtain your surrogate models by generating training data and test data for neural network and then training the network. When running **generate_sampling_data.py**, you have to comment out some codes to generate the training data for object or constraints. For example, if you generate the data for the object, you need to commnet out the code as follow
```
#constraint 1
#fun_result.append((data_result_transpose[0][j]-0.05)**2+(data_result_transpose[1][j]-2.5)**2-4.84)
#fun_test.append((data_test_transpose[0][j]-0.05)**2+(data_test_transpose[1][j]-2.5)**2-4.84)
#constraint 2
#fun_result.append(-(data_result_transpose[0][j])**2-(data_result_transpose[1][j]-2.5)**2+4.84)
#fun_test.append(-(data_test_transpose[0][j])**2-(data_test_transpose[1][j]-2.5)**2+4.84)
```
When the data is generated, you can run **pynet.py** to train the neural network and obtain the corresponding surrogate model. You may also use the data generated by us in the folder **xls_file** to train the network without running the code.
### 2.6 pynet.py
Before running the code, you have to name the network in order to reload the network in **stochastic_ranking.py** or **modified_stochastic_ranking.py**. For example, if you train the network for object, the code should be modified as
```
netuse.save_net('net_for_object.txt')
save_normalization_parameter('net_for_object_normalization.csv', input_min_list, input_max_list, output_min_list,output_max_list)
```
if you train the network for constraint 1, the code should be modified as
```
netuse.save_net('net_for_violation_1.txt')
save_normalization_parameter('net_for_violation_1_normalization.csv', input_min_list, input_max_list, output_min_list,output_max_list)
```
if you train the network for constraint 2, the code should be modified as
```
netuse.save_net('net_for_violation_2.txt')
save_normalization_parameter('net_for_violation_2_normalization.csv', input_min_list, input_max_list, output_min_list,output_max_list)
```
