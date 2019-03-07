import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter  #作为画图的标签使用
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse, Circle
from scipy.linalg import norm
import copy
import random
import math
#生成一个随机拉丁超立方
def latin_hypercube_sampling(n,k,type=1):     #N表示行数，k表示列数
    result = np.empty([n, k])
    temp = np.empty([n])
    d = 1.0                                   #按照列打乱
    if type==1:
        for i in range(k):
            for j in range(n):
                temp[j]=(j*d+(j+1)*d)/2
            np.random.shuffle(temp)
            for j in range(n):
                result[j][i]=temp[j]
    return result
#生成一个对称的拉丁超立方
def generate_sysmetrix_latin_matrix(n,k):
    temp_matrix=np.empty([n,k])
    for j in range(k):
        temp = [0 for i in range(0,n)]
        d = 1.0
        for k in range(n):
            temp[k] = (k * d + (k + 1) * d) / 2
        #对称处理
        #例如n=6,则temp=[0.5,1.5,2.5,3.5,4.5,5.5],满足对称条件
        #如果为偶数
        #floor为地板函数，向下取整
        if n%2==0:
            for i in range(0,math.floor(n/2)):
                select_value=random.sample(temp,1)
                temp.remove(select_value[0])
                temp_matrix[i][j]=select_value[0]
                temp.remove(n-select_value[0])
                temp_matrix[n-1-i][j]=n-temp_matrix[i][j]
        #如果为奇数
        #首先得到中间的那个数，然后其他处理和偶数相同
        else:
            select_value=temp[int(math.floor(n/2))]
            temp_matrix[int(math.floor(n/2))][j]=select_value
            temp.remove(select_value)
            for i in range(0,math.floor(n/2)):
                select_value=random.sample(temp,1)
                temp.remove(select_value[0])
                temp_matrix[i][j]=select_value[0]
                temp.remove(n-select_value[0])
                temp_matrix[n-1-i][j]=n-temp_matrix[i][j]
    return temp_matrix
#得到一个好的初始点
def generate_sysmetrix_latin_matrix_(n,k,q,p,ncount):
    latin_matrix_tmp=generate_sysmetrix_latin_matrix(n,k)
    latin_matrix_best=copy.deepcopy(latin_matrix_tmp)
    w=0.5
    phi_q_l=cal_phi_q_l(n,k,q)
    phi_q_u=cal_phi_q_u(n,k,q)
    latin_matrix_best_phi_q_correlation=(1-w)*(cal_phiq(latin_matrix_best,q,p)-phi_q_l)/(phi_q_u-phi_q_l)+w*cal_correlation(latin_matrix_best)
    for i in range(0,ncount):
        print(i)
        latin_matrix_tmp=generate_sysmetrix_latin_matrix(n,k)
        latin_matrix_tmp_phi_q_correlation=(1-w)*(cal_phiq(latin_matrix_tmp,q,p)-phi_q_l)/(phi_q_u-phi_q_l)+w*cal_correlation(latin_matrix_tmp)
        if latin_matrix_tmp_phi_q_correlation<latin_matrix_best_phi_q_correlation:
            latin_matrix_best=copy.deepcopy(latin_matrix_tmp)
            latin_matrix_best_phi_q_correlation=latin_matrix_tmp_phi_q_correlation
    return latin_matrix_best
#计算两点之间的距离,p=1为rectangle距离,p=2为哦欧几里得距离
def cal_d(sampling_matrix,p):
    n=sampling_matrix.shape[0]
    d=np.empty([int(n*(n-1)/2)])     #计算两点之间的距离
    for i in range(0,n-1):
        for j in range(i+1,n):
            temp=sampling_matrix[i,:]-sampling_matrix[j,:]
            d[int((i)*n-(i+1)*i/2+j-i-1)]=norm(temp,p)
    return d
#得到拉丁超立方矩阵后，计算J和d
def cal_J_d(sampling_matrix,p):      #默认欧几里得距离，p=1为Rectangle距离
    d=cal_d(sampling_matrix,p)
    distinct_d,J=np.unique(d,return_counts=True)
    return distinct_d,J,d
#计算相关系数
def cal_correlation(sampling_matrix):
    k=sampling_matrix.shape[1]
    pho=np.empty([int(k*(k-1)/2)])
    for i in range(0,k-1):
        for j in range(i+1,k):
            col_i=sampling_matrix[:,i]
            col_j=sampling_matrix[:,j]
            col_i_mean=np.array([np.mean(col_i)])
            col_j_mean=np.array([np.mean(col_j)])
            pho_i_j=np.dot(col_i-col_i_mean,col_j-col_j_mean)/(norm(col_i-col_i_mean,2)*norm(col_j-col_j_mean,2))
            pho[int((i)*k-(i+1)*i/2+j-i-1)]=pho_i_j*pho_i_j/(k*(k-1)/2)
    pho_value=sum(pho)
    return pho_value
#计算距离的平均值
def cal_d_mean(n,k):
    return (n+1)*k/3
#计算phi_q的下限
def cal_phi_q_l(n,k,q):
    d_mean=cal_d_mean(n,k)
    if math.ceil(d_mean)==d_mean:
        _d=d_mean
        d_=d_mean+1
    else:
        _d=math.floor(d_mean)
        d_=math.ceil(d_mean)
    phi_q_l=(n*(n-1)/2*((d_-d_mean)/(_d**q)+(d_mean-_d)/(d_**q)))**(1/q)
    return phi_q_l
#计算phi_q的上限
def cal_phi_q_u(n,k,q):
    phi_q_u=0.0
    for i in range(1,n):
        phi_q_u = phi_q_u + (n - i)  / ((i * k) ** q)
    phi_q_u=phi_q_u**(1/q)
    return phi_q_u

def cal_phiq(sampling_matrix,q,p):     #默认欧几里得距离,p=1为Rectangle距离
    '''
    phi_q=0.0
    distinct_d,J,d=cal_J_d(sampling_matrix,p)
    for i in range(0,distinct_d.shape[0]):
        phi_q=phi_q+J[i]*(distinct_d[i]**(-q))
    phi_q=phi_q**(1/q)
    '''
    d=cal_d(sampling_matrix,p)
    phi_q_temp=0.0
    for i in range(0,d.shape[0]):
        phi_q_temp=phi_q_temp+d[i]**(-q)
    phi_q_temp=phi_q_temp**(1/q)
    return phi_q_temp

def simulated_annealing_multiobj(sampling_matrix,T0,T_MIN,I_MAX,FAC_T,q,p):  #默认为欧几里得距离
    w=0.5
    n=sampling_matrix.shape[0]
    k=sampling_matrix.shape[1]
    phi_q_l=cal_phi_q_l(n,k,q)
    phi_q_u=cal_phi_q_u(n,k,q)
    I_MAX=int(math.sqrt(n*k)/k)
    latin=copy.deepcopy(sampling_matrix)
    FLAG=1
    I=1
    T=T0
    latin_best=copy.deepcopy(sampling_matrix)
    latin_best_phi_q_correlation_list=[]
    latin_best_phi_q_correlation_list.append((1-w)*(cal_phiq(latin_best,q,p)-phi_q_l)/(phi_q_u-phi_q_l)+w*cal_correlation(latin_best))

    latin_phi_q_list=[]
    latin_phi_q_list.append(cal_phiq(latin,q,p))
    rho_list=[]
    rho_list.append(cal_correlation(latin))
    latin_phi_q_correlation_list=[]
    latin_phi_q_correlation_list.append((1-w)*(cal_phiq(latin,q,p)-phi_q_l)/(phi_q_u-phi_q_l)+w*cal_correlation(latin))

    while (T>T_MIN):
        FLAG=0
        I=1
        while (I<I_MAX):
            latin_try=copy.deepcopy(latin)
            j=random.randint(0,k-1)
            if n%2==0:
                index=random.sample(list(range(0,n)),2)
            else:
                list_temp=list(range(0,n))
                list_temp.remove(math.floor(n/2))
                index=random.sample(list_temp,2)
            temp=latin_try[index[0]][j]
            latin_try[index[0]][j]=latin_try[index[1]][j]
            latin_try[index[1]][j]=temp
            temp=latin_try[n-1-index[0]][j]
            latin_try[n-1-index[0]][j]=latin_try[n-1-index[1]][j]
            latin_try[n - 1 - index[1]][j]=temp

            latin_phi_q=cal_phiq(latin,q,p)
            latin_correlation=cal_correlation(latin)
            latin_try_phi_q=cal_phiq(latin_try,q,p)
            latin_try_correlation=cal_correlation(latin_try)

            latin_phi_q_correlation=(1-w)*(latin_phi_q-phi_q_l)/(phi_q_u-phi_q_l)+w*latin_correlation
            latin_try_phi_q_correlation=(1-w)*(latin_try_phi_q-phi_q_l)/(phi_q_u-phi_q_l)+w*latin_try_correlation

            if (latin_try_phi_q_correlation<latin_phi_q_correlation or random.uniform(0,1)<math.exp(-(latin_try_phi_q_correlation-latin_phi_q_correlation)/T)):
                latin=copy.deepcopy(latin_try)
                latin_phi_q_correlation_list.append(latin_try_phi_q_correlation)
                latin_phi_q_list.append(latin_try_phi_q)
                rho_list.append(latin_try_correlation)
                FLAG=1
                latin_best_phi_q_correlation=(1-w)*(cal_phiq(latin_best,q,p)-phi_q_l)/(phi_q_u-phi_q_l)+w*cal_correlation(latin_best)
                if (latin_try_phi_q_correlation<latin_best_phi_q_correlation):
                    latin_best=copy.deepcopy(latin_try)
                    I=1
                    latin_best_phi_q_correlation_list.append(latin_try_phi_q_correlation)
                else:
                    I=I+1
            else:
                I=I+1
        T=T*FAC_T
        print(T)
    return latin_best,latin_best_phi_q_correlation_list,latin_phi_q_correlation_list,latin_phi_q_list,rho_list
#得到数据，由拉丁超立方矩阵还有边界
def generate_data(sampling_matrix,bounds):
    n=sampling_matrix.shape[0]
    k=sampling_matrix.shape[1]
    temp_sampling_matrix=sampling_matrix/n              #得到区间（0,1）的数
    lower_bounds=bounds[:,0]                            #得到下界
    upper_bounds=bounds[:,1]                            #得到上界
    data_result=np.multiply(temp_sampling_matrix,upper_bounds-lower_bounds)
    data_result=np.add(data_result,lower_bounds)        #得到结果
    return data_result

if __name__=='__main__':
    n=13
    k=2
    if k==2:
        bounds = np.array([[-1, 1], [-1, 1]])
    elif k==3:
        bounds = np.array([[0, 1], [0, 1], [0, 1]])
    elif k==4:
        bounds=np.array([[0, 1], [0, 1], [0, 1],[0,1]])
    elif k==5:
        bounds=np.array([[1.0, 2.0], [10.0, 20.0], [0.05, 0.15],[5.0,10.0],[45.0,55.0]])
    else:
        bounds = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1],[0,1]])
    latin_example=latin_hypercube_sampling(n,k,1)
    latin_result = generate_sysmetrix_latin_matrix_(n, k, q=30, p=1, ncount=int(n*k*k))
    latin_best, latin_best_phi_q_correlation_list, latin_phi_q_correlation_list, latin_phi_q_list, rho_list = simulated_annealing_multiobj(latin_result, T0=1, T_MIN=1e-18, I_MAX=100, FAC_T=0.95, q=30, p=1)

    print('phi_q')
    print(cal_phi_q_l(n,k,30))
    print(cal_phi_q_u(n,k,30))
    print(cal_phiq(latin_best,30,1))
    print('correlation')
    print(cal_correlation(latin_example))
    print(cal_correlation(latin_result))
    print(cal_correlation(latin_best))

    print('distance:')
    res0=cal_J_d(latin_example,1)
    res1=cal_J_d(latin_result,1)
    res2=cal_J_d(latin_best,1)
    print(res0[0][0],' ',res1[0][0],' ',res2[0][0])

    data_result=generate_data(latin_best,bounds)


    #画图
    fig_for_example = plt.figure()
    x_ticker_labels = range(-1, n + 1)        #x轴标签
    y_ticker_labels = range(-1, n + 1)        #y轴标签
    for i in range(0, k - 1):                 #i按照行进行循环
        for j in range(0, i + 1):             #j按照列进行循环
            ax = fig_for_example.add_subplot(k - 1, k - 1, i * (k - 1) + j + 1)       #python画图从上往下，从左往右
            ax.xaxis.set_major_locator(MultipleLocator(1))   #设置主标签位置
            ax.yaxis.set_major_locator(MultipleLocator(1))   #设置主标签位置
            ax.set_xticklabels(x_ticker_labels, rotation=0)  #设置x轴标签
            ax.set_yticklabels(y_ticker_labels, rotation=90) #设置y轴标签
            ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='k')
            ax.set_xlim(0, n)                                #设置x轴范围
            ax.set_ylim(0, n)                                #设置y轴范围
            ax.scatter(latin_example[:, j], latin_example[:, i + 1], color='', marker='o', edgecolors='k', s=10)
            ax.set_aspect(1)

    fig_for_not_optimization = plt.figure()
    x_ticker_labels = range(-1, n + 1)        #x轴标签
    y_ticker_labels = range(-1, n + 1)        #y轴标签
    for i in range(0, k - 1):                 #i按照行进行循环
        for j in range(0, i + 1):             #j按照列进行循环
            ax = fig_for_not_optimization.add_subplot(k - 1, k - 1, i * (k - 1) + j + 1)       #python画图从上往下，从左往右
            ax.xaxis.set_major_locator(MultipleLocator(1))   #设置主标签位置
            ax.yaxis.set_major_locator(MultipleLocator(1))   #设置主标签位置
            ax.set_xticklabels(x_ticker_labels, rotation=0)  #设置x轴标签
            ax.set_yticklabels(y_ticker_labels, rotation=90) #设置y轴标签
            ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='k')
            ax.set_xlim(0, n)                                #设置x轴范围
            ax.set_ylim(0, n)                                #设置y轴范围
            ax.scatter(latin_result[:, j], latin_result[:, i + 1], color='', marker='o', edgecolors='k', s=10)
            ax.set_aspect(1)

    fig_for_optimization = plt.figure()
    x_ticker_labels = range(-1, n + 1)
    y_ticker_labels = range(-1, n + 1)
    for i in range(0, k - 1):
        for j in range(0, i + 1):
            ax = fig_for_optimization.add_subplot(k - 1, k - 1, i * (k - 1) + j + 1)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.yaxis.set_major_locator(MultipleLocator(1))
            ax.set_xticklabels(x_ticker_labels, rotation=0)
            ax.set_yticklabels(y_ticker_labels, rotation=90)
            ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='k')
            ax.set_xlim(0, n)
            ax.set_ylim(0, n)
            ax.scatter(latin_best[:, j], latin_best[:, i + 1], color='', marker='o', edgecolors='k', s=10)
            ax.set_aspect(1)

    fig_for_best = plt.figure()
    x_list = list(range(0, len(latin_best_phi_q_correlation_list))) #x轴坐标
    ax = fig_for_best.add_subplot(111)
    ax.set_xlim(0, len(latin_best_phi_q_correlation_list))
    ax.plot(x_list, latin_best_phi_q_correlation_list, color='k')
    ax.set_title('fig_for_best')

    fig_for_optimization_process=plt.figure()
    x_list = list(range(0, len(latin_phi_q_correlation_list)))
    ax1=fig_for_optimization_process.add_subplot(311)
    ax1.plot(x_list,latin_phi_q_correlation_list,color='k',linewidth=1)
    ax1.set_title('fig_for_phi_q_correlation')

    x_list = list(range(0, len(latin_phi_q_list)))
    ax2=fig_for_optimization_process.add_subplot(312)
    ax2.plot(x_list,latin_phi_q_list,color='k',linewidth=1)
    ax2.set_title('fig_for_phi_q')
    file_object = open(r'data_for_plot\for_LHD\phi_q.txt', 'w')
    for i in range(0, len(latin_phi_q_list)):
        file_object.write(str(latin_phi_q_list[i]) + '\n')
    file_object.close()

    x_list = list(range(0, len(rho_list)))
    ax3=fig_for_optimization_process.add_subplot(313)
    ax3.plot(x_list,rho_list,color='k',linewidth=1)
    ax3.set_title('fig_for_correlation')
    file_object = open(r'data_for_plot\for_LHD\rho.txt', 'w')
    for i in range(0, len(rho_list)):
        file_object.write(str(rho_list[i]) + '\n')
    file_object.close()

    if k==3:
        x_latin_result = latin_best[:, 0]
        y_latin_result = latin_best[:, 1]
        z_latin_result = latin_best[:, 2]
        fig_for_3d = plt.figure()
        ax_for_3d = fig_for_3d.add_subplot(111, projection='3d')
        x_ticker_labels = range(0, n + 1)
        y_ticker_labels = range(0, n + 1)
        z_ticker_labels = range(0, n + 1)
        ax_for_3d.xaxis.set_major_locator(MultipleLocator(1))
        ax_for_3d.yaxis.set_major_locator(MultipleLocator(1))
        ax_for_3d.zaxis.set_major_locator(MultipleLocator(1))
        ax_for_3d.set_xticklabels(x_ticker_labels, rotation=0)
        ax_for_3d.set_yticklabels(y_ticker_labels, rotation=0)
        ax_for_3d.set_zticklabels(z_ticker_labels, rotation=0)
        ax_for_3d.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='k')
        ax_for_3d.set_xlim(0, n)
        ax_for_3d.set_ylim(0, n)
        ax_for_3d.set_zlim(0, n)
        ax_for_3d.scatter(x_latin_result, y_latin_result, z_latin_result, color='k', marker='o')
        ax_for_3d.set_aspect(1)
    plt.show()