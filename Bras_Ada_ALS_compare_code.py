#from os import confstr_names
import numpy as np
import tensorly as tl
import math
from tensorly.tenalg import khatri_rao
from numpy import linalg
import itertools
import random
from random import choice
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations 
import matplotlib.pyplot as plt

#metric [cost] [done][checked]
def cost(X,bA,shape,rank): 
  ite = np.prod(shape)
  XX = np.ones(shape.tolist())
  TLL = np.ones(rank)
  bAL = (TLL,bA)
  XX = tl.cp_to_tensor(bAL)
  bX = XX
  nX = np.array(X - bX)
  cost1 = np.sum(nX*nX)/ite
  return cost1

#metric [MSE] [done][checked]
def MSE(A,bA,F):
  MSE_r = []
  for i in range(len(A)):
    # X: estimated data
    l = np.array(bA[i]) @ np.diag(1/np.sqrt(np.sum(np.array(bA[i]) ** 2, axis=0)))
    s = np.array(A[i]) @ np.diag(1/(np.sqrt(np.sum(np.array(A[i]) ** 2, axis=0)) + np.finfo(float).eps))
    M = l.T @ s
    MSE_col = np.max(M, axis=0)
    MSE = np.abs(np.mean(2-2*MSE_col))
    MSE_r.append(MSE)
  MSE_r = np.array(MSE_r)
  mean = np.sum(MSE_r)/len(A)
  return mean

#BrasCPD [done][checked]
def BrasCPD(tensor, rank , sample_size, A, bA, J, MSE_plot, Cost, ran, shape, beta_c, alpha_c):
  BETA = beta_c 
  ALPHA = alpha_c 
  F = rank
  N = len(shape)
  B = sample_size
  max = int(60 * (shape[0]**2)/B)

  t = MSE(A, bA, F)
  MSE_plot.append(t)
  l = cost(tensor, bA, shape,F)
  Cost.append(l)
  ran.append(0)
  max_it = max

  G = []
  for i in range(len(shape)):
    G.append(np.zeros((shape[i],F)))
  G = np.array(G)

  #loop
  for i in range(max_it):
    print(f"BrasCPD round {i}")
    np.random.seed(i**2)
    n = choice(range(N))
    print(J[n])
    newlist = random.sample(range(J[n]), B)
    newlist = np.sort(newlist)
    alpha = ALPHA / pow(i+1,BETA)

    #compute gradient and those A
    for j in range(N):
      if (j == n):
        H = khatri_rao(bA, skip_matrix = n)
        sub_H = np.ix_(newlist, range(F))
        HL = H[sub_H]
        t_sub_H = np.transpose(HL)
        X_n = tl.unfold(tensor, j).T
        sub_X_n = np.ix_(newlist, range(shape[j]))
        LL = X_n[sub_X_n]
        t_sub_X_n = np.transpose(LL)
        G[j] = 1/B * (np.matmul(np.matmul(bA[j],t_sub_H),HL) - np.matmul(t_sub_X_n, HL)) 
        bA[j] = np.maximum(bA[j] - alpha * G[j],0)
      else:
        G[j] = np.zeros((shape[j],F))
        bA[j] = bA[j]
    if ((i+1) % (int((shape[0]**2)/B)) == 0):
      t = MSE (A , bA, F)
      MSE_plot.append(t)
      c = cost(tensor, bA,shape,F)
      Cost.append(c)
      ran.append(i/((shape[0]**2)/B))
  return Cost, ran, i , MSE_plot, bA

  #AdaCPD [done] [checked]
def AdaCPD(tensor, rank , sample_size, A, bA, J,MSE_plot,Cost,ran,shape):
  
  F = rank
  N = len(shape)
  B = sample_size
  max = int(60 * (shape[0]**2)/B)
  t = MSE(A, bA, F)
  MSE_plot.append(t)
  l = cost(tensor, bA,shape,F)
  Cost.append(l)
  ran.append(0)
  max_it = max

  G = []
  for i in range(len(shape)):
    G.append(np.zeros((shape[i],F)))
  G = np.array(G)

  lam = []
  for j in range(len(shape)):
    lam.append(np.zeros((shape[j],F)))
  lam = np.array(lam)

  squared_sum_gradient = []
  for k in range(len(shape)):
    squared_sum_gradient.append(np.zeros((shape[k],F)))
  squared_sum_gradient = np.array(squared_sum_gradient)

  #loop
  for i in range(max_it):
    print(f"AdaCPD round {i}")
    print(i)
    np.random.seed(i**2)
    n = choice(range(N))
    newlist = random.sample(range(J[n]), B)
    newlist = np.sort(newlist)

    #compute gradient and those A
    for j in range(N):
      if (j == n):
        H = khatri_rao(bA, skip_matrix = n)
        sub_H = np.ix_(newlist, range(F))
        HL = H[sub_H]
        t_sub_H = np.transpose(HL)
        X_n = tl.unfold(tensor,j).T
        sub_X_n = np.ix_(newlist, range(shape[j]))
        LL = X_n[sub_X_n]
        t_sub_X_n = np.transpose(LL)
        G[j] = 1/B * (np.matmul(np.matmul(bA[j],t_sub_H),HL) - np.matmul(t_sub_X_n, HL))
        squared_sum_gradient[j] = squared_sum_gradient[j] + np.square(G[j])
        lam[j] = np.divide(np.random.rand(shape[j],F), np.sqrt(1e-6+squared_sum_gradient[j])) 
        bA[j] = np.maximum(bA[j] - lam[j] * G[j],0)
      else:
        G[j] = np.zeros((shape[j],F))
        bA[j] = bA[j]
    
    #record MSE and cost
    if ((i+1) % (int((shape[0]**2)/B)) == 0):
      t = MSE (A , bA, F)
      MSE_plot.append(t)
      c = cost(tensor, bA,shape,F)
      Cost.append(c)
      ran.append(i/((shape[0]**2)/B))
  return Cost, ran, i , MSE_plot, bA

#CP-ALS Algorithm [done][checked]
def cp_als(tensor, rank , sample_size, A, bA, J,MSE_plot,Cost,ran,shape):
    F = rank
    N = len(shape)
    B = sample_size
    max = 20
    t = MSE(A, bA, F)
    MSE_plot.append(math.log(t,10))
    l = math.log(cost(tensor, bA,shape,F),10)
    Cost.append(l)
    ran.append(0)
    max_it = max

    #Loop
    for j in range(max):
      for n in range(N):
        V = np.ones((F,F))
        for i in range(N):
          if i != n:
            V = np.matmul(bA[i].T,bA[i])*V
        T = khatri_rao(bA,skip_matrix = n)
        bA[n] = np.matmul(np.matmul(tl.unfold(tensor,mode = n),T),np.linalg.pinv(V))
      t = MSE(A,bA,F)
      MSE_plot.append(math.log(t,10))
      c = math.log(cost(tensor,bA,shape,F),10)
      Cost.append(c)
      ran.append(3*j)
    return Cost,ran,j,MSE_plot,bA


#main function
monte_carlo = 3 #we can choose larger one, like: 50, 100,...
shape = [100,100,100]
shape = np.array(shape)
rank = 10
sample_size = 20
dimension = len(shape)
BETA_CC = 1e-6
ALPHA_CC = 0.1
ALPHA_CC2 = 0.01
ALPHA_CC3=0.05
ALPHA_CC6 = 1.5

#generate ground true latent factors [done]
I = shape # shape
dim = dimension
F = rank   #rank F
B = sample_size  #sample_size

J1 = np.zeros((dim,))
J = []
for j in range(dim):
  J.append(int(J1[j]))
for k in range(dim):
  J[k] = int(np.prod(I)/I[k])

A = [] #true latent factors[done]
for i in range(dim):
  np.random.seed(i)
  A.append(np.random.rand(I[i],F))
A = np.array(A)

bAa = [] #use to approxamate the ground true factors for CP-ALS [done]
for i in range(dim):
  np.random.seed(i**5)
  bAa.append(np.random.rand(I[i],F))
bAa = np.array(bAa)
bA3 = tl.tensor(bAa)

#recover from factors to the original tensor [done]
TLL = np.ones(rank)
L0 = (TLL,A)
X = tl.cp_to_tensor(L0)

#start to compute use BrasCPD-0.1[done]
print("Start BrasCPD")
MSE_bras_o = []
Cost_bras_o = []
ran_bras_o = []
for m in range(monte_carlo):
  MSE_plot = [] 
  Cost = []
  ran = []
  bAz = [] #use to approxamate the ground true factors for BrasCPD-alpha=0.1 [done]
  for i in range(dim):
    np.random.seed(i**5)
    bAz.append(np.random.rand(I[i],F))
  bAz = np.array(bAz)
  bA1 = tl.tensor(bAz)
  Cost, ran, r , MSE_plot, bA1 = BrasCPD(X, F, B, A,bA1, J,MSE_plot, Cost, ran,shape,BETA_CC, ALPHA_CC)
  print(f"MSE_plot{m}{0}={MSE_plot[0]}")
  MSE_bras_o.append(MSE_plot)
  Cost_bras_o.append(Cost)
  ran_bras_o.append(ran)
MSE_plot = np.median(np.array(MSE_bras_o),axis = 0)
Cost = np.median(np.array(Cost_bras_o),axis = 0)
for i in range(len(MSE_plot)):
  print(MSE_plot[i])
  MSE_plot[i] = math.log(MSE_plot[i],10)
  Cost[i] = math.log(Cost[i],10)
ran = np.median(np.array(ran_bras_o),axis = 0)
plt.title("Result")
plt.xlabel("no. of MTTKRP")
plt.ylabel("log(base 10)")
plt.plot(ran,MSE_plot,color = 'blue',marker= 's', label = "BrasCPD-MSE-alpha=0.1")



#BrasCPD-va1
MSE_bras_o1 = []
Cost_bras_o1 = []
ran_bras_o1 = []
for m in range(monte_carlo):
  MSE_plot3 = [] 
  Cost3 = []
  ran3 = []
  bA_v1 = [] #use to approxamate the ground true factors for BrasCPD-alpha=0.01 [done]
  for i in range(dim):
    np.random.seed(i**5)
    bA_v1.append(np.random.rand(I[i],F))
  bA_v1 = np.array(bA_v1)
  bA6 = tl.tensor(bA_v1)
  Cost3, ran3, r , MSE_plot3, bA6 = BrasCPD(X, F, B, A,bA6, J,MSE_plot3, Cost3, ran3,shape,BETA_CC, ALPHA_CC2)
  print(f"MSE_plot{m}{0}={MSE_plot[0]}")
  MSE_bras_o1.append(MSE_plot3)
  Cost_bras_o1.append(Cost3)
  ran_bras_o1.append(ran3)

MSE_plot3 = np.median(np.array(MSE_bras_o1),axis = 0)
Cost3 = np.median(np.array(Cost_bras_o1),axis = 0)
for i in range(len(MSE_plot3)):
  print(MSE_plot3[i])
  MSE_plot3[i] = math.log(MSE_plot3[i],10)
  Cost3[i] = math.log(Cost3[i],10)

ran3 = np.median(np.array(ran_bras_o1),axis = 0)
plt.plot(ran3,MSE_plot3,color = 'blue', marker = 'o', label = "BrasCPD-MSE-alpha=0.01") 


#Bras
#BrasCPD-va2
MSE_bras_o2 = []
Cost_bras_o2 = []
ran_bras_o2 = []
for m in range(monte_carlo):
  MSE_plot4 = [] 
  Cost4 = []
  ran4 = []
  bA_v2 = [] #use to approxamate the ground true factors for BrasCPD-alpha=0.05 [done]
  for i in range(dim):
    np.random.seed(i**5)
    bA_v2.append(np.random.rand(I[i],F))
  bA_v2 = np.array(bA_v2)
  bA7 = tl.tensor(bA_v2)
  Cost4, ran4, r , MSE_plot4, bA7 = BrasCPD(X, F, B, A,bA7, J,MSE_plot4, Cost4, ran4,shape,BETA_CC, ALPHA_CC3)
  print(f"MSE_plot{m}{0}={MSE_plot[0]}")
  MSE_bras_o2.append(MSE_plot4)
  Cost_bras_o2.append(Cost4)
  ran_bras_o2.append(ran4)

MSE_plot4 = np.median(np.array(MSE_bras_o2),axis = 0)
Cost4 = np.median(np.array(Cost_bras_o2),axis = 0)
for i in range(len(MSE_plot4)):
  print(MSE_plot4[i])
  MSE_plot4[i] = math.log(MSE_plot4[i],10)
  Cost4[i] = math.log(Cost4[i],10)

ran4 = np.median(np.array(ran_bras_o2),axis = 0)
plt.plot(ran4,MSE_plot4,color = 'blue', marker = '*', label = "BrasCPD-MSE-alpha=0.05") 


#BrasCPD-va3
MSE_bras_o6 = []
Cost_bras_o6 = []
ran_bras_o6 = []
for m in range(monte_carlo):
  MSE_plot6 = [] 
  Cost6 = []
  ran6 = []
  bA_v6 = [] #use to approxamate the ground true factors for BrasCPD-alpha=0.05 [done]
  for i in range(dim):
    np.random.seed(i**5)
    bA_v6.append(np.random.rand(I[i],F))
  bA_v6 = np.array(bA_v6)
  bA7 = tl.tensor(bA_v6)
  Cost6, ran6, r , MSE_plot6, bA7 = BrasCPD(X, F, B, A,bA7, J,MSE_plot6, Cost6, ran6,shape,BETA_CC, ALPHA_CC6)
  print(f"MSE_plot{m}{0}={MSE_plot[0]}")
  MSE_bras_o6.append(MSE_plot6)
  Cost_bras_o6.append(Cost6)
  ran_bras_o6.append(ran6)

MSE_plot6 = np.median(np.array(MSE_bras_o6),axis = 0)
Cost6 = np.median(np.array(Cost_bras_o6),axis = 0)
for i in range(len(MSE_plot6)):
  print(MSE_plot6[i])
  MSE_plot6[i] = math.log(MSE_plot6[i],10)
  Cost6[i] = math.log(Cost6[i],10)

ran6 = np.median(np.array(ran_bras_o6),axis = 0)
plt.plot(ran6,MSE_plot6,color = 'blue', marker = 'v', label = "BrasCPD-MSE-alpha=1.5") 



#start to compute use AdaCPD[done]
MSE_ada_o = []
Cost_ada_o = []
ran_ada_o = []
for m in range(monte_carlo):
  MSE_plot1 = []
  Cost1 = []
  ran1 = [] 
  bAn = [] #use to approxamate the ground true factors for AdaCPD [done]
  for i in range(dim):
    np.random.seed(i**5)
    bAn.append(np.random.rand(I[i],F))
  bAn = np.array(bAn)
  bA2 = tl.tensor(bAn)
  Cost1, ran1, r1, MSE_plot1,bA2 = AdaCPD(X, F, B, A,bA2, J,MSE_plot1, Cost1, ran1,shape)
  MSE_ada_o.append(MSE_plot1)
  Cost_ada_o.append(Cost1)
  ran_ada_o.append(ran)

MSE_plot1 = np.median(np.array(MSE_ada_o),axis = 0)
Cost1 = np.median(np.array(Cost_ada_o),axis = 0)
ran1 = np.median(np.array(ran_ada_o),axis = 0)
for i in range(len(MSE_plot1)):
  print(MSE_plot1[i])
  MSE_plot1[i] = math.log(MSE_plot1[i],10)
  Cost1[i] = math.log(Cost1[i],10)

plt.plot(ran1,MSE_plot1,color = 'red', label = "AdaCPD-MSE") 

#start to compute CP-ALS[done]
MSE_plot2 = []
Cost2= []
ran2=[]
Cost2,ran2,r2,MSE_plot2,bA3 = cp_als(X ,F, B, A, bA3, J, MSE_plot2, Cost2, ran2,shape)
plt.plot(ran2, MSE_plot2, color ='green', label = "ALS-MSE")

plt.grid(alpha = 0.6)
plt.legend()
plt.savefig('./MSE.png')
plt.close()

plt.title("Cost")
plt.xlabel("No. of MTTKRP")
plt.ylabel("log(base 10)")
plt.plot(ran,Cost,color = 'skyblue', label = 'BrasCPD-Cost,alpha = 0.1')
plt.plot(ran,Cost3,color = 'skyblue', marker = 'o', label = 'BrasCPD-Cost,alpha = 0.01')
plt.plot(ran,Cost4,color = 'skyblue', marker = '*', label = 'BrasCPD-Cost,alpha = 0.05')
plt.plot(ran,Cost6,color = 'skyblue', marker = 's', label = 'BrasCPD-Cost,alpha = 1.5')


plt.plot(ran1, Cost1, color='red',label='AdaCPD-Cost')
plt.plot(ran2, Cost2, color = 'black',label='ALS-Cost')
plt.grid(alpha = 0.6)

plt.legend()
plt.savefig('./Cost.png')
plt.close()
