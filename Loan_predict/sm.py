
# ----------------------AON Network----------------------
import math

class Node:
    def __init__(self, process, exp_time, predecessor):
        self.proces s =process
        self.exp_tim e =exp_time
        self.predecesso r =predecessor
        self.sucesso r =set()
        self.e_start =0
        self.e_finish = 0
        self.l_start = 0
        self.l_finish = 0
        self.slack = -1


node = list()

node.append(Node('A', 7, {}))
node.append(Node('B', 9, {}))
node.append(Node('C', 12, {'A'}))
node.append(Node('D', 8, {'A', 'B'}))
node.append(Node('E', 9, {'D'}))
node.append(Node('F', 6, {'C', 'E'})),
node.append(Node('G', 5, {'E'}))

data_cal = dict()
for i in node:
    data_cal[i.process] = 0

over_mx = -1
while True:
    for i in range(len(node)):
        if data_cal[node[i].process] == 0 and len(node[i].predecessor) == 0:
            node[i].e_start = 0
            node[i].e_finish = node[i].exp_time
            data_cal[node[i].process] = 1
        else:
            mx = -1
            for j in node[i].predecessor:
                if data_cal[j] == 1:
                    for k in node:
                        if k.process == j and mx < k.e_finish:
                            mx = k.e_finish
                            k.sucessor.add(node[i].process)
                else:
                    mx = -1
                    break
            if not (mx == -1):
                node[i].e_start = mx
                node[i].e_finish = node[i].e_start + node[i].exp_time
                data_cal[node[i].process] = 1

        if over_mx < node[i].e_finish:
            over_mx = node[i].e_finish
    if sum(list(data_cal.values())) == len(node):
        break

for i in node:
    data_cal[i.process] = 0

print("Earlier start and finish(After forward Calculation): ")
for i in range(len(node)):
    print(node[i].process, node[i].e_start, node[i].e_finish)

while True:
    for i in range(len(node)):
        if data_cal[node[i].process] == 0 and len(node[i].sucessor) == 0:
            node[i].l_finish = over_mx
            node[i].l_start = node[i].l_finish - node[i].exp_time
            data_cal[node[i].process] = 1
        else:
            mn = 999999

            for j in node[i].sucessor:
                if data_cal[j] == 1:
                    for k in node:
                        if k.process == j and mn > k.l_start:
                            mn = k.l_start
                else:
                    mn = 999999
                    break
            if not (mn == 999999):
                node[i].l_finish = mn
                node[i].l_start = node[i].l_finish - node[i].exp_time
                data_cal[node[i].process] = 1

    if sum(list(data_cal.values())) == len(node):
        break

print("Latest start and finish(After Backward Calculation): ")

for i in range(len(node)):
    print(node[i].process, node[i].l_start, node[i].l_finish)

print("Slack value: ")
for i in range(len(node)):
    node[i].slack = node[i].e_start - node[i].l_start
    print(node[i].slack, end=" ")

print()
print("Critical path: ")
for i in range(len(node)):
    if node[i].slack == 0:
        print(node[i].process, end=" ")


#-----------------------Random Number Generator----------------------------
"""
Implement a combined linear congruential generator of your own with at least a cycle of 512.
Calculate the cycle of your proposed generator
Generate 300 random number
Perform a uniform test and a correlation test on your generated random numbers.
"""
import random as rnd
import math


def CLCG(m1, m2, a1, a2, seed1, seed2, R):
    Y1 = seed1
    Y2 = seed2
    n = 100
    Y1 = a1 * Y1 % m1
    Y2 = a2 * Y2 % m2
    X = (Y1 - Y2) % (m1 - 1)
    preX = X
    cycle = 1
    # for i in range (1, n):
    while True:
        # print(X)
        if (X > 0):
            R.append(X / m1)
        elif (X == 0):
            R.append((m1 - 1) / m1)
        Y1 = a1 * Y1 % m1
        Y2 = a2 * Y2 % m2
        X = (Y1 - Y2) % (m1 - 1)
        if X != preX:
            cycle += 1
        else:
            break
    # print(X)
    print("Cycle length: ", cycle)


# def chi_square_uniformity_test( data_set, confidence_level, num_samples ):
def chi_square_uniformity_test(data_set, num_samples):
    # This is our test statistic, this will be an accumulated value, as we loop through the data set
    chi_sq_value = 0.0
    degrees_of_freedom = num_samples - 1
    expected_val = num_samples / 10.0
    for observed_val in data_set:
        chi_sq_value += (pow((expected_val - data_set[observed_val]), 2) / expected_val)
    return chi_sq_value


def Kolmogorov_Smirnov(R, n):
    R = sorted(R)
    d_plus_max = 0
    d_minus_max = 0
    i = 1
    for value in R:
        d_plus_i_value = ((i / n) - value)
        d_minus_i_value = (value - ((i - 1) / n))
        if d_plus_i_value > d_plus_max:
            d_plus_max = d_plus_i_value
        if d_minus_i_value > d_minus_max:
            d_minus_max = d_minus_i_value
        i += 1
    # print(d_plus_i_value, d_minus_i_value)
    return max(d_plus_i_value, d_minus_i_value)


def autocorrelation_tests(R, n, gap_sequence):
    little_m = gap_sequence
    start_index = 0
    big_n = n
    big_m = 0.0

    # print((big_n - start_index)/little_m )
    while (big_m + 1) < ((big_n - start_index) / little_m):
        big_m = big_m + 1
    # print(big_m)
    one_over_m_plus_one = (1.0 / (big_m + 1.0))
    rho_hat = 0.0
    sum_of_rho_hat = 0.0

    every_m_element = R[0::gap_sequence]

    for value in range(0, (len(every_m_element) - 1)):
        thisValue = float(every_m_element[value])
        nextValue = float(every_m_element[value + 1])
        sum_of_rho_hat = sum_of_rho_hat + (thisValue * nextValue)

    sum_of_rho_hat = (one_over_m_plus_one * sum_of_rho_hat) - 0.25
    variance_of_rho = math.sqrt((13 * big_m + 7)) / (12 * (big_m + 1))
    print("variance of rho: ", variance_of_rho)

    z_statistic = sum_of_rho_hat / variance_of_rho
    return z_statistic


m1 = 2147483563
m1 = 5011
a1 = 4001
m2 = 2147483399
m2 = 1059
a2 = 4069
R = list()
seed1 = m1 - 100
seed2 = m2 - 100
CLCG(m1, m2, a1, a2, seed1, seed2, R)
d = Kolmogorov_Smirnov(R[:301], 300)

print("Uniform test: ")
print("Kolmogorov_Smirnov: ")
print("D value is: ", d)

print("Autocorrelation tests: ")
z = autocorrelation_tests(R[:301], 300, 2)
print("Value of Z: ", z)
# print(R[:301])

print("Chi-square test: ")
# chi_square_uniformity_test(R,len(R))







"""
#-----------------------Bomber Fighter----------------------------

import math


b=[[80,0],[90,-2],[99,-5],[108,-9],[116,-15],[125,-18],[133,-23],[141,-29],[151,-28],[160,-25],[169,-21],[179,-28]]

f=[0,50]
vf=float(input())
distance=0
st=12
d=0.00
for i in range(0,st,1):
  d=math.sqrt(pow(b[i][0]-f[0],2)+ pow(b[i][1]-f[1],2))
  print(d)
  if d<=distance:
    print("Attact")
    #print(d)
    break
  else:
    cost=(b[i][0]-f[0])/d
    sint=(b[i][1]-f[1])/d
    f[0]=f[0]+vf*cost
    f[1]=f[1]+vf*sint

"""

"""
-------------------------Cobweb Model--------------------------
"""
import matplotlib.pyplot as plt
import math

print("Enter the value of a,b,c,d")

a, b, c, d = map(float, input().split())
p = abs((a - c) / (b + d))
Q = a - (b * p)
S = c + (d * p)

if (math.ceil(Q) == math.ceil(S)):
    print("Equilbrium point:", Q, "", p)
    print("Quality Q:", Q)
    print("price P:", p)

else:
    print("there is not equilibrium point")
x = [0, 2 * Q - 0]
y = [0, 2 * p - 0]
x1 = [0, 2 * Q - 0]
y1 = [p * 2, 2 * p - p * 2]

plt.plot(x, y, x1, y1, color="blue", linewidth=2)
plt.scatter(Q, p)

p0 = 1
s1 = c + d * p0
p1 = (a - s1) / b
print(s1, p1)
plt.scatter(s1, p1)
prev_point = [s1, p1]
p0 = p1
for i in range(10):
    s1 = c + d * p0
    p1 = (a - s1) / b
    # print(s1,p1)
    plt.scatter(s1, p1)
    plt.plot([prev_point[0], s1], [prev_point[1], prev_point[1]])
    plt.plot([s1, s1], [prev_point[1], p1])
    prev_point = [s1, p1]
    p0 = p1
# plt.scatter(2,5)
plt.show()
# ---------------------Cobweb End-----------------------------


"""
-------------------------Monte Carlo-------------------------
import random
import math
import numpy as np
import matplotlib.pyplot as plt


def f(x,a,b,c):
  return 3*x*x
#n,a,b,h=200,0,60,150
n,a,b,h=map(int, input("Enter the value of n,a,b,h: ").split())
count=0
xaxis=list()
yaxis=list()

cxaxis=list()
cyaxis=list()

for i in range(a,int(b/2)):
  y_cal=f(i,a,b,h)
  cxaxis.append(i)
  cyaxis.append(y_cal)
print(cxaxis)
print(cyaxis)
dic=dict()
for i in range(a,b+1):
  dic[i]=0
for i in range(n):
  x=random.randint(a,b)
  y=random.randint(0,h)
  y_cal=f(x,a,b,h) 
  if y<=y_cal:
    dic[x]+=1
    count+=1
    xaxis.append(x)
    yaxis.append(y)
    plt.scatter(xaxis,yaxis)
plt.plot([a,a],[0,h])
plt.plot([b,b],[0,h])
plt.plot([a,b],[h,h])
plt.plot(cxaxis,cyaxis)
#plt.plot([a-2,b+4],[3*(a-2),3*(b+4)])


plt.show()
print("Points in Curve: ",count)
for i in dic:
  print("Division",i,":",dic[i])
"""

"""
#-------------------Simple Market Model-------------------
"""

# there is equilibrium point: 600 3000 -100 2000

import matplotlib.pyplot as plt
import math

print("Enter the value of a,b,c,d")

a, b, c, d = map(int, input().split())
p = abs((a - c) / (b + d))
Q = a - (b * p)
S = c + (d * p)

if (math.ceil(Q) == math.ceil(S)):
    print("Equilbrium point:", Q, "", p)
    print("Quality Q:", Q)
    print("price P:", p)

else:
    print("there is not equilibrium point")

x = [0, 2 * Q - 0]
y = [0, 2 * p - 0]
x1 = [0, 2 * Q - 0]
y1 = [0.5, 2 * p - 0.5]

plt.plot(x, y, x1, y1, color="blue", linewidth=2)
plt.scatter(180, 0.14)
plt.show()