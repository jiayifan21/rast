# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 04:32:34 2021

@author: jiayi
"""
import numpy as np
import matplotlib.pyplot as plt


# ract = [0.72,0.42,0.54,0.61,0.66,0.69]
# BSS = [0.72,0.38,0.44,0.53,0.64,0.67]
# epss = [50000,2000,5000,10000,20000,30000]

x = ["PGD","FAST","RACT(2/255)","RACT(4/255)","RACT(8/255)"]

data = [100000,100000,38526,64350,81226]
time = [149.00 ,86.48 ,42.55,57.08 ,66.82]
acc = [0.83 ,0.79 ,0.77 ,0.79 ,0.79 ]
fgsm = [0.59 ,0.72 ,0.66 ,0.69 ,0.71 ]
pgd = [0.56 ,0.74 ,0.65 ,0.70 ,0.74 ]

def plot(data,title):
    plt.plot(np.array(x).astype('str'), np.array(data),'--bo',color = 'blue')
    # plt.plot(np.array(y).astype('str'), np.array(BSS),'--bo', label = "BSS",color = 'blue')
    # plt.plot(np.array(epss).astype('str'), np.array(li_test)[:,1],'--bo', label = "Random",color = 'green')
    plt.title(title)
    plt.grid(True)
    # plt.ylim([0,1])
    for i in range(len(x)):
        plt.annotate(str(round(np.array(data)[i],2)), xy=(i,np.array(data)[i]), ha='right', va='bottom')
    # for i in range(len(epss)):
    #     plt.annotate(str(round(np.array(BSS)[i],2)), xy=(i,np.array(BSS)[i]), ha='left', va='top')
        
    plt.legend()
    plt.show()

plot(data,"Data size")
plot(time,"Training time")
plot(acc, "Testing accuracy")
plot(fgsm,"FGSM robustness")
plot(pgd,"PGD robustness")



plt.plot(np.array(x).astype('str'), np.array(fgsm),'--bo',label = "FGSM",color = 'green')
plt.plot(np.array(x).astype('str'), np.array(pgd),'--bo', label = "PGD",color = 'orange')
# plt.plot(np.array(epss).astype('str'), np.array(li_test)[:,1],'--bo', label = "Random",color = 'green')
plt.title("FGSM and PGD robustness")
plt.grid(True)
plt.ylim([0.5,1])
for i in range(len(x)):
    plt.annotate(str(round(np.array(fgsm)[i],2)), xy=(i,np.array(fgsm)[i]), ha='right', va='bottom')
    plt.annotate(str(round(np.array(pgd)[i],2)), xy=(i,np.array(pgd)[i]), ha='left', va='top')
    
plt.legend()
plt.show()



#################################bss adv#################
# adv_fgsm = [0.37687746,0.43809164,0.4649754,0.5096554,0.5612773]
# bss_fgsm = [0.30998358,0.37965417,0.43253818,0.5400732,0.55925786]
# adv_pgd = [0.37007776,0.42710775,0.44871002,0.4887051,0.5440069]
# bss_pgd = [0.30823356,0.37476856,0.4251327,0.52536726,0.54647577]

adv_fgsm = [0.38,0.44,0.46,0.51,0.56]
bss_fgsm = [0.31,0.38,0.43,0.54,0.56]
adv_pgd = [0.37,0.43,0.45,0.49,0.54]
bss_pgd = [0.31,0.37,0.43,0.52,0.54]

x = [2000,5000,10000,20000,30000]


def plot(data1,data2,title):
    plt.plot(np.array(x).astype('str'), np.array(data1),'--bo',label = "RACT",color = 'red')
    plt.plot(np.array(x).astype('str'), np.array(data2),'--bo', label = "BSS",color = 'blue')
    # plt.plot(np.array(epss).astype('str'), np.array(li_test)[:,1],'--bo', label = "Random",color = 'green')
    plt.title(title)
    plt.grid(True)
    plt.ylim([0,1])
    for i in range(len(x)):
        plt.annotate(str(round(np.array(data1)[i],2)), xy=(i,np.array(data1)[i]), ha='right', va='bottom')
    for i in range(len(x)):
        plt.annotate(str(round(np.array(data2)[i],2)), xy=(i,np.array(data2)[i]), ha='left', va='top')
        
    plt.legend()
    plt.show()

plot(adv_fgsm,bss_fgsm,"Robustness against FGSM attack")
plot(adv_pgd,bss_pgd,"Robustness against PGD attack")





