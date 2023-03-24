import random
import csv
import numpy as np
import os
from operator import itemgetter
import time
from itertools import product
from matplotlib import pyplot as plt

populationSize = 50
numberOfGenerations = 5000
tornumentSize = 2
mutationRate = 0.8  # , 0.01, 0,05, 0,1-.8
sigma = 10  # , 1E-2, 1E-1, 1, 10
mu = 0
parentSelectNum = 50
selectParentSize = 50
finalIndividualfitnes = []
indiv_final = []
newIndiv = []
findMse = []
finalIndividual = []
parent = []
chooseParentIndefinalindividual = []
child = []
child1 = []
child2 = []
min_arr = []
max_arr = []
avg_arr = []

def MseDef(listIndiv, num_pow, data):  # a,b,c,d,

    generateY = np.multiply(num_pow, listIndiv)
    arr_sum = np.sum(generateY, axis = 1, keepdims = True)
    arr_sum = np.around(arr_sum, decimals=3)
    arr_sub = np.subtract(arr_sum, data)
    arr_pow = np.float_power(arr_sub, 2.0)
    result_mse = np.sum(arr_pow)/100
    fitnes_indiv = 1/(1.0 + result_mse)
    #print(fitnes_indiv)
    end = time.time()

    return fitnes_indiv

def fitnessAll(individual, num_pow, data):
    #print('lenindiv',len(individual))
    individual_toaary = np.asarray(individual)
    individual_toaary = np.around(individual_toaary, decimals=4)
    totalFitnes = []
    totalFitnes.clear()
    for i in range(len(individual)):
       totalFitnes.append(MseDef(individual_toaary[i], num_pow, data))

    return totalFitnes


def tournoment( fitcalculated):

   # indi1= random.randint(0, populationSize-1)
    #indi2 = random.randint(0,populationSize-1)
    indi3 = random.sample(range(1, populationSize-1), 2)


    return indi3[0] if fitcalculated[indi3[0]] >= fitcalculated[indi3[1]] else indi3[1]


def selectParent(fitnescalculated):

    for i in range(selectParentSize):
        x = tournoment(fitnescalculated)
        parent.append(x)

    return parent

def doCrossover(parent1, parent2):

    indefinalindividualTocross = random.sample(range(0, 4), 4)
    child1 = [parent1[indefinalindividualTocross[0]], parent2[indefinalindividualTocross[1]], parent2[indefinalindividualTocross[2]], parent1[indefinalindividualTocross[3]]]
    return child1


def crossover(listParent, parent):#parent is index parent
    child.clear()
    for i in range(0, selectParentSize):
        child.append(doCrossover(listParent[random.randint(0, len(parent) - 1)], listParent[random.randint(0, len(parent) - 1)]))

    return child

def do_mutate(one_child):
  for i in range(4):
      if (random.uniform(0.0, 1.0)) <= mutationRate:
          one_child[i] += np.random.normal(mu, sigma, size=None)
  return one_child


def mutation(child):

    for i in range(len(child)):
#        child.append(do_mutate(child[i]))

        r = random.uniform(0.0, 1.0)
       # print(r)
        if r <= mutationRate:
            s = np.random.normal(0, 1, 4)
            t = random.sample(range(0, 4), 4)
            child[i][t[0]] += s[0]
            child[i][t[3]] += s[1]
            child[i][t[2]] += s[3]
            child[i][t[1]] += s[2]


    return child



def finalPopulation(child, individual, num_pow, data, fitcal_input):
    finalIndividual = []
    fitcal1 = []

    temp1 = len(individual)
    individual.extend(child)
    temp2 = len(individual)
    temp = temp2-temp1
    a1 = individual[temp1:len(individual)]
    individual[temp:temp2] = individual[0:temp1]
    individual[0:len(child)] = a1
   # print('lenchild',len(child))
    finalIndividualfitnes = fitnessAll(child,num_pow, data)

    finalIndividualfitnes.extend(fitcal_input)

    indices, finalIndividualfitnes = zip(*sorted(enumerate(finalIndividualfitnes), key=itemgetter(1)))

    c = len(list(indices))
    c2 = len(finalIndividualfitnes)
    min_arr.append(finalIndividualfitnes[0])
    max_arr.append(finalIndividualfitnes[99])
    avg_arr.append(finalIndividualfitnes[49])
    for i in range(populationSize):
        finalIndividual.append(individual[indices[c- 1 - i]])
        fitcal1.append(finalIndividualfitnes[c2 -i -1])
    #print('finalpopulation:',len(fitcal1),fitcal1)
    individual = finalIndividual
    #print('sss',len(individual),'\n',individual,)


    return individual, fitcal1

if __name__ == '__main__':
    os.chdir(r'D:\fatemeh\unversity\term 8\هوش\hw\project1-9431041')
    with open('input.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    data = np.asarray(data)
    data = data.astype(np.float64)
    data2 = np.around(data, 4)
    #numbers = [finalindividual  for finalindividual in range(100)]
    numbers = [finalindividual*0.1  for finalindividual in range(100)]
    num_pow = [[np.float_power(numbers[i], 3), np.float_power(numbers[i], 2), numbers[i], 1] for i in range(100)]
    num_pow = np.around(num_pow, decimals=4)
    individual = []
    for i in range(50):
        individual.append(np.random.uniform(low=-1.0, high=1.0, size=4).tolist())

    start = time.time()
    fitcal = fitnessAll(individual, num_pow, data)
    findres = MseDef([-0.02760047077331451, 0.4008768550158808, -0.25234458190367826, 0.47944639827610247], num_pow, data)

    for i in range (numberOfGenerations):
        print('thisiinmain:', i)
        parent.clear()# 7.867302540352097e-07
        parent = selectParent(fitcal)
        child = crossover(individual,parent)
        child = mutation(child)
        individual, fitcal = finalPopulation(child, individual, num_pow, data, fitcal)

       # print(len(individual),'finalINDIDUAL',individual)
        numberOfGenerations -= 1
    #fitcal = fitnessAll(individual, num_pow, data)
    print('this is the final ans:',individual[0],'max:',max(fitcal),'min:',min(fitcal))
    new_y = np.multiply(num_pow, individual[0])
    new_y = np.sum(new_y, axis = 1, keepdims = True)
    end = time.time()
    print('time is:',end-start)
    plt.scatter(numbers, data, color='Blue')

    plt.scatter(numbers, new_y, color="red")
    plt.show()
    print(len(max_arr),'dd')

    num = [i  for i in range(5000)]
    plt.scatter(num, min_arr, color='Blue')
    plt.show()
    plt.scatter(num, max_arr, color='red')
    plt.show()
    plt.scatter(num, avg_arr, color='black')
    plt.show()



