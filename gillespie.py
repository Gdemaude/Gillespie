import numpy as np
from math import log
from numba import jit
from matplotlib import pyplot
import time

"""
Here is a Numba accelerated implementation of the Gillespie's algorithm that simulates stochastic processes with an example of its use on the SIR epidemiology model.

SIR is a model simulating how many people will be infected and will recovered from an infectious disease. Here is its diagram:

S -> I -> R

With S: susceptible population, I: infected population, R: recovered population  <- 3 categories

Propensity/ Transition rate from S to I : alpha * PopulationOfS * PopulationOfI / TotalPopulation
Propensity/ Transition rate from I to R : beta  * PopulationOfI 

Stoichiometry of first reaction: -1 to PopulationOfS and +1 to PopulationOfI
Stoichiometry of first reaction: -1 to PopulationOfI and +1 to PopulationOfR

"""


@jit(nopython=True, cache= True)
def gillespie_direct(data, stoichiometry, iter, timestop=0):
    """
    Inputs:
    data: preallocated numpy array of size (n+1)xiter with n the number of categories of the stochastic system + 1 for the time. The first column is initialized with initial conditions
    stoichiometry: numpy array of the size mxn with m the number of possible transitions and n the number of categories. The array is used to transfer units from one category to another when a transition is chosen
    iter: max number of iteration the algorithm can do
    timestop: optional, maximum time of the simulation.

    propensity(i, d): user defined function which returns a numpy array with the propensities of length m

    Output:
    res: slice of the data array with the actual number of iterations done (algorithm can stops sooner than iter if no more possible transition),
    warning: res refers to the same memory as data


    See the example for proper initialisation of data stoichiometry and propensity()
    """
    for i in range(iter-1):
        if timestop>0:
            if data[-1,i]>=timestop:
                return data[:,:i]
        propensities= propensity(i, data)
        partition = np.sum(propensities)
        if partition==0.0:
            return data[:,:i]
        r1=np.random.random()
        sojourn = log(1.0 / r1) / partition
        data[-1,i+1]= data[-1, i]+sojourn
        indexes= np.argsort(propensities)
        partition= np.random.random()*partition
        for j in indexes:
            partition-=propensities[j]
            if partition<=0.0:
                data[:-1,i+1]=data[:-1,i]+stoichiometry[j]
                break
    return data

@jit(nopython=True, cache= True)
def propensity(i, d):
    """
    user defined function
    Input
    i: current iteration
    d: data array

    Output:
    1D Numpy array of length m, the number of possible transitions

    Help for defining the function: put parameters of the stochastic process here
    The order of the rows of the output array must be the same as the stoichiometry array.
    Ex: The first reaction/transition is on row 0 of stoichiometry and its propensity (aka its transition rate) must be element 0 of this output array
    Warning: this function must be Numba accelerated thus some python functionalities are unavailable
    """
    # parameters
    alpha = 2.0
    beta = 1.0
    N = 5500
    return np.array([alpha * d[0][i] * d[1][i] / N, #S -> I at transition rate: alpha * S * I / N
                     beta * d[1][i]]) # I -> R at transition rate: beta * I


"""
iter: max number of iteration the algorithm will execute
data: preallocated array upon which gillespie will work. Each row (except the last) stores the population of each category at a given time, the last row is used to store the time.
The first column must be initialized with the initial conditions.
Here S= 5480, I= 20, E= 0 and time = 0
"""
iter=20000
data= np.zeros((4,iter), dtype=float)
data[:, 0]=[5480.0,20.0,0.0,0.0] #intialiase data ->  s i r time


"""
Array used to know how many units are transferred from one category to another for each reaction/transition
"""
stoichiometry = np.array([# s  i  r   <- same column order than initial conditions and same line order as propensities
    [ -1,  1,  0], # First row: S->I thus -1 for S, +1 for I and R not affected
    [0,  -1,  1],# Second row: I-> R thus -1 for I, +1 for R and S not affected
])

pyplot.figure(figsize=(10,10))
# make a subplot for the susceptible, infected and recovered individuals
axes_s = pyplot.subplot(311)
axes_s.set_ylabel("susceptible individuals")

axes_i = pyplot.subplot(312)
axes_i.set_ylabel("infected individuals")

axes_r = pyplot.subplot(313)
axes_r.set_ylabel("recovered individuals")
axes_r.set_xlabel("time (days)")


t1=time.time()
np.random.seed(1235)
timestop=200 #will end at 200 or sooner depending on iter
#execute 100 simulations of our stochastic process
for i in  range(100):
    res=gillespie_direct(data,stoichiometry, iter, timestop=timestop) # first column of data remains untouched, data can be reused for all simulations
    axes_s.plot(res[-1], res[0], color="orange")
    axes_i.plot(res[-1], res[1], color="orange")
    axes_r.plot(res[-1], res[2], color="orange")

t2=time.time()
print(t2-t1)
pyplot.show()
