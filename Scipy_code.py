# -*- coding: utf-8 -*-
"""
@author: David Redondo Quintero
Github: Drq13112
"""

from scipy.optimize import basinhopping
from numpy.random import rand
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from scipy import constants
from numpy import meshgrid
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import root, minimize

"""

Scipy is a scientific computacion library which uses numpy underneath.
The main reason why we don't use numpy to do the task that we do with numpy,
is because it provides few more utilities for optimisation, stats and signal procesing.

So, this library is a layer built over Numpy which offers a lot of new mathmatical and statistical functionalities.

We´ll find functions for linear algebra, optimisation of mathematical functions, signals processing and mathematical distributions.

We can find all the information we need in this path:https://docs.scipy.org/doc/scipy/reference/
"""


# SciPy Constants
# This library includes a long list of constants.
# A easy way to visualise them is printing the module itself.
# print(dir(constants))


# SciPy Optimizers
# Let's start with the interesting part of this library.
# SciPy optimize provides functions for minimizing (or maximizing) objective functions, possibly subject to constraints.
# It includes solvers for nonlinear problems (with support for both local and global optimization algorithms),
# linear programing, constrained and nonlinear least-squares, root finding, and curve fitting.

# ROOT FINDING

# Scipy provide us a pretty useful function called root. We´ll be able to find the roots of
# the equation given.

# Defining the equation as a function:

def eqn(x):
    return x + cos(x)


# The function roots will return an object with information regarding the solution
myroot = root(eqn, 0)

# We will print the solution under atrribute x of the returned object:
print(myroot.x)  # This time, the solution found for this equation is -0.739085

"""
# Let's have a view of some of them:

# Root finding
# Local (multivariate) optimization
# Global optimization
# Least-squares and curve fitting
# Assignment problems

"""

# OPTIMIZATION

# Through the function minimize we made a minimization of scalar function of one
# or more variables.


def eqn(x):
    return x**3 + x + 9


"""
There are three inputs the function requires:
- A function representing an equation
- An initial guess for the root -> This time we chose 0
- The way it'll find the solution.

We have a lot of methods we can use:
    - CG
    - BFGS
    - Newton-CG
    - L-BFGS-B
    - TNC
    - COBYLA
    - SLSQP

The method that we are using is BFGS.
BFGS uses the quasi-Newton method of Broyden, Fletcher, Goldfarb, and Shanno (BFGS).
It uses the first derivatives only. BFGS has proven good performance even for non-smooth
optimizations. This method also returns an approximation of the Hessian inverse,
stored as hess_inv in the OptimizeResult object.

We have more option to take, but the inputs required will change in the same way
as we change the optimisation method.


This function returns an object that stores within it information such as:
    - The jacobian
    - The hessian
    - The hessp
    .
    .
    .
And many other mathematical calutions. Some of them will be given if the solver allow it.

"""
mymin = minimize(eqn, 0, method='BFGS')

print(mymin)

"""
We saw fuction for local optimization, but there are other fuction that we can
use for global ptimization such as basinhopping

There most important functions are:

- basinhopping(func, x0[, niter, T, stepsize, ...])

Find the global minimum of a function using the basin-hopping algorithm.

- brute(func, ranges[, args, Ns, full_output, ...])

Minimize a function over a given range by brute force.

- differential_evolution(func, bounds[, args, ...])

Finds the global minimum of a multivariate function.

- shgo(func, bounds[, args, constraints, n, ...])

Finds the global minimum of a function using SHG optimization.

- dual_annealing(func, bounds[, args, ...])

Find the global minimum of a function using Dual Annealing.

- direct(func, bounds, *[, args, eps, maxfun, ...])

Finds the global minimum of a function using the DIRECT algorithm.

"""
# objective function


def objective(v):
 x, y = v
 return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20

"""
 First, let's have an overview of the function.
 I'm gonna use some fuctions and matplotlib to get a graphic of the function.
 I's no needed to use the function basinhopping, it's just to see how the function is.
"""

"""
Showing the function aspect
"""
# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
xaxis = arange(r_min, r_max, 0.1)
yaxis = arange(r_min, r_max, 0.1)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
v= [x,y]
# compute targets
results = objective(v)
# create a surface plot with the jet color scheme
figure = pyplot.figure()
axis = figure.gca(projection='3d')
axis.plot_surface(x, y, results, cmap='jet')
# show the plot
pyplot.show()

"""
Finding the global minimun.

This time we are using the basinhopping method, 
but there're many other allowed to use.

This method consists in:
    
Basin-hopping is a two-phase method that combines a global stepping algorithm 
with local minimization at each step. 

As the step-taking, step acceptance, and minimization methods are all customizable, 
this function can also be used to implement other two-phase methods.
"""

# define the starting point as a random sample from the domain
pt = r_min + rand(2) * (r_max - r_min)
# perform the basin hopping search
result = basinhopping(objective, pt, stepsize=0.5, niter=200)
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))

"""
Your results may vary given the stochastic nature of the algorithm or evaluation procedure, 
or differences in numerical precision. 
Consider running the example a few times and compare the average outcome.

In this case, we can see that the algorithm located the optima with inputs very close to zero 
and an objective function evaluation that is practically zero.

"""


# SciPy Sparse Data

# Sparse data is data that has mostly unused elements (elements that don't carry any information ).
# It can be an array like this one:
# [1, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0]

# SciPy Graphs

# SciPy Spatial Data

# SciPy Matlab Arrays

# SciPy Interpolation

# SciPy Significance Tests
