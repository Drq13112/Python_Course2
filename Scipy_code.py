# -*- coding: utf-8 -*-
"""
@author: David Redondo Quintero
Github: Drq13112
"""

from scipy.optimize import basinhopping
import numpy as np
from numpy.random import rand
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from scipy import constants
from numpy import meshgrid
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import root, minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
import scipy.interpolate as inter

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
v = [x, y]
# compute targets
results = objective(v)
# create a surface plot with the jet color scheme
figure = plt.figure()
axis = figure.gca(projection='3d')
axis.plot_surface(x, y, results, cmap='jet')
# show the plot
plt.show()

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

# "Sparse data" is data that has mostly unused elements (elements that don't carry any information ).
# It can be an array like this one:
# [1, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0]

# As we can see, It's a datatype which is mostly made by zeros. So, why we need this kind of data?
# In scientific computing, when we are dealing with partial derivatives in linear algebra
# we will come across sparse data.

# We have two options of Sparse Data:
# CSC - Compressed Sparse Column. For efficient arithmetic, fast column slicing.
# CSR - Compressed Sparse Row. For fast row slicing, faster matrix vector products

arr = np.array([1, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0])

print(csr_matrix(arr))


"""
From the result we can see that there are 3 items with value.

The 1. item is in row 0 position 0 and has the value 1.

The 2. item is in row 0 position 2 and has the value 2.

The 3. item is in row 0 position 5 and has the value 3.

"""

# This is a really quick and effienctly way to browse through a numpy array full of zeros.
# We have plenty of methods that we can use with this data type, such as:

arr = np.array([[0, 0, 1], [0, 3, 1], [1, 0, 2]])
print("array in csr format:")
print(csr_matrix(arr))

csr_matrix(arr).count_nonzero()
csr_matrix(arr).eliminate_zeros()

# Converting from csr to csc with the tocsc() method:
newarr = csr_matrix(arr).tocsc()
print("array in csc format:")
print(newarr)
# You can notice that by changing the format, we are printing  the position of each number differently.


# SciPy Graphs

# Well, graphs are essential in data structures, owing to this fact, scipy provides us with a module devoted
# only to graphs.

# Using this module we can define graphs through the adjency matrix. It is an especial matrix whose values
# represent the connection between the elements:

arr = np.array([
    [0, 1, 2],
    [1, 0, 0],
    [2, 0, 0]
])

"""
 This matrix is representing something like this: 
     
      A B C
   A:[0 1 2]  
   B:[1 0 0]
   C:[2 0 0]
   
For a graph like this, with elements A, B and C, the connections are:

A & B are connected with weight 1.

A & C are connected with weight 2.

C & B is not connected.

If one take in the details of this matrix, one will realise that we have a symmetrical matrix.
This actually makes a lot of sense since it's representing the connections between the nodes.
"""
# Once we have designed our graph, we can play a little bit with it.
# This module provides us with the method "dijkstra", which allows us to find the shortest path between the two nodes who are in the vertixes of the matrix.
# The idea that is behind this algorithm is the computer will explore all the shorter paths which take to the goal node.

# Inputs:
"""
csgrapharray, matrix, or sparse matrix, 2 dimensions
The N x N array of non-negative distances representing the input graph.

directed : bool, optional
If True (default), then find the shortest path on a directed graph: 
Only move from point i to point j along paths csgraph[i, j] and from point j to i along paths csgraph[j, i]. 

If False, then find the shortest path on an undirected graph: 
The algorithm can progress from point i to j or j to i along either csgraph[i, j] or csgraph[j, i].

indices : array_like or int, optional
if specified, only compute the paths from the points at the given indices.

return_predecessors: bool, optional
If True, return the size (N, N) predecesor matrix

unweighted: bool, optional
If True, then find unweighted distances. 
That is, rather than finding the path between each point such that the sum of weights is minimized, 
find the path such that the number of edges is minimized.

limit: float, optional
The maximum distance to calculate, must be >= 0. 
Using a smaller limit will decrease computation time by aborting calculations between pairs that are separated by a distance > limit. 
For such pairs, the distance will be equal to np.inf (i.e., not connected).

min_onlybool, optional

If False (default), for every node in the graph, find the shortest path from every node in indices.
If True, for every node in the graph, 
find the shortest path from any of the nodes in indices (which can be substantially faster).

Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html

"""


M = np.array([[ 0,  7,  9,  0, 0, 14],
              [ 7,  0, 10, 15, 0,  0],
              [ 9, 10,  0, 11, 0,  2],
              [ 0, 15, 11,  0, 6,  0],
              [ 0,  0,  0,  6, 0,  9],
              [14,  0,  2,  0, 9,  0]])

"""             A   B   C   D  E   F
            A[[ 0,  7,  9,  0, 0, 14],
            B[ 7,  0, 10, 15, 0,  0],
            C[ 9, 10,  0, 11, 0,  2],
            D[ 0, 15, 11,  0, 6,  0],
            E[ 0,  0,  0,  6, 0,  9],
            F[14,  0,  2,  0, 9,  0]]

Thus, with this method we will find the shortest path between the node A and the node F.

"""


D, Pr = dijkstra(
    M, directed=False, indices=0, return_predecessors=True)

"""
D will show the shortest-distancies-matrix.
D[i, j] gives the shortest distance from node i to node j 
and Pr[i, j] gives the index of the previous node in the shortest path from point i to point j. 
Pr[i,j] = -9999 if there isn't any path from node i to node j. 
"""
print("dist_matrix:", D)
print("predecessor_matrix:", Pr)

"""
Paying attention to "D", we will see that it's an array.
It should look like this: [ 0.  7.  9. 20. 20. 11.]

The stored numbers show the shortest distant between the node A and the rest of the graph.
For example, the shortest distant between A and F is 11. This is the addition of 9+2, because this path
consists of going to the node C and going to the node F, due to the fact that, this path is shorter than going from A to F straight away. 


"""

#Once we know this information, we can design a quick algorithm to make a list with the nodes which made the best path from i to j whatever would be i and j

"""
def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]
"""

# SciPy Spatial Data

"""
The kind of varaible that we'll fnd here is meant to work with geometric spaces.
For instance, points on a coordinate space.
"""

# Triangulation is the first part of this topic that we are going to learn
# Triangulation of a polygon or a shape consists in dividing it into multiple triangules.
# A Triangulation with points means creating surface composed triangles in 
# which all of the given points are on at least one vertex of any triangle in the surface.

# One of the methods we can use is Delaunay().


# Declaring some points in a 2D space 
points = np.array([
  [2, 4],
  [3, 4],
  [3, 0],
  [2, 2],
  [4, 1]
])

simplices = Delaunay(points).simplices # The simplices property creates a generalization of the triangle notation.

# Once we have the result, we can plot it to see it better instead of analysing it directly.

# Triplot draws a unstructured triangular grid as lines and/or markers.
plt.triplot(points[:, 0], points[:, 1], simplices)
plt.scatter(points[:, 0], points[:, 1], color='r')

plt.show()


# Otherwise, if we want to cover a lot of points with just one polygon made of same of these points
# We can use the method convexhull()


points = np.array([
  [2, 4],
  [3, 4],
  [3, 0],
  [2, 2],
  [4, 1],
  [2, 3],
  [1, 3]
])
hull = ConvexHull(points)
hull_points = hull.simplices

plt.scatter(points[:,0], points[:,1])
for simplex in hull_points:
  plt.plot(points[simplex,0], points[simplex,1], 'k-')

plt.show()

# Well, there is a long amount of method we can use to work a coordinate space. 
# We won't take in every one of them, so it will be mentioned some of them below.
# Of course, you can check the scipy website to discover more available methods 
"""
- KDTrees, it's a datastructure optimized for nearest neighbor queries:
In a set of points using KDTrees we can efficiently ask which points are nearest to a certain given point.
One of its methos is query(), which allows us to know the distance to the nearest distance to a given point

- Euclidean() returns the euclidioan distance between given points

- Cityblock() show the manhattan distance between 2 points

- Cosine() finds the cosine distsance between given point.

"""

# SciPy Interpolation

# Let's catch a glimpse of the interplotation posibilities this module provides us:
# The first one is interp1d which is used to interpolate a distribution with 1 variable.

x = np.linspace(0, 10, 30)
y = np.sin(0.5*x)*np.sin(x*np.random.randn(30))

interp_func = inter.interp1d(x, y)
newarr = interp_func(x)

fig = plt.figure()
ax = fig.subplots()
ax.scatter(x, y)
ax.plot(x, newarr, 'r')
plt.show()

"""
As we can see, the result is a piecewise function which connect all the points give
Despite being usefuf, this method is returning a function which is not deriavable.

So best solution to this problem is using splines.

"""
# Now we'll try with the spline interpolation.
# In 1D interpolation the points are fitted for a single curve,
# whereas in Spline interpolation the points are fitted against 
# a piecewise function defined with polynomials called splines.
"""
Once we have defined the initial set of data points, 
we can call the function .UnivariateSpline(), 
from the Scipy package and calculate the spline that best fits our points.

We have some parameters that we can set in this methods:
    
We have the 's' and the 'k' parameters. 
-'s' represents the smoothing factor of the spline. 
If we have a large smoothing factor, the spline will have curves with large radiuses and will probably not reach all points.
Otherwise, if the smoothing factor is small, the function will make more pronounced movements that will allow it to reach most of the given points.

-'k' indicates the grade of the spline. “k” can vary between one and five. 
increasing the degree of the polynomials allows a better fitting of more complicated functions. 
However, in order not to introduce artifacts in our fit.
The best practice is to use the lower degree that allows for the better fitting procedure.
"""

#Creating the spline 1
spline = inter.UnivariateSpline(x, y, s = 5) 
xs = np.linspace(-3, 3, 1000)
x_spline = np.linspace(0, 10, 1000)
y_spline = spline(x_spline)

#Creating the spline 2,less smothig but better fitting
spline = inter.UnivariateSpline(x, y, s = 0.05) 
xs = np.linspace(-3, 3, 1000)
y_spline2 = spline(x_spline)

#Creating the spline 2,less smothig but better fitting
spline = inter.UnivariateSpline(x, y, s = 0.05,k=5) 
xs = np.linspace(-3, 3, 1000)
y_spline3 = spline(x_spline)


#Plotting
fig = plt.figure()
ax = fig.subplots()
ax.scatter(x, y)
ax.plot(x_spline, y_spline, 'g')
ax.plot(x_spline, y_spline2, 'b')
ax.plot(x_spline, y_spline3, 'r')
plt.show()

# As we can see, it returns a function that is able to connect most of the points
# and it's derivable along all its domain.

# SciPy Significance Tests
