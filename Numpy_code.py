import Numpy as np

"""
The main advantage of numpy over the default lists provided is that numpy's processing 
is significantly faster than the default lists. 
This makes numpy a great choice for processing arrays and matrices, 
especially when dealing with large dimensions.

Numpy is such a powerful library that is actully been used in meny other relevant libraries.
Some of case studies that Numpy is been used are quantum computing, statistical compunting, 
image processing, astronomy processes, Chemistry, Mathematical Analysis ...

As we can see, Numpy it's a really powerful library with widespread uses.

Let's stop talking about numpy and start programming some exmaples.

First, we will explain what an array object is.
Basically, it is an organised data structure.

In addticon, we can have more than 1 dimension, 
which is normally known as a vector. This time, 
we can have an array of up to 2 or 3 dimensions.


To create a Numpy array, we must initialise it as an object, 
that is, we must call the constructor of the class by giving it a list as an argument.
"""
# Array de una dimensión
a1 = np.array([1, 2, 3])
print(a1)

# Array de dos dimensiones
a2 = np.array([[1, 2, 3], [4, 5, 6]])
# Array de tres dimensiones
a3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(a3)

"""
In case we need a certain tipe of array, we can create it esaly with numpy library. Some exampl e of it:

np.empty(dimensions) : Generate and return a reference to an empty array with the specified dimensions.

np.zeros(dimensions) : Generate and return a reference to an array en la tupla dimensiones full of zeros.

np.ones(dimensions) : Generate and return a reference to an array en la tupla dimensiones full of ones.

np.full(dimensions, value) :Generate and return a reference to an array en la tupla dimensiones full of numbers with the value given.

np.identity(n) :Generate and return a reference to the indenty matrix of n dimensions.

"""
print(np.zeros(3, 2))
print(np.idendity(3))

"""
Once we have created a numpy array, we can use a wide range of method on it.
Some of them are:

a.ndim : Return the dimensions the numpy array has.

a.shape : Return a tuple with the array dimensions.

a.size :  Return the amount of number the array has inside.

a.dtype: Return what data type the array has inside.
"""

a3.ndim()

a3.shape()

a3.size()

a3.dtype()

"""
We can also filter the array elemnts. 
This provide a fast way extract elements from it.
"""
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a[(a % 2 == 0)])
print(a[(a % 2 == 0) & (a > 2)])

"""
Obviusly numpy give us some fuctions that we can use to operate between Numpy Objects.
For instance, if want to do the product of two matrix, it can be sused np.dot(matrix1,matrix2).

Even you can da a cross-product between two arrays with the command np.cross(x,y) or
getting the gradient from an array with the function np.gradient(x)
"""
a = np.array([[1, 2, 3], [4, 8, 16]])

b = np.array([5, 6, 11]).reshape(-1, 1)
c = np.dot(a, b)

x = [1,2,3]
y = [4,5,6]
z = np.cross(x, y)


np.gradient(x)

"""
Once we'd seen some examples and applications of this library, 
let's do some exercise which show us some real utilities for this library.

"""

"""
Write a NumPy program to convert the values of Centigrade degrees into Fahrenheit degrees and vice versa. 
Values are stored into a NumPy array

Sample Array:
Values in Fahrenheit degrees [0, 12, 45.21, 34, 99.91]
Values in Centigrade degrees [-17.78, -11.11, 7.34, 1.11, 37.73, 0. ]

"""
fvalues = [0, 12, 45.21, 34, 99.91, 32]
F = np.array(fvalues)
print("Values in Fahrenheit degrees:")
print(F)

print("Values in  Centigrade degrees:")
print(np.round((5*F/9 - 5*32/9),2))
"""
Write a NumPy program to find the real and imaginary parts of an array of complex numbers

Original array [ 1.0+0.j 0.70710678+0.70710678j]
Real part of the array:
[ 1. 0.70710678]
Imaginary part of the array:
[ 0. 0.70710678]
"""
x = np.sqrt([[1+0j],[0+1j]])
print("Original array:x ",x)
print("Real part of the array:")
print(x.real)
print("Imaginary part of the array:")
print(x.imag)

"""
Write a NumPy program to get the unique elements of an array

Array from which to draw their unique elements
[10, 10, 20, 20, 30, 40]
[[1, 1], [2, 3]]
"""
x = np.array([10, 10, 20, 20, 30, 40])
print("Original array:")
print(x)
print("Unique elements of the above array:")
print(np.unique(x))

x = np.array([[1, 3], [2, 3]])
print("Original array:")
print(x)
print("Unique elements of the above array:")
print(np.unique(x))
"""
HOMEWORK
Write a NumPy program to find the 3th element of a specified array:

[[2, 4, 6], [6, 8, 10]]
"""

x = np.array([[2, 4, 6], [6, 8, 10]], np.int32)
print(x)
# Notice the interpeter starts to count from 0, 
# so that the third position is actually in the second one: 0 1 ->2<- 
e1 = x.flat[2] 
print("Third e1ement of the array:")
print(e1)