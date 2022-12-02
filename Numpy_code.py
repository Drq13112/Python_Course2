import numpy as np

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
a=np.zeros((3, 2))
print("a:",a)

"""
Once we have created a numpy array, we can use a wide range of method on it.
Some of them are:

a.ndim : Return the dimensions the numpy array has.

a.shape : Return a tuple with the array dimensions.

a.size :  Return the amount of number the array has inside.

a.dtype: Return what data type the array has inside.
"""
print(a3.ndim)

print(a3.shape)

print(a3.size)

print(a3.dtype)

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
    
Values in Fahrenheit degrees :
[0, 12, 45.21, 34, 99.91]


Values in Centigrade degrees :
[-17.78, -11.11, 7.34, 1.11, 37.73, 0. ]
"""
fvalues = [0, 12, 45.21, 34, 99.91, 32]
F = np.array(fvalues)
print("Values in Fahrenheit degrees:")
print(F)

print("Values in  Centigrade degrees:")
print(np.round((5*F/9 - 5*32/9),2))








"""
Write a NumPy program to find the real and imaginary parts of an array of complex numbers


Original array:
[ 1.0+0.j 0.70710678+0.70710678j]


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









# Second Session

"""
Well, we have learnt the numpy basics, how to create a numpy array and how we an work with it.

During this session we are going to use the numpy library to perform more complex exercises.
In this way, we will be able to get a better view of how easy 
it is to use this library for programming.
"""









"""

Write a NumPy program to multiply a matrix by another matrix of complex numbers 
and create a new matrix of complex numbers.

Sample output:
    
    
First array:
[ 1.+2.j 3.+4.j]


Second array:
[ 5.+6.j 7.+8.j]


Product of above two arrays:
(70-8j)

"""

x = np.array([1+2j,3+4j])
print("First array:")
print(x)
y = np.array([5+6j,7+8j])
print("Second array:")
print(y)
z = np.vdot(x, y)
print("Product of above two arrays:")
print(z)












"""
Write a NumPy program to create a random array with 1000 elements 
and compute the average, variance, standard deviation of the array elements. 

Sample output:
    
    
Average of the array elements:
-0.0255137240796


Standard deviation of the array elements:
0.984398282476


Variance of the array elements:
0.969039978542
"""
x = np.random.randn(1000)
print("Average of the array elements:")
mean = x.mean()
print(mean)
print("Standard deviation of the array elements:")
std = x.std()
print(std)
print("Variance of the array elements:")
var = x.var()
print(var)









"""
Write a NumPy program to create a structured array from given student name, height, class and their data types. 
Now sort by class, then height if class are equal.
    
Original array:
[(b'James', 5, 48.5 ) (b'Nail', 6, 52.5 ) (b'Paul', 5, 42.1 ) (b'Pit', 5, 40.11)]


Sort by age, then height if class are equal:
[(b'Pit', 5, 40.11) (b'Paul', 5, 42.1 ) (b'James', 5, 48.5 ) (b'Nail', 6, 52.5 )]
"""
data_type = [('name', 'S15'), ('class', int), ('height', float)]
students_details = [('James', 5, 48.5), ('Nail', 6, 52.5),('Paul', 5, 42.10), ('Pit', 5, 40.11)]
# create a structured array
students = np.array(students_details, dtype=data_type)   
print("Original array:")
print(students)
print("Sort by class, then height if class are equal:")
print(np.sort(students, order=['class', 'height']))




"""
Write a NumPy program to convert angles from degrees to radians for all elements in a given array.

Sample Input: [-180., -90., 90., 180.]
Sample Output:[-3.14159265 -1.57079633  1.57079633  3.14159265]
"""
#Two ways:
    
#1º
degrees=np.array([-180.,  -90.,   90.,  180.])
r1 = np.radians(degrees)
r2 = np.deg2rad(degrees)
assert np.array_equiv(r1, r2)
print("radians_1º:",r1)

#2º
radians=np.round((3.141592659*degrees/180),5)
print("radians_2º:",radians)
