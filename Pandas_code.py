"""
First of all, we need to shed a little light to the Pandas and NumPy relation:
    
Pandas is built on top of NumPy, which means the Python pandas package depends on the NumPy 
package and also pandas intended with many other 3rd party libraries.
So we can say that Numpy is required for operating the Pandas


Pandas is wide used in data analysis:
- This library lets us manage data estructures such as :
    - NumPy arrays
    - CSV files
    - Excel files
    - SQL data bases
- It allows access to data via indexes or names for rows and columns.
- It offers methods for sorting, spliting and combine data sets.
- It brings the posibility of working with temporal series.
- It's pretty efficient.

Inside this library we can work with Series, Dataframes and Panels:
    
Series: One-dimensional structure.
DataFrame: Two-dimensional structure (tables).
Panel: Three-dimensional structure (cubes).
"""
import pandas as pd
from math import log

"""
We have two differents ways to access to series values:

Access by position:  
This is done in a similar way to accessing the elements of an array.
s[i] : Returns the element occupying position i+1 in the series s.
s[positions]: Returns another array with the elements occupying the positions of the list positions.

Access by index:
s[name] : Returns the element with the name name name in the index.
s[names] : Returns another series with the elements corresponding to the names indicated in the list names in the index.

As we've seen during the NumPy session, we can have many aspects and information from it.

s.size : Returns the number of elements in the series s.
s.index : Returns a list of the names of the rows of the DataFrame s.
s.dtype : Returns the data type of the elements of the series s.

s.count() : Returns the number of elements that are neither null nor NaN in the series s.
s.sum() : Returns the sum of the data in the series s when the data are of numeric type, or the concatenation of them when they are of string type str.
s.cumsum() : Returns a series with the cumulative sum of the data in the series s when the data is of a numeric type.
s.value_counts() : Returns a series with the frequency (number of repetitions) of each value in the series s.
s.min() : Returns the smallest of the data in the series s.
s.max() : Returns the largest of the data in the series s.
s.mean() : Returns the mean of the data in the series s when the data is of a numeric type.
s.var() : Returns the variance of the data in series s when the data is of a numeric type.
s.std() : Returns the standard deviation of the data of the series s when the data is of a numeric type.
s.describe(): Returns a series with a descriptive summary including the number of data, their sum, minimum, maximum, mean, standard deviation and quartiles.

Now let's have some operation with Pandas Series:
"""
s = pd.Series([1, 2, 3, 4])
print(s*10)
"""
Notice that python will plot pandas series this way:
    
position    value    
0           2
1           4
2           6
3           8
"""

#As we can see, the mathematical operation applies to all the series elements.
# In case we wanted to do a more complicated operation, we can define a fuction and apply it to the series instead
print(s.apply(log))

"""
In the same way we can filter the series with a condition.
If we use the condition s>2; we'll get all th values greater than 2. 
"""
print(s[s > 2])

# If our series is not sorted in ascending order, we can do it with the fuction below:
print(s.sort_values())
    


"""
The series are pretty useful, but waht happen when we want to store more than one pieces of information to one the indexes
Well, then we have the Dataframes which will 
"""
datos = {'name':['María', 'Luis', 'Carmen', 'Antonio'],
'age':[18, 22, 20, 21],
'degree':['Economy', 'Medicine', 'Architecture', 'Economy'],
'email':['maria@gmail.com', 'luis@yahoo.es', 'carmen@gmail.com', 'antonio@gmail.com'],
'height':[1.64, 1.77, 1.7, 1.85],
'weight':[60, 75, 64, 80],
'gender':['F','M','F','M']}

df = pd.DataFrame(datos)
print(df)


print("Data:")
print('Sample size',df.age.count())  # Sample size
print('Mean',df.age.mean())  # Mean
print('Variance',df.age.var())  # Variance
print('Standart Variance',df.age.std())  # Standart Variance
print('')
print('Covariance Matrix:') # Covariance Matrix
print(df.cov())

# We can also obtain a certain type of data, for instance, 
# in this case we only want to get the males.
print(df.groupby('gender').get_group('M'))



"""
Well, for now, we already know how to create a DataFrame from a dictionary.
However, w'll sure have to get the data from extrenal files, such as CSV files or JSON files.
Let's have a little view of how can we do that'
"""
# This line read the data file:
df_csv = pd.read_csv('data.csv')
df_json = pd.read_json('data.json')

print(df_csv)
print(df_json)

"""
As easy as we've just saw, we can import a data set in JSON or CSV format
"""

