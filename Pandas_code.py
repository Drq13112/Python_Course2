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
import numpy as np
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
As easy as we've just saw, we can import a data set in JSON or CSV format.


Let's do same exercises to consolidate what we've learnt
"""
"""
Write a Pandas program to create and display a DataFrame from a specified dictionary data 
which has the index labels.

Sample DataFrame:
    
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

Sample Output:

   attempts       name qualify  score                                  
a         1  Anastasia     yes   12.5                                  
b         3       Dima      no    9.0                                  
c         2  Katherine     yes   16.5                                  
d         3      James      no    NaN                                  
e         2      Emily      no    9.0                                  
f         3    Michael     yes   20.0                                  
g         1    Matthew     yes   14.5                                  
h         1      Laura      no    NaN                                  
i         2      Kevin      no    8.0                                  
j         1      Jonas     yes   19.0 
"""
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(exam_data , index=labels)

print(df)



"""
Continuing with the previous exercise, we'll find the rows where the score is missing.
We'll know what of them are NaN, due to its value is np.NaN.
"""
df = pd.DataFrame(exam_data , index=labels)
print("Rows where score is missing:")

#So, we define which row we want to look for in and we especify the value none-defined as 'isnull()'
print(df[df['score'].isnull()])

"""
Once we've found the NaN rows, let's selecte by number of attempents.
For instance, we can selecte the people who have attemped more than 2 times
and  have scored more than 10.
"""
print("Number of attempts in the examination is more than 2 and score greater than 15 :")
print(df[(df['attempts'] > 2) & (df['score'] > 15)])




#HOMEWORK


# Well, now will try to select only the people whose qualify ended up in yes

# We have different ways to have this date, these are some of them
print(df.query('qualify == ["yes"]'))
print(df[(df['qualify'] =="yes")])

"""
If we need to get the lowest records of our data, we can use some certain methods such as these:
df.nsmallest(3, 'score') will show us the 3 smallest values within score group
df.nlargest(3, 'score') will do the sime but with the largest values
"""
print("\nLowest 3 records within each group of a DataFrame:")
print(df.nsmallest(3, 'score'))
print("\nLargest 3 records within each group of a DataFrame:")
print(df.nlargest(3, 'score'))






#Sesion 2:








"""
We've working with a only data set of data, but there's going to be situations when we'll have to merge
differents data sets or split our data set into others.

So, we have some methods that Pndas Library includes to do this operations. Let's see some of them:
"""
# In case of merging two data sets, we can do it along the rows or the columns

student_data1 = pd.DataFrame({
        'student_id': ['S1', 'S2', 'S3', 'S4', 'S5'],
         'name': ['Danniella Fenton', 'Ryder Storey', 'Bryce Jensen', 'Ed Bernal', 'Kwame Morin'], 
        'marks': [200, 210, 190, 222, 199]})

student_data2 = pd.DataFrame({
        'student_id': ['S4', 'S5', 'S6', 'S7', 'S8'],
        'name': ['Scarlette Fisher', 'Carla Williamson', 'Dante Morse', 'Kaiser William', 'Madeeha Preston'], 
        'marks': [201, 200, 198, 219, 201]})

# We are going to use the same method for this action, but will use an argument to let the methos knows which action has to do
# The argument axis=1 will let the method know that it has to merge the data sets along columns
result_data = pd.concat([student_data1, student_data2], axis = 1)
print("result_data:")
print(result_data)
# In case we don't define it (use the default value), the method will do it along rows
result_data = pd.concat([student_data1, student_data2])
print("result_data:")
print(result_data)
# As we can see through the console, the option which makes more sense for our data set, it is the second one

# We  can also use this other method instead of the previous one
combined_data = student_data1.append(student_data2, ignore_index = True)
print("combined_data:")
print(combined_data)

# Okey, now let's do it backwards. We'll split it to get two datasets:
    
# splitting dataframe in a particular size
df_split = combined_data.sample(frac=0.6,random_state=200)
df_split.reset_index()
print("df_split:")
print(df_split)


# Splitting dataframe by row index
df_1 = df.iloc[:2,:]
df_2 = df.iloc[2:,:]
print("Shape of new dataframes - {} , {}".format(df_1.shape, df_2.shape))

print(df_1)
print(df_2)


# splitting dataframe by groups
# grouping by particular dataframe column
grouped = df.groupby(df.qualify)
df_yes = grouped.get_group("yes")
df_no = grouped.get_group("no")
print(df_yes)
print(df_no)


"""
Exercise:
    
    
Column      Name	                        Description
price   ->    price in US dollars           (\$326--\$18,823)
carat   ->    weight of the diamond         (0.2--5.01)
cut     ->    quality of the cut            (Fair, Good, Very Good, Premium, Ideal)
color   ->    diamond colour                from J (worst) to D (best)
clarity ->    a measurement 
              of how clear the diamond is:  (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))
              
              
x       ->    length in mm                  (0--10.74)
y       ->    width in mm                   (0--58.9)
z       ->    depth in mm                   (0--31.8)
depth   ->    total depth percentage        z / mean(x, y) = 2 * z / (x + y) (43--79)
table   ->    width of top of diamond 
              relative to widest point      (43--95)

"""
diamonds = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv')
print(diamonds)


"""
Write a Pandas program to find the diamonds that are with a Fair or Good or Premium.
"""
result = diamonds[diamonds.cut.isin(['Fair', 'Good', 'Premium'])]
print(result.head())



"""
Write a Pandas program to iterate through diamonds DataFrame
"""
print("\nIterate through diamonds DataFrame:")
for index, row in diamonds.iterrows():
   print(index, row.carat, row.cut, row.color, row.price)
   
   
   
"""
Write a Pandas program to calculate count, minimum, maximum price for each cut of diamonds DataFrame
"""

print("\nCount, minimum, maximum  price for each cut of diamonds DataFrame:")
print(diamonds.groupby('cut').price.agg(['count', 'min', 'max']))


"""
Write a Pandas program to calculate the multiply of length, width and depth for each cut of diamonds DataFrame
"""

print("\nMultiply of length, width and depth for each cut:")
print((diamonds.x*diamonds.y*diamonds.z).head())

"""
Write a Pandas program to read the diamonds DataFrame and detect duplicate color.

Note: duplicated () function returns boolean Series denoting duplicate rows, optionally only considering certain columns.
"""

print("\nCount the duplicate items:")
print(diamonds.clarity.duplicated().sum())


#HOMEWORK

"""
Write a Pandas program to calculate various summary statistics of cut series of diamonds DataFrame.
"""
print("\nVarious summary statistics of diamonds DataFrame:")
print(diamonds.carat.describe())