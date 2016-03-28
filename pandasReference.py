# Pandas Learning
# http://pandas.pydata.org/pandas-docs/stable/10min.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Index 1-5 of [1.0, 3.0, 5.0, NaN, 6.0, 8.0]
s = pd.Series([1,3,5,np.nan,6,8])

# Creates 6 dates starting from 2013/01/01
# ['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04', '2013-01-05', '2013-01-06']
dates = pd.date_range('20130101', periods=6)

# Creates random number set with rows labeled with dates, columns as 'A', 'B', 'C', 'D'
# 6,4 is shape of random rows,columns
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

# DataFrame from dict instead of array, this has dtypes differing for every column
df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })
#      A          B    C  D      E    F
# 0  1.0 2013-01-02  1.0  3   test  foo
# 1  1.0 2013-01-02  1.0  3  train  foo
# 2  1.0 2013-01-02  1.0  3   test  foo
# 3  1.0 2013-01-02  1.0  3  train  foo

# Displaying DataFrame info
df2.head() 		# first columns
df2.tail(3) 	# last 3 columns
df2.index 		# array of indecies [0, 1, 2, 3]
df2.columns 	# array of column names [u'A', u'B', a'C', u'D'...]
df2.values 		# 2d array of values ignoring indices or column names
df2.describe 	# Summary of mean, count, std, min, max, % ranges for 25 50 75

# DataFrame Transformations
df2.T 			# Transpose
df2.sort_index(axis=1, ascending=false) # Sort by index
df2.sort_values(by='B') # Sorts rows in ascending order of B
df2['E'] = ['one', 'one','two','three','four','three'] # adds new column

# Getting values
df['A'] # Returns column 'A' with indices
df[0:3] # first 3 rows
df['20130102':'20130104'] # rows by indices
df.loc[dates[0]] # get columbs (A-D) for first date as one column
df.loc[:,['A','B']] # only use columns A, B
df.loc['20130102':'20130104',['A','B']] # get 2x2 of those dates A, B
df.loc['20130102',['A','B']] # get A,B for this date as one column
df.loc[dates[0],'A'] # get scalar value at index
# iloc works same but by index not labels

# Filtering
df[df.A > 0] # Only entries where A > 0
df[df > 0] # < 0 entries get replaces with NaN
df2[df2['E'].isin(['two','four'])] # selectes where 'E' is in ['two', 'four']








