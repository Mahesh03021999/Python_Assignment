
# ### Firstly, import these fundamental libraries we'll need for the task
# pip install bokeh
import pandas as pd
import numpy as np


# ### Importing all visualization libraries which we will need in task
# 

import matplotlib.pyplot as plt  # importing matplotlib

plt.figure(figsize=(18,9)) #setting figure size
import seaborn as sns       #import seaborn
sns.set()
from bokeh.plotting import figure, output_file, show
from bokeh.plotting import figure, output_file, show #using bokeh library




# pip install pandas_bokeh

# Ignore warnings from function usage
import warnings;
import numpy as np
warnings.filterwarnings('ignore')


# ### Loading the all three train,test and ideal data sets in jupyter notebook


ideal = pd.read_csv("ideal.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# ## Understanding the Dataset

# ### printing shape of the data


print(test.shape) 
print(train.shape) 
print(ideal.shape)


# ### Test dataset consists two columns and 100 rows.¶
# ### Train dataset consists five columns and 400 rows.
# ### Ideal dataset consists fifty one columns and 400 rows.
# ### each one contains x-y pairs of values

train.head(4) # printing 4 rows of the dataset using head method

test.head(4)
# printing 4 rows of the test dataset using head method


ideal.head(4)
# printing 4 rows of the ideal dataset using head method


## printing last 4 rows of train,test and ideal dataset respectively
print(train.tail(4))
print(test.tail(4))
print(ideal.tail(4))


# ### Implementing the info method to the Train, Test, and Ideal datasets to evaluate the dataset's data types

print(train.info())


print(test.info())

print(ideal.info())

#display the Train data set's statistics analysis!
train.describe()



#display the Test data set's statistics analysis!
test.describe()

#display the Ideal data set's statistics analysis!
ideal.describe()

from bokeh.plotting import figure, output_file, show

# output HTML file
output_file("line.html")

p = figure(title='Plot between test values of x and y',width=500, height=500)

# add a triangle renderer with a size, color, and alpha
p.triangle(test['x'],test['y'], size=20, color="cyan", alpha=0.5)

# showing result
show(p)

# drawing pair plot for y values of train dataframe
sns.pairplot(train,hue="x")
plt.show()

# drawing pair plot for train dataframe
sns.pairplot(train)
plt.show()

sns.pairplot(test,palette='green')
plt.show()

# heatmap for x vs 50 ideal ideal functions
plt.figure(figsize=(60,30))
sns.heatmap(ideal.corr(), annot = True)
plt.show()


#defining variables to calculate mean squared error between y values of train and ideal
predict_train=np.array(list(train.y1))
labl_ideal=np.array(list(ideal.y1))

from sklearn.metrics import mean_squared_error 
# Evaluate the Mean Squared Error (MSE)
_Mse = mean_squared_error(predict_train,labl_ideal)

columns_of_ideal=list(ideal.columns)
print(columns_of_ideal)
columns_ideal_y=columns_of_ideal.pop(0)
print(columns_of_ideal)


list_of_err = []
for column in columns_of_ideal:
    predict_train = np.array(list(train.y1))
    lab_ideal = np.array(list(ideal[column]))
    _Mse = mean_squared_error(predict_train,labl_ideal)
    lab_err_tup = column, _Mse
    list_of_err.append(lab_err_tup)


def Tuple_sorting(tup):
    '''
        Function that arranges the list according to the second tuple item.
        written code will sort a list of tuples with the second Item using the 
        sorted() function.
        The key in this case is set upto the sort using lambda,
        sublist lambda is used.
        Here reverse = None (Sorts in Ascending order)
     '''
    return(sorted(tup, key = lambda a: a[1]))



list_of_err = []
for column in columns_of_ideal:
    predict_train = np.array(list(train.y1))
    lab_ideal = np.array(list(ideal[column]))
    _Mse = mean_squared_error(predict_train,lab_ideal)
    lab_err_tup = column, _Mse
    list_of_err.append(lab_err_tup)

err_lst_sort = Tuple_sorting(list_of_err)
err_lst_sort[0:7]



columns_of_train=list(train.columns)
print(columns_of_train)
columns_train_y=columns_of_train.pop(0)
print(columns_of_train)


'''
  Calculate lowest mean square error value by using following function 
'''
def lowest_mse(train_feat, data_feat):
    err_lst = []
    for i in list(data_feat.columns[1:]):
        predict = np.array(list(data_feat[i]))
        labeled = np.array(list(train[train_feat]))
        error = mean_squared_error(predict, labeled)
        lab_err_tup= i, error
        err_lst.append(lab_err_tup)

    sort_err_lst = Tuple_sorting(err_lst)
    return sort_err_lst[0]


set_of_best_label = set()
for train_col in columns_of_train:
    best_label = lowest_mse(train_col,ideal)
    set_of_best_label.add(best_label)



set_of_best_label


'''
    This class gives the output 
    as dictionary of key value pairs as
    train data set y values i.e. y1,y2,y3,y4
    and values as selected best values of ideal
    i.e. y18,y29,y34,y7
'''
class map:
    def __init__(self):
        pass

    def bestest_labe_map(train_dt, ideal_dt):
        set_of_best_label = set()
        for train_col in columns_of_train:
          best_label = lowest_mse(train_col,ideal)
          set_of_best_label.add(best_label)
        col_list = list(columns_of_train)
    
        return tuple((zip(col_list,set_of_best_label )))


# creating object named err_map_dict of map class
# calling best_label_mapping with the object
err_map_dict= map.bestest_labe_map(train, ideal)
err_map_dict


# ### Creating inheritance to find four ideal functions
# ### and map them with train values



'''
creating a base class(Parent class) named Sel_bst_feat 
and child class named find_lowest_mse 
creat an object of child class and calling method of base class
'''
class Sel_bst_feat:
    def __init__(self, trn_dt, idl_dt):
        self.trn_lngth = trn_dt.shape[0]
        self.idl_lngth = idl_dt.shape[0]
        self.trn_trgt_feat = list(trn_dt.columns[1:])
        self.idl_trgt_feat = list(idl_dt.columns[1:])
        self.idl_trgt_dt = ideal.iloc[0:,1:]
        self.trn_trgt_dt = train.iloc[0:, 1:]
        self.idl_feat_pred = ideal.iloc[0:, 0:1]
        self.trn_feat_pred = train.iloc[0:, 0:1]
        
    def low_mse(train_feat):
       err_lst = []
       for i in list(train.columns[1:]):
        predicted = np.array(list(train[i]))
        labeled = np.array(list(train[train_feat]))
        err = mean_squared_error(predicted, labeled)
        lbl_err_tup = i, err
        err_lst.append(lbl_err_tup)

        sort_err_lst = Tuple_sorting(err_lst)
        return sort_err_lst[0]
    
class find_lowest_meansquarerror(Sel_bst_feat):
    def low_mean_se_labels(self):
        set_of_best_label = set()
        for train_col in columns_of_train:
            best_label = lowest_mse(train_col,ideal)
            set_of_best_label.add(best_label)
        col_list = list(columns_of_train)
    
        return dict((zip(col_list,set_of_best_label )))


new_class_object = find_lowest_meansquarerror(train, ideal)
new_class_object.low_mean_se_labels()




'''
Creating class named Max_Devi which calculates
deviation of four train values and calculated 
ideal values of y
'''

class Max_Devi :
  def __init__ (self):
    pass
  def deviation(predict, label):
    maxdev = np.max(np.abs((predict-label)))
    return maxdev

train_y1_deviation =  Max_Devi.deviation((np.array(list(train.y1))),(np.array(list(ideal.y29))))
train_y2_deviation =  Max_Devi.deviation((np.array(list(train.y2))),(np.array(list(ideal.y34))))
train_y3_deviation =  Max_Devi.deviation((np.array(list(train.y3))),(np.array(list(ideal.y7))))
train_y4_deviation =  Max_Devi.deviation((np.array(list(train.y4))),(np.array(list(ideal.y18))))


print((train_y1_deviation))
print((train_y2_deviation))
print((train_y3_deviation))
print((train_y4_deviation))


allowed_deviation = [train_y1_deviation,train_y2_deviation,train_y3_deviation,train_y4_deviation]
print(allowed_deviation)
values_of_train = ['y1','y2','y3','y4']
print(values_of_train)
tup_train_dev=tuple(zip(values_of_train,allowed_deviation))
print(tup_train_dev)


for i,j in tup_train_dev:
  print(f'Deviation allowed for {i} train is {j*(np.sqrt(2))}')



table_map=dict(ideal[['x','y16']].values) # mapping between test data and slected ideal data
test['y16']=test.x.map(table_map)
test.head()


from numpy import math   
#  importing math library


dev=((test.y)-(ideal.y34))
test['dev']=abs(dev)
test.tail(6)

test.describe()




import pandas as pd
from bokeh.plotting import figure, show

# Create a scatter plot of column 'x' and column 'y'
p = figure(title="Scatter Plot of Test DataSet values x and y ")
p.scatter(x=test['x'], y=test['y'], size=9, color='green')

# Show the plot
show(p)




import pandas as pd
from bokeh.plotting import figure, show

# Create a scatter plot of column 'x' and column 'y'
p = figure(title="Scatter Plot of Train DataSet values x and y1")
p.scatter(x=train['x'], y=train['y1'], size=9, color='orange')

# Show the plot
show(p)


# Create a scatter plot of column 'x' and column 'y'
p = figure(title="Scatter Plot of Train DataSet values x and y2")
p.scatter(x=train['x'], y=train['y2'], size=9, color='red')

# Show the plot
show(p)



# Create a scatter plot of column 'x' and column 'y'
p = figure(title="Scatter Plot of Train DataSet values x and y3")
p.scatter(x=train['x'], y=train['y3'], size=9, color='magenta')

# Show the plot
show(p)




# Create a scatter plot of column 'x' and column 'y'
p = figure(title="Scatter Plot of Train DataSet values x and y4")
p.scatter(x=train['x'], y=train['y4'], size=9, color='black')

# Show the plot
show(p)

import pandas as pd
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot

# Create a scatter plot of column 'x' and column 'y' for mapped test data
p1 = figure(title="Ideal Data Set x vs y16")
p1.scatter(x=ideal['x'], y=ideal['y16'], size=10, color='yellow')



import pandas as pd
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot


# Create a scatter plot of column 'x' and column 'y' for mapped test data
p1 = figure(title="Ideal Data Set x vs y34")
p1.scatter(x=ideal['x'], y=ideal['y16'], size=9, color='red')

# Create a scatter plot of column 'x' and column 'y' for mapped test data
p2 = figure(title="Ideal Data Set x vs y18")
p2.scatter(x=ideal['x'], y=ideal['y17'], size=9, color='cyan')

# Create a scatter plot of column 'x' and column 'y' for mapped test data
p3 = figure(title="Ideal Data Set x vs y29")
p3.scatter(x=ideal['x'], y=ideal['y27'], size=9, color='green')

# Create a scatter plot of column 'x' and column 'y' for mapped test data
p4 = figure(title="Ideal Data Set x vs y7")
p4.scatter(x=ideal['x'], y=ideal['y48'], size=9, color='blue')

# Show the plot

show(p1)
show(p2)
show(p3)
show(p4)


# plot between train x,train y1 and ideal x and ideal y29

plt.rcParams["figure.figsize"] = [14, 7]
plt.rcParams["figure.autolayout"] = True
ax = train.plot(x='x', y='y1')
ideal.plot(ax=ax, x='x', y='y15')


# plot between train x,train y2 and ideal x and ideal y34
plt.rcParams["figure.figsize"] = [14, 7]
plt.rcParams["figure.autolayout"] = True
ax = train.plot(x='x', y='y2')
ideal.plot(ax=ax, x='x', y='y27')


# plot between train x,train y3 and ideal x and ideal y7

plt.rcParams["figure.figsize"] = [14, 7]
plt.rcParams["figure.autolayout"] = True
ax = train.plot(x='x', y='y3')
ideal.plot(ax=ax, x='x', y='y48')


# plot between train x,train y4 and ideal x and ideal y18

plt.rcParams["figure.figsize"] = [14, 7]
plt.rcParams["figure.autolayout"] = True
ax = train.plot(x='x', y='y4')
ideal.plot(ax=ax, x='x', y='y16')


plt.figure(1)
# The subplot() command specifies numrows, numcols, 
# fignum where fignum ranges from 1 to numrows*numcols.
plt.subplot(221)
plt.grid()
plt.plot(train['x'], train['y1'], 'b-')
plt.subplot(222)
plt.plot(ideal['x'], ideal['y15'], 'r--')
plt.show()


plt.figure(1)
# The subplot() command specifies numrows, numcols, 
# fignum where fignum ranges from 1 to numrows*numcols.
plt.subplot(221)
plt.grid()
plt.plot(train['x'], train['y2'], 'b-')
plt.subplot(222)
plt.plot(ideal['x'], ideal['y27'], 'r--')
plt.show()



plt.figure(1)
# The subplot() command specifies numrows, numcols, 
# fignum where fignum ranges from 1 to numrows*numcols.
plt.subplot(221)
plt.grid()
plt.plot(train['x'], train['y3'], 'b-')
plt.subplot(222)
plt.plot(ideal['x'], ideal['y48'], 'r--')
plt.show()



plt.figure(1)
# The subplot() command specifies numrows, numcols, 
# fignum where fignum ranges from 1 to numrows*numcols.
plt.subplot(221)
plt.grid()
plt.plot(train['x'], train['y4'], 'b-')
plt.subplot(222)
plt.plot(ideal['x'], ideal['y16'], 'r--')
plt.show()

plt.figure(1)


# The subplot() command specifies numrows, numcols, 
# fignum where fignum ranges from 1 to numrows*numcols.

plt.subplot(421)
plt.grid()
plt.plot(train['x'], train['y1'], 'b-')
plt.title("Graph for Train Value y1")

plt.subplot(422)
plt.plot(ideal['x'], ideal['y15'], 'g--')
plt.title("Graph for Ideal values y29")

# The subplot() command specifies numrows, numcols, 
# fignum where fignum ranges from 1 to numrows*numcols.

plt.subplot(423)
plt.grid()
plt.plot(train['x'], train['y2'], 'c-')
plt.title("Graph for Train Value y2")

plt.subplot(424)
plt.plot(ideal['x'], ideal['y27'], 'm--')
plt.title("Graph for Ideal values y34")

# The subplot() command specifies numrows, numcols, 
# fignum where fignum ranges from 1 to numrows*numcols.

plt.subplot(425)
plt.grid()
plt.plot(train['x'], train['y3'], 'r-')
plt.title("Graph for Train Value y3")

plt.subplot(426)
plt.plot(ideal['x'], ideal['y48'], 'y--')
plt.title("Graph for Ideal values y7")

# The subplot() command specifies numrows, numcols, 
# fignum where fignum ranges from 1 to numrows*numcols.

plt.subplot(427)
plt.grid()
plt.plot(train['x'], train['y4'], 'b-')
plt.title("Graph for Train Value y4")

plt.subplot(428)
plt.plot(ideal['x'], ideal['y16'], 'k--')
plt.title("Graph for Ideal values y18")

from sqlalchemy import create_engine, Column,ForeignKey, Integer, String, Float, CHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
Base = declarative_base()


import sqlite3


# create a SQLite database in memory
connection_obj=sqlite3.connect('Train4.db')
train.to_sql('Train_DB4',connection_obj)


engine=create_engine("sqlite:///train_database.db",echo=True)
Base.metadata.create_all(bind=engine)
Session=sessionmaker(bind=engine)
session=Session()



cursor_obj=connection_obj.cursor()

statement='''SELECT * FROM TRAIN_DB4'''

cursor_obj.execute(statement)

print("All the data")
output=cursor_obj.fetchall()
for row in output:
   print(row)

connection_obj.commit()

# define the base class for our models

df_train=pd.read_sql_query('''Select * FROM Train_DB4''',connection_obj)
df_train



connection_obj=sqlite3.connect('Test3.db')
test.to_sql('Test_DB3',connection_obj)

engine=create_engine("sqlite:///test_database.db",echo=True)
Base.metadata.create_all(bind=engine)

Session=sessionmaker(bind=engine)
session=Session()


cursor_obj=connection_obj.cursor()
statement='''SELECT * FROM TEST_DB3'''

cursor_obj.execute(statement)

print("All the data")
output= cursor_obj.fetchall()
for row in output:
   print(row)

connection_obj.commit()




df_test=pd.read_sql_query('''SELECT * FROM Test_DB3''',connection_obj)
df_test

connection_obj=sqlite3.connect('Ideal3.db')
ideal.to_sql('Ideal_DB3',connection_obj)


engine=create_engine('sqlite:///ideal_database.db',echo=True)
Base.metadata.create_all(bind=engine)

Session=sessionmaker(bind=engine)
session=Session()

cursor_obj=connection_obj.cursor()

statement='''Select * FROM Ideal_DB3'''

cursor_obj.execute(statement)

print("All the data")
output=cursor_obj.fetchall()
for row in output:
   print(row)

connection_obj.commit()

df_ideal=pd.read_sql_query('''Select * FROM Ideal_DB3''',connection_obj)
df_ideal





