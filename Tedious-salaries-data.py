
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# insight:
# 1) there is no year in negative 
# 2) there are 4 type for employee *full time *part time *Contract *Freelance
# 3) salary not normal distribution.
# 4) All Features have a weak correlation with target [ salary ]

# Target => salarry.

# Read Path of data from user.
print('Hello User Enter Path of Data :',end=" ")
var=input()
df=pd.read_csv(var)
print(df)

my_list=[]

print('*'*100)


# Separating categorical and numerical columns
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

# categorical_data = df[categorical_columns]
# numerical_data = df[numerical_columns]

# print(categorical_columns)
# print(df[categorical_columns])


def unique_data():
    print('*'*100)
    print('Let check for Unique data.')
    # my_list=categorical_data
    my_list=categorical_columns
    print(f'Name of Col. {my_list}')
    len_col=len(my_list)
    # for i in range(0,len_col):
        # print(f'number of unique [ {my_list[i]} ] : {df[my_list[i]].nunique()} ') 
    for i in my_list:
        print(f'number of unique [ {i} ] : {df[i].nunique()} ')
        print('Unique Value:')
        print(f'Value of unique [ {i} ] : {df[i].unique()} ')
        print()
        print(df[i].value_counts()) # Q 
        print('-----------------------------------------------------------------------')




# print(my_list) whyyyyyyyyyyyyyyy


def duplicate():
    print(f'Number of duplicated data = {df.duplicated().sum()}')
    if df.duplicated().sum() >0:
        print('[ Drop Duplicated Data. ]')
        df.drop_duplicates(inplace=True)
        print('[ No duplicated data left. ]')

    elif df.duplicated().sum()==0:
        print('[ There are no duplicated data. ]')
    
    # if 'Unnamed: 0' in df.columns:
    #     df.drop('Unnamed: 0',axis=1,inplace=True)
    #     print(df)



def isnull_data():
    print('*'*100)
    print('Let check for missing data.')
    print(f'Number of Missing Data =\n{df.isna().sum()}')


def hist_plot():
    for i in numerical_columns:
        # range=[0, df[i].value_counts()]
        plt.hist(df[i],bins = 10, edgecolor='black')

        plt.xlabel(i)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {i}')

        

def box_plot():
    for i in numerical_columns:
        plt.boxplot(df[i])

        plt.xlabel(i)
        plt.ylabel('Frequency')
        plt.title(f'Box Plot of {i}')

        

# drop all cat_data
def cat_data():
    for i in categorical_columns:
        df.drop(i,axis=1,inplace=True)



# Features with target

def scatterr():
    for i in numerical_columns:
                
        plt.figure(figsize=(8,6))
        plt.scatter(df[i],df['salary'],c='blue',marker='o',label='Data Points')

        plt.xlabel(i)
        plt.ylabel('salary')
        plt.title(f'Scatter Plot {i} vs Salary ')

        plt.legend() # مفتاح الخريطه
        


# categorical_columns
def cat_pie():
    for i in categorical_columns:
        countt=df[i].value_counts()
        plt.figure(figsize=(8,6))
        plt.pie(countt,labels=countt.index,autopct="%1.1f%%",startangle=140)
        plt.axis('equal')
        plt.title(i)
        

# bar plot
def bar_plot():
    for i in categorical_columns:
        countt=df[i].value_counts()
        count_sort=countt.sort_values(ascending=True)
        plt.figure(figsize=(8,6))
        countt.plot(kind='bar',color='skyblue')
        plt.title(i)
        



duplicate()
unique_data()
isnull_data()
hist_plot()
box_plot()
cat_data()
scatterr()
# cat_pie()
# bar_plot()
plt.show()



# Heatmap:-
# Calculate correlation matrix
correlation_matrix = df.corr()

mask= np.triu(np.ones_like(correlation_matrix,dtype=bool)) # gpt

# # Create a heatmap using Matplotlib
plt.figure(figsize=(8, 6))
plt.title('Correlation Matrix')
sns.heatmap(correlation_matrix,annot=True,linecolor='blue',cmap='BuPu',fmt=".2f",center=0) # color-heatmap : BuPu , Greens , YlGnBu , Blues


plt.show()




# Data-Analysis\ds_salaries.csv