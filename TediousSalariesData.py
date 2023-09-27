
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error ,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# insight:
# 1) there is no year in negative 
# 2) there are 4 type for employee *full time *part time *Contract *Freelance
# 3) salary not normal distribution.
# 4) All Features have a weak correlation with target [ salary ]

# Target => salarry.

# Extract and Read Data With Pandas

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


# Nunarical Features with it self to Know Distribution.
def hist_plot():
    for i in numerical_columns:
        # range=[0, df[i].value_counts()]
        plt.hist(df[i],bins = 10, edgecolor='k')

        plt.xlabel(i)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {i}')

        plt.show()
        

def box_plot():
    plt.boxplot(df[numerical_columns])

    plt.xlabel(numerical_columns.to_list())
    plt.ylabel('Frequency')
    plt.title(f'Box Plot of {numerical_columns.to_list()}')
    plt.show()
        




# Heatmap:-
# # to make heatmap we must drop all categorical data , so befor running the command we should activate cat_data().
#  or use heatmap for numarical data only .

# Relationship between Numarical Features with other Numarical Features
# Calculate correlation matrix
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(8, 6))
plt.title('Correlation Matrix')
sns.heatmap(correlation_matrix,annot=True,linecolor='red',cmap='BuPu',fmt=".2f",center=0) # color-heatmap : BuPu , Greens , YlGnBu , Blues
plt.show()


# Features with target

def scatterr():
    for i in numerical_columns:
                
        plt.figure(figsize=(8,6))
        plt.scatter(df[i],df['salary'],c='blue',marker='s',label='Data Points')

        # Marker Value :
        # 'o' for circular.
        # 's' for square.
        # '^' for triangle.

        # label  => legend

        plt.xlabel(i)
        plt.ylabel('salary')
        plt.title(f'Scatter Plot {i} vs Salary ')

        plt.legend() # مفتاح الخريطه
        


# categorical_columns
def cat_pie():
    for i in categorical_columns:
        countt=df[i].value_counts()
        print(countt)
        plt.figure(figsize=(8,6))
        plt.pie(countt,labels=countt.index,autopct="%1.1f%%")#,autopct="%1.1f%%",startangle=140
        # plt.axis('equal')
        plt.title(f"pie plot {i}")
        

# bar plot
def bar_plot():
    for i in categorical_columns:
        countt=df[i].value_counts()
        x_values = countt.index.tolist()  # List of unique values (categories)
        y_values = countt.tolist() 
        # print(df[i])
        # print(countt.index)
        # print(countt.index.tolist())
        plt.figure(figsize=(8,6))
        plt.bar(x_values,y_values,color='red')
        plt.title(f"bar plot {i}")
# X-axis: Represents the unique categorical values (categories or labels).
# Y-axis: Represents the count or frequency of each unique categorical value.

        

# duplicate()
# unique_data()
# isnull_data()
hist_plot()
box_plot()
scatterr()
cat_pie()
bar_plot()
plt.show()


# ====================================================================================

# Technique for converting categorical data into numerical format

# Initialize the LabelEncoder
label_encoder = LabelEncoder() # instance of LabelEncoder class
df1=df.copy()
# Fit the encoder to the data and transform the data
columns_to_encode = ['experience_level', 'employment_type', 'job_title',
                        'salary_currency','employee_residence', 'company_location', 'company_size']
for column in columns_to_encode:
    df1[column] = label_encoder.fit_transform(df[column])

# Print the original data and the encoded data
# print("Original data:", df)
print("Encoded data:", df1)


# ====================================================================================

# Scaling 

scaler=MinMaxScaler() # instance of MinMaxScaler class.
# print(scaler.fit(df))
numarical_features=numerical_columns.copy().to_list()
scaled_features=scaler.fit_transform(df1[numarical_features])
df1[numerical_columns]=pd.DataFrame(scaled_features,columns=numarical_features)
print('DataFrame after MinMax Scaling:')
print(df1[numerical_columns])
# print(df1)

# ====================================================================================

# Spliting Data for testing & training

df_train = df1.sample(frac=0.8, random_state=42)
df_test = df1.drop(df_train.index)
# print(df1.drop(df_train.index)) # laaag ya ana nasan. mish aref , wallah fhimat ._<
# print(df_train.index)
# print(f'train {df_train}')
# print(f'test {df_test}')
df_train.to_csv('Data-Analysis\Salary\df_train_salry.csv',index=False)
df_test.to_csv('Data-Analysis\Salary\df_test_slary.csv',index=False)
# ====================================================================================

# split train data 
x=df_train.drop('salary',axis=1) # x => new DataFrame without salary features.
y=df_train['salary']  # y => have the predict target variable.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) # had mo kader afhamo :(

# ====================================================================================

# Scaling after Split train data. 
num_col=x_train.select_dtypes(include=['float64']).columns


scaler=MinMaxScaler() # instance of MinMaxScaler class.
# Fit
scaler.fit(x_train[num_col])
# Transform
x_train_scaled=x_train.copy()
x_test_scaled=x_test.copy()
x_train_scaled[num_col]=scaler.transform(x_train[num_col])
x_test_scaled[num_col]=scaler.transform(x_test[num_col])
print(x_train_scaled.head())

# ====================================================================================

# New Distribution [ Histogram ]

for col in num_col:
    plt.figure(figsize=(8,6))
    plt.hist(x_test_scaled[col],bins=20,edgecolor='k')
    plt.title(f'Hisogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    # plt.show()

# ====================================================================================

# LR Model. [ Failed ]

# Initialize models 
lr_model=LinearRegression()

# Fit
lr_model.fit(x_train_scaled,y_train) # Train Model

# Make Prediction 
y_pred=lr_model.predict(x_test_scaled)

# Evaluate the Model
MAE=mean_absolute_error(y_test,y_pred)
MSE=mean_squared_error(y_test,y_pred)
RMSE=np.sqrt(MSE)
R2=r2_score(y_test,y_pred)

# Print the Metrics
print('LR Model')
print(f'mean_absolute_error : {MAE}')
print(f'mean_squared_error  : {MSE}')
print(f'Root mean_squared_error : {RMSE}')
print(f'R-squared  : {R2}')
print('----------------------------------------------------------------------------------------')


# ====================================================================================

# PL Model. [ Failed ]

poly_model=make_pipeline(PolynomialFeatures(degree=2),LinearRegression()) # => ma fihimt

# Fit
poly_model.fit(x_train_scaled,y_train)

# Make Prediction
y_pred=poly_model.predict(x_test_scaled)


# Evaluate the Model
MAE=mean_absolute_error(y_test,y_pred)
MSE=mean_squared_error(y_test,y_pred)
RMSE=np.sqrt(MSE)
R2=r2_score(y_test,y_pred)

# Print the Metrics
print('PL Model')
print(f'mean_absolute_error : {MAE}')
print(f'mean_squared_error  : {MSE}')
print(f'Root mean_squared_error : {RMSE}')
print(f'R-squared  : {R2}')
print('----------------------------------------------------------------------------------------')


# ====================================================================================

# SVM Model 

svr_model=SVR()
dtr_model=DecisionTreeRegressor()

# Fit
svr_model.fit(x_train_scaled,y_train)

# Make Prediction
y_pred=svr_model.predict(x_test_scaled)


# Evaluate the Model
MAE=mean_absolute_error(y_test,y_pred)
MSE=mean_squared_error(y_test,y_pred)
RMSE=np.sqrt(MSE)
R2=r2_score(y_test,y_pred)

# Print the Metrics
print('SVR Model')
print(f'mean_absolute_error : {MAE}')
print(f'mean_squared_error  : {MSE}')
print(f'Root mean_squared_error : {RMSE}')
print(f'R-squared  : {R2}')
print('----------------------------------------------------------------------------------------')

# ====================================================================================

# DT Model

dtr_model=DecisionTreeRegressor()

# Fit
dtr_model.fit(x_train_scaled,y_train)

# Make Predition 
y_pred=dtr_model.predict(x_test_scaled)


# Evaluate the Model
MAE=mean_absolute_error(y_test,y_pred)
MSE=mean_squared_error(y_test,y_pred)
RMSE=np.sqrt(MSE)
R2=r2_score(y_test,y_pred)

# Print the Metrics
print('DT Model')
print(f'mean_absolute_error : {MAE}')
print(f'mean_squared_error  : {MSE}')
print(f'Root mean_squared_error : {RMSE}')
print(f'R-squared  : {R2}')
print('----------------------------------------------------------------------------------------')

# ====================================================================================
# help(PolynomialFeatures())

# 'more' is not recognized as an internal or external command,
# operable program or batch file.

# ====================================================================================

# categorical Data
# ['experience_level', 'employment_type', 'job_title', 'salary_currency',
#        'employee_residence', 'company_location', 'company_size']