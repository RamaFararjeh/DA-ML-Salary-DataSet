# # # About data 
# work_year	: The year the salary was paid.
# experience_level	: The experience level in the job during the year with the following possible values: EN Entry-level / Junior MI Mid-level / Intermediate SE Senior-level / Expert EX Executive-level / Director
# employment_type :	The type of employement for the role: PT Part-time FT Full-time CT Contract FL Freelance
# job_title	: The role worked in during the year.
# salary :	The total gross salary amount paid.
# salary_currency	: The currency of the salary paid as an ISO 4217 currency code.
# salary_in_usd	 : The salary in USD (FX rate divided by avg. USD rate for the respective year via fxdata.foorilla.com).
# employee_residence	: Employee's primary country of residence in during the work year as an ISO 3166 country code.
# remote_ratio	: The overall amount of work done remotely, possible values are as follows: 0 No remote work (less than 20%) 50 Partially remote 100 Fully remote (more than 80%)
# company_location : The country of the employer's main office or contracting branch as an ISO 3166 country code.
# company_size	: The average number of people that worked for the company during the year: S less than 50 employees (small) M 50 to 250 employees (medium) L more than 250 employees (large

# insight:
# 1) there is no year in negative 
# 2) there are 4 type for employee *full time *part time *Contract *Freelance
# 3) salary not normal distribution.

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Extract and Read Data With Pandas

df_=pd.read_csv('Data-Analysis\ds_salaries.csv')
df=pd.DataFrame(df_)
print(df)

df.drop('Unnamed: 0',axis=1,inplace=True)
print(df)

print(df.info())

# q

# print(df['work_year'].sum()) #1226993
print(df.describe())

# Check for duplicated data 
print(df.duplicated().sum()) # [42] => drop
df.drop_duplicates(inplace=True)
print(df.duplicated().sum()) # 0

# Check size of data

print(df.shape) # 565 [ after drop 42 value duplicated. ]
print(df.size) # 6215 wrong 
print(len(df['work_year'])) # 565

# Check for unique value
 
print(df['experience_level'].unique()) # ['MI' 'SE' 'EN' 'EX']
print(df['experience_level'].nunique()) # 4

print(df['employment_type'].unique()) # ['FT' 'CT' 'PT' 'FL']
print(df['employment_type'].nunique()) # 4


print(df['job_title'].unique())
# ['Data Scientist' 'Machine Learning Scientist' 'Big Data Engineer'
#  'Product Data Analyst' 'Machine Learning Engineer' 'Data Analyst'
#  'Lead Data Scientist' 'Business Data Analyst' 'Lead Data Engineer'
#  'Lead Data Analyst' 'Data Engineer' 'Data Science Consultant'
#  'BI Data Analyst' 'Director of Data Science' 'Research Scientist'
#  'Machine Learning Manager' 'Data Engineering Manager'
#  'Machine Learning Infrastructure Engineer' 'ML Engineer' 'AI Scientist'
#  'Computer Vision Engineer' 'Principal Data Scientist'
#  'Data Science Manager' 'Head of Data' '3D Computer Vision Researcher'
#  'Data Analytics Engineer' 'Applied Data Scientist'
#  'Marketing Data Analyst' 'Cloud Data Engineer' 'Financial Data Analyst'
#  'Computer Vision Software Engineer' 'Director of Data Engineering'
#  'Data Science Engineer' 'Principal Data Engineer'
#  'Machine Learning Developer' 'Applied Machine Learning Scientist'
#  'Data Analytics Manager' 'Head of Data Science' 'Data Specialist'
#  'Data Architect' 'Finance Data Analyst' 'Principal Data Analyst'
#  'Big Data Architect' 'Staff Data Scientist' 'Analytics Engineer'
#  'ETL Developer' 'Head of Machine Learning' 'NLP Engineer'
#  'Lead Machine Learning Engineer' 'Data Analytics Lead']
print(df['job_title'].nunique()) # 50


print(df['salary_currency'].unique())
# ['EUR' 'USD' 'GBP' 'HUF' 'INR' 'JPY' 'CNY' 'MXN' 'CAD' 'DKK' 'PLN' 'SGD'
#  'CLP' 'BRL' 'TRY' 'AUD' 'CHF']
print(df['salary_currency'].nunique()) # 17


print(df['employee_residence'].unique())
#['DE' 'JP' 'GB' 'HN' 'US' 'HU' 'NZ' 'FR' 'IN' 'PK' 'PL' 'PT' 'CN' 'GR'
#  'AE' 'NL' 'MX' 'CA' 'AT' 'NG' 'PH' 'ES' 'DK' 'RU' 'IT' 'HR' 'BG' 'SG'
#  'BR' 'IQ' 'VN' 'BE' 'UA' 'MT' 'CL' 'RO' 'IR' 'CO' 'MD' 'KE' 'SI' 'HK'
#  'TR' 'RS' 'PR' 'LU' 'JE' 'CZ' 'AR' 'DZ' 'TN' 'MY' 'EE' 'AU' 'BO' 'IE'
#  'CH'] 
print(df['employee_residence'].nunique()) # 57



print(df['company_location'].unique())
# ['DE' 'JP' 'GB' 'HN' 'US' 'HU' 'NZ' 'FR' 'IN' 'PK' 'CN' 'GR' 'AE' 'NL'
#  'MX' 'CA' 'AT' 'NG' 'ES' 'PT' 'DK' 'IT' 'HR' 'LU' 'PL' 'SG' 'RO' 'IQ'
#  'BR' 'BE' 'UA' 'IL' 'RU' 'MT' 'CL' 'IR' 'CO' 'MD' 'KE' 'SI' 'CH' 'VN'
#  'AS' 'TR' 'CZ' 'DZ' 'EE' 'MY' 'AU' 'IE']
print(df['company_location'].nunique()) # 50


print(df['company_size'].unique()) # ['L' 'S' 'M']
print(df['company_size'].nunique()) # 3


# check if have any missing data in data.

print(df.isnull().sum())
print(df['company_size'].isnull().sum())

# Separating categorical and numerical columns
categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
print(categorical_columns)
print(numerical_columns)




for i in numerical_columns:
        plt.hist(df[i],bins=10, edgecolor='black')

        plt.xlabel(i)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {i}')

        plt.show()



# Calculate correlation matrix
correlation_matrix = df.corr()

# Create a heatmap using Matplotlib
plt.figure(figsize=(8, 6))
plt.title('Correlation Matrix')
sns.heatmap(correlation_matrix,annot=True,camp='coolwarm',linecolor='yellow',fmt=".2f",center=0) #center=0
plt.show()