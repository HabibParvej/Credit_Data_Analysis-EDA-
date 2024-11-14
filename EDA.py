#!/usr/bin/env python
# coding: utf-8

# In[3]:


# importing warnings
import warnings
warnings.filterwarnings("ignore")


# In[4]:


# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_columns",None)


# In[5]:


# Read application csv
app_data = pd.read_csv("application_data.csv")
app_data.head()


# In[6]:


# Data inspection in application Dataset
# Get info and shape on the dataset
app_data.info()


# In[7]:


# Data quality check 
# check for percantage null values in application dataset
pd.set_option('display.max_rows', 200)
app_data.isnull().mean() *100


# In[8]:


# conclusion : columns with null values more than 47% may give wrong insights hence will drop them
# Dropping columns with missing values greater than 47%
percentage = 47
threshold = int(((100-percentage)/100)*app_data.shape[0] + 1)
app_df = app_data.dropna(axis=1, thresh=threshold)
app_df.head()


# In[9]:


app_df.shape


# In[10]:


app_df.isnull().mean() *100


# In[11]:


# impute missing values 
# check the missing values in application dataset before imputing 
app_df.info()


# In[14]:


# occupation type column has 31% missingh values,since its a catagorical column,imputing the missing values with a unknown or other value
app_df.OCCUPATION_TYPE.isnull().mean()*100


# In[16]:


app_df.OCCUPATION_TYPE.value_counts(normalize=True)*100


# In[18]:


app_df.OCCUPATION_TYPE.fillna("Others", inplace=True)


# In[19]:


app_df.OCCUPATION_TYPE.isnull().mean() *100


# In[14]:


app_df.OCCUPATION_TYPE.value_counts(normalize=True)*100


# In[20]:


# External source 3 has 19%
app_df.EXT_SOURCE_3.isnull().mean() *100


# In[21]:


app_df.EXT_SOURCE_3.value_counts(normalize=True)*100


# In[22]:


app_df.EXT_SOURCE_3.describe()


# In[34]:


sns.boxplot(app_df.EXT_SOURCE_3)
plt.show()


# In[23]:


app_df.EXT_SOURCE_3.fillna(app_df.EXT_SOURCE_3.median(), inplace=True)


# In[24]:


app_df.EXT_SOURCE_3.isnull().mean() *100


# In[25]:


app_df.EXT_SOURCE_3.value_counts(normalize=True)*100


# In[26]:


# conclusion since its a numerical columns with no outliers and there is nor much difference between mean and median hence we can impute with mean of median
null_cols = list(app_df.columns[app_df.isna().any()])
len(null_cols)


# In[27]:


app_df.isnull().mean()*100


# In[29]:


# handling missing values in columns with 13% null values
app_df.AMT_REQ_CREDIT_BUREAU_HOUR.value_counts(normalize=True)*100


# In[30]:


app_df.AMT_REQ_CREDIT_BUREAU_DAY.value_counts(normalize=True)*100


# In[42]:


# conclusion  we couls see that 99%  of values in the cols AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_WEEK, AMT_REQ_CREDIT_BUREAU_MONTH HOUR, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_YEAR is 0.0
# hence impute these cols with the mode


# In[31]:


Cols = ["AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR", "AMT_REQ_CREDIT_BUREAU_HOUR" ]


# In[32]:


for col in Cols:
    app_df[col].fillna(app_df[col].mode()[0], inplace = True)


# In[33]:


# handling missin values less than 1%
null_cols = list(app_df.columns[app_df.isna().any()])
len(null_cols)


# In[34]:


app_df.NAME_TYPE_SUITE.value_counts(normalize=True)*100


# In[35]:


app_df.EXT_SOURCE_2.value_counts(normalize=True)*100


# In[55]:


# for catagorical cols impute the missing values with mode
# for catagorical cols impute the missing values with mnedian


# In[36]:


app_df.NAME_TYPE_SUITE.fillna(app_df.NAME_TYPE_SUITE.mode()[0], inplace = True)


# In[37]:


app_df.CNT_FAM_MEMBERS .fillna(app_df.CNT_FAM_MEMBERS.mode()[0], inplace = True)


# In[38]:


# imputing numerical cols


# In[39]:


app_df.EXT_SOURCE_2.fillna(app_df.EXT_SOURCE_2.median(), inplace=True)
app_df.AMT_GOODS_PRICE.fillna(app_df.AMT_GOODS_PRICE.median(), inplace=True)
app_df.AMT_ANNUITY.fillna(app_df.AMT_ANNUITY.median(), inplace=True)
app_df.OBS_30_CNT_SOCIAL_CIRCLE.fillna(app_df.OBS_30_CNT_SOCIAL_CIRCLE.median(), inplace=True)
app_df.DEF_30_CNT_SOCIAL_CIRCLE.fillna(app_df.DEF_30_CNT_SOCIAL_CIRCLE.median(), inplace=True)
app_df.OBS_60_CNT_SOCIAL_CIRCLE.fillna(app_df.OBS_60_CNT_SOCIAL_CIRCLE.median(), inplace=True)
app_df.DEF_60_CNT_SOCIAL_CIRCLE.fillna(app_df.DEF_60_CNT_SOCIAL_CIRCLE.median(), inplace=True)
app_df.DAYS_LAST_PHONE_CHANGE.fillna(app_df.DAYS_LAST_PHONE_CHANGE.median(), inplace=True)


# In[40]:


null_cols = list(app_df.columns[app_df.isna().any()])
len(null_cols)


# In[41]:


app_df.NAME_TYPE_SUITE.value_counts(normalize=True)*100


# In[42]:


app_df.EXT_SOURCE_2.value_counts(normalize=True)*100


# In[61]:


# convert negative values to positive in days variable so that median is not affected


# In[63]:


app_df.DAYS_BIRTH = app_df.DAYS_BIRTH.apply(lambda x: abs(x))
app_df.DAYS_EMPLOYED = app_df.DAYS_EMPLOYED.apply(lambda x: abs(x))
app_df.DAYS_ID_PUBLISH = app_df.DAYS_ID_PUBLISH.apply(lambda x: abs(x))
app_df.DAYS_LAST_PHONE_CHANGE = app_df.DAYS_LAST_PHONE_CHANGE.apply(lambda x: abs(x))
app_df.DAYS_REGISTRATION = app_df.DAYS_REGISTRATION.apply(lambda x: abs(x))


# In[64]:


#binning of continous variables
# standardizing days cols in years for easy binning


# In[65]:


app_df.OBS_30_CNT_SOCIAL_CIRCLE.value_counts(normalize=True)*100


# In[66]:


app_df["YEARS_BIRTH"] = app_df.DAYS_BIRTH.apply(lambda x: int(x//365))
app_df["YEARS_EMPLOYED"] = app_df.DAYS_EMPLOYED.apply(lambda x: int(x//365))
app_df["YEARS_REGISTRATION"] = app_df.DAYS_REGISTRATION.apply(lambda x: int(x//365))
app_df["YEARS_ID_PUBLISH"] = app_df.DAYS_ID_PUBLISH.apply(lambda x: int(x//365))
app_df["YEARS_LAST_PHONE_CHANGE"] = app_df.DAYS_LAST_PHONE_CHANGE.apply(lambda x: int(x//365))


# In[67]:


# binning of AMT credit column 


# In[68]:


app_df.AMT_CREDIT.value_counts(normalize=True)*100


# In[69]:


app_df.AMT_CREDIT.describe()


# In[70]:


app_df["AMT_CREDIT_Category"] = pd.cut(app_df.AMT_CREDIT, [0, 200000, 400000, 600000, 800000, 1000000],
                                       labels=["Very low Credit", "low Credit", "Medium Credit", "High Credit", "Very High Credit"])


# In[71]:


app_df.AMT_CREDIT_Category.value_counts(normalize=True)*100


# In[72]:


app_df["AMT_CREDIT_Category"].value_counts(normalize=True).plot.bar()


# In[73]:


# the credit amount of the loan amount low(2l to 4l) or very high (above 8L)
# Binning YEARS_BIRTH_COLUMN 


# In[78]:


app_df["AGE_Category"] = pd.cut(app_df.YEARS_BIRTH, [0, 25, 45, 65, 85],
                                 labels = ["Below 25", "25-45", "45-65", "65-85"])


# In[79]:


app_df.AGE_category.value_counts(normalize=True)*100


# In[80]:


app_df["AGE_Category"].value_counts(normalize=True).plot.pie(autopct = '%1.2f%%')


# In[81]:


# most of the applicants are between 25-45 age group


# In[82]:


# Data imbalance check 


# In[83]:


app_df.head()


# In[84]:


# Dividing application dataset with target variable as 0 and 1


# In[85]:


tar_0 = app_df[app_df.TARGET == 0]
tar_1 = app_df[app_df.TARGET == 1]


# In[86]:


app_df.TARGET.value_counts(normalize=True)*100


# In[87]:


# 1 out of 9/10 applicants are defaults


# In[88]:


# UNIVARIATE ANALYSIS


# In[92]:


cat_cols = list(app_df.columns[app_df.dtypes == object])
num_cols = list(app_df.columns[app_df.dtypes == np.int64]) + list(app_df.columns[app_df.dtypes == np.float64])


# In[93]:


cat_cols


# In[94]:


num_cols


# In[96]:


for col in cat_cols:
    print(app_df[col].value_counts(normalize=True))
    plt.figure(figsize=[5,5])
    app_df[col].value_counts(normalize=True).plot.pie(labeldistance = None , autopct = '%1.2f%%')
    plt.legend()
    plt.show()


# In[100]:


#   insights on gthe below columns

    #1. NAME_CONTRACT_TYPE - More application have Cash loans than Revolving loans
     #2. CODE_GENDER - Number of Female applicants are twice than that of male applicants
# 3. FLAG_OWN_CAR - Most(70%) of the applicants do not own a car
# 4. FLAG_OWN_REALTY - Most(70%) of the applicants do not own a house
# 5. NAME_TYPE_SUITE - Most(81%) of the applicants are Unaccompanied
# 6. NAME_INCOME_TYPE - Most(51%) of the applicants are earning their income from Work
# 7. NAME_EDUCATION_TYPE - 71% of the applicants have completed Secondary / secondary special education
# 8. NAME_FAMILY_STATUS - 63% of the applicants are married
# 9. NAME_HOUSING_TYPE - 88% of the housing type of applicants are House/apartment
# 10. OCCUPATION_TYPE - Most(31%) of the applicants have other Occupation type
# 11. WEEKDAY_APPR_PROCESS_START- Most of the applicant have applied the loan on Tuseday
# 12. ORGANIZATION_TYPE - Most of the Organization type of employees are Business Entity Type 3



# In[102]:


# plot on numerical coulmns
 #catagorizing columns with and without flags 


# In[104]:


num_cols_withoutflag = []
num_cols_withflag = []
for col in num_cols:
    if col.startswith("FLAG"):
        num_cols_withflag.append(col)
    else:
        num_cols_withoutflag.append(col)


# In[105]:


num_cols_withoutflag


# In[106]:


num_cols_withflag


# In[107]:


for col in num_cols_withoutflag:
    print(app_df[col].describe())
    plt.figure(figsize = [8,5])
    sns.boxplot(data=app_df, x=col)
    plt.show()
    print("----------------------")


# In[108]:


# Few columns with outliers are below
# 1. AMT_INCOME_TOTAL Column has a few outliers and there is a huge difference between the 99th percentile and the max value, also we could see huge variation in mean and median due to outliers
# 2. AMT_CREDIT Column has a few outliers and there is a huge difference between the 99th percentile and the max value, also we could see huge variation in mean and median due to outliers
# 3. AMT_ANNUITY Column has a few outliers and there is a huge difference between the 99th percentile and the max value, also we could see significant variation in mean and median due to outliers
# 4. AMT_GOODS_PRICE Column has a few outliers and there is a huge difference between the 99th percentile and the max value, also we could see significant variation in mean and median due to outliers
# 5. REGION_POPULATION_RELATIVE Column has a one outliers and there not much difference between mean and median


# In[109]:


# Univarient analysis on coulmns with target 1 and 0


# In[117]:


for col in cat_cols:
    print("plot on [col] for Target 0 and 1")
    plt.figure(figsize=[10,7])
    plt.subplot(1,2,1)
    tar_0[col].value_counts(normalize=True).plot.bar()
    plt.title("Target 0")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.subplot(1,2,2)
    tar_1[col].value_counts(normalize=True).plot.bar()
    plt.title("Target 1")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()
    print("\n\n-----------------------------------------------------------\n\n")


# In[119]:


# below are the column insights
# 1. NAME_CONTRACT TYPE. The Applicants are receiving more of Cash loans than Revolving loans both for Target 0 and 1
# 2. CODE_GENDER - Number of Female applicants are twice than that of male applicants both for Target 0 and 1
# 3. FLAG_OWN_CAR - Most(70%) of the applicants do not own a car both for Target 0 and 1
# 6. NAME_INCOME_TYPE - For both Target 0 and 1, Most(51%) of the applicants are earning their income from Work
# 2. NAME FAML SATUS - 6. or te ape s art amos 7r it aero cans nave complete Seconday / secondary special ecuaton
# 10 0CCUPATUN TE 106(351 %) r me picans fae frer cupatus ype are on cet et and Labore Sale start Divers and core statare
# not able to repay the loan on time
# 11. WEEKDAY_APPR_PROCESS_START- Most of the applicant have applied the loan on Tuseday and the least on Sunday
# 12. ORGANIZATION_TYPE - Most of the Applicants are working in Business Entity Type 3, Self Employed and other Organization type


# In[120]:


# analysis on AMT_GOODS_PRICE on target 0 and 1


# In[122]:


plt.figure(figsize=(10,6))
sns.distplot(tar_0["AMT_GOODS_PRICE"], label='tar_0', hist=False)
sns.distplot(tar_1["AMT_GOODS_PRICE"], label='tar_1', hist=False)
plt.legend()
plt.show()


# In[123]:


# the price of the goods for which loan is given has the same variation for target 0 and 1


# In[125]:


# Bivariate and Multivariate Analysis
 # Bivariate analysis between WEEKDAY_APPR_PROCESS_START vs HOUR_APPR_PROCESS_START


# In[127]:


plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
sns.boxplot(x='WEEKDAY_APPR_PROCESS_START', y = 'HOUR_APPR_PROCESS_START', data = tar_0)
plt.subplot(1,2,2)
sns.boxplot(x='WEEKDAY_APPR_PROCESS_START', y = 'HOUR_APPR_PROCESS_START', data= tar_1)
plt.show()


# In[128]:


# 1. The Bank operates between 10am to 3pm except for Saturday and Sunday, its between 10am to 2pm.
# 2. We can observe that around 11:30am to 12pm around 50% of Customers visit the branch for loan application on all the days except for Saturday where the time is between 10am to 11am for both Target 0 and 1
# 3. The loan defaulters have applied for the loan between 9:30am-10am and 2pm where as the applicants who repay the loan on time have applied for the loan between 10am to 3pm


# In[130]:


# analysis between age catago9ry and amt catagory


# In[131]:


plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
sns.boxplot(x='AGE_Category', y = 'AMT_CREDIT', data = tar_0)
plt.subplot(1,2,2)
sns.boxplot(x='AGE_category', y = 'AMT_CREDIT', data= tar_1)
plt.show()


# In[132]:


# 1. The applicants between age group 25 to 65 have Credit amount of the loan less than 2500000 and are able to repay the loan properly
# 2. The applicants with less than 100000 Credit amount are with age group greater than 65 may be considered as loan defaulters
# 3. Most applicants who have Credit amount of the loan less than 1700000 are loan defaulters with 25 and less age


# In[133]:


# pair plots for Amount columns for target 0


# In[139]:


sns.pairplot(tar_0[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]])
plt.show()


# In[140]:


#* Conclusion >> For Applicants who are able to replay the loan on time
# 1. AMT_CREDIT Increases or varies linearly with AMT_GOODS_PRICE and AMT_CREDIT Increases with AMT_ANNUITY
# 2. AMT_ANNUITY Increases with Increases in AMT_GOODS_PRICE and AMT_Credit
# 3. AMT_GOODS_PRICE Increases with Increases in AMT_Credit and AMT_ANNUITY
# 4. AMT_INCOME_TOTAL has a drastic Increase with slight increase in AMT_CREDIT,AMT_ANNUITY,AMT_GOODS_PRICE


# In[141]:


sns.pairplot(tar_1[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]])
plt.show()


# In[142]:


# corelation between numericals columns 


# In[145]:


corr_data = app_df[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","YEARS_BIRTH","YEARS_EMPLOYED","YEARS_REGISTRATION","YEARS_ID_PUBLISH","YEARS_LAST_PHONE_CHANGE"]]
corr_data.corr()


# In[146]:


plt.figure(figsize=(10,10))
sns.heatmap(corr_data.corr(),annot=True,cmap="RdYlGn")
plt.show()


# In[147]:


#* Conclusion >>
#1. AMT_INCOME_TOTAL - It has a positive corelation index of 0.16,0. 19,0.16 with AMT_CREDIT.AMT_ANNUITY,AMT_GOODS_PRICE respectively.
#2. AMT_CREDIT - Is has negative coreltaion index of 0.064 with YEARS_EMPLOYED and positive coreltaion index of 0.99,0.77 with AMT_GOODS_PRICE, AMT_ANNUITY respectively.
#3. AMT_ANNUITY - Is has negative coreltaion index of 0.1 with YEARS_EMPLOYED and positive coreltaion index of 0.77 with AMT_CREDIT
#4. AMT_GOODS_PRICE - It has a positive corelation with AMT_CREDIT,AMT_ANNUITY
#5. YEARS_BIRTH - It has a positive corelation with YEARS_EMPLOYED, AMT_GOODS_PRICE and negative coreitaion with AMT_ANNUITY AMT_INCOME_TOTAL
#6. YEARS_EMPLOYED - Is has negative coreltaion index of 0.1 with AMT_ANNUITY and has a positive corelation with YEARS_REGISTRATION
     #YEARS_ID_PUBLISH
#7. YEARS_REGISTRATION - It has a positive corelation with YEARS_ID_PUBLISH, YEARS_BIRTH, YEARS_EMPLOYED
#8. YEARS_ID_PUBLISH - It has a positive corelation with YEARS_REGISTRATION and negative coreltaion with AMT_INCOME_TOTAL, AMT_ANNUITY
#9. YEARS_LAST_PHONE_CHANGE - It has negative coreltaion with YEARS_EMPLOYED and positive corelation with AMT_GOODS_PRICE, YEARS_ID_PUBLISH


# In[148]:


#split the numerical variables based on target 1 and target 0 to find the corelation


# In[149]:


corr_data_0 = tar_0[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","YEARS_BIRTH","YEARS_EMPLOYED","YEARS_REGISTRATION","YEARS_ID_PUBLISH","YEARS_LAST_PHONE_CHANGE"]]
corr_data_0.corr()


# In[150]:


corr_data_1 = tar_0[["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","YEARS_BIRTH","YEARS_EMPLOYED","YEARS_REGISTRATION","YEARS_ID_PUBLISH","YEARS_LAST_PHONE_CHANGE"]]
corr_data_1.corr()


# In[151]:


plt.figure(figsize=(10,10))
sns.heatmap(corr_data_0.corr(),annot=True,cmap="RdYlGn")
plt.show()

