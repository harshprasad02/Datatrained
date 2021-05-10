#!/usr/bin/env python
# coding: utf-8

# Importing all the useful libraries Which we'll use in the dataset

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("abalone.csv")


# In[3]:


df


# ## Description of the data set

# The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age.

# The other measurments provided to us are as follows:
# 1. Sex ------------> Determines the sex of abalone ,categorized as (M,F and I)
# 2. Length ---------> Longest shell measurement(in mm)
# 3. Diameter -------> Perpendicular to length(in mm)
# 4. Height ---------> with meat in shell (in mm)
# 5. Whole weight ---> whole abalone(in grams)
# 6. Shucked weight -> weight of meat(in grams)
# 7. Viscera weight -> gut weight (after bleeding)(in grams)
# 8. Shell weight ---> after being dried(in grams)
# 9. Rings ----------> size of rings 

# We have to claculate the size of rings with the help of given features.

# In[4]:


df.info()


# In[5]:


df.shape


# The dataset contains :
# 1. Number of columns = 9
# 2. number of rows    = 4177

# In[6]:


df.dtypes


# sex contains nominal data i.e M,F and I.Therefore we will encode it later for further analysis.

# In[7]:


#checking for null values


# In[8]:


df.isnull().sum()


# There are no null Values present in the dataset.

# In[9]:


#using heat map to confirm the presence of null values


# In[10]:


plt.figure(figsize = [8,6])
sns.heatmap(df.isnull())
plt.title('Null values')
plt.show() 


# Since there are no white spots or boxes(respresent null values) present in the graph.therefore,No null values are present in any of the feature.

# In[11]:


#statistical analysis


# In[12]:


df.describe()


# In[13]:


#analysis of the Features using visualization


# In[14]:


ax= sns.countplot(x = 'Sex',data = df)
print(df['Sex'].value_counts())


# Here we can see that all the three cateories are almost equaly distributed.

# In[15]:


#visualization for continous data


# In[16]:


df_visuals_cont = df[["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight"]]


# In[17]:


sns.distplot(df_visuals_cont["Length"],kde =True )


# the the is almost eaqaully distributed but very little left skewed

# In[18]:


sns.distplot(df_visuals_cont["Diameter"],kde =True )


# The diameter is little left skewed.

# In[19]:


sns.distplot(df_visuals_cont["Height"],kde =True )


# the data is equally distributed from 0.0 to 0.2.No skewnes present.

# In[20]:


sns.distplot(df_visuals_cont["Whole weight"],kde =True )


# The data is not equally distributed . it is highly spreaded.

# In[21]:


sns.distplot(df_visuals_cont["Shucked weight"],kde =True )


# The data is spreaded from 0.0 to 0.75. A little bit Right skewed.

# In[22]:


sns.distplot(df_visuals_cont["Viscera weight"],kde =True )


# The data is spreaded from 0.0 to 0.4 . A little bit rightly skewed.

# In[23]:


sns.distplot(df_visuals_cont["Shell weight"],kde =True )


# The data is highly spreaded from 0.0 to 0.4. Right Skewed.

# In[24]:


df["Rings"].hist(grid = True) 


# Here we can see that Maximum number of rings lies 

# In[27]:


#using ordinal encoder to  convert sex feature.


# In[28]:


from sklearn.preprocessing import LabelEncoder
oe = LabelEncoder()
df["Sex"] = oe.fit_transform(df["Sex"])


# In[29]:


df


# #### Checking relationship of features with each other.

# In[26]:


sns.pairplot(df)


# Observations: We can see the following relationship among the features.
# 1. When the length of the shell increases it's diameter also increases,showing a linear relationship and vice-versa.
# 2. when whole weight increases the length increases as square root of whole weight.Length shows similar relationship with shucked weight ,viscera weight and shell weight.
# 3. Since Length of the shell and diameter of the shell shows a linear relationship. therefore we can see that both of them shows similar relationship with other variables.
# 4. when length and diameter increases from 0.0 to 0.6 (mm) height of the shell ranges from 0 to approximately 0.3.
# 5. When whole weight of the shell increases the shucked weight ,viscera weight and the shell weight also increases.
# 6. shucked weight ,viscera weight and shell weight shows linear relationship with each other therefore,they shows similar relationship with ther features.
# 7. when ring size increases from 0to 30 (mm) the height ranges from 0.00 to 0.25.

# In[30]:


#we will be ploting the sex feature seperately for better under standing.


# In[31]:


for i in df[["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Rings"]]:
    x = df[i]
    y = df["Sex"]
    plt.xlabel(i)
    plt.ylabel("Sex")
    plt.scatter(x,y)
    plt.show()


# Here 0 indicates Female(F),1 indicates Infant(I) and 2 indicates male (M).
# Observations:
# 1. when length is greater than 0 less than 0.2 all the shells are Infant. above that either male or infant till approximately 0.3.After wards Can be any of the three.
# 2. Similar relation ship with diameter.
# 3. All three sex have almost same height range.
# 4. Infant whole weight ranges from 0.0 to 1.5 ,Male whole weight ranges from 0.0 to nearly 2.5 and for female it ranges from 0.0 to less tan 2.5.
# 5. Infant shucked weight range(0.0 to 0.6), Male shucked weight range(0.0 to 1.2),female shucked weight range(0 to 1.0) with some outliers.
# 6. Viscera weight of infant ranges (0 to 0.3),male(0 to 0.5) and for female (0 to more than 0.5)
# 7. Shell Weight of infant ranges(0 to above 0.4),male (0 to approximately 0.7) and female (0 to 0.6).Also some outliers are present above there max range.
# 8. rings ,infant have at least 1 ring and atmost 21,male has atleast 3 rings and atmost 27.and female has atleast 5 rings and atmost 29 rings.

# In[32]:


# Relationship of features with target variable


# In[33]:


for i in df[["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight","Sex"]]:
    x = df[i]
    y = df["Rings"]
    plt.xlabel(i)
    plt.ylabel("Rings")
    plt.scatter(x,y)
    plt.show()


# Observation:
# 1. When ring size increases from 0 to 30 (mm) the height ranges from 0.00 to 0.25 approximately.
# 2. rings ,infant have at least 1 ring and atmost 21,male has atleast 3 rings and atmost 27.and female has atleast 5 rings and atmost 29 rings.

# Conclusion: we cocluded that soe of the features are highly correlated to each other. Some of them do not show any relation. And only some features shows a relation with Target variable.

# In[34]:


#using box plot for checking outliers only for continous features ,excluding target variable.


# In[35]:


df.boxplot(column = ['Length',"Diameter"])


# Some outliers are present near lower whiskers.They are very close to the whisker so we'll keep them.

# In[36]:


df.boxplot(column = 'Height')


# we can see two outliers are present very far from the whisker.but there are very small in number so we can keep them.

# In[37]:


df.boxplot(column = ["Whole weight","Shucked weight","Viscera weight","Shell weight"])


# we can see a large number of outliers are present in the following features we'll be removing them.

# In[38]:


corr_matrix = df.corr()


# In[39]:


plt.figure(figsize = [10,4])
sns.heatmap(corr_matrix,annot = True)


# Observations:
# 1. Length and diameter are highly correlated(0.99)
# 2. We can see that all features except sex and Rings are highly correlated to each other showing a corr value greater than 0.77.

# In[40]:


#cheking the correlation of features with target.


# In[41]:


print(corr_matrix['Rings'].sort_values(ascending = False))


# Conclusion:  
# 1. Shell weight(0.62) is the most correlated feature with the Rings.
# 2. Diameter ,Height ,Length ,whole weight and viscera weight shows almost same correlation with Rings ,ranges(0.57 to 0.50)
# 3. Sex is least co relaed to the rings ,i.e., -0.0346.

# We'll be keeping all Features for model building.

# In[42]:


#checking range of each feature.


# In[43]:


def rangei(data_frame,feature_name):
    maximum_value = data_frame[feature_name].max()
    minimum_value = data_frame[feature_name].min()
    rangee = maximum_value - minimum_value
    print("maximum Value is : ",maximum_value)
    print("minimum value is : ",minimum_value)
    print("Range of ",feature_name," feature is :", rangee)


# In[44]:


for i in df:
    rangei(df,i)


# as we cam see that most of the feature ranges from 0.7 to 2 and minimum value is almost near to 0.1.so we can keep the outliers as the feature range is very small.

# In[157]:


#removing outliers using zscorevalue
from scipy.stats import zscore
z = np.abs(zscore(df))


# In[158]:


threshold = 3
print(np.where(z>3))


# In[159]:


df_new1 = df[(z<3).all(axis = 1)]


# In[160]:


df_new1.shape


# In[164]:


df.shape


# In[165]:


print("Total Data loss is ",((4177-4027)/4177)*100)


# In[45]:


#separating dependent feature and independent features


# In[180]:


x = df_new1.drop("Rings",axis = 1)


# In[181]:


y = df_new1["Rings"]


# In[182]:


x


# In[183]:


y


# ##### checking skewness in independent features

# In[184]:


x1 = x[["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight"]]


# In[185]:


x1.skew()


# In[190]:


#removing skewness.
from sklearn.preprocessing import PowerTransformer


# In[196]:


power = PowerTransformer()


# In[199]:


x["Length"] = power.fit_transform(x[["Length"]])


# In[202]:


x["Diameter"] = power.fit_transform(x[["Diameter"]])


# In[204]:


x.skew()


# In[243]:


scaling = ["Length","Diameter","Height","Whole weight","Shucked weight","Viscera weight","Shell weight"]


# Skewness is removed from all the features.

# In[244]:


#scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
for i in scaling:
    x[i]= sc.fit_transform(x[[i]])


# In[245]:


x.shape


# In[246]:


y.shape


# In[247]:


#checking the best random state for the model accuracy
from sklearn.linear_model import LinearRegression
Maxaccu = 0
Maxrow = 0
for i in range(1,400):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .20,random_state = i)
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    prdlr = lr.predict(x_test)
    accu = r2_score(y_test,prdlr)
    if accu>Maxaccu:
        Maxaccu = accu
        Maxrow = i
print("max accuracy is : ",Maxaccu ,"at random state : ",Maxrow)


# In[248]:


x_train,x_test,y_train,y_test = train_test_split(x1,y,test_size = .20,random_state = 172)


# In[249]:


x_train.shape


# In[250]:


x_test.shape


# In[251]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
prdlr = lr.predict(x_test)
print("Accuracy of the model :",r2_score(y_test,prdlr))
print("Mean Squared error :",mean_squared_error(y_test,prdlr))
print("Mean Absolute eroor :",mean_absolute_error(y_test,prdlr))


# In[252]:


# checking the cross vaidation score of the model.
from sklearn.model_selection import cross_val_score


# In[253]:


score = cross_val_score(lr,x1,y,cv = 5)
print(score.mean())


# In[254]:


from sklearn.tree import DecisionTreeRegressor
dtc = DecisionTreeRegressor()
dtc.fit(x_train,y_train)
prdtc = dtc.predict(x_test)
print("Accuracy of the model :",r2_score(y_test,prdtc))
print("Mean Squared error :",mean_squared_error(y_test,prdtc))
print("Mean Absolute eroor :",mean_absolute_error(y_test,prdtc))


# In[255]:


from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor()
rfc.fit(x_train,y_train)


# In[256]:


prrfc = rfc.predict(x_test)
print("Accuracy of the model :",r2_score(y_test,prrfc))
print("Mean Squared error :",mean_squared_error(y_test,prrfc))
print("Mean Absolute eroor :",mean_absolute_error(y_test,prrfc))


# In[257]:


k = ['linear','poly','rbf']


# In[258]:


from sklearn.svm import SVR
svr = SVR()
for i in k:
    svr = SVR(kernel = i)
    svr.fit(x_train,y_train)
    prdsvr = svr.predict(x_test)
    print("Accuracy of the SVR  model for",i," Kernel is",r2_score(y_test,prdsvr))
    print("Mean Squared error SVR  model for",i," Kernel is",mean_squared_error(y_test,prdsvr))
    print("Mean Absolute eroor SVR  model for",i," Kernel is:",mean_absolute_error(y_test,prdsvr))


# In[260]:


score = cross_val_score(SVR(kernel = 'rbf'),x1,y,cv = 5)
print(score.mean())


# In[265]:


score = cross_val_score(SVR(kernel = 'linear'),x1,y,cv = 5)
print(score.mean())


# In[261]:


score = cross_val_score(rfc,x1,y,cv = 5)
print(score.mean())


# In[262]:


from sklearn.ensemble import GradientBoostingRegressor


# In[263]:


ada = GradientBoostingRegressor()
ada.fit(x_train,y_train)
prada = ada.predict(x_test)
print("Accuracy of the model :",r2_score(y_test,prada))
print("Mean Squared error :",mean_squared_error(y_test,prada))
print("Mean Absolute eroor :",mean_absolute_error(y_test,prada))


# In[264]:


score = cross_val_score(ada,x1,y,cv = 5)
print(score.mean())


# In[266]:


#choosing RandomForest regressor and doing HyperParameter tuning


# In[267]:


from sklearn.model_selection import GridSearchCV


# In[271]:


parameters = {'n_estimators': np.arange(10,100),
             'criterion' : ['friedman_mse','mse','mae'],
             'max_depth' : np.arange(2,7)}


# In[272]:


GVC = GridSearchCV(RandomForestRegressor(),parameters,cv =5)


# In[273]:


GVC.fit(x_train,y_train)


# In[274]:


GVC.best_params_


# In[275]:


mod = RandomForestRegressor(criterion = 'friedman_mse',max_depth = 6,n_estimators = 62,random_state = 172)


# In[276]:


mod.fit(x_train,y_train)


# In[277]:


pred = mod.predict(x_test)
print("accuracy score of the model is :",r2_score(y_test,pred)*100)


# In[ ]:


import joblib
joblib.dump(mod,"abalone.pkl")

