#!/usr/bin/env python
# coding: utf-8

# #                         Analytic Report Project

# In[108]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# # Data

# In[97]:


df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[98]:


df


# ## Data attrition 

# In[5]:


att = df[df['Attrition']=="Yes"]


# In[6]:


att


# In[86]:


cont_col = []
for column in att.columns:
    if att[column].dtypes != object and att[column].nunique() > 30:
        print(f"{column} : Minimum: {att[column].min()}, Maximum: {att[column].max()}")
        cont_col.append(column)
        print("====================================")


# In[8]:


att.describe()


# In[106]:


## Data attrition of DistanceFromHome


# In[85]:


att.set_index('DistanceFromHome')


# In[10]:


plt.figure(figsize=([10,20]))
plt.subplots(12,figsize=[12,6])
plt.subplot(121)
sns.boxplot(x='Gender',y='DistanceFromHome',data=att)
plt.title("Gender Rate by Distance From Home")
plt.subplot(122)
sns.distplot(df['DistanceFromHome'])
plt.xlim(0,30)
sns.despine()


# ## Data attrition of DailyRate

# In[87]:


att.set_index('DailyRate')


# In[12]:


plt.subplots(12,figsize=[12,6])
plt.subplot(121)
sns.boxplot(x='Gender',y='DailyRate',data=att)
plt.title("Gender Rate by Daily Rate")
plt.subplot(122)
sns.distplot(df['DailyRate'],color='green')
plt.xlim(0,1750)
sns.despine()


# ## Data attrition of Age

# In[88]:


att.set_index('Age')


# In[13]:


plt.subplots(12,figsize=[12,6])
plt.subplot(121)
sns.boxplot(x='Gender',y='Age',data=att)
plt.title("Gender Rate by Age")
plt.subplot(122)
sns.distplot(df['Age'],color='purple')
plt.xlim(18,58)
sns.despine()


# # Heatmap to display overall attrition rate 

# In[14]:


plt.figure(figsize=(20,20))
sns.heatmap(att.corr(),annot=True)


# # Apply method t-test
# ## Pop1: Attrition Rate
# ## Pop2: Satisfaction Rate
# ## -----------------------
# ### Pop1: Attrition Rate
# ### H0: difference between of Gender playing into current attrtion rate
# ### H1: the same between of Gender playing into current attrition rate

# ### data list attrition of male and female 

# In[89]:


Male = att[att['Gender']=='Male']
FM = att[att['Gender']=='Female']


# ### data attrition of Gender rate by distance from home
# 

# In[90]:


A1 = Male["DistanceFromHome"]

def lstm():
    lst =[]
    for x in A1:
        lst.append(x)
    return lst

A2 = FM['DistanceFromHome']
def lstfm():
    lst1 =[]
    for x in A2:
        lst1.append(x)
    return lst1


# ### list data distance from home to be calculated by t-tes

# In[18]:


MDFH=np.array(lstm())
FMDFH=np.array(lstfm())


# ### Call t-test

# In[20]:


stats.ttest_ind(MDFH,FMDFH)


# ##### We have 
# ##### t_value < t_crit (-0.3967 < 2,76)
# ##### p_value > 0.05 (0.7 > 0.05)
# ##### ==> We can't reject the H0 ( Male and Female have a same result at Attrition rate by DistanceFromHome)

# ## data attrition of Gender rate by Daily Rate

# In[22]:


B1 = Male["DailyRate"]
def lstm1():
    lstB1 =[]
    for x in B1:
        lstB1.append(x)
    return lstB1
B2 = FM['DailyRate']
def lstfm1():
    lstB2 =[]
    for x in B2:
        lstB2.append(x)
    return lstB2


# ### List data Daily Rate to be calculated by t-test

# In[24]:


MDR=np.array(lstm1())
FMDR=np.array(lstfm1())


# ### Call t-test

# In[25]:


stats.ttest_ind(MDR,FMDR)


# ##### We have 
# ##### t_value < t_crit (-0.55 < 2,76)
# ##### p_value > 0.05 (0.583 > 0.05)
# ##### ==> We can't reject the H0 ( Male and Female have a same result at Attrition rate by DailyRate)

# # Data attrition of Gender rate by Age

# In[26]:


C1 = Male["Age"]
def lstm2():
    lstC1 =[]
    for x in C1:
        lstC1.append(x)
    return lstC1
C2 = FM['Age']
def lstfm2():
    lstC2 =[]
    for x in C2:
        lstC2.append(x)
    return lstC2


# 
# ### List data distance from home to be calculated by t-tes

# In[28]:


MAge=np.array(lstm2())
FMAge=np.array(lstfm2())


# ### Call t-test

# In[29]:


stats.ttest_ind(MAge,FMAge)


# ##### We have 
# ##### t_value < t_crit (1.25 < 2,76)
# ##### p_value > 0.05 (0.212 > 0.05)
# ##### ==> We can't reject the H0 ( Male and Female have a same result at Attrition rate by Age)

# # Analysis and apply the method t-test fof playing into Satisfaction Rates

# ## Data Satisfaction Rate

# In[52]:


att1 = df[df['Attrition']=="No"]


# In[53]:


att1


# In[54]:


att1.describe()


# In[82]:


cont_col = []
for column in att1.columns:
    if att1[column].dtypes != object and att1[column].nunique() > 30:
        print(f"{column} : Minimum: {att1[column].min()}, Maximum: {att1[column].max()}")
        cont_col.append(column)
        print("====================================")


# ## Data Satisfaction of Distance From Home

# In[55]:


att1.set_index('DistanceFromHome')


# In[56]:


plt.figure(figsize=([10,20]))
plt.subplots(12,figsize=[12,6])
plt.subplot(121)
sns.boxplot(x='Gender',y='DistanceFromHome',data=att1)
plt.title("Gender Rate by Distance From Home")
plt.subplot(122)
sns.distplot(df['DistanceFromHome'])
plt.xlim(0,30)
sns.despine()


# ## Data Satisfaction of Daily Rate

# In[83]:


att1.set_index('DailyRate')


# In[58]:


plt.subplots(12,figsize=[12,6])
plt.subplot(121)
sns.boxplot(x='Gender',y='DailyRate',data=att1)
plt.title("Gender Rate by Daily Rate")
plt.subplot(122)
sns.distplot(df['DailyRate'],color='green')
plt.xlim(102,1500)
sns.despine()


# ## Data Satisfaction of Age

# In[84]:


att1.set_index('Age')


# In[59]:


plt.subplots(12,figsize=[12,6])
plt.subplot(121)
sns.boxplot(x='Gender',y='Age',data=att1)
plt.title("Gender Rate by Age")
plt.subplot(122)
sns.distplot(df['Age'],color='purple')
plt.xlim(18,58)
sns.despine()


# # Heatmap to display overall satisfaction rate 

# In[60]:


plt.figure(figsize=(20,20))
sns.heatmap(att1.corr(),annot=True)


# # Apply method t-test
# ## Pop1: Attrition Rate
# ## Pop2: Satisfaction Rate
# ## -----------------------
# ### Pop2: Satisfacetion Rate
# ### H0: difference between of Gender playing into current satisfaction rate
# ### H1: the same between of Gender playing into current satisfaction rate

# ## Data list satisfaction of male and female 
# 

# In[61]:


Male1 = att1[att1['Gender']=='Male']
FM1 = att1[att1['Gender']=='Female']


# ### Data satisfaction of Gender rate by distance from home

# In[92]:


X1 = Male1["DistanceFromHome"]

def lstmx():
    lst =[]
    for x in X1:
        lst.append(x)
    return lst

X2 = FM1['DistanceFromHome']

def lstfmx():
    lst1 =[]
    for x in X2:
        lst1.append(x)
    return lst1


# 
# ### List data distance from home to be calculated by t-tes

# In[93]:


X1DFH=np.array(lstmx())
X2DFH=np.array(lstfmx())


# ### Call t-test

# In[94]:


stats.ttest_ind(X1DFH,X2DFH)


# ##### We have 
# ##### t_value < t_crit (-0.005 < 2,76)
# ##### p_value > 0.05 (0.9957 > 0.05)
# ##### ==> We can't reject the H0 ( Male and Female have a same result at satisfaction rate by DistanceFromHome)

# # Data Satisfaction of Gender rate by DailyRate

# In[79]:


Y1 = Male1["DailyRate"]

def lstmy():
    lstB1 =[]
    for x in Y1:
        lstB1.append(x)
    return lstB1

Y2 = FM1['DailyRate']

def lstfmy():
    lstB2 =[]
    for x in Y2:
        lstB2.append(x)
    return lstB2


# ### List Satisfaction of Gender rate by DailyRate

# In[80]:


Y1DR=np.array(lstmy())
Y2DR=np.array(lstfmy())


# ## Call t-test method

# In[81]:


stats.ttest_ind(Y1DR,Y2DR)


# ##### We have 
# ##### t_value < t_crit (-0.656 < 2,76)
# ##### p_value > 0.05 (0.0512 > 0.05)
# ##### ==> We can't reject the H0 ( Male and Female have a same result at satisfaction rate by DailyRate)

# # Apply t-test method with data Satisfaction of Gender rate by Age

# In[95]:


Z1 = Male1["Age"]
def lstmz():
    lstC1 =[]
    for x in Z1:
        lstC1.append(x)
    return lstC1
Z2 = FM1['Age']
def lstfmz():
    lstC2 =[]
    for x in Z2:
        lstC2.append(x)
    return lstC2


# ### List Satisfaction of Gender rate by Age

# In[72]:


Z1Age=np.array(lstmz())
Z2Age=np.array(lstfmz())


# ## Call t-test method

# In[74]:


stats.ttest_ind(Z1Age,Z2Age)


# ### We have 
# ### t_value < t_crit (-1,945 < 2,76)
# ### p_value > 0.05 (0.052 > 0.05)
# ### ==> We can't reject the H0 ( Male and Female have a same result at satisfaction rate by Age)

# # ---------------------------------------------------------------
# # Conclude

# # 1. What are key factors that are playing into current attrition rate ?
# ## - In the attrition rate, it has some difference factors that effect the attrion but in this test the Gender was used that isn't the thing playing into current attrition rate.
# # 2. What are key factors that are playing into current satisfaction rate ?
# ## - Same the result of question 1, the Gender is not effect much to the satisfaction rate .
# 
