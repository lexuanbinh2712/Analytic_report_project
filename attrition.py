#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from scipy import stats


# In[47]:


#date 
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[4]:


df


# In[5]:


#chose the attrition to get analysis and abuse to t-test
att = df[df['Attrition']=="Yes"]


# In[6]:


att


# In[7]:


att.set_index('DistanceFromHome')


# In[8]:


att.describe()


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


# In[11]:


cont_col = []
for column in att.columns:
    if att[column].dtypes != object and att[column].nunique() > 30:
        print(f"{column} : Minimum: {att[column].min()}, Maximum: {att[column].max()}")
        cont_col.append(column)
        print("====================================")


# In[12]:


plt.subplots(12,figsize=[12,6])
plt.subplot(121)
sns.boxplot(x='Gender',y='DailyRate',data=att)
plt.title("Gender Rate by Daily Rate")
plt.subplot(122)
sns.distplot(df['DailyRate'],color='green')
plt.xlim(0,1750)
sns.despine()


# In[13]:


plt.subplots(12,figsize=[12,6])
plt.subplot(121)
sns.boxplot(x='Gender',y='Age',data=att)
plt.title("Gender Rate by Age")
plt.subplot(122)
sns.distplot(df['Age'],color='purple')
plt.xlim(18,58)
sns.despine()


# In[14]:


plt.figure(figsize=(20,20))
sns.heatmap(att.corr(),annot=True)


# In[15]:


#data list attrition of male and female 
Male = att[att['Gender']=='Male']
FM = att[att['Gender']=='Female']


# In[16]:


#data attrition of Gender rate by distance from home
A1 = Male["DistanceFromHome"]
A2 = FM['DistanceFromHome']
def lstm():
    lst =[]
    for x in A1:
        lst.append(x)
    return lst


# In[17]:


def lstfm():
    lst1 =[]
    for x in A2:
        lst1.append(x)
    return lst1


# In[18]:


MDFH=np.array(lstm())


# In[19]:


FMDFH=np.array(lstfm())


# In[20]:


stats.ttest_ind(MDFH,FMDFH)


# In[21]:


Male["DailyRate"]


# In[22]:


#data attrition of Gender rate by Daily Rate
B1 = Male["DailyRate"]
B2 = FM['DailyRate']
def lstm1():
    lstB1 =[]
    for x in B1:
        lstB1.append(x)
    return lstB1


# In[23]:


def lstfm1():
    lstB2 =[]
    for x in B2:
        lstB2.append(x)
    return lstB2


# In[24]:


MDR=np.array(lstm1())
FMDR=np.array(lstfm1())


# In[25]:


stats.ttest_ind(MDR,FMDR)


# In[26]:


#data attrition of Gender rate by Age
C1 = Male["Age"]
C2 = FM['Age']
def lstm2():
    lstC1 =[]
    for x in C1:
        lstC1.append(x)
    return lstC1


# In[27]:


def lstfm2():
    lstC2 =[]
    for x in C2:
        lstC2.append(x)
    return lstC2


# In[28]:


MAge=np.array(lstm2())
FMAge=np.array(lstfm2())


# In[ ]:





# In[29]:


stats.ttest_ind(MAge,FMAge)


# In[52]:


att1 = df[df['Attrition']=="No"]


# In[53]:


att1


# In[54]:


att1.describe()


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


# In[57]:


cont_col = []
for column in att1.columns:
    if att1[column].dtypes != object and att1[column].nunique() > 30:
        print(f"{column} : Minimum: {att1[column].min()}, Maximum: {att1[column].max()}")
        cont_col.append(column)
        print("====================================")


# In[58]:


plt.subplots(12,figsize=[12,6])
plt.subplot(121)
sns.boxplot(x='Gender',y='DailyRate',data=att1)
plt.title("Gender Rate by Daily Rate")
plt.subplot(122)
sns.distplot(df['DailyRate'],color='green')
plt.xlim(102,1500)
sns.despine()


# In[59]:


plt.subplots(12,figsize=[12,6])
plt.subplot(121)
sns.boxplot(x='Gender',y='Age',data=att1)
plt.title("Gender Rate by Age")
plt.subplot(122)
sns.distplot(df['Age'],color='purple')
plt.xlim(18,58)
sns.despine()


# In[60]:


plt.figure(figsize=(20,20))
sns.heatmap(att1.corr(),annot=True)


# In[61]:


#data list satisfaction of male and female 
Male1 = att1[att1['Gender']=='Male']
FM1 = att1[att1['Gender']=='Female']


# In[62]:


#data satisfaction of Gender rate by distance from home
X1 = Male1["DistanceFromHome"]
X2 = FM1['DistanceFromHome']
def lstmx():
    lst =[]
    for x in X1:
        lst.append(x)
    return lst


# In[63]:


def lstfmx():
    lst1 =[]
    for x in X2:
        lst1.append(x)
    return lst1


# In[64]:


X1DFH=np.array(lstmx())
X2DFH=np.array(lstfmx())


# In[65]:


stats.ttest_ind(X1DFH,X2DFH)


# In[66]:


#data attrition of Gender rate by Daily Rate
Y1 = Male1["DailyRate"]
Y2 = FM1['DailyRate']
def lstmy():
    lstB1 =[]
    for x in Y1:
        lstB1.append(x)
    return lstB1


# In[67]:


def lstfmy():
    lstB2 =[]
    for x in Y2:
        lstB2.append(x)
    return lstB2


# In[68]:


Y1DR=np.array(lstmy())
Y2DR=np.array(lstfmy())


# In[69]:


stats.ttest_ind(Y1DR,Y2DR)


# In[70]:


#data attrition of Gender rate by Age
Z1 = Male1["Age"]
Z2 = FM1['Age']
def lstmz():
    lstC1 =[]
    for x in Z1:
        lstC1.append(x)
    return lstC1


# In[71]:


def lstfmz():
    lstC2 =[]
    for x in Z2:
        lstC2.append(x)
    return lstC2


# In[72]:


Z1Age=np.array(lstmz())
Z2Age=np.array(lstfmz())


# In[73]:


stats.ttest_ind(Z1Age,Z2Age)

