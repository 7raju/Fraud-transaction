#!/usr/bin/env python
# coding: utf-8

# # Fraudulent Transaction Prediction

# The Dataset is to be identify about a transaction to predict whether it is Fraudulent or Not

# We are presented with a labeled dataset of financial transactions, some of which are fraudulent. We will be performing exploratory data analysis on this data, and then creating a classifier model to predict whether a transaction is fraudulent given the included features. The objective of this project is to explain my thought processes in solving this problem, as well as addressing some of the issues that inherently face machine learning models. ("All models are wrong, but some are useful.") Using this notebook, I hope to focus primarily on transparency and clarity rather than raw predictive performance, and readability for an audience without a specialization in data science.

# **Import Libraries**

# In[3]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score
import xgboost as xgb
import sklearn.metrics as metrics

import math
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
#We want our plots to appear in the Notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Read The Dataset
data = pd.read_csv("C:\\Users\\Asus\\raju\\fraud transcation ml project\\Fraud.csv")


# In[5]:


data.head()


# In[6]:


data.tail(10)


# In[7]:


# describe the dataset
data.describe()


# In[8]:


#data structure
print(type(data))
data.shape


# In[9]:


data.columns


# In[10]:


#data types of the features
data.info()


# In[11]:


#count the duplicates
data[data.duplicated()].shape


# In[12]:


#To identify the unique values
data.type.unique()


# In[13]:


numeric_data = data.select_dtypes(include=[np.number])
skew_data = numeric_data.skew()
print(skew_data)


# In[ ]:





# In[14]:


print('Data does not have any NULL value.')
data.isnull().any()


# In[15]:


data.rename(columns={'newbalanceOrig':'newbalanceOrg'},inplace=True)
data.drop(labels=['nameOrig','nameDest'],axis=1,inplace=True)


# The provided data has the financial transaction data as well as the target variable isFraud, which is the actual fraud status of the transaction and isFlaggedFraud is the indicator which the simulation is used to flag the transaction using some threshold value.

# In[16]:


print('Minimum value of Amount, Old/New Balance of Origin/Destination:')
data[[ 'amount','oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 'newbalanceDest']].min()


# In[17]:


print('Maximum value of Amount, Old/New Balance of Origin/Destination:')
data[[ 'amount','oldbalanceOrg', 'newbalanceOrg', 'oldbalanceDest', 'newbalanceDest']].max()


# Data analysis
# 
# Since there are no missing and junk values, there is no need for additional data cleansing, but we still need to perform data analysis since the data contains huge variations in the value in different columns. Normalization will also improve the overall accuracy of the machine learning model.

# # Data analysis

# In[18]:


var = data.groupby('type').amount.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
var.plot(kind='bar')
ax1.set_title("Total amount per transaction type")
ax1.set_xlabel('Type of Transaction')
ax1.set_ylabel('Amount');


# In[19]:


data.loc[data.isFraud == 1].type.unique()


# In[20]:


# Pairwise Pearson correlations
numeric_data = data.select_dtypes(include=[np.number])
correlations = numeric_data.corr(method='pearson')
print(correlations)


# In[21]:


# Drop non-numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Calculate the correlation of 'isFraud' with other numeric columns
correlation_with_is_fraud = numeric_data.corr()['isFraud']
print(correlation_with_is_fraud)


# In[22]:


# Drop non-numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Create a heatmap of the correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), cmap='RdBu', annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[23]:


# Select only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Get the index of the correlation matrix (column names)
top_corr_features = correlation_matrix.index

# Visualize the correlation matrix for specific features
plt.figure(figsize=(10,10))
heatmap = sns.heatmap(data[top_corr_features].corr(), annot=True, cmap='YlGnBu')
plt.title('Correlation Matrix for Specific Features')
plt.show()


# What can we do with this heatmap:
# 
# OldbalanceOrg and NewbalanceOrg are highly correlated.
# OldbalanceDest and NewbalanceDest are highly correlated.
# The sum correlates with isFraud(target variable).
# There is not much relationship between these features, so we need to understand where the relationship between them depends on the type of transaction and the amount. To do this, we need to see the heatmap of fraudulent and non-fraudulent transactions differently.

# In[24]:


fraud = data.loc[data.isFraud == 1]
nonfraud = data.loc[data.isFraud == 0]
fraudcount = fraud.isFraud.count()
nonfraudcount = nonfraud.isFraud.count()


# In[25]:


# Select only numeric columns
numeric_data = fraud.select_dtypes(include=[np.number])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Visualize the correlation matrix using seaborn's heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='RdBu')
plt.title('Correlation Matrix')
plt.show()


# There are 2 flags that stand out to me that are interesting to look at: isFraud and isFlaggedFraud column. Based on the hypothesis, isFraud is an indicator that indicates actual fraudulent transactions, while isFlaggedFraud is that the system is preventing a transaction due to some thresholds being triggered. From the heatmap above, we can see that there is some relationship between the other columns and isFlaggedFraud, hence there must be a relationship between isFraud.

# In[26]:


print('The total number of fraud transaction is {}.'.format(data.isFraud.sum()))
print('The total number of fraud transaction which is marked as fraud {}.'.format(data.isFlaggedFraud.sum()))
print('Ratio of fraud transaction vs non-fraud transaction is 1:{}.'.format(int(nonfraudcount//fraudcount)))


# In[27]:


print('Thus in every 773 transaction there is 1 fraud transaction happening.')
print('Amount lost due to these fraud transaction is ${}.'.format(int(fraud.amount.sum())))


# In[28]:


piedata = fraud.groupby(['isFlaggedFraud']).sum()


# In[29]:


f, axes = plt.subplots(1,1, figsize=(6,6))
axes.set_title("% of fraud transaction detected")
piedata.plot(kind='pie',y='isFraud',ax=axes, fontsize=14,shadow=False,autopct='%1.1f%%');
axes.set_ylabel('');
plt.legend(loc='upper left',labels=['Not Detected','Detected'])
plt.show()


# In[30]:


fig = plt.figure()
axes = fig.add_subplot(1,1,1)
axes.set_title("Fraud transaction which are Flagged Correctly")
axes.scatter(nonfraud['amount'],nonfraud['isFlaggedFraud'],c='g')
axes.scatter(fraud['amount'],fraud['isFlaggedFraud'],c='r')
plt.legend(loc='upper right',labels=['Not Flagged','Flagged'])
plt.show()


# In[31]:


fraud= data.groupby('isFraud').size()
print(fraud)


# In[32]:


fraud=data.isFraud.value_counts(normalize=True)*100
fraud


# In[33]:


false=data[data['isFraud']==1]
true=data[data['isFraud']==0]
n=len(false)/float(len(true))
print('false detection:{}'.format(len(data[data['isFraud']==1])))
print('true detection:{}'.format(len(data[data['isFraud']==0])))


# In[34]:


false=data[data['isFraud']==1]
true=data[data['isFraud']==0]
print('false detection')
print(false.amount.describe()/100,"\n")

print('true detection')
print(true.amount.describe()/100)


# From the above data we can infer that less than 0.13% of the total transaction are fraudulent

# The plot above clearly shows the need for a system that can be fast and reliable to flag a transaction as a fraud. Because the current system allows fraudulent transactions to go through a system that does not label them as fraud. Some data exploration can be useful for testing relationships between objects.

# # Data Visualization for descrete data

# In[35]:


data.isFraud.value_counts().plot(kind='bar')


# In[36]:


#histogram of types of transaction
data['type'].hist()


# In[37]:


data.isFlaggedFraud.value_counts()


# In[38]:


print("individual type of transactions:")
print((data.type.value_counts()/data.type.value_counts().sum())*100)


# In[39]:


data.type.value_counts().plot(kind='pie')
plt.title('Types of Transactions')


# In[40]:


data.step.hist(grid=False)


# # Data exploration

# In[41]:


fraud = data.loc[data.isFraud == 1]
nonfraud = data.loc[data.isFraud == 0]
fraudcount = fraud.isFraud.count()
nonfraudcount = nonfraud.isFraud.count()


# In[42]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(nonfraud['oldbalanceOrg'],nonfraud['amount'],c='g')
ax.scatter(fraud['oldbalanceOrg'],fraud['amount'],c='r')
plt.show()


# In[43]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['oldbalanceOrg'])
ax.scatter(fraud['step'],fraud['oldbalanceDest'])
plt.show()


# In[44]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['newbalanceOrg'])
ax.scatter(fraud['step'],fraud['newbalanceDest'])
plt.show()


# In[45]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['newbalanceDest'])
ax.scatter(fraud['step'],fraud['oldbalanceDest'])
plt.show()


# In[46]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(fraud['step'],fraud['oldbalanceOrg'])
ax.scatter(fraud['step'],fraud['newbalanceOrg'])
plt.show()


# In[47]:


pd.crosstab(data.isFraud, data.type)


# In[48]:


data.type[data.isFraud == 1].value_counts().plot(kind="bar", color=["blue","red"])

plt.title("Number of Fraudulent Transactions");


# From the above crosstab we can infer that fraudulent transactions take place only in CASH_OUT and TRASNFER type of transactions where 4116 of CASH_OUT and 4097 of TRANSFER transactions where fraudulent.

# # To find out the Target variable using manual prediction

# In[50]:


data=pd.read_csv("C:\\Users\\Asus\\raju\\fraud transcation ml project\\Fraud.csv")


# In[51]:


data['merchant'] = data['nameDest'].str.contains('M')
data.head()


# In[52]:


data[['isFraud','merchant']].value_counts()


# In[53]:


data[data['isFraud']==1].head(10)


# In[54]:


# Counts of each transaction type for fraudulent transactions
data[data['isFraud']==1]['type'].value_counts()


# TO identify in PAYMENT MODE

# In[55]:


payment=data[data['type']=='PAYMENT']
payment


# In[56]:


payment.shape


# In[57]:


payment.info()


# In[58]:


data['balancediffOrig'] = data['newbalanceOrig'] - data['oldbalanceOrg']
data['balancediffDest'] = data['newbalanceDest'] - data['oldbalanceDest']
data.head()


# In[59]:


data['Orig_diff_amount']=data['amount']+data['balancediffOrig']
data['dest_diff_amount']=data['amount']+data['balancediffDest']
data.head()


# In[60]:


def not_fraud(data):
    lab=[]
    for i in range(len(data)):
        l=int(0)
        lab.append(l)
    return lab


# In[61]:


def fraud(data):
    lab=[]
    for i in range(len(data)):
        l=int(1)
        lab.append(l)
    return lab


# In[62]:


payment["Fraud_Id"]=data[data["type"]=="PAYMENT"]['isFraud']


# In[63]:


payment["Fraud_Id"].value_counts()


# In[64]:


payment["Fraud_Id"].unique()


# There is no Fraud cases in PAYMENT MODE

# # Cash in

# In[65]:


cashin=data[data['type']=='CASH_IN']
cashin


# In[66]:


cashin["Fraud_Id"]=data[data["type"]=="CASH_IN"]['isFraud']


# In[67]:


cashin.Fraud_Id.value_counts()


# In[68]:


cashin.Fraud_Id.unique()


# # Debit

# In[69]:


debit=data[data['type']=='DEBIT']
debit


# In[70]:


debit["Fraud_Id"]=data[data["type"]=="DEBIT"]['isFraud']


# In[71]:


debit.Fraud_Id.value_counts()


# In[72]:


debit.Fraud_Id.unique()


# # Cash out

# In[73]:


cashout=data[data['type']=='CASH_OUT']
cashout


# In[74]:


cashout.shape


# Cash out NOT FRAUD TRANSACTION

# In[75]:


cashout_notfraud=cashout[cashout['oldbalanceOrg']==0]
cashout_notfraud1=cashout[(cashout['amount']>=cashout['oldbalanceOrg'])&
          (cashout['balancediffDest']<0) & cashout['oldbalanceOrg']!=0]
cashout_notfraud2=cashout[(cashout['amount']<cashout['oldbalanceOrg'])&
          (cashout['balancediffDest']<0) & cashout['oldbalanceOrg']!=0]
cashout_notfraud3=pd.concat([cashout_notfraud,cashout_notfraud1,cashout_notfraud2],axis=0)


# In[76]:


cashout_notfraud3['Fraud_Id']=not_fraud(cashout_notfraud3)
cashout_notfraud3


# cash out Fraud Transactions

# In[77]:


cashout_fraud=cashout[(cashout['amount']>=cashout['oldbalanceOrg'])&
                 (cashout['amount']==cashout['balancediffDest']) & cashout['oldbalanceOrg']!=0]
cashout_fraud1=cashout[(cashout["amount"]>=cashout["oldbalanceOrg"]) & 
                          (cashout["amount"]<cashout["balancediffDest"]) & cashout["oldbalanceOrg"]!=0]
cashout_fraud2=pd.concat([cashout_fraud,cashout_fraud1],axis=0)


# In[78]:


cashout_fraud2['Fraud_Id']=fraud(cashout_fraud2)
cashout_fraud2.head()


# In[79]:


cashout_true_fraud=pd.concat([cashout_notfraud3,cashout_fraud2],axis=0)
cashout_true_fraud


# In[80]:


cashout_true_fraud.Fraud_Id.value_counts()


# In[81]:


cashout_true_fraud.Fraud_Id.unique()


# # Transfer Mode

# In[82]:


transfer=data[data['type']=='TRANSFER']
transfer


# In[83]:


fraud_transfer1=transfer[(transfer["balancediffOrig"]<=0) & (transfer["balancediffDest"]<=0) & 
             (transfer["Orig_diff_amount"]>transfer["dest_diff_amount"])]
fraud_transfer2=transfer[(transfer["balancediffOrig"]>=0) & (transfer["balancediffDest"]>=0) & 
             (transfer["Orig_diff_amount"]>=transfer["dest_diff_amount"])]
fraud_transfer3=pd.concat([fraud_transfer1,fraud_transfer2],axis=0)
fraud_transfer3.head()


# In[84]:


fraud_transfer3.shape


# In[85]:


transfer[(transfer["balancediffOrig"]>=0) & (transfer["balancediffDest"]>=0) &
             (transfer["Orig_diff_amount"]==transfer["dest_diff_amount"])]
transfer[(transfer["balancediffOrig"]>=0) & (transfer["balancediffDest"]>=0) \
             & (transfer["Orig_diff_amount"]!=transfer["dest_diff_amount"])]
transfer[(transfer["oldbalanceOrg"]==0)]


# In[86]:


transfer["Fraud_Id"]=data[data["type"]=="TRANSFER"]["isFraud"]
transfer


# In[87]:


transfer.Fraud_Id.value_counts()


# In[88]:


transfer.Fraud_Id.unique()


# # final transaction data

# In[89]:


df=pd.concat([payment,debit,cashin,transfer,cashout_true_fraud],axis=0)
df


# In[90]:


df['Fraud_Id'].value_counts()


# In[91]:


df['Fraud_Id'].unique()


# In[92]:


df['type'].value_counts()


# # Data Cleaning

# In[93]:


import pandas as pd
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[94]:


data= df.drop(['nameOrig','nameDest','isFraud','balancediffOrig','balancediffDest','Orig_diff_amount','dest_diff_amount','merchant'],axis=1)
data.tail(20)


# In[95]:


data1=data.copy()
data1['Fraud_Id']=data1['Fraud_Id'].astype(int)


# In[96]:


data1.head()


# In[97]:


px=sns.countplot(x='Fraud_Id',data=data1)
print(data1['Fraud_Id'].value_counts())


# # undersampling

# In[98]:


target='Fraud_Id'


# In[99]:


x=data1.loc[:,data1.columns!=target]
y=data1.loc[:,data1.columns==target]


# In[100]:


fraud_df_len=len(y[y[target]==1])
print (fraud_df_len)


# In[101]:


fraud_df = data1[data1[target]==1].index
print (fraud_df)


# In[102]:


non_fraud_df = data1[data1[target] == 0].index
print (non_fraud_df)


# In[103]:


random_df=np.random.choice(non_fraud_df,fraud_df_len,replace=False)
print(len(random_df))


# In[104]:


sampling = np.concatenate([random_df, fraud_df])
under_sampling=data1.loc[sampling]
under_sampling


# In[105]:


ax=sns.countplot(x=target,data=under_sampling)
print(data1[target].value_counts())
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# # model performance

# In[106]:


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
label_encoder=preprocessing.LabelEncoder()

under_sampling['type']=label_encoder.fit_transform(under_sampling['type'])
under_sampling['Fraud_Id']=label_encoder.fit_transform(under_sampling['Fraud_Id'])
under_sampling['Fraud_Id'].unique()


# In[107]:


x=under_sampling.drop(['Fraud_Id'],axis=1)
y=under_sampling['Fraud_Id']


# In[108]:


from sklearn.model_selection import train_test_split

np.random.seed(42)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[109]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[110]:


x_train


# In[111]:


x_test


# In[112]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# # Logistic Regression

# In[113]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix

np.random.seed(42)

lr = LogisticRegression().fit(x_train, y_train)
y_pred_lr=lr.predict(x_test)

lr.score(x_test,y_test)


# In[114]:


print(confusion_matrix(y_test,y_pred_lr))
print(classification_report(y_test,y_pred_lr))
print(roc_auc_score(y_test,lr.predict_proba(x_test)[:,1]))


# In[115]:


sns.heatmap(confusion_matrix(y_test,y_pred_lr))


# # Confusion Matrix

# In[116]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score

sns.set(font_scale=1.5)

y_pred = lr.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)

def plot_conf_mat(y_test, y_pred):
  fig, ax = plt.subplots(figsize=(10,7))
  ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cbar=False)
  plt.xlabel("True label")
  plt.ylabel("Predicted label")

plot_conf_mat(y_test, y_pred)


# In[117]:


import sklearn.metrics as metrics
print(metrics.classification_report(y_test,y_pred))


# In[118]:


import numpy
import math
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[119]:


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
label_encoder=preprocessing.LabelEncoder()

under_sampling['type']=label_encoder.fit_transform(under_sampling['type'])
under_sampling['Fraud_Id']=label_encoder.fit_transform(under_sampling['Fraud_Id'])
under_sampling['Fraud_Id'].unique()


# # Decision Tree

# In[120]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix


# In[121]:


from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=1)

dt_clf = dt_clf.fit(x_train,y_train)

y_pred_dt = dt_clf.predict(x_test)


# In[122]:


print(confusion_matrix(y_test,y_pred_dt))
print(classification_report(y_test,y_pred_dt))
print(roc_auc_score(y_test,dt_clf.predict_proba(x_test)[:,1]))


# # XGBoost classifier

# In[123]:


xgbclassifier=xgb.XGBClassifier()
xgbclassifier.fit(x_train,y_train)
y_pred_xgb=xgbclassifier.predict(x_test)


# In[124]:


print(confusion_matrix(y_test,y_pred_xgb))
print(classification_report(y_test,y_pred_xgb))
print(roc_auc_score(y_test,xgbclassifier.predict_proba(x_test)[:,1]))


# In[125]:


sns.heatmap(confusion_matrix(y_test,y_pred_xgb))


# # Neuaral Network 

# In[126]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, roc_curve, auc,\
precision_score
ncols = len(x.columns)
hidden_layers = (ncols,ncols,ncols)
max_iter = 1000
MLP = MLPClassifier(hidden_layer_sizes=hidden_layers,max_iter=1000,random_state=42)

# training model
MLP.fit(x_train,y_train)
    
# evaluating model on how it performs on balanced datasets
predictionsMLP = MLP.predict(x_test)
CM_MLP = confusion_matrix(y_test,predictionsMLP)
CR_MLP = classification_report(y_test,predictionsMLP)
fprMLP, recallMLP, thresholdsMLP = roc_curve(y_test, predictionsMLP)
AUC_MLP = auc(fprMLP, recallMLP)
    
resultsMLP = {"Confusion Matrix":CM_MLP,"Classification Report":CR_MLP,"Area Under Curve":AUC_MLP}


# In[127]:


for measure in resultsMLP:
    print(measure,": \n",resultsMLP[measure])


# # Random Forest

# In[128]:


from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
rf = RandomForestClassifier().fit(x_train, y_train)
y_pred_rf=rf.predict(x_test)
rf.score(x_test, y_test)


# In[129]:


# 25 estimators
rf = RandomForestClassifier(n_estimators=25).fit(x_train, y_train)
rf.score(x_test, y_test)


# In[130]:


print(confusion_matrix(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))
print(roc_auc_score(y_test,rf.predict_proba(x_test)[:,1]))


# In[131]:


sns.heatmap(confusion_matrix(y_test,y_pred_rf))


# In[132]:


models={
     "Logistic Regression" : LogisticRegression()
} 

for name,model in models.items():
    model.fit(x_train,y_train)
    print(name + "trained")
    
for name,model in models.items():
    print(name + "{:.2f}%".format(model.score(x_test,y_test)*100))


# In[133]:


models={
     "K-Nearest Neighbors":KNeighborsClassifier()
} 

for name,model in models.items():
    model.fit(x_train,y_train)
    print(name + "trained")
    
for name,model in models.items():
    print(name + "{:.2f}%".format(model.score(x_test,y_test)*100))


# In[134]:


models={
     "Decision Tree"   : DecisionTreeClassifier()
} 

for name,model in models.items():
    model.fit(x_train,y_train)
    print(name + "trained")
    
for name,model in models.items():
    print(name + "{:.2f}%".format(model.score(x_test,y_test)*100))


# In[135]:


models={
    "Random Forest": RandomForestClassifier()
    
} 

for name,model in models.items():
    model.fit(x_train,y_train)
    print(name + "trained")
    
for name,model in models.items():
    print(name + "{:.2f}%".format(model.score(x_test,y_test)*100))


# In[136]:


models={
    "Neural Network": MLPClassifier()
    
} 

for name,model in models.items():
    model.fit(x_train,y_train)
    print(name + "trained")
    
for name,model in models.items():
    print(name + "{:.2f}%".format(model.score(x_test,y_test)*100))


# In[137]:


models={
    "Gradient Boosting":GradientBoostingClassifier()
    
} 

for name,model in models.items():
    model.fit(x_train,y_train)
    print(name + "trained")
    
for name,model in models.items():
    print(name + "{:.2f}%".format(model.score(x_test,y_test)*100))


# In[138]:


# compare models
models = []
models.append(('LR', LogisticRegression(max_iter=400)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('MC', MLPClassifier()))
models.append(('XGB',xgb.XGBClassifier()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10)
	cv_results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# In[139]:


# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(222)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# In[140]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


scaler = StandardScaler().fit(x_train)
x_train = scaler.fit_transform(x_train)
model = xgb.XGBClassifier()
model.fit(x_train, y_train)


# In[141]:


# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(x_test)
predictions = model.predict(rescaledValidationX)
predictions


# In[142]:


print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))


# As per our model building roc_auc_score is high in XG BOOST
# 
# 
# so the finalized best model is XGBOOST

# # Deployment

# Model Saving

# In[143]:


from pickle import dump
from pickle import load


# In[144]:


# save the model to disk
filename = 'trained_model.sav'
dump(model, open(filename, 'wb'))


# In[145]:


# load the model from disk
loaded_model = load(open('trained_model.sav', 'rb'))



# In[146]:


input_data=(1,181.0,181.0,0.0,21182.0,0,0,2,)

#Changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
#scaler=StandardScaler()
#std_data=scaler.fit_transform(input_data_reshaped)
#print(std data)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
    print('The Transaction is not Fraudulent')
else:
    print('The Transaction is Fraudulent')


# In[147]:


x


# In[148]:


lr_class= LogisticRegression()
lr_class.fit(x,y)


# In[149]:


print('Accuracy Score:', np.round(lr_class.score(x, y), decimals = 3))


# In[150]:


from joblib import dump, load
import joblib


# In[151]:


joblib.dump(lr_class, 'fraudmodel.pkl') 


# In[152]:


loaded_model = joblib.load('fraudmodel.pkl')


# In[153]:


x.columns


# In[154]:


real_values = np.array([743, 2, 339682.13, 339682.13, 0.0, 0.00, 339682.13,0.098,]).reshape(1, -1)


# In[155]:


loaded_model.predict(real_values)


# In[ ]:




