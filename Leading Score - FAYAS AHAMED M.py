#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler


# # 1 : Loading and Cleaning Data

# # 1.1 Import Data

# In[2]:


ld = pd.read_csv("leads.csv")


# In[3]:


ld


# # 1.2 Inspect the dataframe

# This helps to give a good idea of the dataframes.

# In[19]:


ld.info()


# In[5]:


ld.head()


# In[7]:


ld.shape


# In[8]:


ld.describe()


# # 1.3 Cleaning the dataframe

# In[26]:


ld.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[27]:


ld = ld.replace('Select', np.nan)


# In[28]:


#checking null values in each rows

ld.isnull().sum()


# In[29]:


#checking percentage of null values in each column

round(100*(ld.isnull().sum()/len(ld.index)), 2)


# In[31]:


#dropping cols with more than 45% missing values

cols=ld.columns

for i in cols:
    if((100*(ld[i].isnull().sum()/len(ld.index))) >= 45):
        ld.drop(i, 1, inplace = True)


# In[32]:


#checking null values percentage

round(100*(ld.isnull().sum()/len(ld.index)), 2)


# # Categorical Attributes Analysis:

# In[33]:


#checking value counts of Country column

ld['Country'].value_counts(dropna=False)


# In[35]:


#plotting spread of Country columnn 
plt.figure(figsize=(15,5))
s1=sns.countplot(ld.Country, hue=ld.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[36]:


# Since India is the most common occurence among the non-missing values we can impute all missing values with India

ld['Country'] = ld['Country'].replace(np.nan,'India')


# In[37]:


#plotting spread of Country columnn after replacing NaN values

plt.figure(figsize=(15,5))
s1=sns.countplot(ld.Country, hue=ld.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# As we can see the Number of Values for India are quite high (nearly 97% of the Data), this column can be dropped

# In[38]:


#creating a list of columns to be droppped

cols_to_drop=['Country']


# In[39]:


ld['City'].value_counts(dropna=False)


# In[40]:


ld['City'] = ld['City'].replace(np.nan,'Mumbai')


# In[41]:


#plotting spread of City columnn after replacing NaN values

plt.figure(figsize=(10,5))
s1=sns.countplot(ld.City, hue=ld.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[42]:


#checking value counts of Specialization column

ld['Specialization'].value_counts(dropna=False)


# In[43]:


# Lead may not have mentioned specialization because it was not in the list or maybe they are a students 
# and don't have a specialization yet. So we will replace NaN values here with 'Not Specified'

ld['Specialization'] = ld['Specialization'].replace(np.nan, 'Not Specified')


# In[44]:


#plotting spread of Specialization columnn 

plt.figure(figsize=(15,5))
s1=sns.countplot(ld.Specialization, hue=ld.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# We see that specialization with Management in them have higher number of leads as well as leads converted. So this is definitely a significant variable and should not be dropped.

# In[49]:


ld['Specialization'] = ld['Specialization'].replace(['Finance Management','Human Resource Management',
                                                           'Marketing Management','Operations Management',
                                                           'IT Projects Management','Supply Chain Management',
                                                    'Healthcare Management','Hospitality Management',
                                                           'Retail Management'] ,'Management_Specializations')  


# In[50]:


#visualizing count of Variable based on Converted value


plt.figure(figsize=(15,5))
s1=sns.countplot(ld.Specialization, hue=ld.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[51]:


#What is your current occupation

ld['What is your current occupation'].value_counts(dropna=False)


# In[52]:


#imputing Nan values with mode "Unemployed"

ld['What is your current occupation'] = ld['What is your current occupation'].replace(np.nan, 'Unemployed')


# In[53]:


#checking count of values
ld['What is your current occupation'].value_counts(dropna=False)


# In[54]:


#visualizing count of Variable based on Converted value

s1=sns.countplot(ld['What is your current occupation'], hue=ld.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# Working Professionals going for the course have high chances of joining it.
# Unemployed leads are the most in terms of Absolute numbers.

# In[55]:


#checking value counts

ld['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[56]:


#replacing Nan values with Mode "Better Career Prospects"

ld['What matters most to you in choosing a course'] = ld['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[57]:


#visualizing count of Variable based on Converted value

s1=sns.countplot(ld['What matters most to you in choosing a course'], hue=ld.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[58]:


#checking value counts of variable
ld['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[60]:


#Here again we have another Column that is worth Dropping. So we Append to the cols_to_drop List
cols_to_drop.append('What matters most to you in choosing a course')
cols_to_drop


# In[61]:


#checking value counts of Tag variable
ld['Tags'].value_counts(dropna=False)


# In[62]:


#replacing Nan values with "Not Specified"
ld['Tags'] = ld['Tags'].replace(np.nan,'Not Specified')


# In[63]:


#visualizing count of Variable based on Converted value

plt.figure(figsize=(15,5))
s1=sns.countplot(ld['Tags'], hue=ld.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[66]:


#replacing tags with low frequency with "Other Tags"
ld['Tags'] = ld['Tags'].replace(
    ['In confusion whether part time or DLP', 'in touch with EINS', 'Diploma holder (Not Eligible)',
     'Approached upfront', 'Graduation in progress', 'number not provided', 'opp hangup', 'Still Thinking',
     'Lost to Others', 'Shall take in the next coming month', 'Lateral student', 'Interested in Next batch',
     'Recognition issue (DEC approval)', 'Want to take admission but has financial problems',
     'University not recognized'],
    'Other_Tags')

ld['Tags'] = ld['Tags'].replace(
    ['switched off', 'Already a student', 'Not doing further education', 'invalid number',
     'wrong number given', 'Interested  in full time MBA'],
    'Other_Tags')


# In[67]:


#checking percentage of missing values
round(100*(ld.isnull().sum()/len(ld.index)), 2)


# In[68]:


#checking value counts of Lead Source column

ld['Lead Source'].value_counts(dropna=False)


# In[69]:


#replacing Nan Values and combining low frequency values
ld['Lead Source'] = ld['Lead Source'].replace(np.nan,'Others')
ld['Lead Source'] = ld['Lead Source'].replace('google','Google')
ld['Lead Source'] = ld['Lead Source'].replace('Facebook','Social Media')
ld['Lead Source'] = ld['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM'] ,'Others')   


# We can group some of the lower frequency occuring labels under a common label 'Others'

# In[70]:


#visualizing count of Variable based on Converted value
plt.figure(figsize=(15,5))
s1=sns.countplot(ld['Lead Source'], hue=ld.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# Inference
# Maximum number of leads are generated by Google and Direct traffic.
# Conversion Rate of reference leads and leads through welingak website is high.
# To improve overall lead conversion rate, focus should be on improving lead converion of olark chat, organic search, direct traffic, and google leads and generate more leads from reference and welingak website.

# In[71]:


# Last Activity:

ld['Last Activity'].value_counts(dropna=False)


# In[72]:


#replacing Nan Values and combining low frequency values

ld['Last Activity'] = ld['Last Activity'].replace(np.nan,'Others')
ld['Last Activity'] = ld['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                        'Had a Phone Conversation', 
                                                        'Approached upfront',
                                                        'View in browser link Clicked',       
                                                        'Email Marked Spam',                  
                                                        'Email Received','Resubscribed to emails',
                                                         'Visited Booth in Tradeshow'],'Others')


# In[73]:


# Last Activity:

ld['Last Activity'].value_counts(dropna=False)


# In[74]:


#Check the Null Values in All Columns:
round(100*(ld.isnull().sum()/len(ld.index)), 2)


# In[75]:


#Drop all rows which have Nan Values. Since the number of Dropped rows is less than 2%, it will not affect the model
ld = ld.dropna()


# In[76]:


#Checking percentage of Null Values in All Columns:
round(100*(ld.isnull().sum()/len(ld.index)), 2)


# In[77]:


#Lead Origin
ld['Lead Origin'].value_counts(dropna=False)


# In[78]:


#visualizing count of Variable based on Converted value

plt.figure(figsize=(8,5))
s1=sns.countplot(ld['Lead Origin'], hue=ld.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# Inference
# API and Landing Page Submission bring higher number of leads as well as conversion.
# Lead Add Form has a very high conversion rate but count of leads are not very high.
# Lead Import and Quick Add Form get very few leads.
# In order to improve overall lead conversion rate, we have to improve lead converion of API and Landing Page Submission origin and generate more leads from Lead Add Form.

# In[79]:


#Do Not Email & Do Not Call
#visualizing count of Variable based on Converted value

plt.figure(figsize=(15,5))

ax1=plt.subplot(1, 2, 1)
ax1=sns.countplot(ld['Do Not Call'], hue=ld.Converted)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)

ax2=plt.subplot(1, 2, 2)
ax2=sns.countplot(ld['Do Not Email'], hue=ld.Converted)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
plt.show()


# In[80]:


#checking value counts for Do Not Call
ld['Do Not Call'].value_counts(dropna=False)


# In[81]:


#checking value counts for Do Not Email
ld['Do Not Email'].value_counts(dropna=False)


# We Can append the Do Not Call Column to the list of Columns to be Dropped since > 90% is of only one Value

# In[82]:


cols_to_drop.append('Do Not Call')
cols_to_drop


# In[83]:


# IMBALANCED VARIABLES THAT CAN BE DROPPED
ld.Search.value_counts(dropna=False)


# In[84]:


ld.Magazine.value_counts(dropna=False)


# In[96]:


ld['Newspaper Article'].value_counts(dropna=False)


# In[88]:


ld['X Education Forums'].value_counts(dropna=False)


# In[89]:


ld['Newspaper'].value_counts(dropna=False)


# In[90]:


ld['Digital Advertisement'].value_counts(dropna=False)


# In[91]:


ld['Through Recommendations'].value_counts(dropna=False)


# In[92]:


ld['Receive More Updates About Our Courses'].value_counts(dropna=False)


# In[93]:


ld['Update me on Supply Chain Content'].value_counts(dropna=False)


# In[94]:


ld['Get updates on DM Content'].value_counts(dropna=False)


# In[95]:


ld['I agree to pay the amount through cheque'].value_counts(dropna=False)


# In[97]:


ld['A free copy of Mastering The Interview'].value_counts(dropna=False)


# In[98]:


#adding imbalanced columns to the list of columns to be dropped

cols_to_drop.extend(['Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
                 'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                 'Update me on Supply Chain Content',
                 'Get updates on DM Content','I agree to pay the amount through cheque'])


# In[99]:


#checking value counts of last Notable Activity
ld['Last Notable Activity'].value_counts()


# In[102]:


#clubbing lower frequency values

ld['Last Notable Activity'] = ld['Last Notable Activity'].replace(['Had a Phone Conversation',
                                                                       'Email Marked Spam',
                                                                         'Unreachable',
                                                                         'Unsubscribed',
                                                                         'Email Bounced',                                                                    
                                                                       'Resubscribed to emails',
                                                                       'View in browser link Clicked',
                                                                       'Approached upfront', 
                                                                       'Form Submitted on Website', 
                                                                       'Email Received'],'Other_Notable_activity')


# In[103]:


#visualizing count of Variable based on Converted value

plt.figure(figsize = (14,5))
ax1=sns.countplot(x = "Last Notable Activity", hue = "Converted", data = ld)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
plt.show()


# In[104]:


#checking value counts for variable

ld['Last Notable Activity'].value_counts()


# In[105]:


#list of columns to be dropped
cols_to_drop


# In[106]:


#dropping columns
ld = ld.drop(cols_to_drop,1)
ld.info()


# # Numerical Attributes Analysis:

# In[107]:


#Check the % of Data that has Converted Values = 1:

Converted = (sum(ld['Converted'])/len(ld['Converted'].index))*100
Converted


# In[108]:


#Checking correlations of numeric values
# figure size
plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(ld.corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[109]:


#Total Visits
#visualizing spread of variable

plt.figure(figsize=(6,4))
sns.boxplot(y=ld['TotalVisits'])
plt.show()


# In[110]:


#checking percentile values for "Total Visits"

ld['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[111]:


#Outlier Treatment: Remove top & bottom 1% of the Column Outlier values

Q3 = ld.TotalVisits.quantile(0.99)
ld = ld[(ld.TotalVisits <= Q3)]
Q1 = ld.TotalVisits.quantile(0.01)
ld = ld[(ld.TotalVisits >= Q1)]
sns.boxplot(y=ld['TotalVisits'])
plt.show()


# In[112]:


ld.shape


# # Check for the Next Numerical Column:

# In[113]:


#checking percentiles for "Total Time Spent on Website"

ld['Total Time Spent on Website'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[114]:


#visualizing spread of numeric variable

plt.figure(figsize=(6,4))
sns.boxplot(y=ld['Total Time Spent on Website'])
plt.show()


# Since there are no major Outliers for the above variable we don't do any Outlier Treatment for this above Column

# # Check for Page Views Per Visit:

# In[116]:


#checking spread of "Page Views Per Visit"

ld['Page Views Per Visit'].describe()


# In[117]:


#visualizing spread of numeric variable

plt.figure(figsize=(6,4))
sns.boxplot(y=ld['Page Views Per Visit'])
plt.show()


# In[118]:


#Outlier Treatment: Remove top & bottom 1% 

Q3 = ld['Page Views Per Visit'].quantile(0.99)
ld = ld[ld['Page Views Per Visit'] <= Q3]
Q1 = ld['Page Views Per Visit'].quantile(0.01)
ld = ld[ld['Page Views Per Visit'] >= Q1]
sns.boxplot(y=ld['Page Views Per Visit'])
plt.show()


# In[119]:


ld.shape


# In[120]:


#checking Spread of "Total Visits" vs Converted variable
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = ld)
plt.show()


# Inference
# 
# Median for converted and not converted leads are the close.
# 
# Nothng conclusive can be said on the basis of Total Visits

# In[121]:


#checking Spread of "Total Time Spent on Website" vs Converted variable

sns.boxplot(x=ld.Converted, y=ld['Total Time Spent on Website'])
plt.show()


# Inference
# 
# Leads spending more time on the website are more likely to be converted.
# 
# Website should be made more engaging to make leads spend more time.

# In[122]:


#checking Spread of "Page Views Per Visit" vs Converted variable

sns.boxplot(x=ld.Converted,y=ld['Page Views Per Visit'])
plt.show()


# Inference
# 
# Median for converted and unconverted leads is the same.
# 
# Nothing can be said specifically for lead conversion from Page Views Per Visit

# In[123]:


#checking missing values in leftover columns/

round(100*(ld.isnull().sum()/len(ld.index)),2)


# There are no missing values in the columns to be analyzed further

# # Dummy Variable Creation:

# In[124]:


#getting a list of categorical columns

cat_cols= ld.select_dtypes(include=['object']).columns
cat_cols


# In[125]:


# List of variables to map

varlist =  ['A free copy of Mastering The Interview','Do Not Email']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
ld[varlist] = ld[varlist].apply(binary_map)


# In[127]:


#getting dummies and dropping the first column and adding the results to the master dataframe
dummy = pd.get_dummies(ld[['Lead Origin','What is your current occupation',
                             'City']], drop_first=True)

ld = pd.concat([ld,dummy],1)


# In[128]:


dummy = pd.get_dummies(ld['Specialization'], prefix  = 'Specialization')
dummy = dummy.drop(['Specialization_Not Specified'], 1)
ld = pd.concat([ld, dummy], axis = 1)


# In[135]:


dummy = pd.get_dummies(ld['Lead Source'], prefix  = 'Lead Source')
dummy = dummy.drop(['Lead Source_Others'], 1)
ld = pd.concat([ld, dummy], axis = 1)


# In[138]:


dummy = pd.get_dummies(ld['Tags'], prefix  = 'Tags')
dummy = dummy.drop(['Tags_Not Specified'], 1)
ld = pd.concat([ld, dummy], axis = 1)


# In[140]:


#dropping the original columns after dummy variable creation

ld.drop(cat_cols,1,inplace = True)


# In[141]:


ld.head()


# # Train-Test Split & Logistic Regression Model Building:

# In[142]:


from sklearn.model_selection import train_test_split

# Putting response variable to y
y = ld['Converted']

y.head()

X=ld.drop('Converted', axis=1)


# In[143]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[144]:


X_train.info()


# # Scaling of Data:

# In[145]:


#scaling numeric columns

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_train.head()


# # Model Building using Stats Model & RFE:

# In[152]:


import statsmodels.api as sm


# In[153]:


from sklearn.feature_selection import RFE


# In[156]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
rfe = RFE(estimator=logreg, n_features_to_select=15)  # Running RFE with 15 variables as output
rfe.fit(X_train, y_train)  # Fit RFE on the training data


# In[157]:


rfe.support_


# In[158]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[159]:


#list of RFE supported columns
col = X_train.columns[rfe.support_]
col


# In[160]:


X_train.columns[~rfe.support_]


# In[161]:


#BUILDING MODEL #1

X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[163]:


#BUILDING MODEL #2

X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# Since 'All' the p-values are less we can check the Variance Inflation Factor to see if there is any correlation between the variables

# In[164]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[165]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# There is a high correlation between two variables so we drop the variable with the higher valued VIF value

# In[167]:


#BUILDING MODEL #3
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[168]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# So the Values all seem to be in order so now, Moving on to derive the Probabilities, Lead Score, Predictions on Train Data:

# In[169]:


# Getting the Predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[170]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[171]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[172]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[173]:


from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[174]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[175]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[176]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[177]:


# Let us calculate specificity
TN / float(TN+FP)


# In[178]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[179]:


# positive predictive value 
print (TP / float(TP+FP))


# In[180]:


# Negative predictive value
print (TN / float(TN+ FN))


# # PLOTTING ROC CURVE

# In[181]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[182]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[183]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# The ROC Curve should be a value close to 1. We are getting a good value of 0.97 indicating a good predictive model.

# # Finding Optimal Cutoff Point

# Above we had chosen an arbitrary cut-off value of 0.5. We need to determine the best cut-off value and the below section deals with that:

# In[184]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[185]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[186]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[187]:


#### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[188]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# In[189]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[190]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[191]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[192]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[193]:


# Let us calculate specificity
TN / float(TN+FP)


# Observation:
# Upon reviewing the results, it's evident that the model is performing quite effectively. The ROC curve displays a commendable value of 0.97, indicating robust performance. For the Training Data, the performance metrics are as follows:
# 
# - Accuracy: 92.29%
# - Sensitivity: 91.70%
# - Specificity: 92.66%
# 
# Further analysis includes calculation of additional statistics such as False Positive Rate, Positive Predictive Value, Negative Predictive Value, Precision, and Recall. These measures provide a comprehensive understanding of the model's performance.

# In[194]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[195]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[196]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[197]:


#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion


# In[198]:


##### Precision
TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[199]:


##### Recall
TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[200]:


from sklearn.metrics import precision_score, recall_score


# In[201]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted)


# In[202]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[203]:


from sklearn.metrics import precision_recall_curve


# In[204]:


y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[205]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[206]:


#scaling test set

num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()


# In[207]:


X_test = X_test[col]
X_test.head()


# In[208]:


X_test_sm = sm.add_constant(X_test)


# # PREDICTIONS ON TEST SET

# In[209]:


y_test_pred = res.predict(X_test_sm)


# In[210]:


y_test_pred[:10]


# In[211]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[212]:


# Let's see the head
y_pred_1.head()


# In[213]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[214]:


# Putting CustID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[215]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[216]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[217]:


y_pred_final.head()


# In[218]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})


# In[219]:


y_pred_final.head()


# In[220]:


# Rearranging the columns
y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))


# In[221]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[222]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.3 else 0)


# In[223]:


y_pred_final.head()


# In[224]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[225]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[226]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[227]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[228]:


# Let us calculate specificity
TN / float(TN+FP)


# In[229]:


precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)


# In[233]:


recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# Observation:
# Upon evaluating the model's performance on the Test Data, the following metrics were recorded:
# 
# - Accuracy: 92.78%
# - Sensitivity: 91.98%
# - Specificity: 93.26%
# 
# Final Assessment:
# A comparison between the values obtained for the Training and Test Datasets reveals the following insights:
# 
# Train Data: 
# - Accuracy: 92.29%
# - Sensitivity: 91.70%
# - Specificity: 92.66%
# 
# Test Data: 
# - Accuracy: 92.78%
# - Sensitivity: 91.98%
# - Specificity: 93.26%
# 
# These results indicate that the model effectively predicts the Conversion Rate. As a result, the CEO can have increased confidence in making informed decisions based on the insights provided by this model.
