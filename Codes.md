# <span style="color:darkred"><u><center> **Lead Scoring Case Study**


# <span style="color:IndianRed ;">1. Project Requirement and Artifacts

#### **Online Course Leads**


The online courses company sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website through various websites and search engines to browse for the courses. </br>
Once a person lands on the website, they might indulge in various activities like browsing the courses or filling up a form for the course or watch some videos. When a person fill up a form providing their email address or phone number, they are classified to be a **Lead**.</br>
Once a visit has ben converted into lead, the sales team starts communicating with them to get them converted into buying their courses. From the past experience, the `typical lead conversion rate` at *X education* is noted to be around `30%`.

To improve the conversion rate and hence making the sales team more efficient, the company is intended to identify the most potential leads, as **Hot Leads**. With this, the sales team can focus more on conversion of a potential customer rather than wasting time on those whose chances of conversion is very less.



#### **Working of Lead Conversion Process**

An ideal lead conversion process to be followed at X-education is completed in multiple steps: </br>


1.   Social Media Marketing pulls the visitors on the website
2.   The Visitor fills the form with their contact and interest details
3.   Sales team collects the initial pool of the data 
4.   **Lead Nurturing (Identify the Hot Leads after educating them about the products and benefits)** 
5.   Lead Conversion

Here, as we progress in the stages we are left only with concrete leads, which have higher chances of getting converted. Since, we are already given with the lead pool, **the main focus of our case study will revolve around the Lead Nurturing / Hot Leads Identification**.





#### **Business Objective**
This Lead scoring project for a course content will help them with less time wastage over the poor conversion score. With this, we will try:
1.	To build a model wherein we can assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance. 

2.	The model build shall be able to profide atleast 80% efficient leads (conversion rate ~ 80%) after they have passed through the model.


#### **Project Goal**
With this Case study, we will try:

1.	Build a Logistic Regression model to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. A higher score means higher chances of getting converted and vice-versa.
2.	Identify the challenges and strategy to overcome them. The model should be self-adjusting to identify such challenges and overcome them as and when required in the future.

3.	Present the recommendations to better on identification and conversion on their merit..


# <span style="color:IndianRed ;">2. Reading and Understanding the Data

Let us import all the required libraries, and read the dataset

# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# Importing all required packages

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
    
```
pd.options.display.float_format = '{:.2f}'.format
```
    
    
# Setting display options for pandas columns and rows 
```
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)
```
    
    
# Importing dataset
```
lead_df = pd.read_csv('Leads.csv')
lead_df.head()
```

    
    
# inspect the dataframe
```
print(lead_df.info())
```


    
# shape of dataframe

    
```
print(lead_df.shape)
```

# inspect the dataframe to check null values percentage in each column


```
percent_missing = round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)
print(percent_missing)
```

# inspect the dataframe for Numerical variables

```
lead_df.describe()
```

# <span style="color:IndianRed ;">3. Data Cleaning

# check for percentage of null values in each column


```
percent_missing = round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)
print(percent_missing)
```


# Check the duplicate values


```
lead_df[lead_df.duplicated(keep=False)]
```
There are no duplicate values in the data.


#### Handling **Select** in variables
Following columns have a value called select. 
Convert those values as nan since the customer has not selected any options for these columns while entering the data.

•	Specialization </br>
•	How did you hear about X Education</br>
•	Lead Profile</br>
•	City



# Converting 'Select' values to NaN.

lead_df = lead_df.replace('Select', np.nan)


# Re-check for percentage of null values in each column

percent_missing = round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)
print(percent_missing)

# Let us drop the following columns that have more than 30% null values
cols = lead_df.columns

for i in cols:
    if((100*(lead_df[i].isnull().sum()/len(lead_df.index))) >= 30):
        lead_df.drop(i, 1, inplace = True)


# check for percentage of null values in each column after dropping columns with more than 30% null values

percent_missing = round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)
print(percent_missing)

Following columns have null values :
- Lead Source
- Total Visits
- Page Views Per Visit
- Last Activity    
- Country
- What is your current occupation
- What matters most to you in choosing a course
    
Let us see and decide whather we need to impute values in the above column or drop the columns.

# Lets check the number of unique value counts for each values of remaining variables

lead_df.nunique().sort_values()

Drop the Not Required columns, which are clearly having no impact on conversion. The first five with only one option for variables will not help in modeling while Prospect ID and Lead Number are just indicative key for each leads.


•	Get updates on DM Content </br>
•	I agree to pay the amount through cheque </br>
•	Receive More Updates About Our Courses  </br>
•	Magazine </br>
•	Update me on Supply Chain Content</br>
•	Prospect ID</br>
•	Lead Number</br>


# Dropping the above mentioned columns
lead_df.drop(['Get updates on DM Content', 'I agree to pay the amount through cheque', 'Receive More Updates About Our Courses', 'Magazine','Update me on Supply Chain Content','Prospect ID', 'Lead Number'], 1, inplace = True)

# Lets check the value counts for each values in the country column

lead_df.Country.value_counts()

# check the percentage of India as value in the country column

country_percentage = round(100*len(lead_df[lead_df['Country'] == 'India'])/len(lead_df['Country']),2)
print(country_percentage)

Since **India** occurs around 70% of times in the Country column, and country column also has around 27% as missing values, the mode imputation will clearly result in data imbalance, hence we can drop this variable.

lead_df = lead_df.drop(['Country'], axis=1)

# Check the value counts for the column Lead Source

lead_df['Lead Source'].value_counts()

It is noticed that Google appears twice with different cases. Hence we shall convert all rows with value "Google" to the same case. </br>
Also since "Google" has the major chunk of data, we can impute the null values with Google

# Change the google and nan to Google
lead_df['Lead Source'] = lead_df['Lead Source'].replace('google', 'Google')

lead_df['Lead Source'] = lead_df['Lead Source'].replace(np.nan, 'Google')

# Re-check the value counts for the column Lead Source

lead_df['Lead Source'].value_counts()

# Check the value counts for the column Total Visits
lead_df['TotalVisits'].value_counts()

# impute the null values in TotalVisitsby the median of column 

lead_df['TotalVisits'] = lead_df['TotalVisits'].fillna(lead_df['TotalVisits'].median())



# Check the value counts for Page Views Per Visit

lead_df['Page Views Per Visit'].value_counts()

# impute the null values in 'Page Views Per Visit' by the median value

lead_df['Page Views Per Visit'] = lead_df['Page Views Per Visit'].fillna(lead_df['Page Views Per Visit'].median())




# Check the value counts for the column Last Activity

lead_df['Last Activity'].value_counts()

# Since we do not have any information of what the last activity of the customer would have been, we can add a new category called 'Not Sure' for the null values

lead_df['Last Activity'] = lead_df['Last Activity'].replace(np.nan, 'Not Sure')

# Check the value counts for the column "What is your current Occupation"

lead_df['What is your current occupation'].value_counts()

# Since no information has been provided Current Occupation, we can add a new category called No Information and set that as value for the null columns

lead_df['What is your current occupation'] = lead_df['What is your current occupation'].replace(np.nan, 'No Information')

# Check the value counts for the column What matters most to you in choosing a course 

lead_df['What matters most to you in choosing a course'].value_counts()

matters_most_percentage = round(100*len(lead_df[lead_df['What matters most to you in choosing a course'] 
                            == 'Better Career Prospects'])/len(lead_df['What matters most to you in choosing a course']),2)
print(matters_most_percentage)

The Better Career Prospects occurs around 70% of times in the What matters most to you in choosing a course column. Also, column has around 29% as missing values, we shall go ahead and drop the column due to data imbalance.

# Drop the variable

lead_df = lead_df.drop(['What matters most to you in choosing a course'], axis=1)

# Re-check for percentage of null values in each column

percent_missing = round(100*(lead_df.isnull().sum()/len(lead_df.index)), 2)
print(percent_missing)

All the null values in the columns now have either been imputed or we have dropped the columns which have more than 70% data 
concentrated towards one value

# Final Shape:
lead_df.shape

##### Outlier Treatment

# Checking outliers at 25%,50%,75%,90%,95% and above
lead_df.describe(percentiles=[.25,.5,.75,.90,.95,.99])

From the above, it can be seen that outlier exists in the columns TotalVisits and Page Views Per Visit columns.

# Check the outliers in all the numeric columns

plt.figure(figsize=(20, 25))
plt.subplot(4,3,1)
sns.boxplot(y = 'TotalVisits', data = lead_df)
plt.subplot(4,3,2)
sns.boxplot(y = 'Total Time Spent on Website', data = lead_df)
plt.subplot(4,3,3)
sns.boxplot(y = 'Page Views Per Visit', data = lead_df)
plt.show()

# Removing values beyond 99% for Total Visits

nn_quartile_total_visits = lead_df['TotalVisits'].quantile(0.99)
lead_df = lead_df[lead_df["TotalVisits"] < nn_quartile_total_visits]
lead_df["TotalVisits"].describe(percentiles=[.25,.5,.75,.90,.95,.99])

# Checking outliers at 25%,50%,75%,90%,95% and above

lead_df.describe(percentiles=[.25,.5,.75,.90,.95,.99])

# Removing values beyond 99% for page Views Per Visit

nn_quartile_page_visits = lead_df['Page Views Per Visit'].quantile(0.99)
lead_df = lead_df[lead_df["Page Views Per Visit"] < nn_quartile_page_visits]
lead_df["Page Views Per Visit"].describe(percentiles=[.25,.5,.75,.90,.95,.99])

# Checking outliers at 25%,50%,75%,90%,95% and above
lead_df.describe(percentiles=[.25,.5,.75,.90,.95,.99])

# Determine the percentage of data retained

num_data = round(100*(len(lead_df)/9240),2)
print(num_data)

At this point, the data has been cleaned and around 98% of data has been retained

# <span style="color:IndianRed ;">4. Data Analysis
Let us try and understand the data now based on each columns effect on the conversion rates

# Conversion Rate 

plot = sns.catplot(x="Converted", kind="count", data=lead_df, aspect= 0.4);

plt.title('Converted Vs Count', fontsize = 14)
plt.xlabel("Converted", fontsize = 12)
plt.ylabel("Count", fontsize = 12)

# Print the counts

ax = plot.facet_axis(0,0)
for p in ax.patches:        
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), p.get_height(), fontsize=12, va='bottom')
    
plt.show()

From the above graph, there has been a overall conversion rate of around 39%

# Lead Origin

plot = sns.catplot(x="Lead Origin", hue = "Converted", kind="count", data=lead_df, aspect= 1.7);

plt.title('Lead Origin Vs Converted', fontsize = 14)
plt.xlabel("Lead Origin", fontsize = 12)
plt.ylabel("Count", fontsize = 12)

# Print the counts

ax = plot.facet_axis(0,0)
for p in ax.patches:
    
    if np.isnan(p.get_height()):
        height = 0
    else:
        height = p.get_height()
    
    height = int(height)
    ax.text(p.get_x()+p.get_width()/2., height, height, fontsize=12, ha='center', va='bottom')
    
plt.show()

The maximum conversion is for Landing Page Submission, followed by API.
Also there was only one request from quick add form which got converted.

# Lead Source

plot = sns.catplot(x="Lead Source", hue = "Converted", kind="count", data=lead_df, aspect = 3.5);

plt.title('Lead Source Vs Converted', fontsize = 14)
plt.xlabel("Lead Source", fontsize = 12)
plt.ylabel("Count", fontsize = 12)
plt.xticks(rotation=90)

# Print the counts

ax = plot.facet_axis(0,0)
for p in ax.patches:
    
    if np.isnan(p.get_height()):
        height = 0
    else:
        height = p.get_height()
    
    height = int(height)
    ax.text(p.get_x()+p.get_width()/2., height, height, fontsize=12, ha='center', va='bottom')
   
plt.show()

The major conversion in the lead source is from google, direct traffic, and Olarck Chat, ant they are well distributed.

# Do not Email

plot = sns.catplot(x="Do Not Email", hue = "Converted", kind="count", data=lead_df, aspect = 0.6);

plt.title('Do Not Email Vs Converted', fontsize = 14)
plt.xlabel("Do Not Email", fontsize = 12)
plt.ylabel("Count", fontsize = 12)

ax = plot.facet_axis(0,0)
for p in ax.patches:        
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), p.get_height(), fontsize=12, ha='center', va='bottom')

plt.show()

Most of the conversion are from the customers who are marked with emails can be been sent

# Do not Call

plot = sns.catplot(x="Do Not Call", hue = "Converted", kind="count", data=lead_df, aspect = 0.6);

plt.title('Do Not Call Vs Converted', fontsize = 14)
plt.xlabel("Do Not Call", fontsize = 12)
plt.ylabel("Count", fontsize = 12)

# Print the counts

ax = plot.facet_axis(0,0)
for p in ax.patches:
    
    if np.isnan(p.get_height()):
        height = 0
    else:
        height = p.get_height()
    
    height = int(height)
    ax.text(p.get_x()+p.get_width()/2., height, height, fontsize=12, ha='center', va='bottom')

plt.show()

The major conversions happened after calls were made.

# Last Activity

plot = sns.catplot(x="Last Activity", hue = "Converted", kind="count", data=lead_df, aspect = 3.0);

plt.title('Last Activity Vs Converted', fontsize = 14)
plt.xlabel("Last Activity", fontsize = 12)
plt.ylabel("Count", fontsize = 12)
plt.xticks(rotation=90)

# Print the counts

ax = plot.facet_axis(0,0)
for p in ax.patches:
    
    if np.isnan(p.get_height()):
        height = 0
    else:
        height = p.get_height()
    
    height = int(height)
    ax.text(p.get_x()+p.get_width()/2., height, height, fontsize=12, ha='center', va='bottom')

plt.show()

The last activity value of SMS Sent, and email opened  had a major conversion.

# What is your current occupation

plot = sns.catplot(x="What is your current occupation", hue = "Converted", kind="count", data=lead_df, 
                   aspect = 2.0);

plt.title('Current Occupation Vs Converted', fontsize = 14)
plt.xlabel("Current Occupation", fontsize = 12)
plt.ylabel("Count", fontsize = 12)

# Print the counts

ax = plot.facet_axis(0,0)
for p in ax.patches:
    
    if np.isnan(p.get_height()):
        height = 0
    else:
        height = p.get_height()
    
    height = int(height)
    ax.text(p.get_x()+p.get_width()/2., height, height, fontsize=12, ha='center', va='bottom')

plt.show()

Unemployed and persons with missing current occupation data were among the major conversion

# Search

plot = sns.catplot(x="Search", hue = "Converted", kind="count", data=lead_df, aspect = 0.7);

plt.title('Search Vs Converted', fontsize = 14)
plt.xlabel("Search", fontsize = 12)
plt.ylabel("Count", fontsize = 12)

ax = plot.facet_axis(0,0)
for p in ax.patches:        
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), p.get_height(), fontsize=12, ha='center', va='bottom')

plt.show()

Conversion rate is high on leads who are not through search.

# Newspaper Article

plot = sns.catplot(x="Newspaper Article", hue = "Converted", kind="count", data=lead_df, aspect = 0.5);

plt.title('Newspaper Article Vs Converted', fontsize = 14)
plt.xlabel("Newspaper Article", fontsize = 12)
plt.ylabel("Count", fontsize = 12)

ax = plot.facet_axis(0,0)
for p in ax.patches:        
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), p.get_height(), fontsize=12, ha='center', va='bottom')

plt.show()

Since "Newspaper Article" column now has only one value for all rows - "No" , it is safe to drop this column

# Dropping Newspaper Article

lead_df = lead_df.drop(['Newspaper Article'], axis=1)

# X Education Forums

plot = sns.catplot(x="X Education Forums", hue = "Converted", kind="count", data=lead_df, aspect = 0.5);

plt.title('X Education Forums Vs Converted', fontsize = 14)
plt.xlabel("X Education Forums", fontsize = 12)
plt.ylabel("Count", fontsize = 12)

ax = plot.facet_axis(0,0)
for p in ax.patches:        
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), p.get_height(), fontsize=12, ha='center', va='bottom')

plt.show()

Since "X Education Forums" column now has only one value for all rows - "No" , it is safe to drop this column

# Dropping X Education Forum column

lead_df = lead_df.drop(['X Education Forums'], axis=1)

plot = sns.catplot(x="Newspaper", hue = "Converted", kind="count", data=lead_df, aspect = 0.7);

plt.title('Newspaper Vs Converted', fontsize = 14)
plt.xlabel("Newspaper", fontsize = 12)
plt.ylabel("Count", fontsize = 12)

# Print the counts

ax = plot.facet_axis(0,0)
for p in ax.patches:
    
    if np.isnan(p.get_height()):
        height = 0
    else:
        height = p.get_height()
    
    height = int(height)
    ax.text(p.get_x()+p.get_width()/2., height, height, fontsize=12, ha='center', va='bottom')

plt.show()

Since Newspaper column has only one row with "Yes" as the value and further since this lead did not get converted and rest of all the values are "No", we can safely drop the column

# Dropping Newspaper column

lead_df = lead_df.drop(['Newspaper'], axis=1)

# Digital Advertisement

plot = sns.catplot(x="Digital Advertisement", hue = "Converted", kind="count", data=lead_df, aspect = 0.7);

plt.title('Digital Advertisement Vs Converted', fontsize = 14)
plt.xlabel("Digital Advertisement", fontsize = 12)
plt.ylabel("Count", fontsize = 12)

ax = plot.facet_axis(0,0)
for p in ax.patches:        
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), p.get_height(), fontsize=12, ha='center', va='bottom')

plt.show()

It can be noticed above that there were 2 leads that came from digital advertisement of which one lead got converted

# Through Recommendations

plot = sns.catplot(x="Through Recommendations", hue = "Converted", kind="count", data=lead_df, aspect = 0.7);

plt.title('Through Recommendations Vs Converted', fontsize = 14)
plt.xlabel("Through Recommendations", fontsize = 12)
plt.ylabel("Count", fontsize = 12)

ax = plot.facet_axis(0,0)
for p in ax.patches:        
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), p.get_height(), fontsize=12, ha='center', va='bottom')

plt.show()

It can be seen that a total of 6 leads came through recommendations of which 5 leads got converted

# A free copy of Mastering The Interview

plot = sns.catplot(x="A free copy of Mastering The Interview", hue = "Converted", kind="count", data=lead_df,
                   aspect = 0.7);

plt.title('Mastering Interview Copy Vs Converted', fontsize = 14)
plt.xlabel("Mastering Interview Copy", fontsize = 12)
plt.ylabel("Count", fontsize = 12)

ax = plot.facet_axis(0,0)
for p in ax.patches:        
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), p.get_height(), fontsize=12, ha='center', va='bottom')

plt.show()

Conversion rate is high on leads who do not want a free copy of Mastering Interviews

# Last Notable Activity

plot = sns.catplot(x="Last Notable Activity", hue = "Converted", kind="count", data=lead_df, aspect = 3.0);

plt.title('Last Notable Activity Vs Converted', fontsize = 14)
plt.xlabel("Last Notable Activity", fontsize = 12)
plt.ylabel("Count", fontsize = 12)
plt.xticks(rotation=90)

# Print the counts

ax = plot.facet_axis(0,0)
for p in ax.patches:
    
    if np.isnan(p.get_height()):
        height = 0
    else:
        height = p.get_height()
    
    height = int(height)
    ax.text(p.get_x()+p.get_width()/2., height, height, fontsize=12, ha='center', va='bottom')

plt.show()

It can be noticed that the conversion rate is high for "SMS Sent"

# Now check the conversions for all numeric values

plt.figure(figsize=(20,20))
plt.subplot(4,3,1)
sns.barplot(y = 'TotalVisits', x='Converted', data = lead_df)
plt.subplot(4,3,2)
sns.barplot(y = 'Total Time Spent on Website', x='Converted', data = lead_df)
plt.subplot(4,3,3)
sns.barplot(y = 'Page Views Per Visit', x='Converted', data = lead_df)
plt.show()

The conversion rated were high for Total Visits, Total Time Spent on Website and Page Views Per Visit

# <span style="color:IndianRed ;">5. Data Preparation - Creating Dummies

Converting some binary variables (Yes/No) to 0/1

variablelist = ['Do Not Email', 'Do Not Call', 'Search', 'Digital Advertisement', 'Through Recommendations','A free copy of Mastering The Interview']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the columns
lead_df[variablelist] = lead_df[variablelist].apply(binary_map)

###Create Dummy for applicable categorical columns

# Create a dataset with all the variables which needs dummy data creation
lead_dummy=lead_df[['Lead Source','Last Activity','Last Notable Activity','Lead Origin','What is your current occupation']]


# Create dummies
dummy=pd.get_dummies(lead_dummy, drop_first=True)

# Verify
dummy.head()

# concat the lead data with the dummy dataset
lead_df=pd.concat([dummy, lead_df],axis=1)

###Drop the dummyfied columns


#Drop the columns for which we have already created dummy variables
lead_df.drop(['Lead Source','Last Activity','Last Notable Activity','Lead Origin','What is your current occupation'],axis=1,inplace=True)

###Finalised Dataset

#Verify the Finalised Dataset
lead_df.head()

lead_df.shape

# <span style="color:IndianRed ;">6. Correlation Check
Assessing the model with StatsModels

lead_df.corr()

#plotting the heatmap to identify the correlation
plt.figure(figsize=(50,50))


mask = np.array(lead_df.corr())
mask[np.tril_indices_from(mask)] = False
sns.heatmap(lead_df.corr(), mask=mask, vmax=.7,vmin=-.7, square=True, cmap = "YlGnBu");  #removed annot=True for bettter view


**INSIGHTS**:
- Following are the pairs having a little high correlation:
    - Lead_Source_Reference and Lead_Origin_Lead_Add_Form
    - Last_Activity_Email_Link_Clicked and Last_Notable_Activity_Email_Link_Clicked
    - Last_Activity_Email opened and Last_Notable_Activity_Email opened
    - Last_Activity_Had a phone conversation and Last_Notable_Activity_Had a phone conversation
    - Last_Activity_Sms sent and Last_Notable_Activity_Email_Sms sent
    - Last_Activity_Unsubscribed and Last_Notable_Activity_Unsubscribed
    - Occupation_Unemployed and Occupation_Other
- This can be handled during Recursive Function Elimination and while building the model.


# <span style="color:IndianRed ;">7. Test-Train Split


from sklearn.model_selection import train_test_split

# Putting feature variable to X

X = lead_df.drop(['Converted'], axis=1)
X.head()

# Putting response variable to y

y = lead_df['Converted']
y.head()

# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=42)

# <span style="color:IndianRed ;">8. Rescaling the features with MinMax Scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits', 'Total Time Spent on Website','Page Views Per Visit']])

X_train.head()

### Checking the Conversion Rate percentage

converted = (sum(lead_df['Converted'])/len(lead_df['Converted'].index))*100
converted

#### We have almost 39% conversion rate

# <span style="color:IndianRed ;">9. Model Building
Assessing the model with StatsModels

#### First Model

import statsmodels.api as sm

# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()

##Feature Selection Using RFE

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE

# running RFE with 20 variables as output

rfe = RFE(logreg, 20)            
rfe = rfe.fit(X_train, y_train)

rfe.support_

list(zip(X_train.columns, rfe.support_, rfe.ranking_))

# variables shortlisted by RFE

col = X_train.columns[rfe.support_]
col

## Rebuilding Model - Model 2

X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

#drop the most insignificant variable
col = col.drop('What is your current occupation_Housewife',1)


## Rebuilding Model - Model 3

X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()

vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

#drop the most insignificant variable
col = col.drop('Search',1)

## Rebuilding Model - Model 4

X_train_sm = sm.add_constant(X_train[col])
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()

vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

#drop the most insignificant variable
col = col.drop('Page Views Per Visit',1)

## Rebuilding Model - Model 5

X_train_sm = sm.add_constant(X_train[col])
logm5 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm5.fit()
res.summary()

vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

#####Insight:
This Model looks stable with significant p-values and below 5 VIF.

##Preparing the Model for prediction based on Model

# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]

# Reshape

y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]

##### Creating a dataframe with the actual converted flag and the predicted probabilities

y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final['LeadId'] = y_train.index
y_train_pred_final.head()

##### Creating new column 'predicted' with 1 if Converted_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()

## Confusion metrics and Other Parameters
Lets check the confusion metrics and accuracy

from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)

# Predicted     not_converted    converted
# Actual
# not_converted        3397      461
# converted            725       1737

# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))

Accuracy rate is around 81% which is good. However we will also need to calculate the other metrics as we cannot depend only 
on the accuracy metrics

##### Predictive Values Metrices

Let's work on finding the Predictive Metrices like Sensitivity, Specificity, False Positive Rate, Postitive Predictive Value and Negative Predictive Values

TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives

# Let's see the sensitivity of our logistic regression model

TP / float(TP+FN)

# Let us calculate specificity

TN / float(TN+FP)

# Calculate false postive rate - predicting non conversion when leads have converted

print(FP/ float(TN+FP))

# positive predictive value 

print (TP / float(TP+FP))

# Negative predictive value

print (TN / float(TN+ FN))

# <span style="color:IndianRed ;">10. ROC Curve

Plotting the ROC Curve

An ROC curve demonstrates several things:

- It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
- The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
- The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

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

fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_Prob, 
                                         drop_intermediate = False )

draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)

# <span style="color:IndianRed ;">11. Model Evalauation


###Finding Optimal Cutoff Point

Optimal cut off probability is that prob where we get balanced sensitivity and specificity

# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()

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

# Let's plot accuracy, sensitivity and specificity for various probabilities.

cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()

#####Insight:
 From the curve above, 0.38 is the optimum point to take it as a cutoff probability.

# Let us make the final prediction using 0.38 as the cut off

y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_Prob.map( lambda x: 1 if x > 0.38 else 0)
y_train_pred_final.head(20)

# Now let us calculate the lead score

y_train_pred_final['lead_score'] = y_train_pred_final.Converted_Prob.map(lambda x: round(x*100))
y_train_pred_final.head(20)

# checking if 80% cases are correctly predicted based on the converted column.

# get the total of final predicted conversion / non conversion counts from the actual converted rates

checking_df = y_train_pred_final.loc[y_train_pred_final['Converted']==1,['Converted','final_predicted']]
checking_df['final_predicted'].value_counts()

# check the precentage of final_predicted conversions

1979/float(1979+483)

#####Insight on prediction rate of Train data
The final prediction of conversions have a target of 80.38% conversion which should be accepted as per the X Educations CEO's requirement . Hence this is a good model.

### Model Evaluation:
Now, lets evaluate our model for the Accuracy, Confusion Metrics, Sensitivity, Specificity, False Postive Rate, Positive Predictive Value, Negative Predicitive Value on the final prediction of the train set.

#### Overall Accuracy

# overall accuracy

metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)

confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives

#### Sensitivity

# Sensitivity of our logistic regression model

TP / float(TP+FN)

#### Specificity

# specificity

TN / float(TN+FP)

# Calculate false postive rate - predicting conversions when leads has not converted

print(FP/ float(TN+FP))

# Positive predictive value 

print (TP / float(TP+FP))

# Negative predictive value

print (TN / float(TN+ FN))

#### Confusion matrix

#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
confusion

#### Precision and recall Score

from sklearn.metrics import precision_score, recall_score

# precision

precision_score(y_train_pred_final.Converted, y_train_pred_final.predicted)

# recall

recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted)

#### Precision and recall tradeoff

from sklearn.metrics import precision_recall_curve

y_train_pred_final.Converted, y_train_pred_final.predicted

p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)

plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()

# <span style="color:IndianRed ;">12. Making predictions on the test set

X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_test = X_test[col]
X_test.head()

X_test_sm = sm.add_constant(X_test)

y_test_pred = res.predict(X_test_sm)

y_test_pred[:10]

# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)

# Let's see the head

y_pred_1.head()

# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)

# Putting LeadId to index

y_test_df['LeadId'] = y_test_df.index

# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)

y_pred_final.head()

# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})

y_pred_final.head()

# Based on cut off threshold using accuracy, sensitivity and specificity of 0.38%

y_pred_final['final_predicted'] = y_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.38 else 0)

y_pred_final.head()



## Lead Scoring

# Now let us calculate the lead score

y_pred_final['lead_score'] = y_pred_final.Converted_Prob.map(lambda x: round(x*100))
y_pred_final.head(20)

# checking if 80% cases are correctly predicted based on the converted column.

# get the total of final predicted conversion or non conversion counts from the actual converted rates

checking_test_df = y_pred_final.loc[y_pred_final['Converted']==1,['Converted','final_predicted']]
checking_test_df['final_predicted'].value_counts()

# check the precentage of final_predicted conversions on test data

807/float(807+208)

#####**Insight on prediction rate of Test data:**
 The final prediction of conversions have a target rate of 79.5% and is accptable against 80.3% of the predictions rate on training data set

##### Overall Accuracy


# Let's check the accuracy.

metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)

##### Confusion Metrics

confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion2

TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives

##### Sensitivity

# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)

##### Specificity

# Let us calculate specificity
TN / float(TN+FP)


##### Precision and Recall Score

# precision
print('precision ',precision_score(y_pred_final.Converted, y_pred_final.final_predicted))

# recall
print('recall ',recall_score(y_pred_final.Converted, y_pred_final.final_predicted))

##### Precision and Recall Tradeoff


p, r, thresholds = precision_recall_curve(y_pred_final.Converted, y_pred_final.Converted_Prob)

plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()

# <span style="color:IndianRed ;">13. Conclusion
    
- Built the model with above 80% predicition rate.
- Tested the model with 79.5% prediction rate.
- Created a Lead score for each lead, which indicates the chances of it being converted. Higher Lead score depicts a better chances of being converted.
- Checked and evaluated the model on below parameters for both train and test dataset:
    - Overall Accuracy
    - Sensitivity
    - Specificity 
    - Confusion Matrix
    - Precision and Recall Score

- Calculated the final prediction based on the optimal cut-off obtained through above parameters.
- Also the lead score calculated in the trained set of data shows the conversion rate on the final predicted model is around 80%.
- Overall, the model looks good on all the parameters.
