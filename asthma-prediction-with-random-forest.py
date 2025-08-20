#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


data = pd.read_csv("C:/DATA SET/Asthma prediction/asthma_patient_dataset.csv")


# In[4]:


data.head()


# In[5]:


df = data.drop(columns=['Patient_ID'])


# In[6]:


df.head(3)


# In[7]:


df.isnull().sum()


# In[8]:


df['Allergies'] = df['Allergies'].fillna(df['Allergies'].mode()[0])
df['Comorbidities'] = df['Comorbidities'].fillna(df['Comorbidities'].mode()[0])
df['Asthma_Control_Level'] = df['Asthma_Control_Level'].fillna(df['Asthma_Control_Level'].mode()[0])


# In[9]:


col = list(df.columns)
for i in col:
    if df[i].dtype == object:
        print(i, '------->   ', df[i].unique())


# In[10]:


"""
Doing label encoding using for loop. 
"""

target = ['Gender', 'Smoking_Status', 'Allergies', 'Comorbidities']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in target:
    df[i] = le.fit_transform(df[i])


# In[11]:


"""
Doing ordinal encoding using for loop.
"""

target2 = ['Air_Pollution_Level', 'Physical_Activity_Level', 'Occupation_Type', 'Asthma_Control_Level']
from sklearn.preprocessing import OrdinalEncoder
Or = OrdinalEncoder()
for i in target2:
    df[i] = Or.fit_transform(df[[i]])


# In[12]:


df.describe()


# In[13]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Data = df.drop(columns=['Has_Asthma'])
target3 = list(Data.columns)
for i in target3:
    df[i] = sc.fit_transform(df[[i]])


# In[14]:


x = df.drop(columns=['Has_Asthma'])
y = df['Has_Asthma']


# In[15]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

log = LogisticRegression( penalty='l2', C=1.0,solver='lbfgs',max_iter=1000,random_state=42)
svm  = SVC( kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42)
tree = RandomForestClassifier()

clf = [log, svm, tree]
for model in clf:
    print(model)
    model.fit(x_train, y_train)
    print(f'score --> {model.score(x_test, y_test)}')


# In[17]:


final_model = RandomForestClassifier()
final_model.fit(x_train, y_train)


# In[18]:


# Get predictions & probabilities
prediction = final_model.predict(x_test)
probs = final_model.predict_proba(x_test)[:, 1]

# Match IDs to test rows
ids_for_test = data.loc[x_test.index, 'Patient_ID']

# Create DataFrame
output_df = pd.DataFrame({
    'id': ids_for_test,
    'prediction': prediction,
    'probability': probs
})

# Save
output_df.to_csv('predictions.csv', index=False)
print("Saved predictions.csv")


# In[ ]:




