#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[53]:


df = pd.read_csv('UpdatedResumeDataSet.csv')


# In[54]:


df.head()


# In[55]:


df.shape


# In[56]:


df['Category']


# In[57]:


df['Category'].value_counts()


# In[58]:


df['Category'].unique()


# In[61]:


plt.figure(figsize=(15,5))
#plt.xticks(rotation=90)
sns.countplot(x="Category", data=df)
plt.xticks(rotation=90)
plt.show()


# In[12]:


counts = df['Category'].value_counts()
labels = df['Category'].unique()
plt.figure(figsize=(15,10))
plt.pie(counts,labels=labels)


# In[13]:


counts = df['Category'].value_counts()
labels = df['Category'].unique()
plt.figure(figsize=(15,10))
plt.pie(counts,labels=labels,autopct='%1.1f%%',shadow=True,colors=plt.cm.plasma(np.linspace(0,1,3)))
plt.show()


# In[14]:


df['Category'][0]


# In[15]:


df['Category'][1]


# In[16]:


df['Category'][67]


# In[17]:


df['Category'][100]


# In[18]:


df['Resume']


# In[19]:


df['Resume'][0]


# In[20]:


import re
def cleanResume(txt):
    cleanText = re.sub('http\S+\s',' ',txt)
    
    return cleanText


# In[21]:


cleanResume("my website like is this http://helloworld and access it")


# In[22]:


import re
def cleanResume(txt):
    cleanText = re.sub('http\S+\s',' ',txt)
    cleanText = re.sub('@\S+',' ',cleanText)
    
    return cleanText


# In[23]:


cleanResume("my website like is this http://helloworld and access it @gmail.com")


# In[24]:


import re
def cleanResume(txt):
    cleanText = re.sub('http\S+\s',' ',txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    
    return cleanText


# In[25]:


cleanResume("my ####### $  website like is this http://helloworld and access it @gmail.com")


# In[26]:


df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))


# In[27]:


df['Resume'][0]


# In[28]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[29]:


le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])


# In[30]:


df.Category.unique()


# In[31]:


#array(['Data Science', 'HR', 'Advocate', 'Arts', 'Web Designing',
#       'Mechanical Engineer', 'Sales', 'Health and fitness',
#       'Civil Engineer', 'Java Developer', 'Business Analyst',
#       'SAP Developer', 'Automation Testing', 'Electrical Engineering',
#       'Operations Manager', 'Python Developer', 'DevOps Engineer',
#       'Network Security Engineer', 'PMO', 'Database', 'Hadoop',
#       'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing'],
#      dtype=object)


# In[32]:


df


# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

tfidf.fit(df['Resume'])
requiredText  = tfidf.transform(df['Resume'])


# In[34]:


df


# In[35]:


requiredText


# In[36]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(requiredText, df['Category'], test_size=0.2, random_state=42)


# In[42]:


X_train.shape


# In[43]:


X_test.shape


# In[44]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train,y_train)
ypred = clf.predict(X_test)
print(accuracy_score(y_test,ypred))


# In[45]:


ypred


# In[46]:


import pickle
pickle.dump(tfidf,open('tfidf.pkl','wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))


# In[47]:


myresume = """ Senior Web Developer specializing in front end development. Experienced with all stages of the development cycle for dynamic web projects. Well-versed in numerous programming languages including HTML5, PHP OOP, JavaScript, CSS, MySQL. Strong background in project management and customer relations.
Phone:
+49 800 600 600

E-Mail:	
christoper.morgan@gmail.com

Linkedin:
linkedin.com/christopher.morgan

Skill Highlights
•	Skill Highlights
•	Project management
•	Strong decision maker
•	Complex problem solver
•	Creative design
•	Innovative
•	Service-focused
Languages
Spanish – C2
Chinese – A1

Experience
09/2015 to 05/2019
Web Developer -  Luna Web Design, New York
•	Cooperate with designers to create clean interfaces and simple, intuitive interactions and experiences.
•	Develop project concepts and maintain optimal workflow.
•	Work with senior developer to manage large, complex design projects for corporate clients.
•	Complete detailed programming and development tasks for front end public and internal websites as well as challenging back-end server code.
•	Carry out quality assurance tests to discover errors and optimize usability.
Education
2014-2019 
Bachelor of Science: Computer Information Systems -  
Columbia University, NY
Certifications
PHP Framework (certificate): Zend, Codeigniter, Symfony.
Programming Languages: JavaScript, HTML5, PHP OOP, CSS, SQL, MySQL.
References
References available on request
"""


# In[49]:


import pickle

# Load the trained classifier
clf = pickle.load(open('clf.pkl', 'rb'))

# Clean the input resume
cleaned_resume = cleanResume(myresume)

# Transform the cleaned resume using the trained TfidfVectorizer
input_features = tfidf.transform([cleaned_resume])

# Make the prediction using the loaded classifier
prediction_id = clf.predict(input_features)[0]

# Map category ID to category name
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

category_name = category_mapping.get(prediction_id, "Unknown")

print("Predicted Category:", category_name)
print(prediction_id)

