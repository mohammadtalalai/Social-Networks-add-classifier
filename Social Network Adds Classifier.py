#!/usr/bin/env python
# coding: utf-8

# # Import the libraries

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("Social_Network_Ads.csv")


# In[3]:


df


# # EDA

# In[4]:


df.isna().sum()


# Not have messing values

# In[5]:


df = df.drop("User ID" , axis=1)


# In[6]:


df


# In[7]:


from sklearn.preprocessing import LabelEncoder


# In[8]:


df['Gender'] = LabelEncoder().fit_transform(df['Gender'])


# In[9]:


df


# Incode DataFrame To Transform male or female to 1/0

# In[10]:


sns.pairplot(df , hue="Purchased")
plt.show()


# Have a seperate between data with Estimated salary and age

# # Prepare Data For Training

# In[11]:


x=df.iloc[: , :-1]


# In[12]:


x


# In[13]:


y = df.iloc[: , -1]


# In[14]:


y


# Scaled The Data 

# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


scl = StandardScaler()


# In[17]:


x_scl = scl.fit_transform(x)


# In[18]:


x_scl


# # Split the Data

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


x_train , x_test , y_train , y_test = train_test_split(x_scl , y , test_size=0.25 , shuffle=True)


# # Train The Model

# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


classifier = LogisticRegression()


# In[23]:


classifier.fit(x_train , y_train)


# In[24]:


print("The Accuracy of the Model Training is: " , classifier.score(x_train , y_train)*100 , "%")


# In[25]:


pred_train = classifier.predict(x_train)
pred_test = classifier.predict(x_test)


# In[26]:


pred_train


# In[27]:


y_train.value_counts()


# In[28]:


pred_test  


# In[29]:


y_test.value_counts()


# # Evaluate The Model

# In[30]:


from sklearn.metrics import confusion_matrix


# In[31]:


Train_Evaluation = confusion_matrix (y_train , pred_train)


# In[32]:


Train_Evaluation


# In[33]:


sns.heatmap(Train_Evaluation)
plt.xlabel("Predict Value")
plt.ylabel("True Value")
plt.show()


# In[34]:


Test_Evaluation = confusion_matrix (y_test , pred_test)


# In[35]:


Test_Evaluation


# In[36]:


sns.heatmap(Test_Evaluation)
plt.xlabel("Predict Value")
plt.ylabel("True Value")
plt.show()


# In[37]:


from sklearn.metrics import accuracy_score , precision_score  , recall_score


# In[38]:


print("The Accuarcy Score of The Model Training is :" , accuracy_score(y_train  ,pred_train)*100 ,  "%")


# In[39]:


print("The Accuarcy Score of The Model Test is :" , accuracy_score(y_test  ,pred_test)*100 ,  "%")


#  # Visualize The Model Training 

# In[40]:


import pylab as pl


# In[41]:


age_min , age_max = x["Age"].min()-1 , x["Age"].max()-1
es_min ,  es_max  = x["EstimatedSalary"].min()-1 , x["EstimatedSalary"].max()-1
age_grid , es_grid = np.meshgrid(np.arange(age_min , age_max , 0.2) , (np.arange(es_min , es_max , 0.2)))
pl.figure(figsize=(15,15))
pl.set_cmap(pl.cm.cividis)


# In[42]:


print("Unique values in y_train:", np.unique(y_train))
print("Unique values in y_test:", np.unique(y_test))


# In[49]:


#Creating Boundries and grids
age_min, age_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
es_min, es_max = x_train[:, 2].min() - 1, x_train[:, 2].max() + 1
age_grid, es_grid = np.meshgrid(np.arange(age_min, age_max, 0.01), np.arange(es_min, es_max, 0.01))
pl.figure(figsize=(15, 30))  
pl.set_cmap(pl.cm.cividis)



no_of_iterations = [1, 2, 5, 10, 50, 100, 200, 500, 1000]
i = 1

for iteration in no_of_iterations:
    clf = LogisticRegression(max_iter=iteration, solver='saga')
    clf.fit(x_train[:, 1:], y_train)

    # Predict for training and test data
    train_pred = clf.predict(x_train[:, 1:])
    test_pred = clf.predict(x_test[:, 1:])

    
    # Print accuracy scores
    print(f"Iteration Number: {iteration}")
    print(f"The Training Score: {accuracy_score(y_train, train_pred) * 100:.2f}%")
    print(f"The Test Score: {accuracy_score(y_test, test_pred) * 100:.2f}%")


    z = clf.predict(np.c_[age_grid.ravel(), es_grid.ravel()])
    z = z.reshape(age_grid.shape)

    
    #Train Plot
    pl.subplot(9, 2, i)  # Adjust grid size to 5x2 for 10 subplots
    pl.contourf(age_grid, es_grid, z, alpha=0.5)  # Contour plot for decision regions
    pl.scatter(x_train[:, 1], x_train[:, 2], c=y_train,  marker='o', s=20, alpha=0.7)  # Training data scatter
    pl.title(f"Iteration {iteration} train")
    i += 1
    
    

    #Test Plot
    pl.subplot(9, 2, i)  # Adjust grid size to 5x2 for 10 subplots
    pl.contourf(age_grid, es_grid, z, alpha=0.5)  # Contour plot for decision regions
    pl.scatter(x_test[:, 1], x_test[:, 2], c=y_test,  marker='o', s=20, alpha=0.7)  # Training data scatter
    pl.title(f"Iteration {iteration} test")
    i += 1
    
    


# Show all plots
pl.tight_layout()
pl.show()


# # Using Another Algorithim "SVM - SVC"  

# In[50]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[51]:


SVM_C = SVC()


# In[52]:


SVM_C.fit(x_train , y_train)


# In[53]:


SVC_train = SVM_C.predict(x_train)
SVC_test = SVM_C.predict(x_test)


# In[54]:


print("The Accuracy of Training is :" , SVM_C.score(x_train , y_train)*100 , "%")
print("The Accuracy of Test is :" , SVM_C.score(x_test , y_test)*100 , "%")


# it's better

# Let's going and try the another Kernel

# In[55]:


for k in ['linear' , 'poly' , 'rbf']:
    clf_SVM = SVC(kernel=k)
    clf_SVM.fit(x_train , y_train)
    train_pred_SVM = clf_SVM.predict(x_train)
    test_pred_SVM = clf_SVM.predict(x_test)
    print("The Kernal Type is :" , k)
    print("The Score Of Training" , clf_SVM.score(x_train , y_train)*100 , "%")
    print("The Score Of Test is :" , clf_SVM.score(x_test , y_test)*100  ,"%")
    print('--------------------------------------------------')


# The Best Type of kernel is 'rbf'

# Using Kernel=Poly with different degree

# In[56]:


#Creating Boundries and grids
age_min, age_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
es_min, es_max = x_train[:, 2].min() - 1, x_train[:, 2].max() + 1
age_grid, es_grid = np.meshgrid(np.arange(age_min, age_max, 0.01), np.arange(es_min, es_max, 0.01))
pl.figure(figsize=(15, 30))  
pl.set_cmap(pl.cm.Blues)


i = 1
for d in range(1,10):
    clf = SVC(kernel='poly' , degree=d)
    clf.fit(x_train[:, 1:], y_train)

    # Predict for training and test data
    train_pred = clf.predict(x_train[:, 1:])
    test_pred = clf.predict(x_test[:, 1:])

    
    # Print accuracy scores
    print(f"degrees Number: {d}")
    print(f"The Training Score: {accuracy_score(y_train, train_pred) * 100:.2f}%")
    print(f"The Test Score: {accuracy_score(y_test, test_pred) * 100:.2f}%")
    print('------------------------------------------------------------')


    z = clf.predict(np.c_[age_grid.ravel(), es_grid.ravel()])
    z = z.reshape(age_grid.shape)

    
    #Train Plot
    pl.subplot(9, 2, i)  # Adjust grid size to 5x2 for 10 subplots
    pl.contourf(age_grid, es_grid, z, alpha=0.5)  # Contour plot for decision regions
    pl.scatter(x_train[:, 1], x_train[:, 2], c=y_train,  marker='o', s=20, alpha=0.7)  # Training data scatter
    pl.title(f"degree {d} train")
    i += 1
    
    

    #Test Plot
    pl.subplot(9, 2, i)  # Adjust grid size to 5x2 for 10 subplots
    pl.contourf(age_grid, es_grid, z, alpha=0.5)  # Contour plot for decision regions
    pl.scatter(x_test[:, 1], x_test[:, 2], c=y_test,  marker='o', s=20, alpha=0.7)  # Training data scatter
    pl.title(f"degree {d} test")
    i += 1
    
    


# Show all plots
pl.tight_layout()
pl.show()


# In[59]:


#Creating Boundries and grids
age_min, age_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
es_min, es_max = x_train[:, 2].min() - 1, x_train[:, 2].max() + 1
age_grid, es_grid = np.meshgrid(np.arange(age_min, age_max, 0.01), np.arange(es_min, es_max, 0.01))
pl.figure(figsize=(15, 30))  
pl.set_cmap(pl.cm.brg)


i = 1
for k in ['poly' , 'linear' , 'rbf']:
    clf = SVC(kernel=k)
    clf.fit(x_train[:, 1:], y_train)

    # Predict for training and test data
    train_pred = clf.predict(x_train[:, 1:])
    test_pred = clf.predict(x_test[:, 1:])

    
    # Print accuracy scores
    print(f"Kernel Type: {k}")
    print(f"The Training Score: {accuracy_score(y_train, train_pred) * 100:.2f}%")
    print(f"The Test Score: {accuracy_score(y_test, test_pred) * 100:.2f}%")
    print('------------------------------------------------------------')


    z = clf.predict(np.c_[age_grid.ravel(), es_grid.ravel()])
    z = z.reshape(age_grid.shape)

    
    #Train Plot
    pl.subplot(3, 2, i)  # Adjust grid size to 5x2 for 10 subplots
    pl.contourf(age_grid, es_grid, z, alpha=0.5)  # Contour plot for decision regions
    pl.scatter(x_train[:, 1], x_train[:, 2], c=y_train,  marker='o', s=20, alpha=0.7)  # Training data scatter
    pl.title(f"Kernel Type {k} train")
    i += 1
    
    

    #Test Plot
    pl.subplot(3, 2, i)  # Adjust grid size to 5x2 for 10 subplots
    pl.contourf(age_grid, es_grid, z, alpha=0.5)  # Contour plot for decision regions
    pl.scatter(x_test[:, 1], x_test[:, 2], c=y_test,  marker='o', s=20, alpha=0.7)  # Training data scatter
    pl.title(f"Kernel Type {k} test")
    i += 1
    
    


# Show all plots
pl.tight_layout()
pl.show()


# Still rbf kernel is best accuracy and it's better than logistic Regression Model

# In[ ]:




