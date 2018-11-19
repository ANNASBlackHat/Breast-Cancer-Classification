
# coding: utf-8

# ## IMPORTING DATA

# In[72]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[73]:


from sklearn.datasets import load_breast_cancer


# In[74]:


cancer = load_breast_cancer()


# In[75]:


cancer


# In[76]:


cancer.keys()


# In[77]:


print(cancer["DESCR"])


# In[78]:


print(cancer["target_names"])


# In[79]:


print(cancer["feature_names"])


# In[80]:


# we have 569 rows and 30 column (feature)
cancer['data'].shape


# In[81]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'],['target']))
df_cancer.head()


# ## VISUALIZING DATA

# In[82]:


sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture' ,'mean perimeter', 'mean area'
 ,'mean smoothness'])


# In[83]:


sns.countplot(df_cancer['target'])


# In[84]:


sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer)


# In[85]:


plt.figure(figsize = (20,10))
sns.heatmap(df_cancer.corr(), annot = True)


# ## TRAINING DATA

# In[86]:


#remove target column
X = df_cancer.drop(['target'], axis = 1)
X.head()


# In[87]:


y = df_cancer['target']
y


# **Split Data** (for training & testing the model)

# In[88]:


from sklearn.model_selection import train_test_split


# In[89]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# **Train Model**

# In[90]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# In[91]:


svc_model = SVC()
svc_model.fit(X_train, y_train)


# ## EVALUATING THE MODEL

# In[92]:


y_predict = svc_model.predict(X_test)


# In[93]:


y_predict


# In[94]:


cm = confusion_matrix(y_test, y_predict)
cm


# In[95]:


sns.heatmap(cm, annot = True)


# ## IMPROVE THE MODEL

# ### Normalizarion (Model improvement part 1)
# Using feature scaling (Unity-based normalization) : Brings all values into range 0 and 1
# 
# *Formula :*
# ![](https://raw.githubusercontent.com/ANNASBlackHat/Breast-Cancer-Classification/master/images/Data%20normalization%20formula.png)

# In[96]:


min_train = X_train.min()
range_train = (X_train - min_train).max()
X_train_scaled = (X_train - min_train)/range_train


# Before Normalization

# In[97]:


sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)


# After Normalization

# In[98]:


sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train)


# **Train the model**

# In[99]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[100]:


svc_model.fit(X_train_scaled, y_train)


# In[101]:


y_predict = svc_model.predict(X_test_scaled)
cn = confusion_matrix(y_test, y_predict)
sns.heatmap(cn, annot=True)


# In[102]:


print(classification_report(y_test, y_predict))


# ### Update Parameter (Model improvement part 2)

# In[103]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel':['rbf']}


# In[104]:


from sklearn.model_selection import GridSearchCV


# In[105]:


grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=4)
grid.fit(X_train_scaled, y_train)


# In[106]:


#show best value
grid.best_params_


# In[107]:


grid_prediction = grid.predict(X_test_scaled)


# In[108]:


cm = confusion_matrix(y_test, grid_prediction)
sns.heatmap(cm, annot=True)


# In[109]:


print(classification_report(y_test, grid_prediction))


# **We end up with 0.97 accuracy, which is great. 
# The only error that we get, is error type one, that's mean not too bad.**

# In[121]:


from sklearn import metrics
def pretty_cm(y_pred, y_truth, labels):
  cm = metrics.confusion_matrix(y_truth, y_pred)
  ax = plt.subplot()
  sns.heatmap(cm, annot=True, fmt='d', linewidths=.5, square=True, cmap='RdBu_r')
  
  # lables, title and ticks
  ax.set_xlabel('Predicted label')
  ax.set_ylabel('Actual label')
  ax.set_title('Accuracy : {0}'.format(metrics.accuracy_score(y_truth, y_pred)), size = 15)
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)


# In[123]:


pretty_cm(grid_prediction, y_test, ['malignant', 'benign'])

