
# coding: utf-8

# # Linear Classification
# 
# In this lab you will implement parts of a linear classification model using the regularized empirical risk minimization principle. By completing this lab and analysing the code, you gain deeper understanding of these type of models, and of gradient descent.
# 
# 
# ## Problem Setting
# 
# The dataset describes diagnosing of cardiac Single Proton Emission Computed Tomography (SPECT) images. Each of the patients is classified into two categories: normal (1) and abnormal (0). The training data contains 80 SPECT images from which 22 binary features have been extracted. The goal is to predict the label for an unseen test set of 187 tomography images.

# In[ ]:

import urllib
import pandas as pd
import numpy as np
# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

testfile = urllib.URLopener()
testfile.retrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train", "SPECT.train")
testfile.retrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test", "SPECT.test")

df_train = pd.read_csv('SPECT.train',header=None)
df_test = pd.read_csv('SPECT.test',header=None)

train = df_train.as_matrix()
test = df_test.as_matrix()

y_train = train[:,0]
X_train = train[:,1:]
y_test = test[:,0]
X_test = test[:,1:]


# ### Exercise 1
# 
# Analyze the function learn_reg_ERM(X,y,lambda) which for a given $n\times m$ data matrix $\textbf{X}$ and binary class label $\textbf{y}$ learns and returns a linear model $\textbf{w}$.
# The binary class label has to be transformed so that its range is $\left \{-1,1 \right \}$. 
# The trade-off parameter between the empirical loss and the regularizer is given by $\lambda > 0$. 
# Try to understand each step of the learning algorithm and comment each line.

# In[ ]:

def learn_reg_ERM(X,y,lbda):
    max_iter = 200
    e  = 0.001
    alpha = 1.

    w = np.random.randn(X.shape[1]);
    for k in np.arange(max_iter):
        h = np.dot(X,w)
        l,lg = loss(h, y)
        print 'loss: {}'.format(np.mean(l))
        r,rg = reg(w, lbda)
        g = np.dot(X.T,lg) + rg 
        if (k > 0):
            alpha = alpha * (np.dot(g_old.T,g_old))/(np.dot((g_old - g).T,g_old))
        w = w - alpha * g
        if (np.linalg.norm(alpha * g) < e):
            break
        g_old = g
    return w


# ### Exercise 2
# 
# Fill in the code for the function loss(h,y) which computes the hinge loss and its gradient. 
# This function takes a given vector $\textbf{y}$ with the true labels $\in \left \{-1,1\right \}$ and a vector $\textbf{h}$ with the function values of the linear model as inputs. The function returns a vector $\textbf{l}$ with the hinge loss $\max(0, 1 − y_{i} h_{i})$ and a vector $\textbf{g}$ with the gradients of the hinge loss at the points $h_i$. The partial derivative of the hinge loss $h_i$ with respect to the $i$-th position of the weight vector $\textbf{w}$ is $g_{i} = −y x_{i}$ if $l_{i} > 0$, else $g_{i} = 0$).

# In[ ]:

def loss(h, y):
    ##################
    #INSERT CODE HERE#
    ##################
    return l, g


# ### Exercise 3
# 
# Fill in the code for the function reg(w,lambda) which computes the $\mathcal{L}_2$-regularizer and the gradient of the regularizer function at point $\textbf{w}$. 
# 
# 
# $$r = \frac{\lambda}{2} \textbf{w}^{T}\textbf{w}$$
# 
# $$g = \lambda \textbf{w}$$

# In[ ]:

def reg(w, lbda):
    ##################
    #INSERT CODE HERE#
    ##################
    return r, g


# ### Exercise 4
# 
# Fill in the code for the function predict(w,x) which predicts the class label $y$ for a data point $\textbf{x}$ or a matrix $X$ of data points (row-wise) for a previously trained linear model $\textbf{w}$. If there is only a data point given, the function is supposed to return a scalar value. If a matrix is given a vector of predictions is supposed to be returned.

# In[ ]:

def predict(w, X):
    ##################
    #INSERT CODE HERE#
    ##################
    return preds


# ### Exercise 5
# 
# #### 5.1 
# Train a linear model on the training data and classify all 187 test instances afterwards using the function predict. 
# Please note that the given class labels are in the range $\left \{0,1 \right \}$, however the learning algorithm expects a label in the range of $\left \{-1,1 \right \}$. Then, compute the accuracy of your trained linear model on both the training and the test data. 

# In[ ]:

##################
#INSERT CODE HERE#
##################


# #### 5.2
# Compare the accuracy of the linear model with the accuracy of a random forest and a decision tree on the training and test data set.

# In[ ]:

##################
#INSERT CODE HERE#
##################

