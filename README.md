# UdacityProjet1 : Telecom  costumer percentage of  churn analysis

Customer churn occurs when a customer decides to stop using a company's services, content or products. There are many examples and instances of churn:

- Cancellation of a contracted or uncontracted service ;
- Purchase from another competitor's shop;
- Unsubscribing from a newsletter;
- Closing a bank account Etc.

In today's business environment, with many competitors, the cost of acquiring new customers is very high. Therefore, the retention of existing customers is more important for its companies. Thus, the company needs to better understand the behaviour of its customers in order to retain them. One way to do this is to create a Machine Learning model that can predict which customers are likely to unsubscribe. This allows the company to better target and retain those specific customers who are at higher risk of churn.

In this project, we will explore a dataset from a telecommunications company and create a model to predict which customers are at higher risk of churn. We will use different machine.

## Questions to be answered

* Question 1: What are the types of each of the variables? and Which variables have missing values ? 
* Question 2: Is customer churn influenced by gender (the gender variable) ?
* Question 3: In your opinion, which variable has the greatest impact on a customer's susceptibility to churn ?
* Question 4: Which regression model can be used to approach the problem logically ?
* Question 5: Can we choose a model and test its performance ?

## Importation of libraries
```python
import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sn
import matplotlib.pyplot as plt
%matplotlib inline
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
```

## Answer 

### Importation of dataset
```python 
df = pd.read_csv('Data_telecom.csv')
df.head()
```
#### Answer 1 :
We use this command
``` python
# Description (includ type) of the data set
df.info()

# Know which columns have missing values
df.columns[np.sum(df.isnull())==0]
```
#### Answer 2 and 3:

We applied this code after we Exploratory data analysis, Pre-processing of data (clean, encoding of categorical variables etc...) and Segmentation (Segmentation into explanatory variables and variables to be explained)

##### Segmentation of the dataset
```python
# Segmentation into explanatory variables and variables to be explained

X = data.drop('Churn', axis = 1)
y = data['Churn']
```
##### Modelelling via Logistic regression
- `Definition` : We use logistic regression because of the interpretation of our variable to be explained which is either 0 or 1. It is therefore more logical to turn to the logistic regression method whose variable to be explained in this model obeys the logic True or False (0 or 1).

- `choice of the metric to evaluate the performance of our model`:There is the accuracy metric, the precision metric and the recall metric see -> [Metric choices](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

##### Selection of the best variables to predict the result
There are several ways to select the best predictors. ***Note that models based on the decision tree method have an attribute that gives the importance of each of the predictor variables. This allows us to select the best predictors of the outcome.

We will create a random forest model without looking for the best hyperparameters. From this model we will determine the most important variables that will be used to train the machine learning algorithm.

* Selection of the best predictors
```python 
# We will train the data without searching for the best hyperparameters

rf = RandomForestClassifier()

rf.fit(train_features, train_labels)

print(classification_report(y_test, rf.predict(X_test)))
```

* Visualisation of important explanatory variables
```python
plt.figure(figsize=(14, 6))

# the variables are stored in the variable 'features_importances_' (attribute giving the degree of importance of each predictor variable)
#selections of column names ("index")
#we sort from the largest value to the smallest one we do "sort_values(ascending =False)
vars_imp = pd.Series(rf.feature_importances_, index = train_features.columns).sort_values(ascending=False)

#Let's draw a bar chart 
#variable or column names are in row indices
#"X is the panda series object we just created
#"y are the oms of the columns that are in row index"
sns.barplot(x = vars_imp.index, y=vars_imp)
plt.xticks(rotation=90)
plt.xlabel("Variables")
plt.ylabel("Importance score of the variable")
plt.title("Importance of predictor or explanatory variables")
plt.show()
```

* Display of explanatory values in descending order
```python
vars_imp
```
We will select the variables that best predict the model by defining a threshold.

We note from the data of the important variables below that the majority of the variables are lower than 0.009 our threshold will be equal to 0.009, i.e. 1% of estimation error by the randomforets model.

#### answer 4 
**Modelling proper**
Our objective is to develop a model to predict in advance which customers will unsubscribe. This will allow us to better retain customers so that they do not unsubscribe due to the reduced costs compared to the cost of acquiring new customers

- Training of the model : we research the best model wich dictionary of hyperparameter, we applied the GridSearchCV object wich estimator equal ```LogisticRegression``` and param_grid equal to *dictionary of hyperparameters*
- Training the algorithm : ```logreg_model = modele_logreg_class.fit(train_features, train_labels)```
- Print Best score and best hyperparameter

```python
round allows rounding to 3 digits after the decimal point
print(round(logreg_model.best_score_, 3))

print(logreg_model.best_estimator_)
```
In this case we have used logistic regression as described above but other regression models can be applied
#### Answer 5
Yes,we can test other models outside logistic regression in order to choose the best model.
Yes we can do that, but we have many model that we can train to take best model

