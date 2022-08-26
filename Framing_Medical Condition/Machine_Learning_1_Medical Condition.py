# importing libraries and functions
import numpy as np
import pandas as pd
import seaborn as sns

#modules and important packages
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from scipy.stats import describe

df = pd.read_csv("E:\Machine Learning\Framing_Medical Condition/framingham2.csv")
df.head(25)
df.shape

df=pd.DataFrame.sample(df,3000,random_state=27)
df
df.dtypes

# check for dupicates
df1 = df[df.duplicated()]
df1

# checking for missing values
df.isna().sum()
null = df[df.isna().any(axis=1)]

# separate independent & dependent variables
X = df.iloc[:,0:15]  #independent columns
y = df.iloc[:,-1] #target
X
y

import statistics as sc
chol=df.iloc[:,9]
chol
describe(chol)
m=sc.mean(chol)
m
s=sc.stdev(chol) #sample standard deviation
s
psd=np.std(chol) #population standard deviation
print("population sd=",psd,"and Sample sd=",s)

import scipy.stats as ss
m=sc.mean(chol)
m0=240
s=sc.stdev(chol)
n=len(chol)
df=n-1
alpha=0.05
sem=s/np.sqrt(n)
tcal=(m-m0)/sem
tc=ss.t.ppf(1-alpha,df)
tcal
tc
if tcal>tc:
    print("The null hypothesis is rejected at 5% level of significance")
else:
    print("The null hypothesis is not rejected at 5% level of significance")


# P-VALUE APPROACH----------------------------------------------------
print("PVALUE APPROACH")
PVALUE=ss.t.sf(tcal,df)
print("P-value is",PVALUE)
if PVALUE<alpha:
    print("The null hypothesis is rejected at 5% level of significance")
else:
    print("The null hypothesis is not rejected at 5% level of significance")

#95% confidence interval
heart_rate = X['heartRate']
m=sc.mean(heart_rate)
s=sc.stdev(heart_rate)
n=len(heart_rate)
df=n-1
alpha=0.05
tc=ss.t.ppf(1-alpha,df)
tc
u=m+tc*s/np.sqrt(n);u
l=m-tc*s/np.sqrt(n);l
print("95% confidence interval for population mean is (",l,",",u,")")

#Correlation
# create_new dataset for medical condition
df3 = X[['sysBP', 'glucose','age','totChol','cigsPerDay','diaBP','prevalentHyp','diabetes','BPMeds']]
df3.head()

df_corr = df3.corr()
df_corr
sns.heatmap(df_corr)

df = pd.read_csv("E:\Machine Learning\Framing_Medical Condition/framingham2.csv")
# separate independent & dependent variables
X = df.iloc[:,0:15]  #independent columns
y = df.iloc[:,-1] #target

# Split the training test set
#train is based on 80% of the dataset, test is based on 20% of dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=27)

# Fit a Decision Tree model as comparison
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
tree.plot_tree(clf)

# Fit a Random Forest model, " compared to "Decision Tree model, accuracy go up by 7%
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
clf = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=27)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# Fit a AdaBoost model, " compared to "Decision Tree model, accuracy go up by 6%
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# Fit a Gradient Boosting model, " compared to "Decision Tree model, accuracy go up by 6%
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# Fit logistic Regression Model
from sklearn.linear_model import LogisticRegression
clf= LogisticRegression(max_iter=3000, random_state=27)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
clf.intercept_
clf.coef_
clf.predict_proba(X_test)
print (X_test) #test dataset
print (y_pred) #predicted values

# Vooting Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
clf1 = LogisticRegression(max_iter=3000, random_state=27)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],voting='hard')
for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
        scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5)
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

# In[ ]:




