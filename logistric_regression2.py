import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv(r"C:\Users\tanpu\Downloads\logit classification.csv")

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)



from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)


from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)



from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)


bias = classifier.score(X_train,y_train)
print(bias)

variance = classifier.score(X_test,y_test)
print(variance)


print(X.shape)
print(y.shape)
print(X_test.shape)
print(y_test.shape)



---------Future Prediction------------
         

dataset = pd.read_csv(r"C:\Users\tanpu\Downloads\15. Logistic regression with future prediction\15. Logistic regression with future prediction\Future prediction1.csv")
          
d2 = dataset.copy()
dataset = dataset.iloc[:,[2,3]].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
M = sc.fit_transform(dataset)

y_pred = pd.DataFrame()         
    
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

d2['y_pred'] = classifier.predict(M)

d2.to_csv('final1.csv', index=False)





      
          


from sklearn.metrics import roc_auc_score,roc_curve
y_pred_prob = classifier.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test,y_pred_prob)
print(auc_score)

fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show() 













