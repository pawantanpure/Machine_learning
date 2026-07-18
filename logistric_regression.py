import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset1 = pd.read_csv(r"C:\Users\tanpu\Downloads\final1.csv")
d2 = dataset1.copy()

dataset1 = dataset1.iloc[:,[2,3]]
dataset1 = pd.get_dummies(dataset1, columns=['Gender'], drop_first=True)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(dataset1)

import pandas as pd

y_pred1 = pd.DataFrame()



d2['y_pred1'] = classifier.predict(M)

d2.to_csv('final.csv', index=False)
d2 ['y_pred1'] = classifier.predict(M)

d2.to_csv('final.csv')

from sklearn.metrics import roc_auc_score,roc_curve
y_pred_prob = classifier.predict_proba(X_test)[:,1]

auc_score = roc_auc_score(y_test,y_pred_prob)
print(auc_score)

fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(fpr,tpr,label=f'Logistric Regression (AUC = {auc_score:.2})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate ')
plt.title('roc')
plt,legend(loc='lower')










