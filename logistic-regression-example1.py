
# CODE FROM REAL PYTHON LOGISTIC REGRESSION TUTORIAL


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x = np.arange(10).reshape(-1,1) # -1 mean unknown dimension
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
# x = np.arange(10).reshape(10, 1)

print(x)
print(y)

model1 = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)
# improved model

model2 = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(x, y)

print(model.classes_)
print(model.intercept_)
print(model.coef_)
print(model.predict_proba(x))
print(model.predict(x))
print(model.score(x,y))
print(confusion_matrix(y, model.predict(x)))

# confusion matrix visualization

cm = confusion_matrix(y, model.predict(x))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


print(classification_report(y, model.predict(x)))
