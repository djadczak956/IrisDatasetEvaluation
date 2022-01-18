from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
# split it in features and labels
X = iris.data
y = iris.target # ground truth labels

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -----------------------------
model = svm.SVC()   # svm model
model.fit(X_train, y_train) # fit the model with training data
#print(model)

predictions = model.predict(X_test) # predicted labels
acc = accuracy_score(y_test, predictions)   

print("predictions --> ", predictions)
print("actual --> ", y_test)
print("accuracy --> ", acc)

for i in range(len(predictions)):
    print(classes[predictions[i]])
