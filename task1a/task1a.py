import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import csv

# Read data
train_data = np.genfromtxt('./data/train.csv', skip_header=1, delimiter=",")

y = train_data[:, 0]
x = train_data[:, 1:]

# Split data
kfold = KFold(n_splits=10, shuffle=True, random_state=8)

alpha = [0.1, 1, 10, 100, 200]
score = []

# Train
for i in range(5):
    armse = 0
    for train, test in kfold.split(x):
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]
        clf = Ridge(alpha[i])
        clf.fit(x_train,y_train)
        # Compute Validation error
        y_pred = clf.predict(x_test)
        rmse = mean_squared_error(y_pred, y_test, squared=False)
        # Compute cross-validation error
        armse+=rmse
    armse = armse/10.0
    print(armse)
    score.append(armse)

# Write to csv
f = open('result.csv','w')
writer = csv.writer(f)
for i in range(5):
    writer.writerow([score[i]])
f.close()






