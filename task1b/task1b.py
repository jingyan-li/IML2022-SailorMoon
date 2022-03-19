import numpy as np
from sklearn.linear_model import LinearRegression
#%%
# Read data
data = np.genfromtxt('./data/train.csv', skip_header=1, delimiter=",")

y = data[:,1]
x = data[:,2:]
#%%
# Generate features
quadratic = x**2
exponential = np.exp(x)
cosine = np.cos(x)
constant = np.ones_like(y)[:, np.newaxis]

X = np.hstack((x, quadratic, exponential, cosine, constant))

#%%
# Fit the model
reg = LinearRegression(fit_intercept=False).fit(X, y)

#%%
# Show regression results
print(reg.score(X,y))
params = reg.coef_
print(reg.intercept_)

#%%
# Save
np.savetxt("submit.csv", params, delimiter='\n')


