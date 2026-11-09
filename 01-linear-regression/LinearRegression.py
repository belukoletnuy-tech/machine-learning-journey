import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt

X, y, coef = make_regression(
    n_samples=100,
    n_features=1,
    n_informative=1,
    n_targets=1,
    coef=True,
    random_state=42,
    noise=0.1
)

df = pd.DataFrame(X, columns=["column_1"])

print(f"Істинні ваги: {coef}")

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print(mae(y, y_pred))

print(model.coef_)

sorted_indices = np.argsort(X.flatten())
X_sorted = X[sorted_indices]
y_sorted = y[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

plt.scatter(X_sorted[:100], y_sorted[:100], label='Істинні значення (y)' ,color="red")
plt.plot(X_sorted[:100], y_pred_sorted[:100], label='Прогноз (y_pred)', color="red")

plt.grid(True)
plt.legend()
plt.show()