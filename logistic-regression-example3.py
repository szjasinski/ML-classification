import numpy as np
import statsmodels.api as sm

x = np.arange(10).reshape(-1, 1)
y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])
x = sm.add_constant(x)

print(x)
print(y)

model = sm.Logit(y, x)
# .fit_regularized() for regularization
result = model.fit(method='newton')
print(result.params)

result.predict(x)
(result.predict(x) >= 0.5).astype(int)
result.pred_table()
result.summary()
result.summary2()