from sklearn import datasets, model_selection, linear_model, preprocessing, pipeline
import matplotlib.pyplot as plt
import numpy as np
from random import randrange


data_matrix = datasets.fetch_openml(name="mnist_784", version=1)
X, y = data_matrix.data, data_matrix.target


sample = randrange(start=0, stop=2000)
row_1 = X.iloc[sample]
row_1_reshaped = row_1.values.reshape(28, 28)

plt.imshow(row_1_reshaped, cmap="viridis")  # 'cmap' sets the color scheme
plt.colorbar()  # Adds a legend for the colors
plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Cause non-scaled causes the sample to exhaust iterations
pipe = pipeline.make_pipeline(
    preprocessing.StandardScaler(),
    linear_model.LogisticRegression(solver="lbfgs", max_iter=200),
)
pipe.fit(X_train, y_train)

# Test Prediction
pred = pipe.predict(row_1.values.reshape(1, -1))
print("prediction:", pred)
try:
    print("pred as int:", int(pred[0]))
except Exception:
    pass
