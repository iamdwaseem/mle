# ML Algorithms - Code Only 🚀
**Sorted: Easy → Hard**

---

## 1. LINEAR REGRESSION

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = load_diabetes()
X = data.data[:, 2].reshape(-1, 1)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.title("Linear Regression")
plt.show()
```

---

## 2. LOGISTIC REGRESSION

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

data = load_iris()
X = data.data[:, :1]
y = data.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

print("Accuracy:", model.score(X, y))

x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_prob = model.predict_proba(x)[:, 0]

plt.scatter(X, y)
plt.plot(x, y_prob)
plt.xlabel("Feature")
plt.ylabel("Probability")
plt.title("Logistic Regression")
plt.show()
```

---

## 3. NAIVE BAYES

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 4. K-NEAREST NEIGHBORS

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

data = load_iris()
x = data.data
y = data.target

x = StandardScaler().fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.3, random_state=42
)

best_k = 1
best_acc = 0

for k in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(xtrain, ytrain)
    acc = accuracy_score(ytest, model.predict(xtest))

    if acc > best_acc:
        best_acc = acc
        best_k = k

final = KNeighborsClassifier(n_neighbors=best_k)
final.fit(xtrain, ytrain)

print("Best k:", best_k)
print("Accuracy:", accuracy_score(ytest, final.predict(xtest)))
```

---

## 5. PERCEPTRON (Single)

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

model = Perceptron()
model.fit(X, y)

y_pred = model.predict(X)

print("Accuracy:", accuracy_score(y, y_pred))
print("Weights:", model.coef_)
print("Bias:", model.intercept_)
```

---

## 6. PERCEPTRON (Multi-Layer)

```python
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data
y = data.target

model = MLPClassifier(
    hidden_layer_sizes=(5,),
    max_iter=3000,
    random_state=42
)

model.fit(X, y)
y_pred = model.predict(X)

print("Accuracy:", accuracy_score(y, y_pred))
```

---

## 7. ID3 (Decision Tree - Entropy)

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("ID3 Accuracy:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(12, 8))
plot_tree(model)
plt.show()
```

---

## 8. CART (Decision Tree - Gini)

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = DecisionTreeClassifier(criterion="gini")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("CART Accuracy:", accuracy_score(y_test, y_pred))

plot_tree(model, filled=True)
plt.show()
```

---

## 9. SVM (Linear)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, random_state=1)

model = SVC(kernel='linear')
model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y)

w = model.coef_[0]
b = model.intercept_[0]
x = np.linspace(X[:, 0].min(), X[:, 0].max())
y_line = -(w[0] * x + b) / w[1]

plt.plot(x, y_line)
plt.title("Linear SVM")
plt.show()
```

---

## 10. SVM (Non-Linear RBF)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=100, noise=0.1)

model = SVC(kernel='rbf')
model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y)

xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 100),
                     np.linspace(-1.5, 1.5, 100))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.show()
```

---

## 11. K-MEANS

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

data = load_iris()
X = data.data[:, :2]

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker='x'
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.show()
```

---

## 12. EM (GMM)

```python
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

X = load_iris().data

kmeans = KMeans(n_clusters=3, random_state=42)
k_labels = kmeans.fit_predict(X)

gmm = GaussianMixture(n_components=3, random_state=42)
g_labels = gmm.fit_predict(X)

k_score = silhouette_score(X, k_labels)
g_score = silhouette_score(X, g_labels)

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=k_labels)
plt.title("K-Means")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=g_labels)
plt.title("EM (GMM)")

plt.show()

print("K-Means Score:", k_score)
print("EM Score:", g_score)
```

---

## 13. BAGGING

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 14. RANDOM FOREST

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Random Forest Accuracy:", model.score(X_test, y_test))
```

---

## 15. ADABOOST

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = AdaBoostClassifier(n_estimators=50)
model.fit(X_train, y_train)

print("AdaBoost Accuracy:", model.score(X_test, y_test))
```

---

## 16. XGBOOST

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

print("XGBoost Accuracy:", model.score(X_test, y_test))
```

---

## 17. LINEAR REGRESSION (From Scratch)

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 6]
n = len(x)

m = (n * sum([x[i] * y[i] for i in range(n)]) - sum(x) * sum(y)) / (n * sum([i * i for i in x]) - sum(x) ** 2)
c = (sum(y) - m * sum(x)) / n

y_pred = [m * i + c for i in x]

print("m:", m, "c:", c)

plt.scatter(x, y)
plt.plot(x, y_pred)
plt.show()
```

---

## 18. HOLDOUT VALIDATION

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("Holdout Accuracy:", model.score(X_test, y_test))
```

---

## 19. K-FOLD VALIDATION

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=5)
print("K-Fold Avg Accuracy:", scores.mean())
```

---

## 20. BOOTSTRAP VALIDATION

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

n = len(X)
idx = np.random.choice(n, n, replace=True)

X_train, y_train = X[idx], y[idx]
mask = ~np.isin(range(n), idx)
X_test, y_test = X[mask], y[mask]

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("Bootstrap Accuracy:", model.score(X_test, y_test))
```

---

## 21. COMPLETE PIPELINE

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv("spine.csv")

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = LabelEncoder().fit_transform(data[col])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Classification
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Classification Accuracy:", accuracy_score(y_test, y_pred))

# Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)
print("Regression MSE:", mean_squared_error(y_test, y_pred_reg))

# Clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
print("Silhouette Score:", silhouette_score(X_scaled, labels))
```

---

**Copy, paste, run. Good luck! ⚡**
