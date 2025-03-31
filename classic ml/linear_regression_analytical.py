import numpy as np

def linear_regression_analytical(X, y):
    # Добавляем столбец единичных значений для смещения (bias)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # Вычисление коэффициентов по аналитической формуле
    # w = (X^T X)^-1 X^T y
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    return theta

# Пример данных
# X - матрица признаков (например, 2 признака)
X = np.array([[5, 2], [2, 3], [3, 4], [4, 5]])

# y - целевая переменная
y = np.array([5, 7, 9, 11])

# Нахождение коэффициентов (включая смещение)
theta = linear_regression_analytical(X, y)

print("Коэффициенты модели (включая смещение):")
print(theta)
