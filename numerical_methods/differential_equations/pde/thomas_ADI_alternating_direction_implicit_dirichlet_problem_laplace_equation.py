import numpy as np
import matplotlib.pyplot as plt

N = 50
h = 1.0 / N
x = np.linspace(0, 1, N + 1)
y = np.linspace(0, 1, N + 1)
u = np.zeros((N + 1, N + 1))

# Граничные условия
u[0, :] = 1
u[N, :] = np.exp(y)
u[:, 0] = 1
u[:, N] = np.exp(x)


def thomas(a, b, c, d):
    n = len(d)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)

    # Прямая прогонка
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * c_[i - 1]
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / denom
    d_[-1] = (d[-1] - a[-2] * d_[-2]) / (b[-1] - a[-2] * c_[-2])

    # Обратная прогонка
    x = np.zeros(n)
    x[-1] = d_[-1]
    for i in reversed(range(n - 1)):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x


def adi(u, max_iter=1000, tol=1e-6):
    N = u.shape[0] - 1
    h2 = h * h
    for it in range(max_iter):
        u_old = u.copy()

        # Прогонка по x (строки)
        for j in range(1, N):
            a = np.ones(N - 1)
            b = -4 * np.ones(N - 1)
            c = np.ones(N - 1)
            d = -(u[1:N, j + 1] + u[1:N, j - 1])
            # Граничные условия
            d[0] -= u[0, j]
            d[-1] -= u[N, j]
            u[1:N, j] = thomas(a, b, c, d)

        # Прогонка по y (столбцы)
        for i in range(1, N):
            a = np.ones(N - 1)
            b = -4 * np.ones(N - 1)
            c = np.ones(N - 1)
            d = -(u[i + 1, 1:N] + u[i - 1, 1:N])
            d[0] -= u[i, 0]
            d[-1] -= u[i, N]
            u[i, 1:N] = thomas(a, b, c, d)

        if np.linalg.norm(u - u_old, ord=np.inf) < tol:
            break

    print(f"Сошлось за {it+1} итераций")

    return u


u = adi(u)

# Визуализация
X, Y = np.meshgrid(x, y, indexing="ij")
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, u, cmap="viridis")
ax.set_title("Решение уравнения Лапласа в квадрате")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u(x,y)")
plt.show()
