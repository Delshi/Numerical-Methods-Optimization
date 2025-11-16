import numpy as np
import matplotlib.pyplot as plt

# Параметры
L = 1.0  # длина стержня
T = 0.1  # общее время
Nx = 50  # число шагов по x
Nt = 10000  # число шагов по времени
dx = L / Nx
dt = T / Nt
alpha = dt / dx**2

# Сетка
x = np.linspace(0, L, Nx + 1)
t = np.linspace(0, T, Nt + 1)


# Начальное условие
def f(x):
    return np.where(x <= 0.75, 16 * x + 15, -68 * x + 78)


# Начальная температура
u = np.zeros((Nt + 1, Nx + 1))
u[0, :] = f(x)

# Граничные условия
u[:, 0] = 15
u[:, -1] = 10

# Коэффициенты системы
a = -alpha * np.ones(Nx - 1)
b = (1 + 2 * alpha) * np.ones(Nx - 1)
c = -alpha * np.ones(Nx - 1)


# Прогонка (метод Томаса)
def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_star = np.zeros(n)
    d_star = np.zeros(n)

    # Прямая прогонка
    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_star[i - 1]
        c_star[i] = c[i] / denom if i < n - 1 else 0
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / denom

    # Обратная прогонка
    x = np.zeros(n)
    x[-1] = d_star[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]

    return x


# Расчёт по времени
for n in range(0, Nt):
    d = u[n, 1:-1]  # правая часть
    # Корректируем с учётом граничных условий
    d[0] += alpha * u[n + 1, 0]  # левая граница
    d[-1] += alpha * u[n + 1, -1]  # правая граница
    u[n + 1, 1:-1] = thomas_algorithm(a, b, c, d)

# Визуализация
plt.figure(figsize=(10, 6))
for n in range(0, Nt + 1, Nt // 10):
    plt.plot(x, u[n, :], label=f"t={t[n]:.3f}")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.title("Решение уравнения теплопроводности (неявная схема)")
plt.legend()
plt.grid(True)
plt.show()
