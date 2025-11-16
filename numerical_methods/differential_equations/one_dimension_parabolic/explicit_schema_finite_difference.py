import numpy as np
import matplotlib.pyplot as plt

# Параметры сетки
L = 1.0  # длина стержня (в уравнении этот коэфф. стоит перед второй производной по x)
T = 0.1
Nx = 50
Nt = 20000
dx = L / Nx
dt = T / Nt
alpha = dt / dx**2  # число Куранта

# Устойчивость схемы по числу Куранта
if alpha > 0.5:
    print("Cхема неустойчива, alpha =", alpha)

x = np.linspace(0, L, Nx + 1)
t = np.linspace(0, T, Nt + 1)


# Начальное распределение температуры f(x)
def f(x):
    return np.where(x <= 0.75, 16 * x + 15, -68 * x + 78)


# Начальное условие
u = np.zeros((Nt + 1, Nx + 1))
u[0, :] = f(x)

# Граничные условия
u[:, 0] = 15
u[:, -1] = 10

for n in range(0, Nt):
    for i in range(1, Nx):
        u[n + 1, i] = u[n, i] + alpha * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])

plt.figure(figsize=(10, 6))
for n in range(0, Nt + 1, Nt // 10):
    plt.plot(x, u[n, :], label=f"t={t[n]:.3f}")
plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.title("Смешанная задача для уравнения теплопроводности")
plt.legend()
plt.grid(True)
plt.show()
