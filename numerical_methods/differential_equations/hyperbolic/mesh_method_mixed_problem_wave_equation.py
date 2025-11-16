import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


L = 1.0
T = 5
Nx = 200
Nt = 5000
c = 1.0
x = np.linspace(0, L, Nx + 1)
dx = x[1] - x[0]
dt = T / Nt
r = c * dt / dx

H_values = [0.5, 0.6, 0.7]


# Ищем параметры для начального условия f(x)
def DefineInitialCondition(H):
    x1, y1 = 0.4, 0.0
    x2, y2 = 0.8, 0.0
    xH, yH = H, 8.0

    k1 = yH / (xH - x1)
    k2 = yH / (xH - x2)

    def f(x):
        if 0.4 <= x <= H:
            return k1 * (x - 0.4)
        elif H < x <= 0.8:
            return k2 * (x - 0.8)
        else:
            return 0.0

    return np.vectorize(f), k1, k2


all_solutions = []
labels = []


for H in H_values:
    f_func, k1, k2 = DefineInitialCondition(H)
    u0 = f_func(x)
    u1 = np.zeros_like(u0)
    u_next = np.zeros_like(u0)

    # Решаем начальное условие: du/dt = 0
    for i in range(1, Nx):
        u1[i] = u0[i] + 0.5 * r**2 * (u0[i + 1] - 2 * u0[i] + u0[i - 1])

    # Эволюция во времени
    u_prev = u0.copy()
    u_curr = u1.copy()
    solution = [u0.copy(), u1.copy()]

    # Решаем уравнение
    for n in range(2, Nt):
        for i in range(1, Nx):
            u_next[i] = (
                2 * (1 - r**2) * u_curr[i]
                - u_prev[i]
                + r**2 * (u_curr[i + 1] + u_curr[i - 1])
            )
        u_next[0] = u_next[Nx] = 0
        u_prev, u_curr = u_curr, u_next.copy()
        solution.append(u_curr.copy())

    all_solutions.append(solution)
    labels.append(f"H = {H}, k1 = {k1:.2f}, k2 = {k2:.2f}")


colors = ["red", "blue", "green"]

fig, ax = plt.subplots(figsize=(10, 6))
lines = []
for color, label in zip(colors, labels):
    (line,) = ax.plot(x, all_solutions[0][0], color=color, label=label)
    lines.append(line)

ax.set_ylim(-10, 10)
ax.set_xlim(0, 1)
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.grid(True)
ax.legend()


def update(frame):
    for sol, line in zip(all_solutions, lines):
        line.set_ydata(sol[frame])
    ax.set_title(f"Смешанная задача для волнового уравнения")
    return lines


ani = FuncAnimation(fig, update, frames=range(0, Nt, Nt // 300), interval=70, blit=True)
plt.tight_layout()
plt.show()
