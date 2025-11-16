import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Параметры задачи
Q = 1  # Мощность источника, кг
H = 1  # Высота источника, м
Kx, Ky, Kz = 38, 32, 35  # Коэффициенты турбулентной диффузии
Vx = 1.07  # Скорость ветра, м/с
t = 5  # Время расчета, с
dx = dy = 5  # Шаг сетки, м
x_max = 40  # Максимальное расстояние по оси OX, м

# Создание сетки
x = np.arange(-10, 40 + dy, dx)
y = np.arange(-x_max, 25 + dy, dy)
X, Y = np.meshgrid(x, y)


# Расчет концентрации по решению Робертса
def roberts_concentration(Q, H, Kx, Ky, Kz, Vx, x, y, t):
    sigma_x = np.sqrt(2 * Kx * t)
    sigma_y = np.sqrt(2 * Ky * t)
    sigma_z = np.sqrt(2 * Kz * t)

    C = (Q / ((2 * np.pi) ** (3 / 2) * sigma_x * sigma_y * sigma_z)) * np.exp(
        -(
            (x - Vx * t) ** 2 / (2 * sigma_x**2)
            + y**2 / (2 * sigma_y**2)
            + H**2 / (2 * sigma_z**2)
        )
    )
    return C


# Расчет концентрации
Z = roberts_concentration(Q, H, Kx, Ky, Kz, Vx, X, Y, t)

# Построение 3D поверхности
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")

ax.set_xlabel("X (м)")
ax.set_ylabel("Y (м)")
ax.set_zlabel("Концентрация (кг/м^3)")
ax.set_title("Концентрация примеси (Решение Робертса)")

plt.show()
