import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Параметры задачи
Q = 0.1  # Мощность источника, кг
H = 1.0  # Высота источника, м
Kx = 12  # Коэффициент турбулентной диффузии по оси X
Ky = 11  # Коэффициент турбулентной диффузии по оси Y
Kz = 11  # Коэффициент турбулентной диффузии по оси Z
Vx = 2.68  # Скорость ветра по оси X, м/с

# Временные параметры
t_max = 5  # Максимальное время, с
dt = 1  # Шаг по времени, с
time_steps = np.arange(0, t_max + dt, dt)

# Пространственные параметры
x_max = 40  # Максимальное расстояние по оси X, м
dx = 5  # Шаг по оси X, м
x_steps = np.arange(0, x_max + dx, dx)

y_max = 40  # Максимальное расстояние по оси Y, м
dy = 5  # Шаг по оси Y, м
y_steps = np.arange(-y_max, y_max + dy, dy)

z = 0  # Высота, на которой рассчитывается концентрация


# Расчет концентрации по Гауссовой модели рассеивания
def calculate_concentration(x, y, t):
    sigma_x = np.sqrt(2 * Kx * t)
    sigma_y = np.sqrt(2 * Ky * t)
    sigma_z = np.sqrt(2 * Kz * t)

    C = (Q / (2 * np.pi * sigma_x * sigma_y * sigma_z)) * np.exp(
        -0.5 * ((x - Vx * t) ** 2 / sigma_x**2 + y**2 / sigma_y**2 + H**2 / sigma_z**2)
    )
    return C


X, Y = np.meshgrid(x_steps, y_steps)

# Расчет концентрации для каждого момента времени
for t in time_steps:
    Z = calculate_concentration(X, Y, t)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis")

    ax.set_xlabel("Расстояние по оси X, м")
    ax.set_ylabel("Расстояние по оси Y, м")
    ax.set_zlabel("Концентрация, кг/м^3")
    ax.set_title(f"Концентрация примеси в момент времени t = {t} с")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()
