import matplotlib.pyplot as plt
import numpy as np

x4 = np.linspace(50000, 80000, 5)
x3 = np.linspace(10000, 50000, 5)
x2 = np.linspace(5000, 10000, 5)
x1 = np.linspace(0, 5000, 5)

y4 = np.linspace(0.04, 0, 5)
y3 = np.linspace(0.0814, 0.04, 5)
y2 = np.linspace(1.1416, 0.0814, 5)
y1 = np.linspace(1.3084, 1.1416, 5)

ar1_x = np.hstack((x1, x2, x3, x4))
ar1_y = np.hstack((y1, y2, y3, y4))

ar2_x = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 8, 10, 12, 15, 20, 50, 100, 120])
ar2_y = np.array([1.225, 1.219, 1.213, 1.202, 1.190, 1.167, 1.112, 1.007, 0.909, 0.736, 0.526, 0.414, 0.312, 0.195, 0.089,
     1.027 * 10 ** -3, 5.550 * 10 ** -7, 2.440 * 10 ** -8])
ar3_x = np.array([0, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000,
        5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000])
ar3_y = np.array([1.225, 1.213, 1.201, 1.190, 1.178, 1.167, 1.111, 1.006, 0.909,
        0.819, 0.736, 0.659, 0.589, 0.525, 0.466, 0.412, 0.363, 0.310])

ar2_x = ar2_x * 1000


fig = plt.figure()
plt.xlabel("Высота(м)")
plt.ylabel("Плотность(кг/м^3)")
plt.title("Образное описание")
plt.grid(True)
plt.plot(ar1_x, ar1_y)

fig = plt.figure()
plt.xlabel("Высота(м)")
plt.ylabel("Плотность(кг/м^3)")
plt.title("Таблица 1")
plt.grid(True)
plt.plot(ar2_x, ar2_y)

fig = plt.figure()
plt.xlabel("Высота(м)")
plt.ylabel("Плотность(кг/м^3)")
plt.title("Таблица 2")
plt.grid(True)
plt.plot(ar3_x, ar3_y)

fig = plt.figure()
plt.xlabel("Высота(м)")
plt.ylabel("Плотность(кг/м^3)")
plt.title("сравнение описаний")
plt.grid(True)
plt.plot(ar1_x, ar1_y, label="Образное описание")
plt.plot(ar2_x, ar2_y, label="Таблица 1")
plt.plot(ar3_x, ar3_y, label="Таблица 2")
fig.legend()
plt.show()