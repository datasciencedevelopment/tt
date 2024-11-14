import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#
# # Определяем диапазон значений для x и y
# x_values = np.linspace(-10, 10, 100)
# y_values = np.linspace(-10, 10, 100)
#
# # Создаем сетку координат
# X, Y = np.meshgrid(x_values, y_values)
#
# # Коэффициенты уравнения плоскости
# a = 2
# b = -3
# c = 5
#
# # Вычисляем значения Z на основе X и Y
# Z = 2 * X + Y * Y
#
# # Создаем фигуру и 3D-ось
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
#
# # Отображаем поверхность плоскости
# ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
#
# # Настраиваем подписи осей
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # Добавляем заголовок
# ax.set_title('График плоскости z = 2x - 3y + 5')
#
#
# ##############################
#
# fig, ax = plt.subplots()  # this creates a figure paired with axes
#
# x = np.linspace(0, 10, 200)  # x data
# y = np.sin(x) + x            # y data
#
# ax.plot(x, y)                # creates the plot
# # fig.show()                   # this will show your plot in case your environment is not interactive
#
#
# fig, ax = plt.subplots(figsize=(12, 8))
#
#
# fig.suptitle('Our beautiful plot', fontsize=40)
#
#
# color = 'y'
# linestyle = 'dotted'
# linewidth = 4
#
#
# ax.plot(x, y, c=color, linestyle=linestyle, linewidth=linewidth)
#
#
# ax.set_xlabel('argument', fontsize = 30)
# ax.set_ylabel('function', fontsize = 30)
#
# ax.tick_params(axis='both', which='major', labelsize=18)
#
#
#
#
# fig, axes = plt.subplots(1, 2, figsize=(12,8))
# ax1, ax2 = axes
#
# x = np.linspace(0,10,200)
# y1 = np.sin(x)
# y2 = np.cos(x)
#
#
# ax1.set_ylabel('sin(x)', fontsize=30)
# ax1.set_xlabel('x', fontsize=30)
# ax2.set_ylabel('cos(x)', fontsize=30)
# ax2.set_xlabel('x', fontsize=30)
#
# # fig.tight_layout(pad=2)
#
# ax1.plot(x, y1, c='r', linewidth=4)
# ax2.plot(x, y2, c='g', linewidth=4)
#
#

############
############
############
#
# fig, axes = plt.subplots(2, 2, figsize=(12,8))
# (ax1, ax2), (ax3, ax4) = axes
#
#
# x = np.linspace(0,10,200)
# y1 = np.sin(x)
# y2 = np.cos(x)
# y3 = np.tan(x)
# y4 = 2 * np.tan(x)
#
# ax1.set_ylabel('sin(x)', fontsize=30)
# ax1.set_xlabel('x', fontsize=30)
# ax2.set_ylabel('cos(x)', fontsize=28)
# ax2.set_xlabel('x', fontsize=28)
# ax3.set_ylabel('tg(x)', fontsize=24)
# ax3.set_xlabel('x', fontsize=30)
# ax4.set_ylabel('ctg(x)', fontsize=24)
# ax4.set_xlabel('x', fontsize=30)
#
# fig.tight_layout(pad=4)
#
# ax1.plot(x, y1, c='r', linewidth=4)
# ax2.plot(x, y2, c='g', linewidth=4)
# ax3.plot(x, y3, c='b', linewidth=4)
# ax4.plot(x, y4, c='black', linewidth=4)



#############################################
# Создание столбчатой диаграммы
#############################################


#
# films = ['Wonder Woman', 'Sonic', '1917', 'Star Wars', 'Onward']
# box_office = [16.7, 26.1, 37.0, 34.5, 10.6]
#
# plt.bar(films, box_office)
#
# plt.grid(color='grey', linestyle=':', linewidth=1.0, axis='both', alpha=0.5)
#
#
# plt.ylabel('Box Office (mil. $)')  # labling y-axis
# plt.xlabel('Film title')           # labling x-axis
# plt.title('Box office of 5 different films of 2020 in the USA')  # giving chart a title
#
# plt.show()



#############################################
# Горизонтальная столбчатая диаграмма
#############################################
#
# films = ['Wonder Woman', 'Sonic', '1917', 'Star Wars', 'Onward']
# box_office = [16.7, 26.1, 37.0, 34.5, 10.6]
#
# # plotting the chart horizontally
# plt.barh(films, box_office)
#
# plt.xlabel('Box Office (mil. $)')
# plt.ylabel('Film title')
# plt.title('Box office of 5 different films of 2020 in the USA')
#
# plt.show()
#

#
# #############################################
# # Сгруппированная столбчатая диаграмма
# #############################################
#
#
# years = ["2016", "2017", "2018", "2019"]
# cats = [57, 50, 47, 30]
# dogs = [43, 50, 53, 70]
# birds = [3, 5, 5, 7]
# robots = [30, 50, 50, 200]
#
#
# # increase the figure size
# plt.figure(figsize=(10, 6))
#
# # create x-axis values depending on the number of years
# x_axis = np.arange(len(years))
#
# plt.bar(x_axis - 0.2, cats, width=0.2, label='Cats')
# plt.bar(x_axis - 0, dogs, width=0.2, label='Dogs')
# plt.bar(x_axis + 0.2, birds, width=0.2, label='Birds')
# plt.bar(x_axis + 0.4, robots, width=0.2, label='robots')
#
# # set tick labels and their location
# plt.xticks(x_axis, years)
#
# plt.xlabel('Years', fontsize=14)
# plt.ylabel('Preference (%)', fontsize=14)
# plt.title('The results of cat/dog survey', fontsize=20)
#
# # add legend
# plt.legend()
#
# plt.show()


#
# #############################################
# # Сгруппированная столбчатая диаграмма
# #############################################
#
# plt.figure(figsize=(10, 6))
#
# years = ['2016', '2017', '2018', '2019']
# cats = [57, 50, 47, 30]
# dogs = [43, 50, 53, 70]
# birds = [3, 5, 5, 7]
# robots = [30, 50, 50, 80]
#
# arr_cats = np.array(cats)
# arr_dogs = np.array(dogs)
# arr_birds = np.array(birds)
# arr_robots = np.array(robots)
#
# plt.bar(years, arr_cats, label='Cats')
# plt.bar(years, arr_dogs, bottom=arr_cats, label='Dogs')
# plt.bar(years, arr_birds, bottom=arr_cats+arr_dogs, label='birds')
# plt.bar(years, arr_robots, bottom=arr_cats+arr_dogs+arr_birds, label='robots')
#
#
# plt.xlabel('Years', fontsize=14)
# plt.ylabel('Preference (%)', fontsize=14)
# plt.title('The results of cat/dog survey', fontsize=20)
# plt.legend()
# plt.show()




# months = ['January', 'February', 'March']
# transport = np.array([760, 575, 955])
# food = np.array([597, 710, 675])
# healthcare = np.array([395, 210, 450])
#
# plt.bar(months, transport, width=0.9, label='Transport')
# plt.bar(months, food, width=0.9, label='Food', bottom=transport)
# # YOUR CODE HERE #
# plt.bar(months, healthcare, width=0.9, label='Healthcare', bottom=transport+food)
#
# plt.xlabel("Period", fontsize=12)
# plt.ylabel("Amount ($) ", fontsize=12)
# plt.legend()
# plt.title("My monthly expenses", fontsize=14)
#
# plt.show()



#
# #############################################
# # Matplotlib histogram
# # Количество повторяющихся данных масива
# #############################################
#
#
# my_data = [163, 163, 164, 170, 170, 172, 173, 190]
# andy_data = [161, 172, 174, 175, 181, 183, 186, 190]
# bins = [160, 170, 180, 190]
# # bins = 3 # делит на равные части
#
# names = ["my friends", "Andy's friends"]
#
#
# plt.figure()
#
# plt.title("Mine and Andy's friends' height")
# plt.ylabel("Number of people")
# plt.xlabel("Height in cm")
#
# plt.hist(my_data, bins=bins, color="orange", edgecolor='white', range=(170, 180))
#
#
# plt.figure()
#
# plt.title("Mine and Andy's friends' height")
# plt.ylabel("Number of people")
# plt.xlabel("Height in cm")
#
# plt.hist(my_data)
#
# # plt.hist([my_data, andy_data], bins=bins, label=names, color=['blue', 'green'])
#
# # plt.hist([my_data, andy_data], bins=bins, label=names, stacked=True, edgecolor='white')
#
# plt.legend()
# plt.show()

#
# #####################################
# #####################################
# #
# # ПОИСК САМОГО БЫСТРОГО СПУСКА И ШАГА ДЛЯ ЛЮБОЙ ФУНКЦИИ
# #
# #####################################
# #####################################
#
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sympy import symbols, lambdify, diff
#
# # Определяем символические переменные
# x_sym, y_sym = symbols('x y')
#
# # Задайте вашу функцию символически
# # Замените эту функцию на любую другую по вашему выбору
# #####################################
#
# f_sym = x_sym ** 2 + y_sym ** 2
#
# #####################################
#
# # Вычисляем градиент
# grad_f_sym = [diff(f_sym, var) for var in (x_sym, y_sym)]
#
# # Вычисляем матрицу Гессе (если необходимо)
# hessian_f_sym = [[diff(grad, var) for var in (x_sym, y_sym)] for grad in grad_f_sym]
#
# # Преобразуем символические функции в численные
# f = lambdify((x_sym, y_sym), f_sym, 'numpy')
# grad_f = lambdify((x_sym, y_sym), grad_f_sym, 'numpy')
# hessian_f = [[lambdify((x_sym, y_sym), hess_elem, 'numpy') for hess_elem in row] for row in hessian_f_sym]
#
#
# # Функция для вычисления оптимального шага gamma_k
# def optimal_gamma(xk, yk):
#     grad = np.array(grad_f(xk, yk))
#     grad_norm_sq = np.dot(grad, grad)
#
#     # Вычисляем Гессиан в точке (xk, yk)
#     H = np.array([[hessian_f[i][j](xk, yk) for j in range(2)] for i in range(2)])
#
#     hess_grad = np.dot(grad, H @ grad)
#     if hess_grad == 0:
#         return 0  # Избегаем деления на ноль
#     else:
#         gamma_k = grad_norm_sq / hess_grad
#         return gamma_k
#
#
# # Начальная точка
#
# #####################################
# x0, y0 = 2, 3
#
# #####################################
#
# # Параметры алгоритма
# iterations = 20  # количество итераций
#
# # Метод градиентного спуска с фиксированным шагом gamma = 0.01
# gamma_0_01 = 0.01
# x_vals_0_01 = [x0]
# y_vals_0_01 = [y0]
# z_vals_0_01 = [f(x0, y0)]
# xk, yk = x0, y0
# for _ in range(iterations):
#     grad = np.array(grad_f(xk, yk))
#     xk = xk - gamma_0_01 * grad[0]
#     yk = yk - gamma_0_01 * grad[1]
#     x_vals_0_01.append(xk)
#     y_vals_0_01.append(yk)
#     z_vals_0_01.append(f(xk, yk))
#
# # Метод градиентного спуска с фиксированным шагом gamma = 0.1
# gamma_0_1 = 0.1
# x_vals_0_1 = [x0]
# y_vals_0_1 = [y0]
# z_vals_0_1 = [f(x0, y0)]
# xk, yk = x0, y0
# for _ in range(iterations):
#     grad = np.array(grad_f(xk, yk))
#     xk = xk - gamma_0_1 * grad[0]
#     yk = yk - gamma_0_1 * grad[1]
#     x_vals_0_1.append(xk)
#     y_vals_0_1.append(yk)
#     z_vals_0_1.append(f(xk, yk))
#
# # Метод наискорейшего спуска
# gamma_vals_opt = []
# x_vals_opt = [x0]
# y_vals_opt = [y0]
# z_vals_opt = [f(x0, y0)]
# xk, yk = x0, y0
# for _ in range(iterations):
#     grad = np.array(grad_f(xk, yk))
#     gamma_k = optimal_gamma(xk, yk)
#     gamma_vals_opt.append(gamma_k)
#     xk = xk - gamma_k * grad[0]
#     yk = yk - gamma_k * grad[1]
#     x_vals_opt.append(xk)
#     y_vals_opt.append(yk)
#     z_vals_opt.append(f(xk, yk))
#
# # Построение графиков
# # Создаем сетку значений для построения поверхности функции
# x_grid = np.linspace(-5, 5, 100)
# y_grid = np.linspace(-2, 6, 100)
# X, Y = np.meshgrid(x_grid, y_grid)
# Z = f(X, Y)
#
# # Создаем 3D-рисунок
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Поверхность функции
# ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
#
# # Траектории градиентного спуска
# ax.plot(x_vals_0_01, y_vals_0_01, z_vals_0_01, 'o-', color='red', label='γ = 0.01')
# ax.plot(x_vals_0_1, y_vals_0_1, z_vals_0_1, 'o-', color='blue', label='γ = 0.1')
#
# # Отображаем значение γₖ на графике для наискорейшего спуска
# gamma_k_text = f'γₖ = {gamma_vals_opt[0]:.2f}'
#
# ax.plot(x_vals_opt, y_vals_opt, z_vals_opt, 'o-', color='green', label=f'Самый быстрый спуск ({gamma_k_text})')
#
# # Начальная точка
# ax.plot([x0], [y0], [f(x0, y0)], 'ko', label='Начальная точка')
#
# # Настройки графика
# ax.set_title('Градиентный спуск для заданной функции')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('f(x, y)')
# ax.legend()
# plt.show()


#####################################
#####################################
#
# СРАВНЕНИЕ ПОВЕДЕНИЯ N ШАГОВ ДЛЯ ЛЮБОЙ ФУНКЦИИ
#
#####################################
#####################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, lambdify, diff, sqrt

# Определяем символические переменные
x_sym, y_sym = symbols('x y')

# Задаем функцию символически
f_sym = sqrt(x_sym**2 + y_sym**2)

# Вычисляем градиент символически
grad_f_sym = [diff(f_sym, var) for var in (x_sym, y_sym)]

# Преобразуем символические функции в численные функции
f = lambdify((x_sym, y_sym), f_sym, 'numpy')

# Обработка случая x = y = 0 для избежания деления на ноль в градиенте
def grad_f_num(x, y):
    denom = np.sqrt(x**2 + y**2)
    epsilon = 1e-8  # Малое число для избежания деления на ноль
    denom = np.where(denom == 0, epsilon, denom)
    df_dx = x / denom
    df_dy = y / denom
    return np.array([df_dx, df_dy])

# Начальная точка
x0, y0 = 2, 3

# Шаги градиентного спуска для сравнения
gamma_list = [0.8, 2]

# Количество итераций
iterations = 50

# Словарь для хранения траекторий
trajectories = {}

for gamma in gamma_list:
    x_vals = [x0]
    y_vals = [y0]
    z_vals = [f(x0, y0)]
    xk, yk = x0, y0
    for _ in range(iterations):
        grad = grad_f_num(xk, yk)
        # Проверяем норму градиента для избежания бесконечных циклов
        grad_norm = np.linalg.norm(grad)
        if grad_norm == 0:
            break  # Градиент равен нулю, достигли стационарной точки
        xk = xk - gamma * grad[0]
        yk = yk - gamma * grad[1]
        x_vals.append(xk)
        y_vals.append(yk)
        z_vals.append(f(xk, yk))
    trajectories[gamma] = (x_vals, y_vals, z_vals)

# Построение графиков
# Создаем сетку значений для построения поверхности функции
x_grid = np.linspace(-5, 5, 200)
y_grid = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = f(X, Y)

# Создаем 3D-график
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Поверхность функции
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Цвета для разных шагов
colors = ['red', 'blue', 'green', 'orange']

# Построение траекторий для каждого шага
for idx, gamma in enumerate(gamma_list):
    x_vals, y_vals, z_vals = trajectories[gamma]
    label = f'γ = {gamma}'
    ax.plot(x_vals, y_vals, z_vals, 'o-', color=colors[idx % len(colors)], label=label)

# Начальная точка
ax.plot([x0], [y0], [f(x0, y0)], 'ko', label='Начальная точка')

# Настройки графика
ax.set_title('Градиентный спуск для функции $f(x, y) = \\sqrt{x^2 + y^2}$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.legend()
plt.show()



