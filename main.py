#!/usr/bin/env python
# coding=utf8
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, sqrt

start_point = 0
way = []

X = [0, 2, 5, 6, 8]
Y = [2, 5, 2, 6, 3]
n = len(X)  # кол-о пунктов

M = np.zeros([n, n])  # Шаблон матрицы относительных расстояний между пунктами

for i in np.arange(0, n):
    for j in np.arange(0, n):
        if i != j:
            M[i, j] = sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)  # Заполнение матрицы
        else:
            M[i, j] = float('inf')  # Заполнение главной диагонали матрицы

way.append(start_point)
print(M)
for i in np.arange(1, n):
    cur_row = []
    for j in np.arange(0, n):
        cur_row.append(M[way[i - 1], j])
    mininum = min(cur_row)
    mininum_i = cur_row.index(mininum)

    way.append(mininum_i)  # Индексы пунктов ближайших городов соседей
    for j in np.arange(0, i):
        print( M[way[i], way[j]])
        print('step: ')
        print(i,j)

        print(f'Point 1: {M[way[i], way[j]]}')
        print(f'Point 2: {M[way[j], way[i]]}')
        M[way[i], way[j]] = float('inf')
        M[way[j], way[i]] = float('inf')


    print(f'Way: {way}')
    # print(cur_row)
    # print(f'Minimum: {mininum}')

S = sum(
    [sqrt((X[way[i]] - X[way[i + 1]]) ** 2 + (Y[way[i]] - Y[way[i + 1]]) ** 2) for i in np.arange(0, n - 1, 1)]) + sqrt(
    (X[way[n - 1]] - X[way[0]]) ** 2 + (Y[way[n - 1]] - Y[way[0]]) ** 2)

plt.title('Общий путь = %s.Всего городов: %i.\n ' % (
    round(S, 3), n), size=14)

X1 = [X[way[i]] for i in np.arange(0, n, 1)]
Y1 = [Y[way[i]] for i in np.arange(0, n, 1)]
plt.plot(X1, Y1, color='black', linestyle=' ', marker='x')
plt.plot(X1, Y1, color='red', linewidth=1)
X2 = [X[way[n - 1]], X[way[0]]]
Y2 = [Y[way[n - 1]], Y[way[0]]]
plt.plot(X2, Y2, color='blue', linewidth=1, linestyle='--', label='Путь от последнего пункта до \n '
                                                                 'почтового отделения')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# print(S)
# print(route)
