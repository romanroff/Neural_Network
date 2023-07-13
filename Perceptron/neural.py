# ------------------------ Сеть с одним параметром ------------------------ #
# weight = .1                                # вес
#
#
# def neural_network(input, weight):         # мини-сеточка :)
#     prediction = input * weight            # предсказание (входные данные * на вес)
#     return prediction
#
# number_of_toes = [8.5, 9.5, 10., 9.]       # входные данные
# input = number_of_toes[0]                  # входной данный :)
# pred = neural_network(input, weight)       # создание НС и загрузка данных
# print(pred)                                # вывод предсказания (0.85)

# ------------------------ Сеть с несколькими входами ------------------------ #
# weights = [.1, .2, 0.]                       # веса
#
#
# def w_sum(input, weights):                   # функция вычисления веса
#     assert(len(input) == len(weights))       # массивы равны, иначе ошибка
#     output = 0                               # здесь будет итоговое значение
#     for i in range(len(input)):
#         output += (input[i] * weights[i])
#     return output
#
#
# def neural_network(input, weights):
#     pred = w_sum(input, weights)
#     return pred
#
#
# toes = [8.5, 9.5, 9.9, 9.0]                 # среднее кол-во игр
# wlrec = [0.65, 0.8, 0.8, 0.9]               # процент побед
# nfans = [1.2, 1.3, 0.5, 1.0]                # число болельщиков
# input = [toes[0], wlrec[0], nfans[0]]       # входные данные
# pred = neural_network(input, weights)       # создание НС и загрузка данных
# print(pred)                                 # результат (0.98, ого-го)

# ------------------------ ЗАДАЧА: ВЕКТОРНАЯ МАТЕМАТИКА ------------------------ #
# def elementwise_multiplication(vec_a, vec_b): # поэлементное умножение векторов
#     c = []
#     assert(len(vec_a) == len(vec_b))
#     # for i in range(len(vec_a)):
#     #     c.append(vec_a[i] * vec_b[i])
#     result = [a * b for a, b in zip(vec_a, vec_b)]
#     return result
#
#
# def elementwise__addition(vec_a, vec_b):      # поэлементное сложение векторов
#     c = []
#     assert (len(vec_a) == len(vec_b))
#     # for i in range(len(vec_a)):
#     #     c.append(vec_a[i] + vec_b[i])
#     result = [a + b for a, b in zip(vec_a, vec_b)]
#     return result
#
#
# def vector_sum(vec_a):                        # сумма элементов вектора
#     return sum(vec_a)
#
#
# def vector_average(vec_a):                    # среднее значение элементов вектора
#     n = len(vec_a)
#     return sum(vec_a) / n
#
#
# def dot_product(vec_a, vec_b):                # скалярное произведение векторов
#     multi = elementwise_multiplication(vec_a, vec_b)
#     result = vector_sum(multi)
#     return result
#
#
# a = [1, 2, 3]
# b = [4, 5, 6]
# print(elementwise_multiplication(a, b))
# print(elementwise__addition(a, b))
# print(vector_sum(a))
# print(vector_average(a))
# print(dot_product(a, b))

# ------------------------ Ура, пришел NumPy ------------------------ #
# import numpy as np
#
# weights = np.array([.1, .2, 0.])
#
#
# def neural_network(input, weights):
#     pred = input.dot(weights)               # скалярное произведение
#     return pred
#
#
# toes = np.array([8.5, 9.5, 9.9, 9.0])
# wlrec = np.array([0.65, 0.8, 0.8, 0.9])
# nfans = np.array([1.2, 1.3, 0.5, 1.0])
# input = np.array([toes[0], wlrec[0], nfans[0]])
# pred = neural_network(input, weights)
# print(pred)

# ------------------------ Прогнозирование с несколькими выходами ------------------------ #
# weights = [.3, .2, .9]                           # веса
#
#
# def multiple_outputs(input, weights):            # функция вычисления значений выходов
#     output = [0 for i in range(len(weights))]    # количество выходов = количество весов
#     for i in range(len(weights)):
#         output[i] = input * weights[i]
#     return output
#
#
# def neural_network(input, weights):               # НС, ничего нового
#     pred = multiple_outputs(input, weights)
#     return pred
#
#
# wlrec = [.65, .8, .9]
# input = wlrec[0]
# pred = neural_network(input, weights)
# print(pred)

# ------------------------ Несколько входов и выходов ------------------------ #
#         игр % побед # болельщиков
# weights = [[0.1, 0.1, -0.3],                        # травмы?
#             [0.1, 0.2, 0.0],                        # победа?
#             [0.0, 1.3, 0.1]]                        # печаль?
#
#
# def w_sum(vector, weights):                         # векторная сумма
#     assert(len(vector) == len(weights))
#     output = 0
#     for i in range(len(vector)):
#         output += (vector[i] * weights[i])
#     return  output
#
#
# def vect_mat_mul(vector, matrix):                   # типо матричное умножение
#     assert(len(vector) == len(matrix))
#     output = [0 for i in range(len(weights))]
#     for i in range(len(vector)):
#         output[i] = w_sum(vector, matrix[i])
#     return output
#
#
# def neural_network(input, weights):                 # НС, ничего нового
#     pred = vect_mat_mul(input, weights)
#     return pred
#
#
# toes = [8.5, 9.5, 9.9, 9.0]                         # текущее среднее число игр
# wlrec = [0.65, 0.8, 0.8, 0.9]                       # процент побед
# nfans = [1.2, 1.3, 0.5, 1.0]                        # число болельщиков
# input = [toes[0],wlrec[0],nfans[0]]
# pred = neural_network(input, weights)
# print(pred)

# ------------------------ Несколько входов и выходов ------------------------ #
