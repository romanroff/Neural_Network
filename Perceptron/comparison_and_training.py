# ------------------------ Измерим ошибку ------------------------ #
# knob_weight = .5                        # настраиваемый вес
# input = .5                              # входные данные
# goal_pred = .8                          # целевое значение
# pred = input * knob_weight              # результат
# error = (pred - goal_pred) ** 2         # среднеквадратичная ошибка
# print(error)                            # вывод ошибки

# ------------------------ Простейшая форма нейронного обучения ------------------------ #
# weight = .1                                   # вес
# lr = .1
#
#
# def neural_network(input, weight):            # как всегда, НС
#     prediction = input * weight
#     return prediction
#
#
# number_of_toes = [8.5]
# win_or_lose_binary = [1]
# input = number_of_toes[0]
# true = win_or_lose_binary[0]
# pred = neural_network(input, weight)
# error = (pred - true) ** 2
# print(error)
#
# p_up = neural_network(input, weight) + lr     # предсказание с изменением ошибки
# e_up = (p_up - true) ** 2
# p_dn = neural_network(input, weight) - lr     # предсказание с изменением ошибки
# e_dn = (p_dn- true) ** 2
# print(e_up, '\n', e_dn)                       # смотрим, добавить или вычесть, чтобы уменьшиьб ошибку

# ------------------------ «холодно/горячо» ------------------------ #
# weight = .5                                                               # вес
# input = .5                                                                # входные данные
# goal_prediction = .8                                                      # целевое значение
# step_amount = .001                                                        # шаг изменения ошибки
# for i in range(1101):                                                     # цикл для изменения веса
#     predictiton = input * weight                                          # предсказание
#     error = (predictiton - goal_prediction) ** 2                          # MSE
#     print('Error: ' + str(error) + ' Prediction: ' + str(predictiton))    #
#     up_prediction = input * (weight + step_amount)                        # предсказание с + шагом
#     up_error = (goal_prediction - up_prediction) ** 2                     # MSE после изменения веса
#     down_prediction = input * (weight - step_amount)                      # предсказание с - шагом
#     down_error = (goal_prediction - down_prediction) ** 2                 # MSE после изменения веса
#     if down_error < up_error:                                             # сравнение ошибок,
#         weight -= step_amount                                             # чтобы изменить вес
#     if down_error > up_error:                                             #
#         weight += step_amount                                             #

# ------------------------ Вычисление направления и величины из ошибки ------------------------ #
# weight = .5                                                                 # вес
# goal_pred = .8                                                              # целевое значение
# input = .5                                                                  # входные данные
#
# for i in range(20):                                                         # цикл для вычисления веса
#     pred = input * weight                                                   # предсказание
#     error = (goal_pred - pred) ** 2                                         # MSE
#     direction_and_amount = (pred - goal_pred) * input                       # градиентный спуск
#     weight -= direction_and_amount                                          # изменение веса
#     print('Error: ' + str(error) + ' Prediction: ' + str(pred))             #

# ------------------------ Одна итерация градиентного спуска ------------------------ #
# weight = .1                                         # вес
# alpha = .01                                         # коэффициент скорости обучения
#
#
# def neural_network(input, weight):                  # НС
#     prediction = input * weight                     #
#     return prediction                               #
#
#
# number_of_toes = [8.5]                              #
# win_or_lose_binary = [1]                            #
# input = number_of_toes[0]                           #
# goal_pred = win_or_lose_binary[0]                   #
# pred = neural_network(input,weight)                 #
# error = (pred - goal_pred) ** 2                     # MSE
# delta = pred - goal_pred                            # чистая ошибка
# weight_delta = input * delta                        # градиентный спуск
# weight -= weight_delta * alpha                      # изменение веса градиентным спуском с учетом коэффициента

# ------------------------ Создание градиентного спуска ------------------------ #
# weight, goal_pred, input = (.0, .8, 2)                                      # вес, решение, входные данные (двойка, что ты творишь)
#
# for i in range(10):                                                         #
#     print("---------\nWeight:" + str(weight))                               #
#     pred = input * weight                                                   # предсказание
#     error = (pred - goal_pred) ** 2                                         # MSE
#     delta = pred - goal_pred                                                # чистая ошибка
#     weight_delta = delta * input                                            # градиентный спуск
#     weight = weight - weight_delta                                          # новый вес
#     print("Error:" + str(error) + " Prediction:" + str(pred))               #
#     print("Delta:" + str(delta) + " Weight delta:" + str(weight_delta))     #

# ------------------------ Создание градиентного спуска с альфа-коэффициентом ------------------------ #
# weight, goal_pred, input, alpha = (.0, .8, 2, .2)                           # вес, решение, входные данные, коэффициент
#
# for i in range(10):                                                         #
#     print("---------\nWeight:" + str(weight))                               #
#     pred = input * weight                                                   # предсказание
#     error = (pred - goal_pred) ** 2                                         # MSE
#     derivative = input * (pred - goal_pred)                                 # производная
#     weight = weight - (derivative * alpha)                                  # новый вес
#     print("Error:" + str(error) + " Prediction:" + str(pred))               #
#     print("Derivative:" + str(derivative))

# ------------------------ Обучение методом градиентного спуска с несколькими входами ------------------------ #
def w_sum(a, b):
    assert (len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output


weights = [.1, .2, -.1]


def neural_network(input, weights):
    pred = w_sum(input, weights)
    return pred


toes = [8.5 , 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2 , 1.3, 0.5, 1.0]
win_or_lose_binary = [1, 1, 0, 1]
true = win_or_lose_binary[0]
input = [toes[0], wlrec[0], nfans[0]]
pred = neural_network(input, weights)
error = (pred - true) ** 2
delta = pred - true


def ele_mul (number, vector):
    output = [0, 0, 0]
    assert (len(output) == len(vector))
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output


weight_deltas = ele_mul(delta, input)
alpha = 0.01
for i in range(len(weights)):
    weights[i] -= alpha * weight_deltas[i]
print(f"Weights: {weights}")
print(f"Weights deltas: {weight_deltas}")
















