# ------------------------ Создание нейронной сети ------------------------ #
# import numpy as np
# weights = np.array([0.5,0.48,-0.7])
# alpha = 0.1
# streetlights = np.array([[1, 0, 1],
#                          [0, 1, 1],
#                          [0, 0, 1],
#                          [1, 1, 1],
#                          [0, 1, 1],
#                          [1, 0, 1]])
# walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])
# input = streetlights[0]
# goal_pred = walk_vs_stop[0]
# for i in range(20):
#     pred = input.dot(weights)
#     error = (pred - goal_pred) ** 2
#     delta = pred - goal_pred
#     weights -= delta * input * alpha
#     print("Error:" + str(error) + " Prediction:" + str(pred))

# ------------------------ Обучение на полном наборе данных ------------------------ #
import numpy as np
weights = np.array([0.5,0.48,-0.7])
alpha = 0.1
streetlights = np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 1, 1],
                         [1, 0, 0],
                         [1, 0, 1],
                         [1, 1, 0]])
walk_vs_stop = np.array([0, 1, 1, 0, 0, 1])
for i in range(31):
    error_for_all_lights = 0
    for row_index in range(len(walk_vs_stop)):
        input = streetlights[row_index]
        goal_pred = walk_vs_stop[row_index]
        pred = input.dot(weights)
        error = (goal_pred - pred) ** 2
        error_for_all_lights += error
        delta = pred - goal_pred
        weights -= alpha * input * delta
        print(f"Prediction: {pred:.2f}")
    print(f"Error: {error_for_all_lights:.5f}\n")









