import numpy as np
import matplotlib.pyplot as plt


# Function to be learned: A * arctan((s-O)/B) + A = d
# Begin by finding the inflection point by using linear regression on consecutive triples of training points. This
# will return the slope of the line at the middle point in each triple.
# The inflection point of the graph is located at (O, A). Once you have this point, it acts as a "pivot," staying
# constant while B is altered to change the shape of the graph around it.
def target(A, B, O, s):
    return A * np.arctan((s - O) / B) + A


# Using mean-squared error loss function
def error(A, B, O, points):
    result = 0
    for s, d in points:
        result += (target(A, B, O, s) - d)**2
    return result / len(points)


# Returns parameters w1, w2 such that every y in regression_poinst is approx. equal to w1 * x + w2
def linear_regression(regression_points):
    sum_x = 0
    sum_x2 = 0
    sum_y = 0
    sum_xy = 0
    n = len(regression_points)
    for x, y in regression_points:
        sum_x += x
        sum_x2 += x**2
        sum_y += y
        sum_xy += x * y

    w1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    w2 = (sum_y - w1 * sum_x) / n
    return w1, w2


def find_inflection_point(training_points):
    training_slopes = []
    for i in range(len(training_points) - 2):
        w1, w2 = linear_regression([training_points[i], training_points[i+1], training_points[i+2]])
        training_slopes.append([training_points[i+1][0], w1])

    max_slope = 0
    max_index = 0
    for i in range(len(training_slopes)):
        if training_slopes[i][1] > max_slope:
            max_slope = training_slopes[i][1]
            max_index = i + 1  # Plus 1 because this will be the index of our original list of training points

    O = training_points[max_index][0]
    A = training_points[max_index][1]
    return O, A, training_slopes


def train(training_points):
    learning_rate = 0.05
    O, A, training_slopes = find_inflection_point(training_points)
    B = 1
    previous_error = error(A, B, O, training_points)
    print('Starting error: ', previous_error)
    current_error = float("inf")
    iterations = 0
    start = True
    # This can also be tweaked; how much error should we allow for the final set of parameters? Requires experimentation
    while start or current_error > 1 and iterations < 5000:
        start = False
        iterations += 1
        gradient_B = 0

        for s, d in training_points:
            gradient_B += (2 * A * (O - s) * (A * np.arctan((s - O) / B) + A - d)) / (B ** 2 + (O - s) ** 2)

        temp_B = B - gradient_B * learning_rate
        temp_error = error(A, temp_B, O, training_points)
        print(temp_error, previous_error, learning_rate)
        if temp_error < previous_error:
            previous_error = current_error
            current_error = temp_error
            B = temp_B
            learning_rate += 0.1
        # If the learning rate is very small we've probably converged at the smallest error rate we're likely to have
        elif learning_rate < 0.001:
            break
        else:
            learning_rate = learning_rate * 0.9

    # Final parameters
    print("A: ", A)
    print("B: ", B)
    print("O: ", O)
    return A, B, O


def plot_curve(A, B, O, color=None):
    X = np.arange(start=0, stop=3000)
    y = []
    for x in X:
        y.append(A * np.arctan((x - O) / B) + A)
    if color is None:
        plt.plot(X, y)
    else:
        plt.plot(X, y, color)


def main():
    # training_points = [[1254, 113.7], [1291, 129.7], [1341, 157.5], [1381, 177.3], [1428, 204.8], [1469, 226.4], [1516, 257.9], [1552, 280.0], [1596, 306.1]]
    # training_points = [[1262, 115.9], [1292, 128.9], [1340, 156.0], [1380, 176.7], [1424, 201.3], [1467, 228.0], [1507, 249.5], [1553, 278.0], [1595, 306.6]]
    # training_points = [[1253, 113.2], [1287, 130.7], [1330, 151], [1377, 175.3], [1422, 203.7], [1460, 224.9], [1503, 247.6], [1551, 279.7], [1588, 306.1]]
    # training_points = [[1262, 115.9], [1292, 128.9], [1340, 156.0], [1380, 176.7], [1424, 201.3], [1467, 228.0], [1507, 249.5], [1553, 278.0], [1595, 306.6]]
    # training_points = [[1256, 116.6], [1287, 128.5], [1339, 154.6], [1385, 178.1], [1431, 204.8], [1472, 228.8], [1515, 253.9], [1559, 284.0], [1594, 304.5]]
    # Failed on these:
    # training_points = [[1229, 96.3], [1282, 116.8], [1328, 137.3], [1370, 157.2], [1409, 179.0], [1446, 198.3], [1481, 219.8], [1515, 242.8], [1548, 261.1], [1583, 278.7]]
    training_points = [[1227, 93.1], [1280, 111.6], [1327, 132.9], [1370, 157.6], [1409, 179.4], [1446, 198.1], [1482, 214.7], [1517, 235.0], [1551, 259.3]]
    A, B, O = train(training_points)
    plot_curve(A, B, O, color='g')
    plt.scatter(np.array(training_points)[:, 0], y=np.array(training_points)[:, 1], c='red')
    plt.show()


if __name__ == '__main__':
    main()