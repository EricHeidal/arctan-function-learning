import numpy as np
import matplotlib.pyplot as plt

# Function to be learned: A * arctan((s-O)/B) + Z = d
# Begin by finding the inflection point by using linear regression on consecutive triples of training points. This
# will return the slope of the line at the middle point in each triple.
# The inflection point of the graph is located at (O, Z). Once you have this point, it acts as a "pivot," staying
# constant while A and B are altered to change the shape of the graph around it.
# The loss function will find the best values A and B such that the difference in slopes between the learned function
# and the slopes returned by the linear regressions is minimized


# Slope of the arctan function is equal to it's first derivative with respect to s:
def target(A, B, O, Z, steps):
    return A * B / (B**2 + (steps - O)**2)


# Using Mean-Squared Error for loss
def loss(A, B, O, Z, training_slopes):
    result = 0
    # training_slopes should have a length 2 shorter than training_points
    for i in range(len(training_slopes)):
        steps = training_slopes[i][0]
        slope = training_slopes[i][1]
        print('Target: ', target(A, B, O, Z, steps), 'Ground: ', slope)
        result += (target(A, B, O, Z, steps) - slope) ** 2
    return result / len(training_slopes)


# Returns parameters w1, w2 such that every y in training points is approx. w1 * x + w2
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
    Z = training_points[max_index][1]
    return O, Z, training_slopes


def train(training_points):
    learning_rate = .1
    O, Z, training_slopes = find_inflection_point(training_points)
    A = Z
    B = 1
    previous_error = loss(A, B, O, Z, training_slopes)
    print('Starting error: ', previous_error)
    current_error = float("inf")
    iterations = 0
    start = True
    # This can also be tweaked; how much error should we allow for the final set of parameters? Requires experimentation
    while start or current_error > 0.001 and iterations < 5000:
        start = False
        iterations += 1
        gradient_A = 0
        gradient_B = 0

        for steps, slope in training_slopes:
            gradient_B += 2 * (A / (B**2 + (steps - O)**2) - (2 * A * B**2 / (B**2 + (steps - O)**2)**2)) * (A * B / (B**2 + (steps - O)**2) - slope)
            # gradient_A += -2 * B * (slope * (B**2 + (steps - O)**2) - A * B) / (B**2 + (steps - O)**2)**2

        temp_A = A - gradient_A * learning_rate
        temp_B = B - gradient_B * learning_rate
        temp_error = loss(temp_A, temp_B, O, Z, training_slopes)
        print(temp_error, previous_error, learning_rate)
        if temp_error < previous_error:
            previous_error = current_error
            current_error = temp_error
            A = temp_A
            B = temp_B
        #     learning_rate += 0.1
        # # If the learning rate is very small we've probably converged at the smallest error rate we're likely to have
        # elif learning_rate < 0.001:
        #     break
        # else:
        #     learning_rate = learning_rate / 2

    # Final parameters
    print("A: ", A)
    print("B: ", B)
    print("O: ", O)
    print("Z: ", Z)
    return A, B, O, Z


def plot_curve(A, B, O, Z, color=None):
    X = np.arange(start=0, stop=3000)
    y = []
    for x in X:
        y.append(A * np.arctan((x - O) / B) + Z)
    if color is None:
        plt.plot(X, y)
    else:
        plt.plot(X, y, color)


def main():
    training_points = [[1254, 113.7], [1291, 129.7], [1341, 157.5], [1381, 177.3], [1428, 204.8], [1469, 226.4],
                       [1516, 257.9], [1552, 280.0], [1596, 306.1]]
    A, B, O, Z = train(training_points)
    plot_curve(A, B, O, Z, color='g')
    plt.scatter(np.array(training_points)[:, 0], y=np.array(training_points)[:, 1], c='red')
    plt.show()


if __name__ == '__main__':
    main()
