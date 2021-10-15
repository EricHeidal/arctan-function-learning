import matplotlib.pyplot as plt

# This function starts by finding B, then learns A and C through linear regression
# To find B, we need to know where the inflection point is in our training points (place where second derivative equals
# 0, or just the point of maximum slope; the "middle" of the arctan curve). We can approximate this by performing
# piecewise linear regressions for each consecutive triple of training points, and finding where the slopes of those
# regressions stop increasing and start decreasing
import numpy as np


def find_B(training_points):
    pairwise_slopes = []
    for i in range(len(training_points) - 1):
        # slope = (y2 - y1) / (x2 - x1)
        slope = (training_points[i + 1][1] - training_points[i][1]) / (training_points[i + 1][0] - training_points[i][0])
        pairwise_slopes.append(slope)
    print(pairwise_slopes)
    print("Average slope: ", np.average(pairwise_slopes))
    print("Standard Deviation:", np.sqrt(np.var(pairwise_slopes)))
    for i in range(len(pairwise_slopes) - 1):
        print(pairwise_slopes[i + 1] - pairwise_slopes[i])

    print("Performing linear regression for consecutive triples: ")
    consecutive_triple_slopes = []
    for i in range(len(training_points) - 2):
        sum_x = training_points[i][0] + training_points[i + 1][0] + training_points[i + 2][0]
        sum_x2 = training_points[i][0]**2 + training_points[i + 1][0]**2 + training_points[i + 2][0]**2
        sum_y = training_points[i][1] + training_points[i + 1][1] + training_points[i + 2][1]
        sum_xy = training_points[i][0] * training_points[i][1] + training_points[i + 1][0] * training_points[i + 1][1] + training_points[i + 2][0] * training_points[i + 2][1]
        slope = (3 * sum_xy - (sum_x * sum_y)) / (3 * sum_x2 - sum_x**2)
        consecutive_triple_slopes.append(slope)
    print(consecutive_triple_slopes)
    print("Average slope: ", np.average(consecutive_triple_slopes))
    print("Standard Deviation:", np.sqrt(np.var(consecutive_triple_slopes)))
    B = 1 / np.max(consecutive_triple_slopes)
    inflection_point = training_points[np.argmax(consecutive_triple_slopes) + 1][0]
    y_offset = training_points[np.argmax(consecutive_triple_slopes) + 1][1]
    O = inflection_point # * B
    Z = y_offset
    return B, O, Z


# Returns parameters w1, w2 such that every y in training points is approx. w1 * x + w2
def linear_regression(training_points):
    sum_x = 0
    sum_x2 = 0
    sum_y = 0
    sum_xy = 0
    n = len(training_points)
    for x, y in training_points:
        sum_x += x
        sum_x2 += x**2
        sum_y += y
        sum_xy += x * y

    w1 = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    w2 = (sum_y - w1 * sum_x) / n
    return w1, w2


def find_A(training_points, B, O):
    transformed_points = []
    for s, d in training_points:
        transformed_points.append([np.arctan((s - O) / B), d])
    A, Z = linear_regression(transformed_points)
    return A, Z


def plot_curve(A, B, O, Z, color=None):
    X = np.arange(start=0, stop=3000)
    y = []
    for x in X:
        y.append(target(A, B, O, Z, x))
    if color is None:
        plt.plot(X, y)
    else:
        plt.plot(X, y, color)


def target(A, B, O, Z, s):
    return A * np.arctan((s - O) / B) + Z


def main():
    training_points = [[1254, 113.7], [1291, 129.7], [1341, 157.5], [1381, 177.3], [1428, 204.8], [1469, 226.4], [1516, 257.9], [1552, 280.0], [1596, 306.1]]
    B, O, Z = find_B(training_points)
    A = 150
    # A, Z = find_A(training_points, B, O)
    print(A, B, O, Z)
    plot_curve(A, B, O, Z, color='g')
    plt.scatter(np.array(training_points)[:, 0], y=np.array(training_points)[:, 1], c='red')
    plt.show()


if __name__ == '__main__':
    main()
