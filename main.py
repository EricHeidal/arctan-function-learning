import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sp

# For now, hardcode B1 and B2 as 750 and 1.7, respectively
B2_INITIAL = 1.7
B1_INITIAL = 875


def train(points, A, B1, B2, O, learn_B=False):
    # Input: Some points (steps, actual_distance) and initial values for A, B, and Flow Start
    # The learning rate can be tweaked as a hyperparameter. There's also a method for finding the "best" learning rate
    # during each iteration of the while loop, but let's start with the simple version for now.
    learning_rate = .1
    # A /= 2
    # B /= 2
    previous_error = error(points, A, B1, B2, O)
    current_error = float("inf")
    iterations = 0
    start = True
    # This can also be tweaked; how much error should we allow for the final set of parameters? Requires experimentation.
    while start or current_error > 1 and iterations < 5000:
        start = False
        iterations += 1
        gradient_A = 0
        # gradient_B = 0
        gradient_B1 = 0
        gradient_B2 = 0
        gradient_O = 0
        for s, d in points:
            if learn_B:
                # New method: learn two parameters for B
                b = B1 - B2 * A
                inner_term = (s - O) / (B1 - B2 * A)
                target_term = A * (np.arctan(inner_term) + 1) - d
                denom = inner_term ** 2 + 1
                gradient_B1 += A * (s - O) * target_term / (b ** 2 * denom)
                # No need to learn the other B, trying to change both at once leads to scariness
                # gradient_B2 += A ** 2 * (s - O) * target_term / (b ** 2 * denom)
                # Old method: learn B as a single param
                # gradient_B += (A * (O - s) * (A * np.arctan((s - O) / B) + A - d)) / (B**2 + (O - s)**2 * np.abs(A * np.arctan((s - O) / B) + A - d))
                # gradient_B += (2 * A * (O - s) * (A * np.arctan((s - O) / B) + A - d)) / (B ** 2 + (O - s) ** 2)
            else:
                # gradient_A += ((np.arctan((s - O) / B) + 1) * (A * np.arctan((s - O) / B) + A - d)) / np.abs(A * np.arctan((s - O) / B) + A - d)
                # gradient_A += 2 * (np.arctan((s-O) / B) + 1) * (A * np.arctan((s-O) / B) + A - d)
                gradient_A += ((np.arctan((s - O) / (B1_INITIAL - B2_INITIAL * A)) + 1) * (A * np.arctan((s - O) / (B1_INITIAL - B2_INITIAL * A)) + A - d))
                # gradient_O += (A * B * (A * np.arctan((O - s) / B) - A + d)) / ((B**2 + (O - s)**2) * np.abs(A * np.arctan((s - O) / B) + A - d))
                # gradient_O += (2 * A * B * (A * np.arctan((O - s) / B) - A + d)) / (B**2 + (O - s)**2)
                gradient_O += (A * (B1_INITIAL - B2_INITIAL * A) * (A * np.arctan((O - s) / (B1_INITIAL - B2_INITIAL * A)) - A + d)) / ((B1_INITIAL - B2_INITIAL * A) ** 2 + (O - s) ** 2)

            # if learn_B:

            # else:
            #     gradient_A += ((A * B1 * (s - O) / (b**2 * denom)) + np.arctan(inner_term) + 1) * target_term
            #     gradient_O += A * target_term / (b * denom)

        temp_A = A - gradient_A * learning_rate
        # temp_B = B - gradient_B * learning_rate
        print('gradient for B1', gradient_B1)
        temp_B1 = B1 - gradient_B1 * learning_rate
        temp_B2 = B2 - gradient_B2 * learning_rate * 0.01
        temp_O = O - gradient_O * learning_rate
        temp_error = error(points, temp_A, temp_B1, temp_B2, temp_O)
        print(temp_error, previous_error, learning_rate)
        if temp_error < previous_error:
            previous_error = current_error
            current_error = temp_error
            A = temp_A
            # B = temp_B
            B1 = temp_B1
            B2 = temp_B2
            O = temp_O
            learning_rate += 0.1
        # If the learning rate is very small we've probably converged at the smallest error rate we're likely to have
        elif learning_rate < 0.001:
            break
        else:
            learning_rate = learning_rate / 2

    # Final parameters
    print("A: ", A)
    # print("B: ", B)
    print("B1", B1)
    print("B2", B2)
    print("O: ", O)
    # Graphing happens here
    # plot_curve(A, B, O, points)
    # plt.show()
    # return A, B, O
    return A, B1, B2, O


def plot_curve(A, B1, B2, O, points, color=None):
    X = np.arange(start=0, stop=3000)
    y = []
    for x in X:
        y.append(target(A, B1, B2, O, x))

    if color is None:
        plt.plot(X, y)
    else:
        plt.plot(X, y, color)
    points = np.array(points)
    plt.scatter(points[:, 0], y=points[:, 1], c='red')


# Function we're trying to find the parameters for
# def target(A, B, O, s):
#     return A * (np.arctan((s - O) / B) + 1)

# def target(A, B, O, s):
#     return A * (np.arctan((s - O) / (750 - B2_INITIAL * A)) + 1)

def target(A, B1, B2, O, s):
    return A * (np.arctan((s - O) / (B1 - B2 * A)) + 1)


# Using mean-squared error loss function
def error(points, A, B1, B2, O):
    result = 0
    for s, d in points:
        result += (target(A, B1, B2, O, s) - d)**2
    return result / len(points)


# def error_sp(params):
#     training_points = [[1254, 113.7], [1291, 129.7], [1341, 157.5], [1381, 177.3], [1428, 204.8], [1469, 226.4], [1516, 257.9], [1552, 280.0], [1596, 306.1]]
#     return error(training_points, params[0], params[1], params[2])

def main():
    # training_points = [[1262, 115.9], [1292, 128.9], [1340, 156.0], [1380, 176.7], [1424, 201.3], [1467, 228.0], [1507, 249.5], [1553, 278.0], [1595, 306.6]]
    # A, B, O = main(training_points, 1000, 100, 100)

    # PSI 64
    # training_points = [[1254, 113.7], [1291, 129.7], [1341, 157.5], [1381, 177.3], [1428, 204.8], [1469, 226.4], [1516, 257.9], [1552, 280.0], [1596, 306.1]]
    # Other PSIs
    # training_points = [[1253, 113.2], [1287, 130.7], [1330, 151], [1377, 175.3], [1422, 203.7], [1460, 224.9], [1503, 247.6], [1551, 279.7], [1588, 306.1]]
    # training_points = [[1262, 115.9], [1292, 128.9], [1340, 156.0], [1380, 176.7], [1424, 201.3], [1467, 228.0],
    #                    [1507, 249.5], [1553, 278.0], [1595, 306.6]]
    training_points = [[1256, 116.6], [1287, 128.5], [1339, 154.6], [1385, 178.1], [1431, 204.8], [1472, 228.8], [1515, 253.9], [1559, 284.0], [1594, 304.5]]
    A, B1, B2, O = train(training_points, 1000, B1_INITIAL, B2_INITIAL, 2000)
    A_1, B1_1, B2_1, O_1 = train(training_points, 10, B1_INITIAL, B2_INITIAL, 2000)
    A_2, B1_2, B2_2, O_2 = train(training_points, 1, B1_INITIAL, B2_INITIAL, 2000)
    plot_curve(A, B1, B2, O, training_points, 'r')
    plot_curve(A_1, B1_1, B2_1, O_1, training_points, 'g')
    plot_curve(A_2, B1_2, B2_2, O_2, training_points, 'b')
    plot_curve(396 / 2, B1_INITIAL, B2_INITIAL, 1411, training_points, 'orange')
    # result = sp.least_squares(error_sp, [100, 100, 1411])
    # print(result)
    # plot_curve(result.x[0], result.x[1], result.x[2], training_points, color='r')
    plt.show()
    # Now that we have better estimates for A and O, relearn B using two params
    A, B1, B2, O = train(training_points, A, B1_INITIAL, B2_INITIAL, O, learn_B=True)
    A_1, B1_1, B2_1, O_1 = train(training_points, A_1, B1_INITIAL, B2_INITIAL, O_1, learn_B=True)
    A_2, B1_2, B2_2, O_2 = train(training_points, A_2, B1_INITIAL, B2_INITIAL, O_2, learn_B=True)
    plot_curve(A, B1, B2, O, training_points, 'r')
    plot_curve(A_1, B1_1, B2_1, O_1, training_points, 'g')
    plot_curve(A_2, B1_2, B2_2, O_2, training_points, 'b')
    plot_curve(396 / 2, B1_INITIAL, B2_INITIAL, 1411, training_points, 'orange')
    plt.show()


if __name__ == '__main__':
    main()

