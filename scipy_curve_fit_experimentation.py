import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def train(points, A, B, O, learn_B=False):
    # Input: Some points (steps, actual_distance) and initial values for A, B, and Flow Start
    # The learning rate can be tweaked as a hyperparameter. There's also a method for finding the "best" learning rate
    # during each iteration of the while loop, but let's start with the simple version for now.
    learning_rate = .1
    # A /= 2
    # B /= 2
    previous_error = error(points, A, B, O)
    current_error = float("inf")
    iterations = 0
    start = True
    # This can also be tweaked; how much error should we allow for the final set of parameters? Requires experimentation
    while start or current_error > 1 and iterations < 5000:
        start = False
        iterations += 1
        gradient_A = 0
        gradient_B = 0
        gradient_O = 0
        for s, d in points:
            gradient_A += 2 * (-1 - (-O + s) / np.sqrt(B**2 + (-O + s)**2)) * (-A * (1 + (-O + s)/np.sqrt(B**2 + (-O + s)**2)) + d)
        temp_A = A - gradient_A * learning_rate
        temp_B = B - gradient_B * learning_rate
        temp_O = O - gradient_O * learning_rate
        temp_error = error(points, temp_A, temp_B, temp_O)
        print(temp_error, previous_error, learning_rate)
        if temp_error < previous_error:
            previous_error = current_error
            current_error = temp_error
            A = temp_A
            B = temp_B
            O = temp_O
            learning_rate += 0.1
        # If the learning rate is very small we've probably converged at the smallest error rate we're likely to have
        elif learning_rate < 0.001:
            break
        else:
            learning_rate = learning_rate / 2

    # Final parameters
    print("A: ", A)
    print("B: ", B)
    print("O: ", O)
    # Graphing happens here
    plot_curve(A, B, O, points)
    plt.show()
    return A, B, O


def plot_curve(A, B, O, points, color=None):
    X = np.arange(start=0, stop=3000)
    y = []
    for x in X:
        y.append(target(A, B, O, x))

    if color is None:
        plt.plot(X, y)
    else:
        plt.plot(X, y, color)
    points = np.array(points)
    plt.scatter(points[:, 0], y=points[:, 1], c='red')


# Function we're trying to find the parameters for
def target(A, B, O, s):
    return A * (s - O) / (np.sqrt(B**2 + (s - O)**2)) + A


# Using mean-squared error loss function
def error(points, A, B, O):
    result = 0
    for s, d in points:
        result += (target(A, B, O, s) - d)**2
    return result / len(points)

def gondola(x, A, B, O):
    numerator = x - O
    denominator = np.sqrt(B ** 2 + ((x - O) ** 2))
    return A * numerator / denominator + A

def main():
    training_points = [[1256, 116.6], [1287, 128.5], [1339, 154.6], [1385, 178.1], [1431, 204.8], [1472, 228.8], [1515, 253.9], [1559, 284.0], [1594, 304.5]]
    popt, other = curve_fit(gondola, [item[0] for item in training_points], [item[1] for item in training_points], p0=[185, 400, 1404])
    print(popt)
    A, B, O = popt
    print('A', A)
    print('B', B)
    print('O', O)
    train(training_points, 200, 400, 1500)



main()

# gradient for A : 2 (1 + (-o + x)/sqrt(b + (-o + x)^2)) (a + (a (-o + x))/sqrt(b + (-o + x)^2) - y)
# gradient for B : -(a (-o + x) (a + (a (-o + x))/sqrt(b + (-o + x)^2) - y))/(b + (-o + x)^2)^(3/2)
# gradient for O : (2 a b (-a (-o + sqrt(b + (o - x)^2) + x) + sqrt(b + (o - x)^2) y))/(b + (o - x)^2)^2
