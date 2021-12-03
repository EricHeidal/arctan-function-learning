import csv
import numpy as np
import matplotlib.pyplot as plt

# Indexes for matrix:
PRESSURE = 1
STEPS = 2
DESIRED = 3
ACTUAL = 4
A_COEF = 5
B_COEF = 6
C_COEF = 7
DIFFERENCES = 8

def main():
    with open('data') as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        matrix = []
        for row in reader:
            matrix.append(row[0:8])  # Strip the final column of differences in A
        matrix = np.array(matrix).T
        matrix = np.vstack((matrix, matrix[ACTUAL] - matrix[DESIRED]))
        print(matrix.shape)
        print('Starting A value: ', matrix[A_COEF, 0])
        print('Standard deviation for desired v. actual: ', np.std(matrix[DIFFERENCES]))
        currentA = matrix[A_COEF, 0]
        # Find all the places where A changes
        change_A_mask = np.zeros_like(matrix[A_COEF])
        for i, val in enumerate(matrix[A_COEF]):
            if val != currentA:
                change_A_mask[i] = 1
                currentA = val
        change_A_mask = change_A_mask > 0
        plt.plot(matrix[DIFFERENCES])
        plt.scatter(np.arange(len(change_A_mask))[change_A_mask], matrix[DIFFERENCES][change_A_mask])
        for i, val in enumerate(matrix[A_COEF]):
            print(matrix[DIFFERENCES, i])
            if change_A_mask[i]:
                plt.annotate(val, (i, matrix[DIFFERENCES, i]))
        indices_of_change = np.where(change_A_mask)
        print(indices_of_change)

        plt.show()


if __name__ == "__main__":
    main()