import numpy as np
import math
import scipy.linalg


#1.Sa se calculeze vectorul b∈R^n:
def calculate_b(A, s, epsilon=0.0001):
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        print("The array has to be squared")
        return

    if s.shape[0] != n:
        print("The vector has to be of the same dimension as the matrix")
        return

    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b[i] += s[j] * A[i][j]
    return b


A = np.array([[0., 0, 4], [1, 2, 3], [0, 1, 2]])
s = np.array([3, 2, 1])
print(f"\033[93mMatrix A:\n\033[00m{A}")
print(f"\033[93mVector s:\n\033[00m{s}")
b = calculate_b(A, s)
print(f"\033[93mVector b:\n\033[00m{b}")


#2.Sa se implementeze descompunerea QR a matricei A folosind algoritmul lui Householder
def calculate_QR(A, epsilon=0.0001):
    A = A.astype(float)
    n = A.shape[0]
    Q = np.identity(n)
    u = np.zeros(n)
    for r in range(n - 1):
        alpha = sum([A[i][r] ** 2 for i in range(r, n)])
        if alpha <= epsilon:
            break
        k = math.sqrt(alpha)  #norm of the vector
        if A[r][r] >= 0:
            k = -k
        beta = alpha - k * A[r][r]
        #u construction
        u[r] = A[r][r] - k
        for i in range(r + 1, n):
            u[i] = A[i][r]
        for j in range(r + 1, n):
            γ = sum([u[i] * A[i][j] for i in range(r, n)]) / beta
            for i in range(r, n):
                A[i][j] = A[i][j] - γ * u[i]
        # Transform column r of A
        A[r][r] = k
        for i in range(r + 1, n):
            A[i][r] = 0

        for j in range(n):
            γ = sum([u[i] * Q[i][j] for i in range(r, n)]) / beta
            for i in range(r, n):
                Q[i][j] = Q[i][j] - γ * u[i]
    return Q.T, A


Q, R = calculate_QR(np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]]))
print(f"\033[93mMatrix Q:\n\033[00m{Q}")
print(f"\033[93mMatrix R:\n\033[00m{R}")

Q, R = scipy.linalg.qr(np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]]))
print(f"\033[93mScipy Matrix Q:\n\033[00m{Q}")
print(f"\033[93mScipy Matrix R:\n\033[00m{R}")


#3.Sa se rezolve sistemul liniar: Ax = b
def solve_system_scipy_QR(A, b):
    Q, R = scipy.linalg.qr(A)
    x_QR = scipy.linalg.solve_triangular(R, Q.T @ b)
    return x_QR

def inverse_substitution_method(A, b):
    if A.shape[0] != A.shape[1]:
        print("The array has to be squared")
        return
    n = A.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]
    return x

def approximate_matrix_inverse_householder(A, epsilon=0.0001):
    A = A.astype(float)
    R = A.copy()
    Q, R = calculate_QR(R, epsilon)
    n = R.shape[0]
    if np.linalg.det(R) == 0:
        print("The matrix is singular")
        return
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1
        b = Q.T @ e_j
        x = inverse_substitution_method(R, b)
        for i in range(n):
            A[i][j] = x[i]
    return A

def solve_system_householder_QR(A, b, epsilon=0.0001):
    Q, R = calculate_QR(A, epsilon)
    x = inverse_substitution_method(R, Q.T @ b)
    return x

A_init = np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]])
s = np.array([3, 2, 1])
b_init = calculate_b(A_init, s)

x_Householder = solve_system_householder_QR(
    np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]]), b_init)
print(b_init)
x_QR = solve_system_scipy_QR(
    np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]]), b_init)

print(f"\033[93mHouseholder QR solution:\n\033[00m", x_Householder)
print(f"\033[93mScipy QR solution:\n\033[00m", x_QR)
print(f"\033[93mNorm of the difference between the two solutions: \033[00m{np.linalg.norm(x_QR - x_Householder)}")

#4.Sa se calculeze si sa se afiseze urmatoarele erori
s = np.array([3, 2, 1])
A = A_init.copy()
b = b_init.copy()
print("\033[93mnorm(A_init @ x_Householder - b_init) =\033[00m",
      np.linalg.norm(A_init @ x_Householder - b_init, ord=2))
A = A_init.copy()
b = b_init.copy()
print("\033[93mnorm(A_init @ x_QR - b_init) =\033[00m",
      np.linalg.norm(A_init @ x_QR - b_init, ord=2))
print("\033[93mnorm(x_Householder - s)/ norm(s) =\033[00m",
      np.linalg.norm(x_Householder - s, ord=2) / np.linalg.norm(s, ord=2))
print("\033[93mnorm(x_QR - s)/ norm(s) =\033[00m",
      np.linalg.norm(x_QR - s, ord=2) / np.linalg.norm(s, ord=2))

#5.Sa se calculeze inversa matricei A folosind descompunerea QR
x = approximate_matrix_inverse_householder(
    np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]]))

print(f"\033[93mNumpy inverse of the matrix:\n\033[00m{np.linalg.inv(np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]]))}")
print(f"\033[93mApproximate inverse of the matrix:\n\033[00m{x}")
print(f"\033[93mNorm of the difference between the two inverses: \033[00m{np.linalg.norm(np.linalg.inv(np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]])) - x, ord=2)}")

def generate_random_square_matrix_and_vector(n):
    A = np.random.randint(1, 10, (n, n))
    s = np.random.randint(1, 10, n)
    return A, s

#Bonus
def multiplication_RQ(R, Q):
    n = R.shape[0]
    result = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            result[i][j] = R[i, i] * Q[i, j]
            for k in range(i + 1, n):
                result[i][j] += R[i][k] * Q[k][j]

    return result


def equal_matrices(A, B, epsilon=1e-8):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - B[i][j]) > epsilon:
                return False
    return True


def approximate_limit(A_init, epsilon=1e-8, k_max=10000):
    A = A_init.copy()
    k = 0
    while True:
        Q, R = calculate_QR(A)
        A_next = multiplication_RQ(R, Q)
        # if equal_matrices(A, A_next, epsilon):
        #     A = A_next
        #     break
        if np.allclose(A, A_next, atol=epsilon):
            A = A_next
            break
        A = A_next
        k += 1
        # print(k)
        if k > k_max:
            break

    return A, k


def generate_random_symmetric_matrix(n):
    matrix = np.random.rand(n, n)
    return (matrix + matrix.T) / 2


matrix = generate_random_symmetric_matrix(5)
print("Random symmetric matrix: \n", matrix)
matrix, k = approximate_limit(matrix)

print(f"\nSame matrix for {k} iterations")
print(matrix)


# import time
# import matplotlib.pyplot as plt
#
#
# def solve_scipy_lu(A, b):
#     lu, piv = scipy.linalg.lu_factor(A)
#     return scipy.linalg.lu_solve((lu, piv), b)
#
#
# def plot_solve_comparison(start, finish):
#     solve_diff_householder_scipyqr = np.array([])
#     solve_diff_householder_solve = np.array([])
#     solve_diff_householder_scipylu = np.array([])
#     solve_diff_householder_npsolve = np.array([])
#
#     time_householder = np.array([])
#     time_scipyqr = np.array([])
#     time_solve = np.array([])
#     time_scipylu = np.array([])
#     time_npsolve = np.array([])
#
#     fig, ax = plt.subplots(6, 1, figsize=(15, 10))
#     for i in range(start, finish):
#         A_init, s = generate_random_square_matrix_and_vector(i)
#         b = calculate_b(A_init, s)
#         A = A_init.copy()
#
#         t_start = time.time()
#         x_Householder = solve_system_householder_QR(A, b)
#         t_end = time.time()
#         time_householder = np.append(time_householder, t_end - t_start)
#
#         t_start = time.time()
#         solve_diff_householder_scipyqr = np.append(
#             solve_diff_householder_scipyqr, np.linalg.norm(x_Householder - solve_system_scipy_QR(A_init, s)))
#         t_end = time.time()
#         time_scipyqr = np.append(time_scipyqr, t_end - t_start)
#
#         t_start = time.time()
#         solve_diff_householder_solve = np.append(
#             solve_diff_householder_solve, np.linalg.norm(
#                 x_Householder - scipy.linalg.solve(A_init, s))
#         )
#         t_end = time.time()
#         time_solve = np.append(time_solve, t_end - t_start)
#
#         t_start = time.time()
#         solve_diff_householder_scipylu = np.append(
#             solve_diff_householder_scipylu, np.linalg.norm(x_Householder - solve_scipy_lu(A_init, s)))
#         t_end = time.time()
#         time_scipylu = np.append(time_scipylu, t_end - t_start)
#
#         t_start = time.time()
#         solve_diff_householder_npsolve = np.append(
#             solve_diff_householder_npsolve, np.linalg.norm(
#                 x_Householder - np.linalg.solve(A_init, s)))
#         t_end = time.time()
#         time_npsolve = np.append(time_npsolve, t_end - t_start)
#
#     ax[0].plot(np.arange(start, finish),
#                solve_diff_householder_scipyqr, label="scipy QR")
#     ax[0].plot(np.arange(start, finish),
#                solve_diff_householder_solve, label="scipy solve")
#     ax[0].plot(np.arange(start, finish),
#                solve_diff_householder_scipylu, label="scipy LU")
#     ax[0].plot(np.arange(start, finish),
#                solve_diff_householder_npsolve, label="np solve")
#     ax[0].set_title("Solve comparison")
#     ax[0].set_xlabel("Dimension")
#     ax[0].set_ylabel("Norm")
#     ax[0].legend()
#
#     ax[1].plot(np.arange(start, finish),
#                solve_diff_householder_scipyqr, label="scipy QR")
#     ax[1].set_title("Solve comparison scipy QR")
#     ax[1].set_xlabel("Dimension")
#     ax[1].set_ylabel("Norm")
#
#     ax[2].plot(np.arange(start, finish),
#                solve_diff_householder_solve, label="scipy solve")
#     ax[2].set_title("Solve comparison scipy solve")
#     ax[2].set_xlabel("Dimension")
#     ax[2].set_ylabel("Norm")
#
#     ax[3].plot(np.arange(start, finish),
#                solve_diff_householder_scipylu, label="scipy LU")
#     ax[3].set_title("Solve comparison scipy LU")
#     ax[3].set_xlabel("Dimension")
#     ax[3].set_ylabel("Norm")
#
#     ax[4].plot(np.arange(start, finish),
#                solve_diff_householder_npsolve, label="numpy solve")
#     ax[4].set_title("Solve comparison numpy solve")
#     ax[4].set_xlabel("Dimension")
#     ax[4].set_ylabel("Norm")
#
#     ax[5].plot(np.arange(start, finish),
#                time_householder, label="Householder")
#     ax[5].plot(np.arange(start, finish),
#                time_scipyqr, label="scipy QR")
#     ax[5].plot(np.arange(start, finish),
#                time_scipylu, label="scipy LU")
#     ax[5].plot(np.arange(start, finish),
#                time_npsolve, label="numpy solve")
#     ax[5].set_title("Time comparison")
#     ax[5].set_xlabel("Dimension")
#     ax[5].set_ylabel("Time")
#     ax[5].legend()
#
#     plt.show()
#
#
# # All plots compare the householder implemented in the homework with the algorithm in the plot legend/title
# plot_solve_comparison(1, 100)
#
#
# def plot_inv_comparison(start, finish):
#     fig, ax = plt.subplots(2, 1, figsize=(15, 10))
#     inv_difference = np.array([])
#     inv_difference_non_singular = np.array([])
#     colors_non_singular = np.array([])
#     for i in range(start, finish):
#         A_init, s = generate_random_square_matrix_and_vector(i)
#         inv_householder = approximate_matrix_inverse_householder(A_init)
#         inv_difference = np.append(inv_difference, np.linalg.norm(
#             np.linalg.pinv(A_init) - inv_householder))
#         if np.linalg.det(A_init) != 0:
#             inv_difference_non_singular = np.append(inv_difference_non_singular, np.linalg.norm(
#                 np.linalg.inv(A_init) - inv_householder))
#             colors_non_singular = np.append(colors_non_singular, 'r')
#         else:
#             inv_difference_non_singular = np.append(
#                 inv_difference_non_singular, inv_difference[-1])
#             colors_non_singular = np.append(colors_non_singular, 'b')
#
#     ax[0].plot(np.arange(start, finish), inv_difference)
#     ax[0].set_title("Difference between scipy inverse and householder inverse")
#     ax[0].set_xlabel("Matrix dimension")
#     ax[0].set_ylabel("Norm of the difference")
#
#     if np.all(colors_non_singular == 'r'):
#         ax[1].plot(np.arange(start, finish), inv_difference_non_singular)
#     else:
#         ax[1].scatter(np.arange(start, finish),
#                       inv_difference_non_singular, c=colors_non_singular)
#     ax[1].set_title(
#         "Difference between scipy inverse and householder inverse for non singular matrices")
#     ax[1].set_xlabel("Matrix dimension")
#     ax[1].set_ylabel("Norm of the difference")
#
#     plt.show()
#
#
# plot_inv_comparison(1, 100)
