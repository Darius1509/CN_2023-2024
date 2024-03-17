import numpy as np
import math
import scipy.linalg


# 1. Să se calculeze vectorul b∈R^n:
def calculate_b(A, s, ε=0.0001):
    # n - system dimension
    # ε - precision
    # A - matrix
    # s - vector

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


# 2. Să se implementeze descompunerea QR a matricei A folosind algoritmul lui Householder


def calculate_QR(A, ε=0.0001):
    A = A.astype(float)
    n = A.shape[0]
    Q = np.identity(n)
    u = np.zeros(n)
    for r in range(n - 1):
        # construct P matrix for the current r, constant β and vector u
        # sum list comprehension

        σ = sum([A[i][r] ** 2 for i in range(r, n)])
        if σ <= ε:
            break
        k = math.sqrt(σ)  # norm of the vector
        if A[r][r] >= 0:
            k = -k
        β = σ - k * A[r][r]
        # u construction
        u[r] = A[r][r] - k
        for i in range(r + 1, n):
            u[i] = A[i][r]

        # print(
        #     f"\033[93mσ:\033[00m{σ},\033[93m β:\033[00m{β},\033[93m k:\033[00m{k}")
        # print(f"\033[93mVector u:\n\033[00m{u}")

        # Transform columns j = r + 1, ...,n of A
        for j in range(r + 1, n):
            γ = sum([u[i] * A[i][j] for i in range(r, n)]) / β
            for i in range(r, n):
                A[i][j] = A[i][j] - γ * u[i]
        # Transform column r of A
        A[r][r] = k
        for i in range(r + 1, n):
            A[i][r] = 0

        for j in range(n):
            γ = sum([u[i] * Q[i][j] for i in range(r, n)]) / β
            for i in range(r, n):
                Q[i][j] = Q[i][j] - γ * u[i]
    return Q.T, A


Q, R = calculate_QR(np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]]))
print(f"\033[93mMatrix Q:\n\033[00m{Q}")
print(f"\033[93mMatrix R:\n\033[00m{R}")

Q, R = scipy.linalg.qr(np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]]))
print(f"\033[93mScipy Matrix Q:\n\033[00m{Q}")
print(f"\033[93mScipy Matrix R:\n\033[00m{R}")


# 3. Să se rezolve sistemul liniar: Ax = b
# folosind descompunerea QR din una din bibliotecile menționate în pagina
# laboratorului(se obţine soluţia x_QR) şi descompunerea QR calculată la punctul 2.
# (se obţine soluţia x_Householder). Calculaţi şi afişaţi:


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


def approximate_matrix_inverse_householder(A, ε=0.0001):
    A = A.astype(float)
    R = A.copy()
    Q, R = calculate_QR(R, ε)
    n = R.shape[0]
    if np.linalg.det(R) == 0:
        print("The matrix is singular")
        return
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1
        b = Q.T @ e_j
        x = inverse_substitution_method(R, b)
        # x = scipy.linalg.solve_triangular(R, b, lower=False)
        # save x in the j-th column of the matrix A
        for i in range(n):
            A[i][j] = x[i]

    return A


def solve_system_householder_QR(A, b, ε=0.0001):
    Q, R = calculate_QR(A, ε)
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
print(
    f"\033[93mNorm of the difference between the two solutions: \033[00m{np.linalg.norm(x_QR - x_Householder)}")

# 4. Să se calculeze şi să se afişeze următoarele erori


s = np.array([3, 2, 1])
A = A_init.copy()
b = b_init.copy()
print("\033[93mnorm(A_init @ x_Householder - b_init) =\033[00m",
      np.linalg.norm(A_init @ x_Householder - b_init, ord=2))
A = A_init.copy()
b = b_init.copy()
print("\033[93mnorm(A_init @ x_QR - b_init) =\033[00m",
      np.linalg.norm(A_init @ x_QR - b_init))
print("\033[93mnorm(x_Householder - s)/ norm(s) =\033[00m",
      np.linalg.norm(x_Householder - s) / np.linalg.norm(s))
print("\033[93mnorm(x_QR - s)/ norm(s) =\033[00m",
      np.linalg.norm(x_QR - s) / np.linalg.norm(s))

# Să se calculeze inversa matricei A folosind descompunerea QR calculată la
# punctul 2. şi să se compare cu inversa calculată folosind funcţia din bibliotecă,
# afişând valoarea următoarei norme:


x = approximate_matrix_inverse_householder(
    np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]]))

print(
    f"\033[93mNumpy inverse of the matrix:\n\033[00m{np.linalg.inv(np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]]))}")
print(f"\033[93mApproximate inverse of the matrix:\n\033[00m{x}")

print(
    f"\033[93mNorm of the difference between the two inverses: \033[00m{np.linalg.norm(np.linalg.inv(np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]])) - x)}")


def generate_random_square_matrix_and_vector(n):
    A = np.random.randint(1, 10, (n, n))
    s = np.random.randint(1, 10, n)
    return A, s


def run_tests(dimension=100, ε=0.0001):
    for i in range(1, dimension):
        A_init, s = generate_random_square_matrix_and_vector(i)
        b = calculate_b(A_init, s)
        A = A_init.copy()
        x_Householder = solve_system_householder_QR(A, b)
        print(x_Householder)
        A = A_init.copy()
        x_QR = solve_system_scipy_QR(A, b)
        print(x_QR)
        if np.linalg.norm(x_QR - x_Householder) > ε:
            print(
                f"The two solutions are different at dimension {i} with norm = ", np.linalg.norm(x_QR - x_Householder))
            return


run_tests(50)

import numpy as np
import math
import scipy.linalg
from enum import Enum


class ValueType(Enum):
    INTEGER = 0
    FLOAT = 1


def calculate_b(A, s):
    A = A.astype(float)
    # n - system dimension
    # A - matrix
    # s - vector

    n = A.shape[0]

    # Error handling
    if A.shape[0] != A.shape[1]:
        print("The array has to be squared")
        return

    if s.shape[0] != n:
        print("The vector s has to be of the same dimension as the matrix A")
        return
    # End of error handling

    return A @ s


# Să se implementeze descompunerea QR a matricei A folosind algoritmul lui Householder
def calculate_own_QR_decomposition(A_init, ε=0.00001):
    A_init = A_init.astype(float)
    n = A_init.shape[0]

    A = A_init.copy()

    Q = np.identity(n)
    for r in range(n - 1):
        # construct P matrix for the current r, constant β and vector u
        # sum list comprehension

        σ = 1.0 * sum([A[i][r] ** 2 for i in range(r, n)])

        if σ <= ε:
            print("I just exited when computing the decomposition!!!")
            break

        k = math.sqrt(σ)  # norm of the vector

        if A[r][r] >= 0:
            k = -k

        β = σ - k * A[r][r]

        u = np.zeros(n)
        # u construction
        u[r] = A[r][r] - k
        for i in range(r + 1, n):
            u[i] = A[i][r]

        # Transform columns j = r + 1, ...,n of A
        for j in range(r + 1, n):
            γ = 1.0 * sum([u[i] * A[i][j] for i in range(r, n)]) / β
            for i in range(r, n):
                A[i][j] = A[i][j] - γ * u[i]

        # Transform column r of A
        A[r][r] = k
        for i in range(r + 1, n):
            A[i][r] = 0

        for j in range(n):
            γ = 1.0 * sum([u[i] * Q[i][j] for i in range(r, n)]) / β
            for i in range(r, n):
                Q[i][j] = Q[i][j] - γ * u[i]

    # Add 0 where we know there must be 0
    # for i in range(n):
    #     for j in range(n):
    #         if i>j:
    #           A[i,j]=0

    return Q.T, A


def calculate_library_QR_decomposition(A_init, ε=0.00001):
    return scipy.linalg.qr(A_init)


def solve_using_scipy_QR(A, b):
    Q, R = scipy.linalg.qr(A, pivoting=False)
    x_QR = scipy.linalg.solve_triangular(R, Q.T @ b)
    return x_QR


def solve_using_householder_QR(A, b, ε=0.0001):
    A = A.astype(float)
    Q, R = calculate_own_QR_decomposition(A, ε)
    x = inverse_substitution_method(R, Q.T @ b)
    return x


def inverse_substitution_method(A, b, ε=10):
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

        # Weird...By increasing ε we get better norm values between the solution found by us and by the scipy library
        # if A[i][i]<ε:
        #   x[i] /= ε
        # else:
        #   x[i] /= A[i][i]

    return x


def approximate_matrix_inverse_householder(A_init, ε=0.00001):
    A = A_init.copy()
    A = A.astype(float)

    Q, R = calculate_own_QR_decomposition(A_init, ε)

    n = R.shape[0]

    if np.linalg.det(R) == 0:
        print("The matrix is singular")
        return

    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        b = Q.T @ e_j
        x = inverse_substitution_method(R, b)
        # x = scipy.linalg.solve_triangular(R, b, lower=False)
        # save x in the j-th column of the matrix A
        for i in range(n):
            A[i][j] = x[i]

    return A


def run_test(A, s):
    print(f"\033[93mMatrice A:\n\033[00m{A}")
    print(f"\033[93mVector s:\n\033[00m{s}")
    b = calculate_b(A, s)
    print(f"\033[93mVector b:\n\033[00m{b}")

    Q, R = calculate_own_QR_decomposition(A)
    print(f"\033[93mFolosind propia implementare a decompozitiei QR:")
    print(f"\033[93mMatrice Q:\n\033[00m{Q}")
    print(f"\033[93mMatrice R:\n\033[00m{R}")

    Q, R = calculate_library_QR_decomposition(A)
    print(f"\033[93mFolosind implementarea decompozitiei QR din scipy:")
    print(f"\033[93mScipy Matrix Q:\n\033[00m{Q}")
    print(f"\033[93mScipy Matrix R:\n\033[00m{R}")

    x_Householder = solve_using_householder_QR(A, b)
    x_QR = solve_using_scipy_QR(A, b)

    print(f"\033[93mHouseholder QR:\n\033[00m", x_Householder)
    print(f"\033[93mScipy QR:\n\033[00m", x_QR)
    print(
        f"\033[93mNorma diferentei dintre cele 2 solutii: \033[00m{np.linalg.norm(x_QR - x_Householder)}")

    # Să se calculeze şi să se afişeze următoarele erori

    print("\033[93mnorm(A_init @ x_Householder - b_init) =\033[00m",
          np.linalg.norm(A @ x_Householder - b))
    print("\033[93mnorm(A_init @ x_QR - b_init) =\033[00m",
          np.linalg.norm(A @ x_QR - b))
    print("\033[93mnorm(x_Householder - s)/ norm(s) =\033[00m",
          np.linalg.norm(x_Householder - s) / np.linalg.norm(s))
    print("\033[93mnorm(x_QR - s)/ norm(s) =\033[00m",
          np.linalg.norm(x_QR - s) / np.linalg.norm(s))

    # Să se calculeze inversa matricei A folosind descompunerea QR calculată la
    # punctul 2. şi să se compare cu inversa calculată folosind funcţia din bibliotecă,
    # afişând valoarea următoarei norme:

    my_A_inverse = approximate_matrix_inverse_householder(A)
    numpy_A_inverse = np.linalg.inv(A)

    print(
        f"\033[93mNumpy inversa matricii:\n\033[00m{numpy_A_inverse}")
    print(f"\033[93mAproximare a inversii matricii:\n\033[00m{my_A_inverse}")

    print(
        f"\033[93mNorma diferentei dintre cele 2 matrici inverse: \033[00m{np.linalg.norm(numpy_A_inverse - my_A_inverse)}")


def generate_matrix_vector_pair(dimension, value_type=ValueType.INTEGER, interval=(-100, 100)):
    # The parameter interval is only used for the integer values
    A = s = None
    if value_type == ValueType.INTEGER:
        A = np.random.randint(interval[0], interval[1], (dimension, dimension))
        s = np.random.randint(interval[0], interval[1], (dimension,))
    elif value_type == ValueType.FLOAT:
        A = np.random.rand(dimension, dimension)
        s = np.random.rand(dimension)
    return A, s


def run_tests(n_interval=(1, 10), value_type=ValueType.INTEGER, values_interval=(1, 10), ε=0.00001):
    # test by generating matrices in the interval [n_interval[0],n_interval[1]]
    n_start = n_interval[0]
    n_end = n_interval[1] + 1

    for n in range(n_start, n_end):
        A, s = generate_matrix_vector_pair(n, value_type, values_interval)

        b = calculate_b(A, s)

        my_Q, my_R = calculate_own_QR_decomposition(A, ε)
        scipy_Q, scipy_R = calculate_library_QR_decomposition(A)

        if np.linalg.norm(my_Q - scipy_Q) > ε:
            print(f"The values are different at dimension {n} with norm = ", np.linalg.norm(my_Q - scipy_Q), " for Q")
            return

        if np.linalg.norm(my_R - scipy_R) > ε:
            print(f"The values are different at dimension {n} with norm = ", np.linalg.norm(my_R - scipy_R), " for R")
            return

        x_Householder = solve_using_householder_QR(A, b)

        x_QR = solve_using_scipy_QR(A, b)

        # print(A)
        # print(b)
        # print(x_Householder)
        # print(x_QR)

        if np.linalg.norm(x_QR - x_Householder) > ε:
            print(
                f"The two solutions are different at dimension {n} with norm = ", np.linalg.norm(x_QR - x_Householder))
            return


def compute_approximate_limit(A_init, max_k=100, ε=0.00001):
    # Compute the approximate limit represented by a matrix A(k+1) and the value k
    Q, R = calculate_library_QR_decomposition(A_init, ε)
    A = A_init
    k = 1
    while k < max_k:

        A_next = R @ Q

        if k % 11 == 0:
            print("k -> " + str(np.linalg.norm(A - A_next)))

        if np.linalg.norm(A_next - A) < ε:
            return True, A_next, k

        Q, R = calculate_library_QR_decomposition(A_next, ε)
        A = A_next.copy()

        k += 2

    return False, None, None


def run_limit_approximation_test(A, max_k=100, ε=0.00001):
    print("Calculam matricea limita folosind valoarea maxima a lui max_k=" + str(max_k) + " si ε=" + str(ε))
    found_limit, A_k, k = compute_approximate_limit(A, max_k, ε)
    if found_limit:
        print("Am gasit pentru k=" + str(k) + " matricea:\n", A_k)
    else:
        print("Nu am gasit o matrice in intervalul si precizia data")


A = np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]])
s = np.array([3, 2, 1])
run_test(A, s)

run_tests((10, 100), ValueType.FLOAT)

# A,s=generate_matrix_vector_pair(20)
# run_limit_approximation_test(A,max_k=1_000_000,ε=1e-5)



# Bonus
def multiplication_RQ(R, Q):
    n = R.shape[0]
    result = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            result[i][j] = R[i, i] * Q[i, j]
            for k in range(i + 1, n):
                result[i][j] += R[i][k] * Q[k][j]

    return result


def equal_matrices(A, B, ε=1e-8):
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - B[i][j]) > ε:
                return False
    return True


def approximate_limit(A_init, ε=1e-8, k_max=10000):
    A = A_init.copy()
    k = 0
    while True:
        Q, R = calculate_QR(A)
        A_next = multiplication_RQ(R, Q)
        # if equal_matrices(A, A_next, ε):
        #     A = A_next
        #     break
        if np.allclose(A, A_next, atol=ε):
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


matrix = generate_random_symmetric_matrix(20)
print("Random symmetric matrix: \n", matrix)
matrix, k = approximate_limit(matrix)

print(f"\nSame matrix for {k} iterations")
print(matrix)


import time
import matplotlib.pyplot as plt


def solve_scipy_lu(A, b):
    lu, piv = scipy.linalg.lu_factor(A)
    return scipy.linalg.lu_solve((lu, piv), b)


def plot_solve_comparison(start, finish):
    solve_diff_householder_scipyqr = np.array([])
    solve_diff_householder_solve = np.array([])
    solve_diff_householder_scipylu = np.array([])
    solve_diff_householder_npsolve = np.array([])

    time_householder = np.array([])
    time_scipyqr = np.array([])
    time_solve = np.array([])
    time_scipylu = np.array([])
    time_npsolve = np.array([])

    fig, ax = plt.subplots(6, 1, figsize=(15, 10))
    for i in range(start, finish):
        A_init, s = generate_random_square_matrix_and_vector(i)
        b = calculate_b(A_init, s)
        A = A_init.copy()

        t_start = time.time()
        x_Householder = solve_system_householder_QR(A, b)
        t_end = time.time()
        time_householder = np.append(time_householder, t_end - t_start)

        t_start = time.time()
        solve_diff_householder_scipyqr = np.append(
            solve_diff_householder_scipyqr, np.linalg.norm(x_Householder - solve_system_scipy_QR(A_init, s)))
        t_end = time.time()
        time_scipyqr = np.append(time_scipyqr, t_end - t_start)

        t_start = time.time()
        solve_diff_householder_solve = np.append(
            solve_diff_householder_solve, np.linalg.norm(
                x_Householder - scipy.linalg.solve(A_init, s))
        )
        t_end = time.time()
        time_solve = np.append(time_solve, t_end - t_start)

        t_start = time.time()
        solve_diff_householder_scipylu = np.append(
            solve_diff_householder_scipylu, np.linalg.norm(x_Householder - solve_scipy_lu(A_init, s)))
        t_end = time.time()
        time_scipylu = np.append(time_scipylu, t_end - t_start)

        t_start = time.time()
        solve_diff_householder_npsolve = np.append(
            solve_diff_householder_npsolve, np.linalg.norm(
                x_Householder - np.linalg.solve(A_init, s)))
        t_end = time.time()
        time_npsolve = np.append(time_npsolve, t_end - t_start)

    ax[0].plot(np.arange(start, finish),
               solve_diff_householder_scipyqr, label="scipy QR")
    ax[0].plot(np.arange(start, finish),
               solve_diff_householder_solve, label="scipy solve")
    ax[0].plot(np.arange(start, finish),
               solve_diff_householder_scipylu, label="scipy LU")
    ax[0].plot(np.arange(start, finish),
               solve_diff_householder_npsolve, label="np solve")
    ax[0].set_title("Solve comparison")
    ax[0].set_xlabel("Dimension")
    ax[0].set_ylabel("Norm")
    ax[0].legend()

    ax[1].plot(np.arange(start, finish),
               solve_diff_householder_scipyqr, label="scipy QR")
    ax[1].set_title("Solve comparison scipy QR")
    ax[1].set_xlabel("Dimension")
    ax[1].set_ylabel("Norm")

    ax[2].plot(np.arange(start, finish),
               solve_diff_householder_solve, label="scipy solve")
    ax[2].set_title("Solve comparison scipy solve")
    ax[2].set_xlabel("Dimension")
    ax[2].set_ylabel("Norm")

    ax[3].plot(np.arange(start, finish),
               solve_diff_householder_scipylu, label="scipy LU")
    ax[3].set_title("Solve comparison scipy LU")
    ax[3].set_xlabel("Dimension")
    ax[3].set_ylabel("Norm")

    ax[4].plot(np.arange(start, finish),
               solve_diff_householder_npsolve, label="numpy solve")
    ax[4].set_title("Solve comparison numpy solve")
    ax[4].set_xlabel("Dimension")
    ax[4].set_ylabel("Norm")

    ax[5].plot(np.arange(start, finish),
               time_householder, label="Householder")
    ax[5].plot(np.arange(start, finish),
               time_scipyqr, label="scipy QR")
    ax[5].plot(np.arange(start, finish),
               time_scipylu, label="scipy LU")
    ax[5].plot(np.arange(start, finish),
               time_npsolve, label="numpy solve")
    ax[5].set_title("Time comparison")
    ax[5].set_xlabel("Dimension")
    ax[5].set_ylabel("Time")
    ax[5].legend()

    plt.show()


# All plots compare the householder implemented in the homework with the algorithm in the plot legend/title
plot_solve_comparison(1, 100)


def plot_inv_comparison(start, finish):
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))
    inv_difference = np.array([])
    inv_difference_non_singular = np.array([])
    colors_non_singular = np.array([])
    for i in range(start, finish):
        A_init, s = generate_random_square_matrix_and_vector(i)
        inv_householder = approximate_matrix_inverse_householder(A_init)
        inv_difference = np.append(inv_difference, np.linalg.norm(
            np.linalg.pinv(A_init) - inv_householder))
        if np.linalg.det(A_init) != 0:
            inv_difference_non_singular = np.append(inv_difference_non_singular, np.linalg.norm(
                np.linalg.inv(A_init) - inv_householder))
            colors_non_singular = np.append(colors_non_singular, 'r')
        else:
            inv_difference_non_singular = np.append(
                inv_difference_non_singular, inv_difference[-1])
            colors_non_singular = np.append(colors_non_singular, 'b')

    ax[0].plot(np.arange(start, finish), inv_difference)
    ax[0].set_title("Difference between scipy inverse and householder inverse")
    ax[0].set_xlabel("Matrix dimension")
    ax[0].set_ylabel("Norm of the difference")

    if np.all(colors_non_singular == 'r'):
        ax[1].plot(np.arange(start, finish), inv_difference_non_singular)
    else:
        ax[1].scatter(np.arange(start, finish),
                      inv_difference_non_singular, c=colors_non_singular)
    ax[1].set_title(
        "Difference between scipy inverse and householder inverse for non singular matrices")
    ax[1].set_xlabel("Matrix dimension")
    ax[1].set_ylabel("Norm of the difference")

    plt.show()


plot_inv_comparison(1, 100)
