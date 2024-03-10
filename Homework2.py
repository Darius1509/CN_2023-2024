import numpy as np

def crout(A):
    L = np.zeros((3, 3))
    U = np.zeros((3, 3))

    for k in range(0, 3):
        U[k, k] = 1

        for j in range(k, 3):
            sum0 = sum([L[j, s] * U[s, k] for s in range(0, j)])
            L[j, k] = A[j, k] - sum0

        for j in range(k+1, 3):
            sum1 = sum([L[k, s] * U[s, j] for s in range(0, k)])
            U[k, j] = (A[k, j] - sum1) / L[k, k]

    return L, U

def determinant_from_lu(L, U):
    det_L = np.prod(np.diag(L))
    det_U = np.prod(np.diag(U))
    det_A = det_L * det_U
    return det_A

def forward_substitution(L, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def solve_approximate(A, b):
    L, U = crout(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

def check_solution(A, b, x):
    residual_norm = np.linalg.norm(np.dot(A, x) - b, ord=2)
    return residual_norm

A = np.array([[2.5, 2, 2], [5, 6, 5], [5, 6, 6.5]])
b = np.array([2, 2, 2])
L, U = crout(A)
det_A = determinant_from_lu(L, U)
x_approx = solve_approximate(A, b)
residual_norm = check_solution(A, b, x_approx)
A_inv = np.linalg.inv(A)
xLIB = np.linalg.solve(A, b)
A_inv_b = np.dot(A_inv, b)
norm_diff_x = np.linalg.norm(x_approx - xLIB, ord=2)
norm_diff_A_inv_b = np.linalg.norm(x_approx - A_inv_b, ord=2)

print("Matricea L (superioara triunghiulara cu 1 pe diagonala):")
print(L)
print("\nMatricea U (inferioara triunghiulara):")
print(U)
print("\nDeterminantul matricei A este:", det_A)
print("\nSolutia aproximativa a sistemului Ax = b este:", x_approx)
print("\nNorma Euclidiana a reziduului ||A*x - b||_2:", residual_norm)
if residual_norm < 1e-9:
    print("Norma reziduului este mai mica decat 10^-9")
else:
    print("Norma reziduului NU este mai mica decat 10^-9")
print("\nSoluția sistemului Ax = b este:", xLIB)
print("\nInversa matricei A este:", A_inv)
print("\nNorma Euclidiana ||xLU - xLIB||_2 este:", norm_diff_x)
print("\nNorma Euclidiana ||xLU - A^-1*b||_2 este:", norm_diff_A_inv_b)
