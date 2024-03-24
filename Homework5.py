import numpy as np
from scipy.linalg import norm, cholesky
from numpy.linalg import svd, norm, inv

#exemplu A_K (converge prin eroare)
# 2 1 1
# 1 2 1
# 1 1 2

#exemplu A_K (converge prin iteratii)
# 100 50 30 20
# 50 100 60 40
# 30 60 100 70
# 20 40 70 100


def jacobi_rotation(A):
    n = A.shape[0]
    p, q = np.unravel_index(np.argmax(np.abs(A - np.diag(np.diagonal(A)))), A.shape)
    if A[p, q] == 0:
        return A, np.eye(n)
    phi = 0.5 * np.arctan2(2 * A[p, q], A[q, q] - A[p, p])
    R = np.eye(n)
    R[p, p] = R[q, q] = np.cos(phi)
    R[p, q] = np.sin(phi)
    R[q, p] = -np.sin(phi)
    return R, p, q

def metoda_jacobi(A, epsilon):
    n = A.shape[0]
    if n != A.shape[1]:
        return None, "Matricea nu este patrata. Metoda Jacobi necesita o matrice patrata."

    U = np.eye(n)
    while True:
        R, p, q = jacobi_rotation(A)
        if abs(A[p, q]) < epsilon:
            break
        A = R.T @ A @ R
        U = U @ R
    Lambda = np.diag(A)
    return U, Lambda

def verifica_relatia(A_init, U, Lambda, mesaj, epsilon):
    if U is None:
        print(mesaj)
        return
    Lambda_aproximate = np.where(np.abs(Lambda) < epsilon, 0, Lambda)
    diferenta = A_init @ U - U @ np.diag(Lambda_aproximate)
    norma = norm(diferenta, 'fro')  #frobenius=norma matriceala
    print("Norma matriceala a diferentei este:", norma)
    print("Valorile proprii aproximative sunt:", Lambda_aproximate)


def calcul_sir_matrice(A, epsilon, k_max):
    n = A.shape[0]
    if n != A.shape[1]:
        print("Matricea nu este patrata. Sirul matriceal nu poate fi calculat.")
        return A
    k = 0
    while k < k_max:
        try:
            L = cholesky(A, lower=True)
        except np.linalg.LinAlgError:
            print(f"Descompunerea Cholesky a esuat la iteratia {k}.")
            return A
        A_next = L.T @ L
        diferenta = norm(A - A_next, 'fro')
        if diferenta < epsilon:
            print(f"Convergenta atinsa la iteratia {k}.")
            return A_next
        A = A_next
        k += 1
    print("Numarul maxim de iteratii a fost atins.")
    return A

def informatii_SVD(A):
    p, n = A.shape
    if p <= n:
        print("Dimensiunea p trebuie sa fie mai mare decat n pentru a calcula informatiile cerute.")
        return
    U, sigma, VT = svd(A, full_matrices=False)
    print("Valorile singulare ale matricei A:", sigma)
    rang_A = np.sum(sigma > 1e-10)
    print("Rangul matricei A:", rang_A)
    k_2 = sigma.max() / sigma[sigma > 1e-10].min()
    print("Numarul de conditionare al matricei A:", k_2)
    S_inv = np.diag(1 / sigma)
    A_I = VT.T @ S_inv @ U.T
    print("Pseudoinversa Moore-Penrose a matricei A:\n", A_I)
    A_J = inv(A.T @ A) @ A.T
    print("Matricea pseudo inversa in sensul celor mai mici patrate:\n", A_J)
    diferenta_norma_1 = norm(A_I - A_J, 1)
    print("||A^I - A^J||_1:", diferenta_norma_1)

def citire_matrice():
    p = int(input("Introduceti numarul de linii p al matricei A: "))
    n = int(input("Introduceti numarul de coloane n al matricei A: "))
    A = np.zeros((p, n))
    print("Introduceti elementele matricei A (linie cu linie): ")
    for i in range(p):
        A[i] = np.array(input().split(), dtype=float)
    epsilon = float(input("Introduceti precizia epsilon: "))
    return A, epsilon

def main():
    A, epsilon = citire_matrice()
    A_init = A.copy()
    U, Lambda_or_mesaj = metoda_jacobi(A, epsilon)
    A_final = calcul_sir_matrice(A, epsilon, 100)
    verifica_relatia(A_init, U, Lambda_or_mesaj, Lambda_or_mesaj if isinstance(Lambda_or_mesaj, str) else "", epsilon)
    print("Ultima matrice calculata este:")
    print(A_final)
    print("Informatii SVD:")
    informatii_SVD(A)

if __name__ == "__main__":
    main()
