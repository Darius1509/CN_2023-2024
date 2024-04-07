import numpy as np

n = int(input("Introduceti n (numarul de puncte - 1): "))
x_nodes = list(map(float, input("Introduceti nodurile x separate prin spatiu: ").split()))
y_values = list(map(float, input("Introduceti valorile corespunzatoare y separate prin spatiu: ").split()))
m = int(input("Introduceti gradul polinomului m (m < 6): "))

if len(x_nodes) != n + 1 or len(y_values) != n + 1:
    print("Numarul de noduri x sau valori y introduse nu corespunde cu n+1.")
else:
    print("Noduri x: ", x_nodes)
    print("Valori y: ", y_values)

def f(x):
    return x ** 4 - 12 * x ** 3 + 30 * x ** 2 + 12

def calculate_differences(y_values, n): #Schema Aitken
    for k in range(1, n + 1):
        for i in range(n, k - 1, -1):
            y_values[i] = y_values[i] - y_values[i - 1]
    return y_values

def newton_forward_interpolation(x, x0, h, y_values, n):
    t = (x - x0) / h
    interpolation_sum = y_values[0]
    t_product = 1
    for i in range(1, n + 1):
        t_product *= (t - i + 1) / i
        interpolation_sum += y_values[i] * t_product
    return interpolation_sum


def least_squares_approximation(x_nodes, y_values, m):
    A = np.zeros((m + 1, m + 1))
    b = np.zeros(m + 1)
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = sum(x ** ((i + j)) for x in x_nodes)
        b[i] = sum(y * (x ** i) for x, y in zip(x_nodes, y_values))

    coef = np.linalg.solve(A, b)
    return coef


def horner_scheme(coef, x):
    result = coef[-1]
    for a in reversed(coef[:-1]):
        result = result * x + a
    return result

h = (x_nodes[n] - x_nodes[0])/n  #Diferenta dintre noduri

#Calculam diferentele finite
y_diff = calculate_differences(y_values.copy(), n)

x_value = float(input("Introduceti valoarea lui x pentru care doriți să aproximați f(x): "))
Ln_x = newton_forward_interpolation(x_value, x_nodes[0], h, y_diff, n)

error = abs(Ln_x - f(x_value))
print("f(x) =", f(x_value))

print(f"Valoarea aproximata L_n({x_value}) = {Ln_x}")
print(f"Eroarea absoluta |L_n({x_value}) - f({x_value})| = {error}")

coef = least_squares_approximation(x_nodes, y_values, m)

x = float(input("Introduceti valoarea lui x pentru care doriti sa calculati Pm(x): "))
Pm_x = horner_scheme(coef, x)
error = abs(Pm_x - f(x))
sum_errors = sum(abs(horner_scheme(coef, xi) - yi) for xi, yi in zip(x_nodes, y_values))

print(f"Valoarea aproximata Pm({x}) = {Pm_x}")
print(f"Eroarea |Pm({x}) - f({x})| = {error}")
print(f"Suma erorilor |Pm(xi) - yi| pentru i=0 pana la n = {sum_errors}")


