#ex1
from matplotlib import pyplot as plt

u = 10.0
m = 1

while not (u ** (-m) + 1 == 1):
    m += 1
print("Ex1")
print(m - 1)
print("\n")

#ex2
#calcul u
print("Ex2")
u = u**(-m)
print(u)

#adunare neasociativa
x = 1.0
y = u / 10
z = u / 10
m=1
while(x + y) + z == x + (y + z):
    m+= 1
    y= 10**(-m)
    z= y

print(f"x = {x}, y = {y}, z = {z}")
print((x + y) + z)
print(x + (y + z))
print((x + y) + z != x + (y + z))

#inmultire neasociativa
x = 2
m = 1
while (x * y) * z == x * (y * z):
  m += 1
  y = 10**(-m)
  z = y

print(f"x = {x}, y = {y}, z = {z}")
print((x * y) * z)
print(x * (y * z))
print((x * y) * z != x * (y * z))

#ex3
import matplotlib.pyplot as plt

print("Ex3\n")
import numpy as np
import math
def T1(a):
    return a

def S1(a):
    return T1(a) / (1 + T1(a) ** 2)**0.5
def C1(a):
    return 1 / (1 + T1(a) ** 2)**0.5

def T2(a):
    return 3 * a / (3 - a**2)

def S2(a):
    return T2(a)/  (1 + T2(a) ** 2)**0.5

def C2(a):
    return 1 /  (1 + T2(a) ** 2)**0.5

def T3(a):
    return (15 * a - a**3) / (15 - 6 * a**2)

def S3(a):
    return T3(a)/  (1 + T3(a) ** 2)**0.5

def C3(a):
    return 1 /  (1 + T3(a) ** 2)**0.5

def T4(a):
    return (105 * a - 10 * a**3) / (105 - 45 * a**2 + a**4)

def S4(a):
    return T4(a)/  (1 + T4(a) ** 2)**0.5

def C4(a):
    return 1 /  (1 + T4(a) ** 2)**0.5


def T5(a):
    return (945 * a - 105 * a**3 + a**5) / (945 - 420 * a**2 + 15 * a**4)

def S5(a):
    return T5(a)/  (1 + T5(a) ** 2)**0.5

def C5(a):
    return 1/  (1 + T5(a) ** 2)**0.5

def T6(a):
    return (10395 * a - 1260 * a**3 + 21 * a**5) / (10395 - 4725 * a**2 + 210 * a**4 - a**6)

def S6(a):
    return T6(a)/  (1 + T6(a) ** 2)**0.5

def C6(a):
    return 1 /  (1 + T6(a) ** 2)**0.5

def T7(a):
    return (135135 * a - 17325 * a**3 + 378 * a**5 - a**7) / (135135 - 62370 * a**2 + 3150 * a**4 - 28 * a**6)

def S7(a):
    return T7(a)/  (1 + T7(a) ** 2)**0.5

def C7(a):
    return 1 /  (1 + T7(a) ** 2)**0.5

def T8(a):
    return (2027025 * a - 270270 * a**3 + 6830 * a**5 - 36 * a**7) / (2027025 - 945945 * a**2 + 51975 * a**4 - 630 * a**6 + a**8)

def S8(a):
    return T8(a)/  (1 + T8(a) ** 2)**0.5

def C8(a):
    return 1 /  (1 + T8(a) ** 2)**0.5

def T9(a):
    return (34459425 * a - 4729725 * a**3 + 135135 * a**5 - 990 * a**7 + a**9) / (34459425 - 16216200 * a**2 + 945945 * a**4 - 13860 * a**6 + 45 * a**8)

def S9(a):
    return T9(a)/  (1 + T9(a) ** 2)**0.5

def C9(a):
    return 1 /  (1 + T9(a) ** 2)**0.5


random_numbers = np.random.uniform(-np.pi/2, np.pi/2, 10000)

values = np.array([T1(random_numbers), T2(random_numbers), T3(random_numbers),
                   T4(random_numbers), T5(random_numbers), T6(random_numbers),
                   T7(random_numbers), T8(random_numbers), T9(random_numbers)])

exact_values = np.tan(random_numbers)

errors = np.abs(values - exact_values.reshape(1, -1))

errorsT = dict()
for i in range(9):
    avg_error = np.mean(errors[i])
    errorsT[i] = avg_error

sorted_errors=sorted(errorsT.items(), key=lambda x: x[1])
print("Errors in ascending order:")
for i in range(9):
    print(f"T({sorted_errors[i][0]+1},a)- Average Error: {sorted_errors[i][1]}")


#errors in ascending order
sorted_errors = sorted(errorsT.items(), key=lambda x: x[1])

plt.figure(figsize=(10, 6))
plt.bar(range(1, 10), [error[1] for error in sorted_errors], color='skyblue')
plt.xlabel('Term (Tn(a))')
plt.ylabel('Average Error')
plt.title('Errors in Ascending Order for Trigonometric Approximations')
plt.xticks(range(1, 10), [f'T({error[0]+1},a)' for error in sorted_errors])
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.xlim(0.5, 9.5)

plt.show()

#bonus
values = np.array([S1(random_numbers), S2(random_numbers), S3(random_numbers),
                   S4(random_numbers), S5(random_numbers), S6(random_numbers),
                   S7(random_numbers), S8(random_numbers), S9(random_numbers)])

exact_values_sin = np.sin(random_numbers)

errors_sin = np.abs(values - exact_values_sin.reshape(1, -1))

errorsSin = dict()
for i in range(9):
    avg_error = np.mean(errors_sin[i])
    errorsSin[i] = avg_error

sorted_errors_sin=sorted(errorsSin.items(), key=lambda x: x[1])
print("Errors in ascending order:")
for i in range(9):
    print(f"S({sorted_errors_sin[i][0]+1},a)- Average Error: {sorted_errors_sin[i][1]}")


#errors in ascending order
sorted_errors_sin = sorted(errorsSin.items(), key=lambda x: x[1])

plt.figure(figsize=(10, 6))
plt.bar(range(1, 10), [error[1] for error in sorted_errors_sin], color='skyblue')
plt.xlabel('Term (Tn(a))')
plt.ylabel('Average Error')
plt.title('Errors in Ascending Order for Trigonometric Approximations of Sin')
plt.xticks(range(1, 10), [f'S({error[0]+1},a)' for error in sorted_errors_sin])
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.xlim(0.5, 9.5)

plt.show()

#cos
values_cos = np.array([C1(random_numbers), C2(random_numbers), C3(random_numbers),
                   C4(random_numbers), C5(random_numbers), C6(random_numbers),
                   C7(random_numbers), C8(random_numbers), C9(random_numbers)])

exact_values_cos = np.cos(random_numbers)

errors_cos = np.abs(values_cos - exact_values_cos.reshape(1, -1))

errorsCos = dict()
for i in range(9):
    avg_error = np.mean(errors_cos[i])
    errorsCos[i] = avg_error

sorted_errors_cos=sorted(errorsCos.items(), key=lambda x: x[1])
print("Errors in ascending order:")
for i in range(9):
    print(f"C({sorted_errors_cos[i][0]+1},a)- Average Error: {sorted_errors_cos[i][1]}")


#errors in ascending order
sorted_errors_cos = sorted(errorsCos.items(), key=lambda x: x[1])

plt.figure(figsize=(10, 6))
plt.bar(range(1, 10), [error[1] for error in sorted_errors_cos], color='skyblue')
plt.xlabel('Term (Tn(a))')
plt.ylabel('Average Error')
plt.title('Errors in Ascending Order for Trigonometric Approximations of Cos')
plt.xticks(range(1, 10), [f'C({error[0]+1},a)' for error in sorted_errors_cos])
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.xlim(0.5, 9.5)

plt.show()


