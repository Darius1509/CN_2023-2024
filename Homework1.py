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
print("Ex3\n")
import numpy as np
import math
def T1(a):
    return a

def T2(a):
    return 3 * a / (3 - a**2)

def T3(a):
    return (15 * a - a**3) / (15 - 6 * a**2)

def T4(a):
    return (105 * a - 10 * a**3) / (105 - 45 * a**2 + a**4)

def T5(a):
    return (945 * a - 105 * a**3 + a**5) / (945 - 420 * a**2 + 15 * a**4)

def T6(a):
    return (10395 * a - 1260 * a**3 + 21 * a**5) / (10395 - 4725 * a**2 + 210 * a**4 - a**6)

def T7(a):
    return (135135 * a - 17325 * a**3 + 378 * a**5 - a**7) / (135135 - 62370 * a**2 + 3150 * a**4 - 28 * a**6)

def T8(a):
    return (2027025 * a - 270270 * a**3 + 6830 * a**5 - 36 * a**7) / (2027025 - 945945 * a**2 + 51975 * a**4 - 630 * a**6 + a**8)

def T9(a):
    return (34459425 * a - 4729725 * a**3 + 135135 * a**5 - 990 * a**7 + a**9) / (34459425 - 16216200 * a**2 + 945945 * a**4 - 13860 * a**6 + 45 * a**8)

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