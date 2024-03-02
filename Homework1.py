#ex1
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