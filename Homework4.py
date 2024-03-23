import numpy as np
import re
import math


def parse_vector(file_name):
    try:
        with open(file_name) as file:
            n = int(file.readline())
            b = np.zeros(n).astype(float)

            for index in range(n):
                b[index] = float(file.readline())

        print(f"Vectorul din {file_name} a fost incarcat cu succes.")
        return b
    except IOError:
        print("Could not open file: ", file_name)
        return None



def parse_sparse_matrix(file_name):
    matrix = None
    try:
        with open(file_name) as file:
            first_line = True
            for text_line in file:
                if first_line:
                    n = int(text_line)
                    matrix = [dict() for i in range(n)]
                    first_line = False
                    continue

                fields = list(re.split("[ ,\n]+", text_line))
                fields = list(filter(lambda x: len(x) > 0, fields))

                if len(fields) != 3:
                    continue

                value = float(fields[0])
                line, column = int(fields[1]), int(fields[2])

                previous_value = matrix[line].setdefault(column, 0)
                matrix[line][column] = previous_value + value

        print(f"Matricea rara din {file_name} a fost incarcata cu succes folosind metoda de memorare dictionar in lista.")
        return matrix
    except IOError:
        print("Could not open file: ", file_name)
        return None


def parse_sparse_matrix_rare(file_name):
    sparse_matrix_rare = []
    try:
        with open(file_name, 'r') as file:
            n = int(file.readline().strip())
            for _ in range(n):
                sparse_matrix_rare.append([])

            for line in file:
                parts = [x.strip() for x in line.strip().split(',')]
                if len(parts) == 3:
                    value, row, column = parts
                    value = float(value)
                    row, column = int(row), int(column)
                    if value != 0:
                        sparse_matrix_rare[row].append((value, column))
                else:
                    print(f"Linie neasteptata (ignorata): {line.strip()}")

        print(f"Matricea rara din {file_name} a fost incarcata cu succes.")
        return sparse_matrix_rare
    except IOError:
        print(f"Nu s-a putut deschide fișierul: {file_name}")
        return None


def print_sparse_matrix_rare(sparse_matrix_rare):
    for row in sparse_matrix_rare:
        print("[", end="")
        for value, column in row:
            print(f"({value}, {column})", end=" ")
        print("]")


def check_diagonal_elements(A, eps=0.00001):
    not_null_main_diagonal = not_null_secondary_diagonal = True

    n = len(A)

    for line in range(n):
        main_element = secondary_element = 0.0
        for column, value in A[line].items():
            if line == column:
                main_element += value
            if line + column == n - 1:
                secondary_element += value

        if abs(main_element) <= eps:
            not_null_main_diagonal = False

        if abs(secondary_element) <= eps:
            not_null_secondary_diagonal = False

        if (not not_null_main_diagonal) and (not not_null_secondary_diagonal):
            break

    return not_null_main_diagonal, not_null_secondary_diagonal


def approximate_solution_using_Gauss_Seidel(A, b, max_k=1000, max_difference=1e8, eps=1e-25):
    not_null_main_diagonal, not_null_secondary_diagonal = check_diagonal_elements(
        A, eps)

    if not not_null_main_diagonal:
        print(
            "Diagonala principala trebuie sa contina elemente nenule pentru a putea aproxima folosind metoda Gauss-Seidel")
        return False, "Divergenta", 0

    n = b.shape[0]
    x = np.zeros(n).astype(float)
    current_k = 0
    euclidean_norm = None

    while True:
        euclidean_norm = 0.0
        for i in range(n):
            new_x_i = b[i]

            new_x_i -= sum([value * x[j] for j, value in A[i].items() if i != j])

            new_x_i /= A[i].get(i, 1.0)

            euclidean_norm += (x[i] - new_x_i) ** 2
            x[i] = new_x_i

        euclidean_norm = math.sqrt(euclidean_norm)
        current_k += 1

        if current_k >= max_k or euclidean_norm < eps or euclidean_norm > max_difference:
            break

    if euclidean_norm < eps:
        return True, x, current_k

    return False, "Divergenta", current_k



def compute_solutions(file_name):
    x = None
    if file_name == "a_1.txt" or file_name == "b_1.txt":
        n = 60690
        x = np.zeros((n,))
        for i in range(n):
            x[i] = 1.0
    elif file_name == "a_2.txt" or file_name == "b_2.txt":
        n = 40460
        x = np.zeros((n,))
        for i in range(n):
            x[i] = 4.0 / 3.0
    elif file_name == "a_3.txt" or file_name == "b_3.txt":
        n = 20230
        x = np.zeros((n,))
        for i in range(n):
            x[i] = 2.0 * (i + 1) / 5.0
    elif file_name == "a_4.txt" or file_name == "b_4.txt":
        n = 10115
        x = np.zeros((n,))
        for i in range(n):
            x[i] = i / 7.0
    elif file_name == "a_5.txt" or file_name == "b_5.txt":
        n = 2023
        x = np.zeros((n,))
        for i in range(n):
            x[i] = 2.0
    else:
        print("Fisier nerecunoscut!Nu avem solutia sa!")
    return x


def own_Chebyshev_norm(a):
    norm = -1
    n = a.shape[0]
    for i in range(n):
        norm = max(norm, abs(a[i]))
    return norm


def multiple_sparse_matrix_with_vector(matrix, vector):
    n = vector.shape[0]

    result = np.zeros(n)
    for line in range(n):
        element = 0
        for column, value in matrix[line].items():
            element += value * vector[column]

        result[line] = element
    return result


def compute_closeness(A, B, eps=1e-5):

    if len(A) != len(B):
        print("Dimensiunea matricelor patratice trebuie sa fie aceeasi!")
        return

    n = len(A)

    for line in range(n):
        columns = A[line].keys() | B[line].keys()

        for column in columns:
            a = A[line].get(column, 0)
            b = B[line].get(column, 0)
            if abs(a - b) > eps:
                return False

    return True


def run_test(a_file_name, b_file_name, max_k=10000, max_difference=1e8, eps=1e-5):
    print("\n\nPentru fisierele " + a_file_name + " si " + b_file_name)
    print("Folosim precizia eps=" + str(eps))

    b = parse_vector(b_file_name)
    A = parse_sparse_matrix(a_file_name)
    A_alternative = parse_sparse_matrix_rare(a_file_name)

    if A is None or b is None:
        return

    if len(A) != len(b):
        print("Dimensiunile pentru matricea A si b trebuie sa fie la fel!")
        return

    print("\nDimensiunea N = " + str(len(b)))
    #print("\nMatricea A salvata economic(metoda enunt):" + print_sparse_matrix_rare(A_alternative))

    not_null_main_diagonal, not_null_secondary_diagonal = check_diagonal_elements(A, eps)
    print("\nAre diagonala principala toate elementele nenule? " + ("Da" if not_null_main_diagonal else "Nu"))
    print("Are diagonala secundara toate elementele nenule? " + ("Da" if not_null_secondary_diagonal else "Nu"))

    converged, x_approximation, iterations = approximate_solution_using_Gauss_Seidel(A, b, max_k, max_difference, eps)
    if converged:
        print(f"Metoda a convergat in {iterations} iteratii.")
        Ax_minus_b = multiple_sparse_matrix_with_vector(A, x_approximation) - b
        infinity_norm = np.linalg.norm(Ax_minus_b, np.inf)
        print("Norma infinita a diferentei dintre Ax_GS si b este: " + str(infinity_norm))

        x = compute_solutions(a_file_name)
        # if x is not None and x_approximation.shape == x.shape:
        #     print("Norma euclideana dintre x si x* este : " + str(np.linalg.norm(x_approximation - x)))
        # else:
        #     print("Cannot compute Euclidean norm due to mismatch in array dimensions.")
        print("Solutia aproximativa este: " + str(x_approximation))
    else:
        print(f"Metoda nu a convergat dupa {iterations} iteratii.")




def run_test_for_files(matrix_file_names: list[(str, str)], max_k: int = 10_000, max_difference: float = 1e8,
                       eps: float = 1e-9):
    for file_name_a, file_name_b in matrix_file_names:
        run_test(file_name_a, file_name_b, max_k=max_k, max_difference=max_difference, eps=eps)


file_names = [("a_1.txt", "b_1.txt"),
    ("a_2.txt", "b_2.txt"),
    ("a_3.txt", "b_3.txt"),
    ("a_4.txt", "b_4.txt"),
    ("a_5.txt", "b_5.txt")]
run_test_for_files(file_names, max_k=10_000, max_difference=1e8, eps=1e-9)

# Bonus
def add_sparse_matrixes(A, B):
    if len(A) != len(B):
        print("Dimensiunea matricelor patratice trebuie sa fie aceeasi!")
        return

    n = len(A)

    C = []
    for line in range(n):
        summed_elements = dict()

        columns = A[line].keys() | B[line].keys()
        for column in columns:
            summed_elements[column] = A[line].get(
                column, 0) + B[line].get(column, 0)

        C.append(summed_elements)

    return C


def run_add_test(A_file_name, B_file_name, C_file_name, eps=1e-5):
    print("\n\nPrecizia folosita eps=" + str(eps))
    print("Din fisierele " + A_file_name + " si " + B_file_name +
          " extragem cele 2 matrici A si B si suma lor fiind C.Acest C il verificam cu matricea din fisierul " + C_file_name)

    A = parse_sparse_matrix(A_file_name)
    B = parse_sparse_matrix(B_file_name)
    if A is None or B is None:
        return
    our_C = add_sparse_matrixes(A, B)
    correct_C = parse_sparse_matrix(C_file_name)
    are_equal = compute_closeness(our_C, correct_C, eps)
    if are_equal:
        print("Matricea obtinuta este egala cu cea din fisierul dat")
    else:
        print("Matricea obtinuta nu este egala cu cea din fisierul dat")

run_add_test("a.txt", "b.txt", "aplusb.txt", eps=1e-9)