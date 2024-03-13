import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QPushButton, QVBoxLayout, QMessageBox

def crout(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(n):
        U[k, k] = 1

        for j in range(k, n):
            sum0 = sum([L[j, s] * U[s, k] for s in range(k)])
            L[j, k] = A[j, k] - sum0

        for j in range(k+1, n):
            sum1 = sum([L[k, s] * U[s, j] for s in range(k)])
            U[k, j] = (A[k, j] - sum1) / L[k, k]

    return L, U


def lu_decomposition_vector(A):
    n = len(A)
    L_vector = np.zeros(n * (n + 1) // 2)
    U_vector = np.zeros(n * (n + 1) // 2)

    def map_to_vector(i, j, n):
        if i > j:
            return (i * (i + 1) // 2) + j
        else:
            return (j * (j + 1) // 2) + i

    for i in range(n):
        for j in range(i, n):
            sum_u = sum(U_vector[map_to_vector(k, j, n)] * L_vector[map_to_vector(i, k, n)] for k in range(i))
            U_vector[map_to_vector(i, j, n)] = A[i, j] - sum_u

        for j in range(i + 1, n):
            sum_l = sum(U_vector[map_to_vector(k, i, n)] * L_vector[map_to_vector(j, k, n)] for k in range(i))
            L_vector[map_to_vector(j, i, n)] = (A[j, i] - sum_l) / U_vector[map_to_vector(i, i, n)]

    return L_vector, U_vector


def compute(L_vector, U_vector, n):
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    def index_in_vector(i, j, n):
        if i >= j:
            return i * (i + 1) // 2 + j
        else:
            return j * (j + 1) // 2 + i

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i, j] = 1
            else:
                L[i, j] = L_vector[index_in_vector(i, j, n)]

    for i in range(n):
        for j in range(i, n):
            U[i, j] = U_vector[index_in_vector(i, j, n)]

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

def solve_approximate(A, b, method):
    if method == 0:
        L, U = crout(A)
    elif method == 1:
        L_vector, U_vector = lu_decomposition_vector(A)
        L, U = compute(L_vector, U_vector, len(A))
    else:
        raise ValueError("Metoda selectata nu este valida. Introduceti 0 pentru metoda initiala sau 1 pentru metoda noua:")

    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

def check_solution(A, b, x):
    residual_norm = np.linalg.norm(np.dot(A, x) - b, ord=2)
    return residual_norm

class LUApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.n_label = QLabel("Dimensiunea matricei A:")
        layout.addWidget(self.n_label)
        self.n_entry = QLineEdit()
        layout.addWidget(self.n_entry)

        self.A_label = QLabel("Elementele matricei A (separate prin spațiu sau rand):")
        layout.addWidget(self.A_label)
        self.A_entry = QTextEdit()
        layout.addWidget(self.A_entry)

        self.b_label = QLabel("Elementele vectorului b (separate prin spatiu):")
        layout.addWidget(self.b_label)
        self.b_entry = QLineEdit()
        layout.addWidget(self.b_entry)

        self.method_label = QLabel("Metoda pentru calculul descompunerii LU (0 pentru metoda initiala sau 1 pentru metoda noua):")
        layout.addWidget(self.method_label)
        self.method_entry = QLineEdit()
        layout.addWidget(self.method_entry)

        self.error_label = QLabel("Eroarea dorita pentru calcul:")
        layout.addWidget(self.error_label)
        self.error_entry = QLineEdit()
        layout.addWidget(self.error_entry)

        self.results_button = QPushButton("Afiseaza rezultatele!")
        self.results_button.clicked.connect(self.show_results)
        layout.addWidget(self.results_button)

        self.setLayout(layout)
        self.setWindowTitle("Calculator LU")
        self.show()

    def show_results(self):
        n = int(self.n_entry.text())
        A_text = self.A_entry.toPlainText()
        A_lines = A_text.split('\n')
        A = np.array([list(map(float, line.split())) for line in A_lines])
        b = np.array(list(map(float, self.b_entry.text().split())))
        method = int(self.method_entry.text())
        error = float(self.error_entry.text())

        L, U = crout(A)
        det_A = determinant_from_lu(L, U)
        x_approx = solve_approximate(A, b, method)
        residual_norm = check_solution(A, b, x_approx)
        xLIB = np.linalg.solve(A, b)
        norm_diff_x = np.linalg.norm(x_approx - xLIB, ord=2)
        inverse_A = np.linalg.inv(A)
        A_inv_b = np.dot(inverse_A, b)
        norm_diff_A_inv_b = np.linalg.norm(x_approx - A_inv_b, ord=2)
        L_vector, U_vector = lu_decomposition_vector(A)

        if method == 0:
            result_text = f"\nMatricea L (superioara triunghiulara):\n{L}"
            result_text += f"\n\nMatricea U (inferioara triunghiulara):\n{U}"
            result_text += f"\n\nDeterminantul matricei A este: {det_A}"
        elif method == 1:
            result_text = f"\nMatricea L (superioara triunghiulara) sub forma de vector:\n{L_vector}"
            result_text += f"\n\nMatricea U (inferioara triunghiulara) sub forma de vector:\n{U_vector}"
        result_text += f"\n\nSolutia aproximativa a sistemului Ax = b este: {x_approx}"
        result_text += f"\n\nNorma Euclidiana a reziduului ||A*x - b||_2: {residual_norm}"
        if residual_norm < error:
            result_text += f"\nNorma reziduului este mai mică decat {error}"
        else:
            result_text += f"\nNorma reziduului NU este mai mică decat {error}"
        result_text += f"\n\nSoluția sistemului Ax = b este: {xLIB}"
        result_text += f"\n\nNorma Euclidiana ||xLU - xLIB||_2 este: {norm_diff_x}"
        result_text += f"\n\nInversa matricei A este: {inverse_A}"
        result_text += f"\n\nNorma Euclidiana ||xLU - A^-1*b||_2 este: {norm_diff_A_inv_b}"

        QMessageBox.information(self, "Rezultate", result_text)


if __name__ == '__main__':
    app = QApplication([])
    lu_app = LUApp()
    app.exec_()
