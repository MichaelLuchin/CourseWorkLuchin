import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QLabel, QLineEdit,
                             QPushButton, QTabWidget, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class WaveguideCalculator:
    """Класс для численного решения методом прогонки (строго по формулам заказчика)."""

    def __init__(self, R, L, lam, n_ref, A_amp, c_frac):
        self.R = R
        self.L = L
        self.lam = lam
        self.n_ref = n_ref
        self.A_amp = A_amp
        self.c_frac = c_frac

    # 1. ОТДЕЛЬНАЯ ФУНКЦИЯ: Получение коэффициентов a, b, c
    def get_matrix_coeffs(self, I, hr, hz):
        k = 2 * np.pi / self.lam
        mu = 1j * hz / (2 * k * self.n_ref)

        # Массивы размера I
        a = np.zeros(I, dtype=np.complex128)
        b = np.zeros(I, dtype=np.complex128)
        c = np.zeros(I, dtype=np.complex128)

        # i = 0 (соответствует a_1 = 0)
        a[0] = 0
        b[0] = 1 - 4 * mu / hr ** 2
        c[0] = 4 * mu / hr ** 2

        # i = 1...I-1
        for i in range(1, I):
            r_i = i * hr
            a[i] = mu / hr ** 2
            b[i] = 1 - mu * (2 / hr ** 2 + 1 / (r_i * hr))
            c[i] = mu * (1 / hr ** 2 + 1 / (r_i * hr))

        # Граничное условие (соответствует c_n = 0)
        c[I - 1] = 0
        return a, b, c

    # 2. Получение прогоночных коэффициентов
    def get_sweep_coeffs(self, a, b, c, f):
        n = len(b)
        b_sweep = np.ones(n, dtype=np.complex128)
        f_sweep = np.ones(n, dtype=np.complex128)

        # Начальное условие
        # На последнем элементе c_n = 0, поэтому b'_n = b_n, f'_n = f_n
        b_sweep[n - 1] = b[n - 1]
        f_sweep[n - 1] = f[n - 1]

        for i in range(n - 1, 1, -1):
            b_sweep[i-1] = (b_sweep[i] * b[i-1] - c[i-1]) / (a[i-1] * b_sweep[i])

            f_sweep[i-1] = (b_sweep[i] * f[i-1] - c[i-1] * f_sweep[i]) / (a[i-1] * b_sweep[i])

        # ОТДЕЛЬНЫЕ ФОРМУЛЫ ДЛЯ ПЕРВОГО ЭЛЕМЕНТА (b'_1 и f'_1)
        b_sweep[0] = b[0] - c[0] / b_sweep[1]
        f_sweep[0] = f[0] - c[0] * f_sweep[1] / b_sweep[1]

        return b_sweep, f_sweep

    # 3. Обратная прогонка (прямой ход решения)
    def backward_substitution(self, b_sweep, f_sweep):
        n = len(b_sweep)
        x = np.zeros(n, dtype=np.complex128)


        # b'_1 * x_1 = f'_1  =>  x_1 = f'_1 / b'_1
        x[0] = f_sweep[0] / b_sweep[0]

        # Для остальных: x_{n-1} + b'_n * x_n = f'_n  =>  x_n = (f'_n - x_{n-1}) / b'_n
        for i in range(1, n):
            x[i] = (f_sweep[i] - x[i - 1]) / b_sweep[i]

        return x

    def solve(self, I, K):
        hr = self.R / I
        hz = self.L / K

        r = np.linspace(0, self.R, I + 1)
        z = np.linspace(0, self.L, K + 1)

        u = np.zeros((K + 1, I + 1), dtype=np.complex128)


        # psi(r) = 12, при r in [0, 0.1R], иначе 0
        for i in range(I):
            if r[i] <= self.c_frac * self.R:
                u[0, i] = self.A_amp
            else:
                u[0, i] = 0.0

        a, b, c = self.get_matrix_coeffs(I, hr, hz)

        for k in range(1, K + 1):
            f = u[k - 1, :I]

            b_sweep, f_sweep = self.get_sweep_coeffs(a, b, c, f)
            u_k = self.backward_substitution(b_sweep, f_sweep)

            u[k, :I] = u_k
            u[k, I] = 0.0

        return r, z, u, hr, hz


class CourseworkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Курсовая работа, Вариант 12 (Метод прогонки)")
        self.resize(1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Левая панель: Глобальные параметры
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)

        param_group = QGroupBox("Глобальные параметры системы")
        self.global_params = QFormLayout()


        self.global_inputs = {
            'R (мкм)': QLineEdit('4.0'),
            'L (макс. z, мкм)': QLineEdit('2.0'),
            'λ (мкм)': QLineEdit('1.0'),
            'n (пок. преломления)': QLineEdit('1.0'),
            'Амплитуда ψ': QLineEdit('12.0'),
            'Доля радиуса ψ': QLineEdit('0.1')
        }

        for label, widget in self.global_inputs.items():
            self.global_params.addRow(QLabel(label), widget)

        param_group.setLayout(self.global_params)
        left_layout.addWidget(param_group)
        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # Правая часть: Вкладки
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.setup_tabs()

    def get_global_params(self):
        try:
            return {
                'R': float(self.global_inputs['R (мкм)'].text()),
                'L': float(self.global_inputs['L (макс. z, мкм)'].text()),
                'lam': float(self.global_inputs['λ (мкм)'].text()),
                'n_ref': float(self.global_inputs['n (пок. преломления)'].text()),
                'A_amp': float(self.global_inputs['Амплитуда ψ'].text()),
                'c_frac': float(self.global_inputs['Доля радиуса ψ'].text())
            }
        except ValueError:
            QMessageBox.critical(self, "Ошибка ввода", "Некорректные глобальные параметры.")
            return None

    def setup_tabs(self):
        # Вкладка 1: График от r
        self.tab_r = QWidget()
        layout_r = QHBoxLayout(self.tab_r)
        controls_r = QWidget()
        controls_r.setFixedWidth(300)
        form_r = QFormLayout(controls_r)

        self.r_z_vals = QLineEdit('0.0, 0.5, 1.0, 1.5, 2.0')
        self.r_Nr = QLineEdit('100')
        self.r_Nz = QLineEdit('100')
        btn_r = QPushButton("Построить график")
        btn_r.clicked.connect(self.plot_r)

        self.r_hr_label = QLabel("—")
        self.r_hz_label = QLabel("—")

        form_r.addRow(QLabel("Z сечения:"), self.r_z_vals)
        form_r.addRow(QLabel("Точек по r:"), self.r_Nr)
        form_r.addRow(QLabel("Точек по z:"), self.r_Nz)
        form_r.addRow(QLabel("Шаг hr (мкм):"), self.r_hr_label)
        form_r.addRow(QLabel("Шаг hz (мкм):"), self.r_hz_label)
        form_r.addRow(btn_r)

        self.fig_r = Figure()
        self.canvas_r = FigureCanvas(self.fig_r)

        plot_layout_r = QVBoxLayout()
        plot_layout_r.addWidget(NavigationToolbar(self.canvas_r, self))
        plot_layout_r.addWidget(self.canvas_r)

        layout_r.addWidget(controls_r)
        layout_r.addLayout(plot_layout_r)
        self.tabs.addTab(self.tab_r, "Распределение от r")

        # Вкладка 2: График от z
        self.tab_z = QWidget()
        layout_z = QHBoxLayout(self.tab_z)
        controls_z = QWidget()
        controls_z.setFixedWidth(300)
        form_z = QFormLayout(controls_z)

        self.z_r_vals = QLineEdit('0.0, 1.0, 2.0, 3.0, 4.0')
        self.z_Nr = QLineEdit('100')
        self.z_Nz = QLineEdit('100')
        btn_z = QPushButton("Построить график")
        btn_z.clicked.connect(self.plot_z)

        self.z_hr_label = QLabel("—")
        self.z_hz_label = QLabel("—")

        form_z.addRow(QLabel("R сечения:"), self.z_r_vals)
        form_z.addRow(QLabel("Точек по r:"), self.z_Nr)
        form_z.addRow(QLabel("Точек по z:"), self.z_Nz)
        form_z.addRow(QLabel("Шаг hr (мкм):"), self.z_hr_label)
        form_z.addRow(QLabel("Шаг hz (мкм):"), self.z_hz_label)
        form_z.addRow(btn_z)

        self.fig_z = Figure()
        self.canvas_z = FigureCanvas(self.fig_z)

        plot_layout_z = QVBoxLayout()
        plot_layout_z.addWidget(NavigationToolbar(self.canvas_z, self))
        plot_layout_z.addWidget(self.canvas_z)

        layout_z.addWidget(controls_z)
        layout_z.addLayout(plot_layout_z)
        self.tabs.addTab(self.tab_z, "Распределение от z")

    def plot_r(self):
        p = self.get_global_params()
        if not p: return
        try:
            Z_vals = [float(x.strip()) for x in self.r_z_vals.text().split(',')]
            Nr = int(self.r_Nr.text())
            Nz = int(self.r_Nz.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Проверьте параметры графика R.")
            return

        I = Nr - 1
        K = Nz - 1
        if I <= 0 or K <= 0:
            QMessageBox.critical(self, "Ошибка", "Количество точек должно быть > 1.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            calc = WaveguideCalculator(p['R'], p['L'], p['lam'], p['n_ref'], p['A_amp'], p['c_frac'])
            r, z, u, hr, hz = calc.solve(I, K)

            self.r_hr_label.setText(f"{hr:.5f}")
            self.r_hz_label.setText(f"{hz:.5f}")

            self.fig_r.clear()
            ax = self.fig_r.add_subplot(111)

            for z_val in Z_vals:
                if z_val < 0 or z_val > p['L']:
                    continue
                idx = int(round(z_val / hz))
                idx = min(max(idx, 0), K)
                u_slice = u[idx, :]
                ax.plot(r, np.abs(u_slice), label=f'z = {z_val:.2f}')

            ax.set_title('Распределение поля вдоль r')
            ax.set_xlabel('Радиус r, мкм')
            ax.set_ylabel('|U(r,z)|')
            ax.grid(True)
            ax.legend()
            self.fig_r.tight_layout()
            self.canvas_r.draw()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка расчетов", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def plot_z(self):
        p = self.get_global_params()
        if not p: return
        try:
            R_vals = [float(x.strip()) for x in self.z_r_vals.text().split(',')]
            Nr = int(self.z_Nr.text())
            Nz = int(self.z_Nz.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Проверьте параметры графика Z.")
            return

        I = Nr - 1
        K = Nz - 1
        if I <= 0 or K <= 0:
            QMessageBox.critical(self, "Ошибка", "Количество точек должно быть > 1.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            calc = WaveguideCalculator(p['R'], p['L'], p['lam'], p['n_ref'], p['A_amp'], p['c_frac'])
            r, z, u, hr, hz = calc.solve(I, K)

            self.z_hr_label.setText(f"{hr:.5f}")
            self.z_hz_label.setText(f"{hz:.5f}")

            self.fig_z.clear()
            ax = self.fig_z.add_subplot(111)

            for r_val in R_vals:
                if r_val < 0 or r_val > p['R']:
                    continue
                idx = int(round(r_val / hr))
                idx = min(max(idx, 0), I)
                u_slice = u[:, idx]
                ax.plot(z, np.abs(u_slice), label=f'r = {r_val:.2f}')

            ax.set_title('Распределение поля вдоль z')
            ax.set_xlabel('Продольная координата z, мкм')
            ax.set_ylabel('|U(r,z)|')
            ax.grid(True)
            ax.legend()
            self.fig_z.tight_layout()
            self.canvas_z.draw()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка расчетов", str(e))
        finally:
            QApplication.restoreOverrideCursor()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CourseworkApp()
    window.show()
    sys.exit(app.exec_())