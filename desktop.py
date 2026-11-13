import sys
import numpy as np
from scipy.special import jv, jn_zeros
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QLabel, QLineEdit,
                             QPushButton, QTabWidget, QMessageBox, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class WaveguideCalculator:
    """Класс для численного решения методом прогонки"""

    def __init__(self, R, L, lam, n_ref, A_amp, c_frac):
        self.R = R
        self.L = L
        self.lam = lam
        self.n_ref = n_ref
        self.A_amp = A_amp
        self.c_frac = c_frac

    # 1. Получение коэффициентов a, b, c
    def get_matrix_coeffs(self, I, hr, hz):
        k = 2 * np.pi / self.lam
        gamma = 1j * hz / (2 * k * self.n_ref * hr ** 2)

        # Массивы размера I
        a = np.zeros(I, dtype=np.complex128)
        b = np.zeros(I, dtype=np.complex128)
        c = np.zeros(I, dtype=np.complex128)

        # i = 0 (первая строка матрицы, соответствует a_1 = 0)
        a[0] = 0
        b[0] = 1 - 4 * gamma
        c[0] = 4 * gamma

        # i = 1...I-1
        for i in range(1, I):
            a[i] = gamma
            # b[i] = 1 - 2y - 1/i
            b[i] = 1 - 2 * gamma - gamma / i
            # c[i] = y * (1 + 1/i)
            c[i] = gamma * (1 + 1 / i)

        # Граничное условие
        c[I - 1] = 0
        return a, b, c

    # 2. Получение прогоночных коэффициентов
    def get_sweep_coeffs(self, a, b, c, f):
        n = len(b)
        # Массивы на 1 длиннее: элемент n - фиктивный узел (u_I = 0).
        # Т.к. c[n-1] = 0, значения b_sweep[n], f_sweep[n] не влияют на результат,
        # а на шаге i = n получаются b'_{n-1} = b_{n-1}/a_{n-1}, f'_{n-1} = f_{n-1}/a_{n-1}
        b_sweep = np.ones(n + 1, dtype=np.complex128)
        f_sweep = np.zeros(n + 1, dtype=np.complex128)

        for i in range(n, 1, -1):
            # b'_{n-1} = (b'_n * b_{n-1} - c_{n-1}) / (a_{n-1} * b'_n)
            b_sweep[i - 1] = (b_sweep[i] * b[i - 1] - c[i - 1]) / (a[i - 1] * b_sweep[i])

            # f'_{n-1} = (b'_n * f_{n-1} - c_{n-1} * f'_n) / (a_{n-1} * b'_n)
            f_sweep[i - 1] = (b_sweep[i] * f[i - 1] - c[i - 1] * f_sweep[i]) / (a[i - 1] * b_sweep[i])

        # ОТДЕЛЬНЫЕ ФОРМУЛЫ ДЛЯ ПЕРВОГО ЭЛЕМЕНТА (b'_1 и f'_1)
        b_sweep[0] = b[0] - c[0] / b_sweep[1]
        f_sweep[0] = f[0] - c[0] * f_sweep[1] / b_sweep[1]

        return b_sweep[:n], f_sweep[:n]

    # 3. Обратная прогонка
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

        # Начальное условие psi(r) = 12, при r in [0, 0.1R], иначе 0
        for i in range(I):
            if r[i] <= self.c_frac * self.R:
                u[0, i] = self.A_amp
            else:
                u[0, i] = 0.0

        # Получаем коэффициенты
        a, b, c = self.get_matrix_coeffs(I, hr, hz)

        for k in range(1, K + 1):
            f = u[k - 1, :I]

            b_sweep, f_sweep = self.get_sweep_coeffs(a, b, c, f)
            u_k = self.backward_substitution(b_sweep, f_sweep)

            u[k, :I] = u_k
            u[k, I] = 0.0

        return r, z, u, hr, hz


class FourierBesselCalculator:
    """Аналитическое решение: ряд Фурье-Бесселя (из предыдущей версии программы)."""

    def __init__(self, R, L, lam, n_ref, A, c_frac, members_count):
        self.R = R
        self.L = L
        self.lam = lam
        self.n_ref = n_ref
        self.A = A
        self.c_frac = c_frac
        self.c = c_frac * R
        self.members_count = max(int(members_count), 1)

        self.mu_m = jn_zeros(0, self.members_count)

    def calculate_u_range(self, r, z, n_start, n_end):
        """
        Вычисляет сумму членов ряда в интервале от n_start до n_end включительно.
        Использует 1-базовую индексацию членов ряда для удобства.
        """
        u_val = np.zeros(np.broadcast(r, z).shape, dtype=np.complex128)
        start_idx = max(int(n_start) - 1, 0)
        end_idx = min(int(n_end), self.members_count)

        if start_idx >= end_idx:
            return u_val

        for i in range(start_idx, end_idx):
            mu = self.mu_m[i]
            amplitude = (2 * self.A * self.c_frac) / (mu * jv(1, mu) ** 2) * jv(1, mu * self.c_frac)
            phase = np.exp(1j * (mu ** 2 * self.lam) / (4 * np.pi * self.n_ref * self.R ** 2) * z)
            bessel_radial = jv(0, mu * r / self.R)

            u_val += amplitude * phase * bessel_radial

        return u_val

    def calculate_u_n_terms(self, r, z, n_terms):
        """Вычисляет поле U(r, z) для заданного количества членов ряда (от 1 до n_terms)."""
        return self.calculate_u_range(r, z, 1, n_terms)


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
        self.setup_analytical_tabs()
        self.setup_convergence_tabs()

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


    # ===================== Аналитическое решение (ряд Фурье-Бесселя) =====================
    def setup_analytical_tabs(self):
        # Вкладка 3: Аналитика, график от r
        self.tab_an_r = QWidget()
        layout_an_r = QHBoxLayout(self.tab_an_r)
        controls_an_r = QWidget()
        controls_an_r.setFixedWidth(300)
        form_an_r = QFormLayout(controls_an_r)

        self.an_r_z_vals = QLineEdit('0.0, 0.5, 1.0, 1.5, 2.0')
        self.an_r_N = QLineEdit('50')
        self.an_r_pts = QLineEdit('1000')
        btn_an_r = QPushButton("Построить график")
        btn_an_r.clicked.connect(self.plot_an_r)

        form_an_r.addRow(QLabel("Z сечения:"), self.an_r_z_vals)
        form_an_r.addRow(QLabel("N членов:"), self.an_r_N)
        form_an_r.addRow(QLabel("Точек:"), self.an_r_pts)
        form_an_r.addRow(btn_an_r)

        self.fig_an_r = Figure()
        self.canvas_an_r = FigureCanvas(self.fig_an_r)

        plot_layout_an_r = QVBoxLayout()
        plot_layout_an_r.addWidget(NavigationToolbar(self.canvas_an_r, self))
        plot_layout_an_r.addWidget(self.canvas_an_r)

        layout_an_r.addWidget(controls_an_r)
        layout_an_r.addLayout(plot_layout_an_r)
        self.tabs.addTab(self.tab_an_r, "Аналитика: от r")

        # Вкладка 4: Аналитика, график от z
        self.tab_an_z = QWidget()
        layout_an_z = QHBoxLayout(self.tab_an_z)
        controls_an_z = QWidget()
        controls_an_z.setFixedWidth(300)
        form_an_z = QFormLayout(controls_an_z)

        self.an_z_r_vals = QLineEdit('0.0, 1.0, 2.0, 3.0, 4.0')
        self.an_z_N = QLineEdit('50')
        self.an_z_pts = QLineEdit('1000')
        btn_an_z = QPushButton("Построить график")
        btn_an_z.clicked.connect(self.plot_an_z)

        form_an_z.addRow(QLabel("R сечения:"), self.an_z_r_vals)
        form_an_z.addRow(QLabel("N членов:"), self.an_z_N)
        form_an_z.addRow(QLabel("Точек:"), self.an_z_pts)
        form_an_z.addRow(btn_an_z)

        self.fig_an_z = Figure()
        self.canvas_an_z = FigureCanvas(self.fig_an_z)

        plot_layout_an_z = QVBoxLayout()
        plot_layout_an_z.addWidget(NavigationToolbar(self.canvas_an_z, self))
        plot_layout_an_z.addWidget(self.canvas_an_z)

        layout_an_z.addWidget(controls_an_z)
        layout_an_z.addLayout(plot_layout_an_z)
        self.tabs.addTab(self.tab_an_z, "Аналитика: от z")

    def plot_an_r(self):
        p = self.get_global_params()
        if not p: return
        try:
            Z_vals = [float(x.strip()) for x in self.an_r_z_vals.text().split(',')]
            N = int(self.an_r_N.text())
            pts = int(self.an_r_pts.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Проверьте локальные параметры графика (аналитика, r).")
            return

        calc = FourierBesselCalculator(p['R'], p['L'], p['lam'], p['n_ref'], p['A_amp'], p['c_frac'], N)
        self.fig_an_r.clear()
        ax = self.fig_an_r.add_subplot(111)
        r_arr = np.linspace(0, p['R'], pts)

        for z in Z_vals:
            U = calc.calculate_u_n_terms(r_arr, z, N)
            ax.plot(r_arr, np.abs(U), label=f'z = {z:.2f}')

        ax.set_title('Аналитическое решение: распределение вдоль r')
        ax.set_xlabel('Радиус r, мкм')
        ax.set_ylabel('|U(r,z)|')
        ax.grid(True)
        ax.legend()
        self.fig_an_r.tight_layout()
        self.canvas_an_r.draw()

    def plot_an_z(self):
        p = self.get_global_params()
        if not p: return
        try:
            R_vals = [float(x.strip()) for x in self.an_z_r_vals.text().split(',')]
            N = int(self.an_z_N.text())
            pts = int(self.an_z_pts.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Проверьте локальные параметры графика (аналитика, z).")
            return

        calc = FourierBesselCalculator(p['R'], p['L'], p['lam'], p['n_ref'], p['A_amp'], p['c_frac'], N)
        self.fig_an_z.clear()
        ax = self.fig_an_z.add_subplot(111)
        z_arr = np.linspace(0, p['L'], pts)

        for r in R_vals:
            U = calc.calculate_u_n_terms(r, z_arr, N)
            ax.plot(z_arr, np.abs(U), label=f'r = {r:.2f}')

        ax.set_title('Аналитическое решение: распределение вдоль z')
        ax.set_xlabel('Продольная координата z, мкм')
        ax.set_ylabel('|U(r,z)|')
        ax.grid(True)
        ax.legend()
        self.fig_an_z.tight_layout()
        self.canvas_an_z.draw()

    # ===================== Сходимость численного решения к аналитическому =====================
    def setup_convergence_tabs(self):
        # Вкладка 5: Сходимость, разрезы по z (график от r)
        self.tab_conv_r = QWidget()
        layout_conv_r = QHBoxLayout(self.tab_conv_r)
        controls_conv_r = QWidget()
        controls_conv_r.setFixedWidth(300)
        form_conv_r = QFormLayout(controls_conv_r)

        self.conv_r_z_vals = QLineEdit('0.0, 0.5, 1.0, 1.5, 2.0')
        self.conv_r_Nr = QLineEdit('100')
        self.conv_r_Nz = QLineEdit('100')
        self.conv_r_N = QLineEdit('200')
        self.conv_r_log = QCheckBox("Логарифмическая шкала по Y")
        btn_conv_r = QPushButton("Построить график")
        btn_conv_r.clicked.connect(self.plot_conv_r)

        self.conv_r_hr_label = QLabel("—")
        self.conv_r_hz_label = QLabel("—")

        form_conv_r.addRow(QLabel("Z сечения:"), self.conv_r_z_vals)
        form_conv_r.addRow(QLabel("Точек по r (числ.):"), self.conv_r_Nr)
        form_conv_r.addRow(QLabel("Точек по z (числ.):"), self.conv_r_Nz)
        form_conv_r.addRow(QLabel("N членов (аналит.):"), self.conv_r_N)
        form_conv_r.addRow(QLabel("Шаг hr (мкм):"), self.conv_r_hr_label)
        form_conv_r.addRow(QLabel("Шаг hz (мкм):"), self.conv_r_hz_label)
        form_conv_r.addRow(self.conv_r_log)
        form_conv_r.addRow(btn_conv_r)

        self.fig_conv_r = Figure()
        self.canvas_conv_r = FigureCanvas(self.fig_conv_r)

        plot_layout_conv_r = QVBoxLayout()
        plot_layout_conv_r.addWidget(NavigationToolbar(self.canvas_conv_r, self))
        plot_layout_conv_r.addWidget(self.canvas_conv_r)

        layout_conv_r.addWidget(controls_conv_r)
        layout_conv_r.addLayout(plot_layout_conv_r)
        self.tabs.addTab(self.tab_conv_r, "Сходимость: от r")

        # Вкладка 6: Сходимость, разрезы по r (график от z)
        self.tab_conv_z = QWidget()
        layout_conv_z = QHBoxLayout(self.tab_conv_z)
        controls_conv_z = QWidget()
        controls_conv_z.setFixedWidth(300)
        form_conv_z = QFormLayout(controls_conv_z)

        self.conv_z_r_vals = QLineEdit('0.0, 1.0, 2.0, 3.0, 4.0')
        self.conv_z_Nr = QLineEdit('100')
        self.conv_z_Nz = QLineEdit('100')
        self.conv_z_N = QLineEdit('200')
        self.conv_z_log = QCheckBox("Логарифмическая шкала по Y")
        btn_conv_z = QPushButton("Построить график")
        btn_conv_z.clicked.connect(self.plot_conv_z)

        self.conv_z_hr_label = QLabel("—")
        self.conv_z_hz_label = QLabel("—")

        form_conv_z.addRow(QLabel("R сечения:"), self.conv_z_r_vals)
        form_conv_z.addRow(QLabel("Точек по r (числ.):"), self.conv_z_Nr)
        form_conv_z.addRow(QLabel("Точек по z (числ.):"), self.conv_z_Nz)
        form_conv_z.addRow(QLabel("N членов (аналит.):"), self.conv_z_N)
        form_conv_z.addRow(QLabel("Шаг hr (мкм):"), self.conv_z_hr_label)
        form_conv_z.addRow(QLabel("Шаг hz (мкм):"), self.conv_z_hz_label)
        form_conv_z.addRow(self.conv_z_log)
        form_conv_z.addRow(btn_conv_z)

        self.fig_conv_z = Figure()
        self.canvas_conv_z = FigureCanvas(self.fig_conv_z)

        plot_layout_conv_z = QVBoxLayout()
        plot_layout_conv_z.addWidget(NavigationToolbar(self.canvas_conv_z, self))
        plot_layout_conv_z.addWidget(self.canvas_conv_z)

        layout_conv_z.addWidget(controls_conv_z)
        layout_conv_z.addLayout(plot_layout_conv_z)
        self.tabs.addTab(self.tab_conv_z, "Сходимость: от z")

    def plot_conv_r(self):
        p = self.get_global_params()
        if not p: return
        try:
            Z_vals = [float(x.strip()) for x in self.conv_r_z_vals.text().split(',')]
            Nr = int(self.conv_r_Nr.text())
            Nz = int(self.conv_r_Nz.text())
            N = int(self.conv_r_N.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Проверьте параметры графика сходимости (от r).")
            return

        I = Nr - 1
        K = Nz - 1
        if I <= 0 or K <= 0:
            QMessageBox.critical(self, "Ошибка", "Количество точек должно быть > 1.")
            return
        if N < 1:
            QMessageBox.critical(self, "Ошибка", "Число членов ряда должно быть >= 1.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            calc_num = WaveguideCalculator(p['R'], p['L'], p['lam'], p['n_ref'], p['A_amp'], p['c_frac'])
            r, z, u, hr, hz = calc_num.solve(I, K)
            calc_an = FourierBesselCalculator(p['R'], p['L'], p['lam'], p['n_ref'], p['A_amp'], p['c_frac'], N)

            self.conv_r_hr_label.setText(f"{hr:.5f}")
            self.conv_r_hz_label.setText(f"{hz:.5f}")

            self.fig_conv_r.clear()
            ax = self.fig_conv_r.add_subplot(111)
            use_log = self.conv_r_log.isChecked()

            for z_val in Z_vals:
                if z_val < 0 or z_val > p['L']:
                    continue
                idx = int(round(z_val / hz))
                idx = min(max(idx, 0), K)
                z_node = z[idx]  # аналитику считаем ровно в узле сетки численного решения

                u_an = calc_an.calculate_u_n_terms(r, z_node, N)
                diff = np.abs(u[idx, :] - u_an)

                label = f'z = {z_node:.3f}, max = {diff.max():.2e}'
                ax.plot(r, np.maximum(diff, 1e-16) if use_log else diff, label=label)

            if use_log:
                ax.set_yscale('log')
            ax.set_title(f'Разность численного и аналитического решений вдоль r (N = {N})')
            ax.set_xlabel('Радиус r, мкм')
            ax.set_ylabel('|U_числ − U_аналит|')
            ax.grid(True)
            ax.legend()
            self.fig_conv_r.tight_layout()
            self.canvas_conv_r.draw()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка расчетов", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def plot_conv_z(self):
        p = self.get_global_params()
        if not p: return
        try:
            R_vals = [float(x.strip()) for x in self.conv_z_r_vals.text().split(',')]
            Nr = int(self.conv_z_Nr.text())
            Nz = int(self.conv_z_Nz.text())
            N = int(self.conv_z_N.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Проверьте параметры графика сходимости (от z).")
            return

        I = Nr - 1
        K = Nz - 1
        if I <= 0 or K <= 0:
            QMessageBox.critical(self, "Ошибка", "Количество точек должно быть > 1.")
            return
        if N < 1:
            QMessageBox.critical(self, "Ошибка", "Число членов ряда должно быть >= 1.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            calc_num = WaveguideCalculator(p['R'], p['L'], p['lam'], p['n_ref'], p['A_amp'], p['c_frac'])
            r, z, u, hr, hz = calc_num.solve(I, K)
            calc_an = FourierBesselCalculator(p['R'], p['L'], p['lam'], p['n_ref'], p['A_amp'], p['c_frac'], N)

            self.conv_z_hr_label.setText(f"{hr:.5f}")
            self.conv_z_hz_label.setText(f"{hz:.5f}")

            self.fig_conv_z.clear()
            ax = self.fig_conv_z.add_subplot(111)
            use_log = self.conv_z_log.isChecked()

            for r_val in R_vals:
                if r_val < 0 or r_val > p['R']:
                    continue
                idx = int(round(r_val / hr))
                idx = min(max(idx, 0), I)
                r_node = r[idx]  # аналитику считаем ровно в узле сетки численного решения

                u_an = calc_an.calculate_u_n_terms(r_node, z, N)
                diff = np.abs(u[:, idx] - u_an)

                label = f'r = {r_node:.3f}, max = {diff.max():.2e}'
                ax.plot(z, np.maximum(diff, 1e-16) if use_log else diff, label=label)

            if use_log:
                ax.set_yscale('log')
            ax.set_title(f'Разность численного и аналитического решений вдоль z (N = {N})')
            ax.set_xlabel('Продольная координата z, мкм')
            ax.set_ylabel('|U_числ − U_аналит|')
            ax.grid(True)
            ax.legend()
            self.fig_conv_z.tight_layout()
            self.canvas_conv_z.draw()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка расчетов", str(e))
        finally:
            QApplication.restoreOverrideCursor()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CourseworkApp()
    window.show()
    sys.exit(app.exec_())