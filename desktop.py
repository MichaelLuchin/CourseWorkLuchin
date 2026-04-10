# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy.special import jv, jn_zeros

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QLabel, QLineEdit,
                             QPushButton, QTabWidget, QMessageBox, QGroupBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class FourierBesselCalculator:
    """
    Класс инкапсулирует физические параметры системы и логику
    вычисления распределения интенсивности света на основе ряда Фурье-Бесселя.
    """

    def __init__(self, R, L, lam, n_ref, A, c_frac, m_modes):
        self.R = R
        self.L = L
        self.lam = lam
        self.n_ref = n_ref
        self.A = A
        self.c_frac = c_frac
        self.c = c_frac * R
        self.m_modes = m_modes

        # Предрасчет корней функции Бесселя для ускорения (вычисляются 1 раз при создании объекта)
        self.mu_m = jn_zeros(0, self.m_modes)

    def calculate_u(self, r, z):
        """
        Вычисляет поле U(r, z).
        r и z могут быть как числами, так и numpy-массивами.
        """
        # Инициализация массива нулей нужной размерности
        if isinstance(r, np.ndarray):
            u_val = np.zeros_like(r, dtype=np.complex128)
        elif isinstance(z, np.ndarray):
            u_val = np.zeros_like(z, dtype=np.complex128)
        else:
            u_val = 0.0 + 0.0j

        for i in range(self.m_modes):
            mu = self.mu_m[i]

            # Интегральный коэффициент Cm
            amplitude = (2 * self.A * self.c) / (self.R * mu * jv(1, mu) ** 2) * jv(1, mu * self.c / self.R)

            # Фазовый множитель
            phase = np.exp(1j * (mu ** 2 * self.lam) / (4 * np.pi * self.n_ref * self.R ** 2) * z)

            # Радиальная функция Бесселя
            bessel_radial = jv(0, mu * r / self.R)

            u_val += amplitude * phase * bessel_radial

        return u_val


class CourseworkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распределение интенсивности света (Ряд Фурье-Бесселя)")
        self.resize(1100, 700)

        # Основной виджет
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Левая панель (ввод параметров)
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)

        # Группа физических параметров
        param_group = QGroupBox("Параметры системы")
        param_layout = QFormLayout()

        self.inputs = {}
        # Задаем дефолтные значения из файла docx
        default_values = {
            'R (мкм)': '4.0',
            'L (макс. z, мкм)': '2.0',
            'λ (мкм)': '1.0',
            'n (пок. преломления)': '1.0',
            'Амплитуда ψ': '12.0',
            'Доля радиуса c (0-1)': '0.1',
            'Число мод (N)': '30',
            'Z сечения (через запятую)': '0.2, 0.5, 1.0, 1.5, 2.0',
            'R сечения (через запятую)': '0.0, 1.0, 2.0, 3.0, 4.0'
        }

        for label_text, default_val in default_values.items():
            line_edit = QLineEdit(default_val)
            self.inputs[label_text] = line_edit
            param_layout.addRow(QLabel(label_text), line_edit)

        param_group.setLayout(param_layout)
        left_layout.addWidget(param_group)

        # Кнопка расчета
        self.calc_btn = QPushButton("Рассчитать")
        self.calc_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.calc_btn.clicked.connect(self.update_plots)
        left_layout.addWidget(self.calc_btn)
        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # Правая панель (графики)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Вкладка 1: U(r) при фиксированных z
        self.tab1 = QWidget()
        self.tab1_layout = QVBoxLayout(self.tab1)
        self.fig1 = Figure()
        self.canvas1 = FigureCanvas(self.fig1)
        self.toolbar1 = NavigationToolbar(self.canvas1, self)
        self.tab1_layout.addWidget(self.toolbar1)
        self.tab1_layout.addWidget(self.canvas1)
        self.tabs.addTab(self.tab1, "График от r (z = const)")

        # Вкладка 2: U(z) при фиксированных r
        self.tab2 = QWidget()
        self.tab2_layout = QVBoxLayout(self.tab2)
        self.fig2 = Figure()
        self.canvas2 = FigureCanvas(self.fig2)
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        self.tab2_layout.addWidget(self.toolbar2)
        self.tab2_layout.addWidget(self.canvas2)
        self.tabs.addTab(self.tab2, "График от z (r = const)")

        # Первоначальная отрисовка
        self.update_plots()

    def update_plots(self):
        try:
            # Считывание параметров
            R = float(self.inputs['R (мкм)'].text())
            L = float(self.inputs['L (макс. z, мкм)'].text())
            lam = float(self.inputs['λ (мкм)'].text())
            n_ref = float(self.inputs['n (пок. преломления)'].text())
            A = float(self.inputs['Амплитуда ψ'].text())
            c_frac = float(self.inputs['Доля радиуса c (0-1)'].text())
            m_modes = int(self.inputs['Число мод (N)'].text())

            z_strs = self.inputs['Z сечения (через запятую)'].text().split(',')
            r_strs = self.inputs['R сечения (через запятую)'].text().split(',')

            z_fixed = [float(z.strip()) for z in z_strs]
            r_fixed = [float(r.strip()) for r in r_strs]

        except ValueError:
            QMessageBox.critical(self, "Ошибка ввода", "Пожалуйста, проверьте корректность введенных чисел.")
            return

        # Инициализация калькулятора с текущими параметрами
        calculator = FourierBesselCalculator(R, L, lam, n_ref, A, c_frac, m_modes)

        Nr = 200
        Nz = 200

        # --- Отрисовка Графика 1 (U от r) ---
        self.fig1.clear()
        ax1 = self.fig1.add_subplot(111)
        r_arr = np.linspace(0, R, Nr)

        for z in z_fixed:
            if z > L or z < 0:
                continue  # Защита от выхода за пределы
            # Вызов метода из нового класса
            U_vals = calculator.calculate_u(r_arr, z)
            ax1.plot(r_arr, np.abs(U_vals), label=f'z = {z:.1f} мкм')

        ax1.set_xlabel('Радиус r, мкм')
        ax1.set_ylabel('|U(r,z)|')
        ax1.set_title('Распределение интенсивности света по радиусу')
        ax1.set_xlim(0, R)
        ax1.legend()
        ax1.grid(True)
        self.fig1.tight_layout()
        self.canvas1.draw()

        # --- Отрисовка Графика 2 (U от z) ---
        self.fig2.clear()
        ax2 = self.fig2.add_subplot(111)
        z_arr = np.linspace(0, L, Nz)

        for r in r_fixed:
            if r > R or r < 0:
                continue  # Защита от выхода за пределы
            # Вызов метода из нового класса
            U_vals = calculator.calculate_u(r, z_arr)
            ax2.plot(z_arr, np.abs(U_vals), label=f'r = {r:.1f} мкм')

        ax2.set_xlabel('Продольная координата z, мкм')
        ax2.set_ylabel('|U(r,z)|')
        ax2.set_title('Изменение интенсивности вдоль оси z')
        ax2.set_xlim(0, L)
        ax2.legend()
        ax2.grid(True)
        self.fig2.tight_layout()
        self.canvas2.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CourseworkApp()
    window.show()
    sys.exit(app.exec_())