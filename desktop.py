import sys
import numpy as np
from scipy.special import jv, jn_zeros

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QLabel, QLineEdit,
                             QPushButton, QTabWidget, QMessageBox, QGroupBox,
                             QProgressBar)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class FourierBesselCalculator:
    """
    Класс инкапсулирует физические параметры системы и логику
    вычисления распределения интенсивности света на основе ряда Фурье-Бесселя.
    """

    def __init__(self, R, L, lam, n_ref, A, c_frac, members_count):
        self.R = R
        self.L = L
        self.lam = lam
        self.n_ref = n_ref
        self.A = A
        self.c_frac = c_frac
        self.c = c_frac * R
        self.members_count = members_count

        # Предрасчет корней функции Бесселя для ускорения
        self.mu_m = jn_zeros(0, self.members_count + 1)

    def calculate_u(self, r, z):
        """
        Вычисляет поле U(r, z) согласно итоговому ряду
        """
        if isinstance(r, np.ndarray):
            u_val = np.zeros_like(r, dtype=np.complex128)
        elif isinstance(z, np.ndarray):
            u_val = np.zeros_like(z, dtype=np.complex128)
        else:
            u_val = 0.0 + 0.0j

        for i in range(1, self.members_count):
            mu = self.mu_m[i]

            # Коэффициент моды с учетом аналитического интеграла
            amplitude = (2 * self.A * self.c) / (self.R * mu * jv(1, mu) ** 2) * jv(1, mu * self.c / self.R)

            # Фазовый множитель и радиальная часть
            phase = np.exp(1j * (mu ** 2 * self.lam) / (4 * np.pi * self.n_ref * self.R ** 2) * z)
            bessel_radial = jv(0, mu * r / self.R)

            u_val += amplitude * phase * bessel_radial

        return u_val


class CourseworkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распределение интенсивности (Ряд Фурье-Бесселя)")
        self.resize(1150, 800)

        # Основной виджет
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Левая панель (ввод параметров)
        left_panel = QWidget()
        left_panel.setFixedWidth(340)
        left_layout = QVBoxLayout(left_panel)

        # Группа физических параметров
        param_group = QGroupBox("Параметры системы")
        param_layout = QFormLayout()

        self.inputs = {}
        # Дефолтные значения из вводных данных + данные для сходимости
        default_values = {
            'R (мкм)': '4.0',
            'L (макс. z, мкм)': '2.0',
            'λ (мкм)': '1.0',
            'n (пок. преломления)': '1.0',
            'Амплитуда ψ, r ∈ R': '12.0',
            'Область задания ψ (доля радиуса)': '0.1',
            'Кол-во членов ряда (N для графиков)': '50',
            'Z сечения (через запятую)': '0.2, 0.5, 1.0, 1.5, 2.0',
            'R сечения (через запятую)': '0.2, 1.0, 2.0, 3.0, 4.0',
            'Число точек на графике': '2000',
            '--- ИССЛЕДОВАНИЕ СХОДИМОСТИ ---': '---',
            'Точки (r, z) через точку с запятой': '0.0, 1.0; 0.4, 2.0',
            'Начальное кол-во членов (A)': '10',
            'Конечное кол-во членов (B)': '100',
            'Шаг по кол-ву членов': '5'
        }

        for label_text, default_val in default_values.items():
            if default_val == '---':
                # Создаем визуальный разделитель
                lbl = QLabel(label_text)
                lbl.setStyleSheet("font-weight: bold; color: gray; margin-top: 10px;")
                param_layout.addRow(lbl, QLabel(""))
                continue

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

        # ПРОГРЕСС-БАР
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(sys.modules['PyQt5.QtCore'].Qt.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid grey; border-radius: 5px; text-align: center; }
            QProgressBar::chunk { background-color: #05B8CC; width: 20px; }
        """)
        left_layout.addWidget(self.progress_bar)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Вкладки
        self.setup_tabs()
        self.update_plots()

    def setup_tabs(self):
        # Вкладка 1: График от r
        self.tab1 = QWidget()
        self.tab1_layout = QVBoxLayout(self.tab1)
        self.fig1 = Figure()
        self.canvas1 = FigureCanvas(self.fig1)
        self.tab1_layout.addWidget(NavigationToolbar(self.canvas1, self))
        self.tab1_layout.addWidget(self.canvas1)
        self.tabs.addTab(self.tab1, "График от r (z = const)")

        # Вкладка 2: График от z
        self.tab2 = QWidget()
        self.tab2_layout = QVBoxLayout(self.tab2)
        self.fig2 = Figure()
        self.canvas2 = FigureCanvas(self.fig2)
        self.tab2_layout.addWidget(NavigationToolbar(self.canvas2, self))
        self.tab2_layout.addWidget(self.canvas2)
        self.tabs.addTab(self.tab2, "График от z (r = const)")

        # Вкладка 3: График сходимости
        self.tab3 = QWidget()
        self.tab3_layout = QVBoxLayout(self.tab3)
        self.fig3 = Figure()
        self.canvas3 = FigureCanvas(self.fig3)
        self.tab3_layout.addWidget(NavigationToolbar(self.canvas3, self))
        self.tab3_layout.addWidget(self.canvas3)
        self.tabs.addTab(self.tab3, "Сходимость ряда")

    def update_plots(self):
        try:
            # Считывание основных параметров
            R = float(self.inputs['R (мкм)'].text())
            L = float(self.inputs['L (макс. z, мкм)'].text())
            lam = float(self.inputs['λ (мкм)'].text())
            n_ref = float(self.inputs['n (пок. преломления)'].text())
            A_amp = float(self.inputs['Амплитуда ψ, r ∈ R'].text())
            c_frac = float(self.inputs['Область задания ψ (доля радиуса)'].text())
            members_count_graph = int(self.inputs['Кол-во членов ряда (N для графиков)'].text())
            points_by_val = int(self.inputs['Число точек на графике'].text())

            z_fixed = [float(z.strip()) for z in self.inputs['Z сечения (через запятую)'].text().split(',')]
            r_fixed = [float(r.strip()) for r in self.inputs['R сечения (через запятую)'].text().split(',')]

            # Считывание параметров для 3-го графика (сходимости)
            points_str = self.inputs['Точки (r, z) через точку с запятой'].text()
            conv_points = []
            for pt in points_str.split(';'):
                if pt.strip():
                    r_str, z_str = pt.split(',')
                    conv_points.append((float(r_str.strip()), float(z_str.strip())))

            A_terms = int(self.inputs['Начальное кол-во членов (A)'].text())
            B_terms = int(self.inputs['Конечное кол-во членов (B)'].text())
            step_terms = int(self.inputs['Шаг по кол-ву членов'].text())

            # Массив значений N для исследования
            N_vals = list(range(A_terms, B_terms + 1, step_terms))

            # Настройка прогресс-бара
            total_steps = len(z_fixed) + len(r_fixed) + (len(conv_points) * len(N_vals))
            self.progress_bar.setMaximum(total_steps)
            self.progress_bar.setValue(0)
            current_step = 0

        except ValueError:
            QMessageBox.critical(self, "Ошибка ввода", "Проверьте корректность числовых параметров.")
            return

        # Инициализация калькулятора для основных графиков
        calculator_graph = FourierBesselCalculator(R, L, lam, n_ref, A_amp, c_frac, members_count_graph)

        # --- График 1 ---
        self.fig1.clear()
        ax1 = self.fig1.add_subplot(111)
        r_arr = np.linspace(0, R, points_by_val)
        for z in z_fixed:
            if 0 <= z <= L:
                U_vals = calculator_graph.calculate_u(r_arr, z)
                ax1.plot(r_arr, np.abs(U_vals), label=f'z = {z:.2f}')
            current_step += 1
            self.progress_bar.setValue(current_step)
            QApplication.processEvents()

        ax1.set_xlabel('Радиус r, мкм')
        ax1.set_ylabel('|U(r,z)|')
        ax1.set_title('Распределение вдоль r')
        ax1.set_xlim(0, R)
        ax1.legend()
        ax1.grid(True)
        self.fig1.tight_layout()
        self.canvas1.draw()

        # --- График 2 ---
        self.fig2.clear()
        ax2 = self.fig2.add_subplot(111)
        z_arr = np.linspace(0, L, points_by_val)
        for r in r_fixed:
            if 0 <= r <= R:
                U_vals = calculator_graph.calculate_u(r, z_arr)
                ax2.plot(z_arr, np.abs(U_vals), label=f'r = {r:.2f}')
            current_step += 1
            self.progress_bar.setValue(current_step)
            QApplication.processEvents()

        ax2.set_xlabel('Продольная координата z, мкм')
        ax2.set_ylabel('|U(r,z)|')
        ax2.set_title('Распределение вдоль z')
        ax2.set_xlim(0, L)
        ax2.legend()
        ax2.grid(True)
        self.fig2.tight_layout()
        self.canvas2.draw()

        # --- График 3: Сходимость ---
        self.fig3.clear()
        ax3 = self.fig3.add_subplot(111)

        print("\n--- Данные для графика сходимости ---")
        for r_val, z_val in conv_points:
            u_results = []
            print(f"\nТочка: r = {r_val}, z = {z_val}")

            for n_terms in N_vals:
                # Инициализируем новый экземпляр для каждого шага N
                calc = FourierBesselCalculator(R, L, lam, n_ref, A_amp, c_frac, n_terms)
                u_val = calc.calculate_u(r_val, z_val)

                # Считаем модуль амплитуды
                abs_u = np.abs(u_val)
                u_results.append(abs_u)

                # Выводим текущую точку в консоль
                print(f"  N = {n_terms:3d}  |  |U| = {abs_u:.8f}")

                current_step += 1
                self.progress_bar.setValue(current_step)
                QApplication.processEvents()

            # Строим кривую для конкретной точки с маркерами, чтобы было видно шаги
            ax3.plot(N_vals, u_results, marker='.', label=f'r = {r_val}, z = {z_val}')

        ax3.set_xlabel('Количество членов ряда')
        ax3.set_ylabel('|U(r,z)|')
        ax3.set_title('Оценка сходимости остатка ряда')
        ax3.legend()
        ax3.grid(True)
        self.fig3.tight_layout()
        self.canvas3.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CourseworkApp()
    window.show()
    sys.exit(app.exec_())