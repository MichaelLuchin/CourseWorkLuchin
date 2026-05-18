import sys
import csv
import math
import numpy as np
from scipy.special import jv, jn_zeros

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QLabel, QLineEdit,
                             QPushButton, QTabWidget, QMessageBox, QGroupBox,
                             QTableWidget, QTableWidgetItem, QFileDialog, QHeaderView)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class FourierBesselCalculator:
    """Класс для вычисления распределения интенсивности и оценок ряда Фурье-Бесселя."""

    def __init__(self, R, L, lam, n_ref, A, c_frac, members_count):
        self.R = R
        self.L = L
        self.lam = lam
        self.n_ref = n_ref
        self.A = A
        self.c_frac = c_frac
        self.c = c_frac * R
        self.members_count = max(int(members_count), 1)

        # Заранее вычисляем корни Бесселя J0(x) = 0
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

    @staticmethod
    def calculate_Mt_theoretical(epsilon, B):
        """
        Вычисляет теоретическое количество членов ряда M_t для заданной точности epsilon
        и максимального количества членов ряда B согласно формуле (34).
        """
        if B < 0.25:
            B = 0.25
        fraction = epsilon / (4.8 * np.sqrt(5))
        sqrt_b = np.sqrt(B - 0.25)

        Mt = 0.75 + (fraction - sqrt_b) ** 2

        return int(np.floor(Mt))


class CourseworkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Распределение интенсивности (Ряд Фурье-Бесселя)")
        self.resize(1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 1. Левая панель: Глобальные параметры системы
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
            'Амплитуда ψ, r ∈ R': QLineEdit('12.0'),
            'Доля радиуса для ψ': QLineEdit('0.1')
        }

        for label, widget in self.global_inputs.items():
            self.global_params.addRow(QLabel(label), widget)

        param_group.setLayout(self.global_params)
        left_layout.addWidget(param_group)
        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # 2. Правая часть: Вкладки
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.setup_tabs()

    def get_global_params(self):
        """Читает общие параметры из левой панели."""
        try:
            return {
                'R': float(self.global_inputs['R (мкм)'].text()),
                'L': float(self.global_inputs['L (макс. z, мкм)'].text()),
                'lam': float(self.global_inputs['λ (мкм)'].text()),
                'n_ref': float(self.global_inputs['n (пок. преломления)'].text()),
                'A_amp': float(self.global_inputs['Амплитуда ψ, r ∈ R'].text()),
                'c_frac': float(self.global_inputs['Доля радиуса для ψ'].text())
            }
        except ValueError:
            QMessageBox.critical(self, "Ошибка ввода", "Некорректные глобальные параметры.")
            return None

    def setup_tabs(self):
        # --- Вкладка 1: График от r ---
        self.tab_r = QWidget()
        layout_r = QHBoxLayout(self.tab_r)
        controls_r = QWidget()
        controls_r.setFixedWidth(250)
        form_r = QFormLayout(controls_r)

        self.r_z_vals = QLineEdit('0.2, 0.5, 1.0, 1.5, 2.0')
        self.r_N = QLineEdit('50')
        self.r_pts = QLineEdit('1000')
        btn_r = QPushButton("Построить график")
        btn_r.clicked.connect(self.plot_r)

        form_r.addRow(QLabel("Z сечения:"), self.r_z_vals)
        form_r.addRow(QLabel("N членов:"), self.r_N)
        form_r.addRow(QLabel("Точек:"), self.r_pts)
        form_r.addRow(btn_r)

        self.fig_r = Figure()
        self.canvas_r = FigureCanvas(self.fig_r)

        plot_layout_r = QVBoxLayout()
        plot_layout_r.addWidget(NavigationToolbar(self.canvas_r, self))
        plot_layout_r.addWidget(self.canvas_r)

        layout_r.addWidget(controls_r)
        layout_r.addLayout(plot_layout_r)
        self.tabs.addTab(self.tab_r, "Распределение от r")

        # --- Вкладка 2: График от z ---
        self.tab_z = QWidget()
        layout_z = QHBoxLayout(self.tab_z)
        controls_z = QWidget()
        controls_z.setFixedWidth(250)
        form_z = QFormLayout(controls_z)

        self.z_r_vals = QLineEdit('0.0, 1.0, 2.0, 3.0, 4.0')
        self.z_N = QLineEdit('50')
        self.z_pts = QLineEdit('1000')
        btn_z = QPushButton("Построить график")
        btn_z.clicked.connect(self.plot_z)

        form_z.addRow(QLabel("R сечения:"), self.z_r_vals)
        form_z.addRow(QLabel("N членов:"), self.z_N)
        form_z.addRow(QLabel("Точек:"), self.z_pts)
        form_z.addRow(btn_z)

        self.fig_z = Figure()
        self.canvas_z = FigureCanvas(self.fig_z)

        plot_layout_z = QVBoxLayout()
        plot_layout_z.addWidget(NavigationToolbar(self.canvas_z, self))
        plot_layout_z.addWidget(self.canvas_z)

        layout_z.addWidget(controls_z)
        layout_z.addLayout(plot_layout_z)
        self.tabs.addTab(self.tab_z, "Распределение от z")

        # --- Вкладка 3: Качество оценки (Таблица) ---
        self.tab_quality = QWidget()
        layout_q = QVBoxLayout(self.tab_quality)

        controls_q = QHBoxLayout()
        self.q_pt = QLineEdit('4.0, 2.0')
        self.q_B = QLineEdit('1000')
        self.q_eps = QLineEdit('1e-02, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10')
        self.q_step_Me = QLineEdit('1')

        btn_q_calc = QPushButton("Рассчитать таблицу")
        btn_q_calc.clicked.connect(self.calc_quality)
        btn_q_save = QPushButton("Экспорт в CSV")
        btn_q_save.clicked.connect(self.save_csv)

        controls_q.addWidget(QLabel("Точка (r, z):"))
        controls_q.addWidget(self.q_pt)
        controls_q.addWidget(QLabel("Предельное B:"))
        controls_q.addWidget(self.q_B)
        controls_q.addWidget(QLabel("Список ε:"))
        controls_q.addWidget(self.q_eps)
        controls_q.addWidget(QLabel("Шаг поиска Me:"))
        controls_q.addWidget(self.q_step_Me)
        controls_q.addWidget(btn_q_calc)
        controls_q.addWidget(btn_q_save)

        self.table_q = QTableWidget()
        self.table_q.setEditTriggers(QTableWidget.NoEditTriggers)

        layout_q.addLayout(controls_q)
        layout_q.addWidget(self.table_q)
        self.tabs.addTab(self.tab_quality, "Качество оценки остатка")

        # --- Вкладка 4: Сходимость ряда (График от N) ---
        self.tab_conv = QWidget()
        layout_c = QHBoxLayout(self.tab_conv)
        controls_c = QWidget()
        controls_c.setFixedWidth(250)
        form_c = QFormLayout(controls_c)

        self.c_points = QLineEdit('(0.0, 0.0), (2.0, 1.0), (4.0, 2.0)')
        self.c_start_N = QLineEdit('1')
        self.c_max_N = QLineEdit('150')
        self.c_step_N = QLineEdit('2')
        btn_conv = QPushButton("Построить сходимость")
        btn_conv.clicked.connect(self.plot_convergence)

        form_c.addRow(QLabel("Точки (r, z):"), self.c_points)
        form_c.addRow(QLabel("Стартовое N членов:"), self.c_start_N)
        form_c.addRow(QLabel("Макс. N членов:"), self.c_max_N)
        form_c.addRow(QLabel("Шаг по N:"), self.c_step_N)
        form_c.addRow(btn_conv)

        self.fig_conv = Figure()
        self.canvas_conv = FigureCanvas(self.fig_conv)

        plot_layout_c = QVBoxLayout()
        plot_layout_c.addWidget(NavigationToolbar(self.canvas_conv, self))
        plot_layout_c.addWidget(self.canvas_conv)

        layout_c.addWidget(controls_c)
        layout_c.addLayout(plot_layout_c)
        self.tabs.addTab(self.tab_conv, "Сходимость ряда")

    # --- Обработчики графиков ---
    def plot_r(self):
        p = self.get_global_params()
        if not p: return
        try:
            Z_vals = [float(x.strip()) for x in self.r_z_vals.text().split(',')]
            N = int(self.r_N.text())
            pts = int(self.r_pts.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Проверьте локальные параметры графика R.")
            return

        calc = FourierBesselCalculator(p['R'], p['L'], p['lam'], p['n_ref'], p['A_amp'], p['c_frac'], N)
        self.fig_r.clear()
        ax = self.fig_r.add_subplot(111)
        r_arr = np.linspace(0, p['R'], pts)

        for z in Z_vals:
            U = calc.calculate_u_n_terms(r_arr, z, N)
            ax.plot(r_arr, np.abs(U), label=f'z = {z:.2f}')

        ax.set_title('Распределение вдоль r')
        ax.set_xlabel('Радиус r, мкм')
        ax.set_ylabel('|U(r,z)|')
        ax.grid(True)
        ax.legend()
        self.fig_r.tight_layout()
        self.canvas_r.draw()

    def plot_z(self):
        p = self.get_global_params()
        if not p: return
        try:
            R_vals = [float(x.strip()) for x in self.z_r_vals.text().split(',')]
            N = int(self.z_N.text())
            pts = int(self.z_pts.text())
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Проверьте локальные параметры графика Z.")
            return

        calc = FourierBesselCalculator(p['R'], p['L'], p['lam'], p['n_ref'], p['A_amp'], p['c_frac'], N)
        self.fig_z.clear()
        ax = self.fig_z.add_subplot(111)
        z_arr = np.linspace(0, p['L'], pts)

        for r in R_vals:
            U = calc.calculate_u_n_terms(r, z_arr, N)
            ax.plot(z_arr, np.abs(U), label=f'r = {r:.2f}')

        ax.set_title('Распределение вдоль z')
        ax.set_xlabel('Продольная координата z, мкм')
        ax.set_ylabel('|U(r,z)|')
        ax.grid(True)
        ax.legend()
        self.fig_z.tight_layout()
        self.canvas_z.draw()

    def plot_convergence(self):
        p = self.get_global_params()
        if not p: return
        try:
            raw_pts = eval(f"[{self.c_points.text()}]")
            points = [(float(pt[0]), float(pt[1])) for pt in raw_pts]
            start_N = max(int(self.c_start_N.text()), 1)
            max_N = int(self.c_max_N.text())
            step_N = max(int(self.c_step_N.text()), 1)
        except Exception:
            QMessageBox.critical(self, "Ошибка", "Проверьте формат точек (r, z) или параметры N.")
            return

        if start_N > max_N:
            QMessageBox.warning(self, "Предупреждение", "Стартовое N не может быть больше максимального N.")
            return

        N_vals = np.arange(start_N, max_N + 1, step_N)
        calc = FourierBesselCalculator(p['R'], p['L'], p['lam'], p['n_ref'], p['A_amp'], p['c_frac'], max_N + 10)

        self.fig_conv.clear()
        ax = self.fig_conv.add_subplot(111)

        for r_val, z_val in points:
            u_amplitudes = []

            current_u = calc.calculate_u_range(r_val, z_val, 1, N_vals[0])
            u_amplitudes.append(np.abs(current_u))

            prev_n = N_vals[0]
            for n in N_vals[1:]:
                current_u += calc.calculate_u_range(r_val, z_val, prev_n + 1, n)
                u_amplitudes.append(np.abs(current_u))
                prev_n = n

            ax.plot(N_vals, u_amplitudes, marker='.', linestyle='-', label=f'r={r_val}, z={z_val}')

        ax.set_title('Сходимость амплитуды ряда от числа членов N')
        ax.set_xlabel('Количество членов ряда N')
        ax.set_ylabel('|U(r,z)|')
        ax.grid(True)
        ax.legend()
        self.fig_conv.tight_layout()
        self.canvas_conv.draw()

    # --- Обработчик вкладки Качество оценки ---
    def calc_quality(self):
        p = self.get_global_params()
        if not p: return
        try:
            r_val, z_val = [float(x.strip()) for x in self.q_pt.text().split(',')]
            B = int(self.q_B.text())
            eps_str_list = [x.strip() for x in self.q_eps.text().split(',')]
            eps_list = [float(x) for x in eps_str_list]
            step_Me = max(int(self.q_step_Me.text()), 1)
        except ValueError:
            QMessageBox.critical(self, "Ошибка", "Неверный формат параметров таблицы.")
            return

        max_theoretical_mt = 100
        for eps in eps_list:
            max_theoretical_mt = max(max_theoretical_mt, FourierBesselCalculator.calculate_Mt_theoretical(eps, B))

        calc = FourierBesselCalculator(p['R'], p['L'], p['lam'], p['n_ref'], p['A_amp'], p['c_frac'],
                                       max_theoretical_mt + 100)

        self.table_q.setColumnCount(len(eps_list))
        self.table_q.setRowCount(4)
        self.table_q.setHorizontalHeaderLabels(eps_str_list)
        self.table_q.setVerticalHeaderLabels(['Mt', 'Me', 'Mt - Me', '|S_Mt - S_Me|'])
        self.table_q.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for col, eps in enumerate(eps_list):
            Mt = FourierBesselCalculator.calculate_Mt_theoretical(eps, B)

            # Разность полей |S_Mt - S_m| — это модуль суммы членов от m + 1 до Mt.
            Me = Mt
            for m in range(Mt - step_Me, 0, -step_Me):
                S_diff = calc.calculate_u_range(r_val, z_val, m + 1, Mt)
                diff = np.abs(S_diff)

                print("m: ", m)
                print("eps: ", eps)

                if diff < eps:
                    Me = m
                else:
                    break

            if Me == Mt:
                actual_diff = 0.0
            else:
                actual_diff = np.abs(calc.calculate_u_range(r_val, z_val, Me + 1, Mt))

            self.table_q.setItem(0, col, QTableWidgetItem(str(Mt)))
            self.table_q.setItem(1, col, QTableWidgetItem(str(Me)))
            self.table_q.setItem(2, col, QTableWidgetItem(str(Mt - Me)))
            self.table_q.setItem(3, col, QTableWidgetItem(f"{actual_diff:.6e}"))

    def save_csv(self):
        if self.table_q.rowCount() == 0:
            QMessageBox.warning(self, "Пусто", "Сначала рассчитайте таблицу.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", "quality_evaluation.csv", "CSV Files (*.csv)")
        if not path: return

        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                headers = ["Параметр"] + [self.table_q.horizontalHeaderItem(i).text() for i in
                                          range(self.table_q.columnCount())]
                writer.writerow(headers)

                for row in range(self.table_q.rowCount()):
                    row_data = [self.table_q.verticalHeaderItem(row).text()]
                    for col in range(self.table_q.columnCount()):
                        item = self.table_q.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)

            QMessageBox.information(self, "Успех", f"Данные сохранены в файл:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CourseworkApp()
    window.show()
    sys.exit(app.exec_())