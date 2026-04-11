import sys
import numpy as np
from scipy.special import jv, jn_zeros

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QLabel, QLineEdit,
                             QPushButton, QTabWidget, QMessageBox, QGroupBox,
                             QProgressBar, QTableWidget, QTableWidgetItem)  # Добавлены классы для таблицы
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
        self.mu_m = jn_zeros(0, self.members_count)

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

        for i in range(self.members_count):
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
        left_panel.setFixedWidth(330)
        left_layout = QVBoxLayout(left_panel)

        # Группа физических параметров
        param_group = QGroupBox("Параметры системы")
        param_layout = QFormLayout()

        self.inputs = {}
        # Дефолтные значения из вводных данных + данные для таблицы
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
            '--- ОЦЕНКА ОСТАТКА ---': '---',
            'Кол-во членов ряда 1 (N1)': '40',
            'Кол-во членов ряда 2 (N2)': '50',
            'Z для таблицы': '0.0, 1.0, 2.0',
            'R для таблицы': '0.0, 0.4, 4.0'
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

        # Вкладка 3: Таблица оценки остатка
        self.tab3 = QWidget()
        self.tab3_layout = QVBoxLayout(self.tab3)
        self.table_widget = QTableWidget()

        # Настройка внешнего вида таблицы
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setStyleSheet("""
            QTableWidget { font-size: 14px; }
            QHeaderView::section { font-weight: bold; background-color: #f0f0f0; }
        """)
        self.tab3_layout.addWidget(QLabel("Таблица модуля разности: | U(N1) - U(N2) |"))
        self.tab3_layout.addWidget(self.table_widget)
        self.tabs.addTab(self.tab3, "Оценка остатка")

    def update_plots(self):
        try:
            # Считывание основных параметров
            R = float(self.inputs['R (мкм)'].text())
            L = float(self.inputs['L (макс. z, мкм)'].text())
            lam = float(self.inputs['λ (мкм)'].text())
            n_ref = float(self.inputs['n (пок. преломления)'].text())
            A = float(self.inputs['Амплитуда ψ, r ∈ R'].text())
            c_frac = float(self.inputs['Область задания ψ (доля радиуса)'].text())
            members_count_graph = int(self.inputs['Кол-во членов ряда (N для графиков)'].text())
            points_by_val = int(self.inputs['Число точек на графике'].text())

            z_fixed = [float(z.strip()) for z in self.inputs['Z сечения (через запятую)'].text().split(',')]
            r_fixed = [float(r.strip()) for r in self.inputs['R сечения (через запятую)'].text().split(',')]

            # Считывание параметров для таблицы
            n1 = int(self.inputs['Кол-во членов ряда 1 (N1)'].text())
            n2 = int(self.inputs['Кол-во членов ряда 2 (N2)'].text())
            z_table = [float(z.strip()) for z in self.inputs['Z для таблицы'].text().split(',')]
            r_table = [float(r.strip()) for r in self.inputs['R для таблицы'].text().split(',')]

            # Настройка прогресс-бара
            total_steps = len(z_fixed) + len(r_fixed) + (len(z_table) * len(r_table))
            self.progress_bar.setMaximum(total_steps)
            self.progress_bar.setValue(0)
            current_step = 0

        except ValueError:
            QMessageBox.critical(self, "Ошибка ввода", "Проверьте корректность числовых параметров.")
            return

        # Инициализация калькулятора для графиков
        calculator_graph = FourierBesselCalculator(R, L, lam, n_ref, A, c_frac, members_count_graph)
        # todo: почистить, или выбрать как основную оценку
        calculator_graph2 = FourierBesselCalculator(R, L, lam, n_ref, A, c_frac, members_count_graph + 10) #для оценки остатка ряда

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

        #todo: почистить, или выбрать как основную оценку
        # для оценки остатка ряда
        U_vals_1= calculator_graph.calculate_u(r_arr, 0.5)
        U_vals_2= calculator_graph2.calculate_u(r_arr, 0.5)
        Sn = U_vals_1-U_vals_2
        print(f"{np.abs(np.mean(Sn)):.4e}")
        # конец оценки остатка ряда

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

        # --- Таблица Оценки Остатка ---
        # Инициализация калькуляторов с разным числом мод
        calc_1 = FourierBesselCalculator(R, L, lam, n_ref, A, c_frac, n1)
        calc_2 = FourierBesselCalculator(R, L, lam, n_ref, A, c_frac, n2)

        self.table_widget.setRowCount(len(r_table))
        self.table_widget.setColumnCount(len(z_table))

        # Подписываем оси таблицы
        self.table_widget.setHorizontalHeaderLabels([f"z = {z}" for z in z_table])
        self.table_widget.setVerticalHeaderLabels([f"r = {r}" for r in r_table])

        # Заполнение ячеек
        for i, r_val in enumerate(r_table):
            for j, z_val in enumerate(z_table):
                # Расчет в конкретной точке (r, z)
                u1 = calc_1.calculate_u(r_val, z_val)
                u2 = calc_2.calculate_u(r_val, z_val)

                # Модуль разности комплексных чисел
                diff = np.abs(u1 - u2)

                # Форматируем в экспоненциальный вид (например, 1.23e-04)
                item = QTableWidgetItem(f"{diff:.4e}")
                item.setTextAlignment(sys.modules['PyQt5.QtCore'].Qt.AlignCenter)
                self.table_widget.setItem(i, j, item)

                current_step += 1
                self.progress_bar.setValue(current_step)
                QApplication.processEvents()

        # Подгоняем размер столбцов под содержимое
        self.table_widget.resizeColumnsToContents()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CourseworkApp()
    window.show()
    sys.exit(app.exec_())
