import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QTextBrowser, QComboBox, QLineEdit, QPushButton, QLabel,
    QMessageBox, QStackedWidget, QFileDialog
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineProfile, QWebEngineSettings

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# GETELEC imports
# ---------------------------------------------------------
if '__file__' in locals():
    current_dir = os.path.dirname(os.path.abspath(__file__))
else:
    current_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from getelec.potential_barrier import SchottkyPotential
from getelec.band_structure import CustomMetal, SmartMetal, Metal
from getelec.transmission_solver import Noumerov
from getelec.electron_supply import FermiDirac, LogFermiDirac
from getelec.electron_emitter import MetalEmitter


# ---------------------------------------------------------
# CalculateTab
# ---------------------------------------------------------
class CalculateTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

        self.func_menu = QComboBox()
        self.func_menu.addItems(["I-F", "I-T", "TED", "NED"])
        self.func_menu.currentTextChanged.connect(self.update_parameter_fields)

        self.xmin_box = QLineEdit()
        self.xmax_box = QLineEdit()
        self.dx_box = QLineEdit()
        for box in (self.xmin_box, self.xmax_box, self.dx_box):
            box.setFixedHeight(28)

        self.paramA = QLineEdit()
        self.paramB = QLineEdit()
        self.paramC = QLineEdit()
        self.paramD = QLineEdit()
        for box in (self.paramA, self.paramB, self.paramC, self.paramD):
            box.setFixedHeight(28)

        self.paramA.setPlaceholderText("Fermi level (eV)")
        self.paramB.setPlaceholderText("Work Function (eV)")
        self.paramC.setPlaceholderText("Electric Field (V/nm)")
        self.paramD.setPlaceholderText("Temperature (K)")

        self.xscale_menu = QComboBox()
        self.yscale_menu = QComboBox()
        self.xscale_menu.addItems(["linear", "log"])
        self.yscale_menu.addItems(["linear", "log"])

        self.compute_button = QPushButton("Compute")
        self.compute_button.clicked.connect(self.compute_function)

        self.save_data_button = QPushButton("Save Data")
        self.save_fig_button = QPushButton("Save Figure")
        self.save_data_button.clicked.connect(self.save_data)
        self.save_fig_button.clicked.connect(self.save_figure)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Function:"))
        layout.addWidget(self.func_menu)

        layout.addWidget(QLabel("X-range:"))
        layout.addWidget(self.xmin_box)
        layout.addWidget(self.xmax_box)
        layout.addWidget(self.dx_box)

        layout.addWidget(QLabel("Parameters:"))
        layout.addWidget(self.paramA)
        layout.addWidget(self.paramB)
        layout.addWidget(self.paramC)
        layout.addWidget(self.paramD)

        layout.addWidget(QLabel("Axis scales:"))
        layout.addWidget(QLabel("X-axis"))
        layout.addWidget(self.xscale_menu)
        layout.addWidget(QLabel("Y-axis"))
        layout.addWidget(self.yscale_menu)

        layout.addWidget(self.compute_button)

        bottom = QHBoxLayout()
        bottom.addWidget(self.save_data_button)
        bottom.addWidget(self.save_fig_button)
        bottom.addStretch()
        layout.addLayout(bottom)

        layout.addStretch()
        self.setLayout(layout)

        self.update_parameter_fields()
        self.last_x = None
        self.last_y = None

    def update_parameter_fields(self):
        func = self.func_menu.currentText()

        self.paramA.hide()
        self.paramB.hide()
        self.paramC.hide()
        self.paramD.hide()

        if func == "I-F":
            self.paramA.show()
            self.paramB.show()
            self.paramD.show()
            self.xmin_box.setPlaceholderText("F_min (V/nm)")
            self.xmax_box.setPlaceholderText("F_max (V/nm)")
            self.dx_box.setPlaceholderText("delta_F (V/nm)")

        elif func == "I-T":
            self.paramA.show()
            self.paramB.show()
            self.paramC.show()
            self.xmin_box.setPlaceholderText("T_min (K)")
            self.xmax_box.setPlaceholderText("T_max (K)")
            self.dx_box.setPlaceholderText("delta_T (K)")

        elif func in ("TED", "NED"):
            self.paramA.show()
            self.paramB.show()
            self.paramC.show()
            self.paramD.show()
            self.xmin_box.setPlaceholderText("E_min (eV)")
            self.xmax_box.setPlaceholderText("E_max (eV)")
            self.dx_box.setPlaceholderText("delta_E (eV)")

    def generate_x(self):
        try:
            func = self.func_menu.currentText()
            xmin = float(self.xmin_box.text())
            xmax = float(self.xmax_box.text())
            dx = float(self.dx_box.text())
            if dx <= 0:
                raise ValueError

            if func in ("TED", "NED"):
                return Metal(xmin, xmax, dx)
            else:
                return np.arange(xmin, xmax + dx, dx)

        except:
            QMessageBox.warning(self, "Error", "Invalid x-range input")
            return None

    def compute_function(self):
        x = self.generate_x()
        if x is None:
            return

        func = self.func_menu.currentText()

        try:
            if func == "I-F":
                A = float(self.paramA.text())
                B = float(self.paramB.text())
                D = float(self.paramD.text())

                current_density = np.zeros_like(x)
                my_grid = SmartMetal(barrier_width=3.0, supply_threshold=1e-14, energy_resolution=0.01)
                my_solver = Noumerov(x_metal=-1.0, x_vac_plus=10, h=0.001, max_barrier_width=3.0)

                for iii, fff in enumerate(x):
                    my_potential = SchottkyPotential(A, B, fff)
                    my_supply = FermiDirac(A, D)
                    emitter = MetalEmitter(my_potential, my_solver, my_supply, my_grid)
                    current_density[iii] = emitter.calculate_current_density()

                y = current_density
                legend = f"Ef={A} eV, wf={B} eV, T={D} K"
                title = "Emitted Current Density as Function of Field"
                x_label = "Field (V/nm)"
                y_label = "Current Density (A/cm2)"

            elif func == "I-T":
                A = float(self.paramA.text())
                B = float(self.paramB.text())
                C = float(self.paramC.text())

                current_density = np.zeros_like(x)
                my_grid = SmartMetal(barrier_width=3.0, supply_threshold=1e-14, energy_resolution=0.01)
                my_solver = Noumerov(x_metal=-1.0, x_vac_plus=10, h=0.001, max_barrier_width=3.0)

                for iii, ttt in enumerate(x):
                    my_potential = SchottkyPotential(A, B, C)
                    my_supply = FermiDirac(A, ttt)
                    emitter = MetalEmitter(my_potential, my_solver, my_supply, my_grid)
                    current_density[iii] = emitter.calculate_current_density()

                y = current_density
                legend = f"Ef={A} eV, wf={B} eV, F={C} V/nm"
                title = "Emitted Current Density as Function of Temperature"
                x_label = "Temperature (K)"
                y_label = "Current Density (A/cm2)"

            elif func == "NED":
                A = float(self.paramA.text())
                B = float(self.paramB.text())
                C = float(self.paramC.text())
                D = float(self.paramD.text())

                my_potential = SchottkyPotential(A, B, C)
                my_solver = Noumerov(x_metal=-1.0, x_vac_plus=10, h=0.001, max_barrier_width=3)
                my_supply_log = LogFermiDirac(A, D)
                emitter = MetalEmitter(my_potential, my_solver, my_supply_log, x)

                energies, ted = emitter.calculate_normal_energy_distribution()
                x = energies
                y = ted
                legend = f"Ef={A} eV, wf={B} eV, F={C} V/nm, T={D} K"
                title = "Normal Energy Distribution"
                x_label = "Energy (eV)"
                y_label = "Electron counts (A/eV*cm2)"

            elif func == "TED":
                A = float(self.paramA.text())
                B = float(self.paramB.text())
                C = float(self.paramC.text())
                D = float(self.paramD.text())

                my_potential = SchottkyPotential(A, B, C)
                my_solver = Noumerov(x_metal=-1.0, x_vac_plus=10, h=0.001, max_barrier_width=3)
                my_supply = FermiDirac(A, D)
                emitter = MetalEmitter(my_potential, my_solver, my_supply, x)

                energies, ted = emitter.calculate_total_energy_distribution()
                x = energies
                y = ted
                legend = f"Ef={A} eV, wf={B} eV, F={C} V/nm, T={D} K"
                title = "Total Energy Distribution"
                x_label = "Energy (eV)"
                y_label = "Electron counts (A/eV*cm2)"

        except:
            QMessageBox.warning(self, "Error", "Invalid parameter values")
            return

        self.last_x = x
        self.last_y = y

        self.main.figure.clear()
        ax = self.main.figure.add_subplot(111)
        ax.plot(x, y, label=legend)
        ax.legend()
        ax.set_xscale(self.xscale_menu.currentText())
        ax.set_yscale(self.yscale_menu.currentText())
        ax.grid()
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title}")
        self.main.canvas.draw()

    def save_data(self):
        if self.last_x is None:
            QMessageBox.warning(self, "Error", "No data to save")
            return

        directory = QFileDialog.getExistingDirectory(self, "Choose save directory")
        if not directory:
            return

        path = os.path.join(directory, "calculation_data.txt")
        np.savetxt(path, np.column_stack([self.last_x, self.last_y]), header="x y")
        QMessageBox.information(self, "Saved", f"Saved data to:\n{path}")

    def save_figure(self):
        directory = QFileDialog.getExistingDirectory(self, "Choose save directory")
        if not directory:
            return

        path = os.path.join(directory, "calculation_plot.png")
        self.main.figure.savefig(path)
        QMessageBox.information(self, "Saved", f"Saved figure to:\n{path}")


# ---------------------------------------------------------
# DocumentationTab
# ---------------------------------------------------------
class DocumentationTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window

        # --- Sidebar (left) ---
        self.sidebar = QTextBrowser()

        # CRITICAL: Sidebar must NEVER load pages
        self.sidebar.setOpenExternalLinks(True)
        self.sidebar.setOpenLinks(False)

        # When clicking a link, sidebar does NOT navigate
        self.sidebar.anchorClicked.connect(self.on_sidebar_click)

        self.sidebar.setHtml("""
            <div align="center">
                <h2>GETELEC</h2>
            </div>
            <p><b>Authors:</b><br>
               Salvador Barranco Cárceles<br>
               Andreas Kyritsakis<br>
               Anthony Ayari</p>
            <p><b>Version:</b> 3.0.0</p>
            <p><b>Contact:</b>
            <a>s [dot] barranco [dot] carceles [at] gmail [dot] com</a>
            </p>
            <hr>
            <p><b>Documentation:</b></p>
            <ul>
                <li><a href='docs/introduction.html'>Introduction</a></li>
                <li><a href='docs/usage.html'>Usage Guide</a></li>
                <li><a href='docs/getelec.html'>GETELEC Documentation</a></li>
            </ul>
        """)

        # --- Main viewer (right) ---
        self.doc_view = QWebEngineView()

        # --- Layout ---
        layout = QHBoxLayout()
        layout.addWidget(self.sidebar, 1)
        layout.addWidget(self.doc_view, 3)
        self.setLayout(layout)

        # Load default page
        self.load_page("docs/introduction.html")

    def on_sidebar_click(self, url: QUrl):
        # Sidebar stays unchanged
        self.load_page(url.toString())

    def load_page(self, path: str):
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        full_path = os.path.join(base_dir, path)
        self.doc_view.load(QUrl.fromLocalFile(full_path))


# ---------------------------------------------------------
# FitTab
# ---------------------------------------------------------
class FitTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.file_path = None

        self.fit_menu = QComboBox()
        self.fit_menu.addItems(["I-V", "I-T", "TED", "NED"])

        self.xscale_menu = QComboBox()
        self.yscale_menu = QComboBox()
        self.xscale_menu.addItems(["linear", "log"])
        self.yscale_menu.addItems(["linear", "log"])

        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(self.run_fit)

        self.save_data_button = QPushButton("Save Fit Data")
        self.save_fig_button = QPushButton("Save Figure")
        self.save_data_button.clicked.connect(self.save_fit_data)
        self.save_fig_button.clicked.connect(self.save_fit_figure)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Fit model:"))
        layout.addWidget(self.fit_menu)

        layout.addWidget(QLabel("Axis scales:"))
        layout.addWidget(QLabel("X-axis"))
        layout.addWidget(self.xscale_menu)
        layout.addWidget(QLabel("Y-axis"))
        layout.addWidget(self.yscale_menu)

        layout.addWidget(self.fit_button)

        bottom = QHBoxLayout()
        bottom.addWidget(self.save_data_button)
        bottom.addWidget(self.save_fig_button)
        bottom.addStretch()
        layout.addLayout(bottom)

        layout.addStretch()
        self.setLayout(layout)

        self.last_x = None
        self.last_y = None
        self.last_fit = None

    def load_file(self, path):
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, header=None)
            elif path.endswith(".txt"):
                df = pd.read_csv(path, header=None, delim_whitespace=True)
            elif path.endswith(".xlsx") or path.endswith(".xls"):
                df = pd.read_excel(path, header=None)
            else:
                raise ValueError("Unsupported file type")

            if df.shape[1] < 2:
                raise ValueError("File must contain at least 2 columns")

            self.last_x = df.iloc[:, 0].values
            self.last_y = df.iloc[:, 1].values
            self.file_path = path

        except Exception as e:
            print("Error loading file:", e)

    def run_fit(self):
        if self.last_x is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return

        mode = self.fit_menu.currentText()

        try:
            if mode == "I-V":
                mask = self.last_y > 0
                V_data = self.last_x[mask]
                I_data = self.last_y[mask]

                mask = I_data > 0

                V = V_data[mask]
                I = np.log(I_data[mask])

                my_potential = SchottkyPotential()
                my_band = SmartMetal()
                my_solver = Noumerov()
                my_supply = FermiDirac()

                emitter = MetalEmitter(my_potential, my_solver, my_supply, my_band)

                initial_guess = [0.01, 1E-18, 9.5, 4.5, 300] # gamma, emission_area, fermi_level, work_function, temperature
                bounds = ([0.001, 1E-20, 7, 3.5, 200], [0.01, 1E-10, 13, 5.5, 700])

                def model(V_array, gamma, area, ef, wf, temp):

                    results = np.zeros_like(V_array)

                    for i, v in enumerate(V_array):
                        f = v * gamma

                        emitter.update_params(field=f, work_function=wf, fermi=ef, temp=temp)
                    
                        j = emitter.calculate_current_density()

                        results[i] = j * area * 1E9

                    results = np.clip(results,1E-100, None)

                    return np.log(results)

                popt, pcov = curve_fit(model, V, I, p0=initial_guess, bounds=bounds,maxfev=1000000)

                legend = f"Fit param: gamma = {popt[0]:.6f} 1/nm, radius = {1/(5*popt[0]):.2f} nm, area = {popt[1]*1E14:.2f} nm2, ef = {popt[2]:.2f} eV, phi = {popt[3]:.2f} eV, T = {popt[4]:.2f} K"
                x_label = "Voltage (V)"
                y_label = "Current (nA)"

                FERMI_LEVEL = popt[2]
                WORK_FUNCTION = popt[3]
                TEMPERATURE = popt[4]

                electric_field = V * popt[0]
                current_density = np.zeros_like(electric_field)

                my_band = SmartMetal(barrier_width=3.0, supply_threshold=1e-14, energy_resolution=0.01)
                my_solver = Noumerov(x_metal=-1.0, x_vac_plus=10, h=0.001, max_barrier_width=3)
                my_potential = SchottkyPotential(fermi_level=FERMI_LEVEL, work_function=WORK_FUNCTION, electric_field=electric_field)    
                my_supply = FermiDirac(fermi_level=FERMI_LEVEL, temperature=TEMPERATURE)
                emitter = MetalEmitter(my_potential, my_solver, my_supply, my_band)

                for i, f in enumerate(electric_field):
                    emitter.update_params(field=f)
                    current_density[i] = emitter.calculate_current_density()

                y_fit = current_density * popt[1] * 1E9
                y_data = I

            elif mode == "I-T":
                mask = self.last_y > 0
                T_data = self.last_x[mask]
                I_data = self.last_y[mask]

                mask = I_data > 0

                T = T_data[mask]
                I = np.log(I_data[mask])

                my_potential = SchottkyPotential()
                my_band = SmartMetal()
                my_solver = Noumerov()
                my_supply = FermiDirac()

                emitter = MetalEmitter(my_potential, my_solver, my_supply, my_band)

                initial_guess = [0.01, 1E-18, 9.5, 4.5, 5] # gamma, emission_area, fermi_level, work_function, field
                bounds = ([0.001, 1E-20, 7, 3.5, 0.01], [0.01, 1E-10, 13, 5.5, 9])

                def model(T_array, gamma, area, ef, wf, field):

                    results = np.zeros_like(T_array)

                    for i, temp in enumerate(T_array):

                        emitter.update_params(field=field, work_function=wf, fermi=ef, temp=temp)
                    
                        j = emitter.calculate_current_density()

                        results[i] = j * area * 1E9

                    results = np.clip(results,1E-100, None)

                    return np.log(results)

                popt, pcov = curve_fit(model, T, I, p0=initial_guess, bounds=bounds,maxfev=1000000)

                legend = f"Fit param: gamma = {popt[0]:.6f} 1/nm, radius = {1/(5*popt[0]):.2f} nm, area = {popt[1]*1E14:.2f} nm2, ef = {popt[2]:.2f} eV, phi = {popt[3]:.2f} eV, F = {popt[4]:.2f} V/nm"
                x_label = "Temperature (T)"
                y_label = "Current (nA)"

                FERMI_LEVEL = popt[2]
                WORK_FUNCTION = popt[3]
                FIELD = popt[4]

                current_density = np.zeros_like(T)

                my_band = SmartMetal(barrier_width=3.0, supply_threshold=1e-14, energy_resolution=0.01)
                my_solver = Noumerov(x_metal=-1.0, x_vac_plus=10, h=0.001, max_barrier_width=3)

                for i, temp in enumerate(T):

                    my_potential = SchottkyPotential(fermi_level=FERMI_LEVEL, work_function=WORK_FUNCTION, electric_field=FIELD)
                    
                    my_supply = FermiDirac(fermi_level=FERMI_LEVEL, temperature=temp)

                    emitter = MetalEmitter(my_potential, my_solver, my_supply, my_band)

                    current_density[i] = emitter.calculate_current_density()

                y_fit = current_density * popt[1] * 1E9
                y_data = I

            elif mode == "TED":
                energy_data = self.last_x + 90
                counts_data = self.last_y

                my_potential = SchottkyPotential()
                my_band = CustomMetal(energy_data)
                my_solver = Noumerov()
                my_supply = FermiDirac()

                emitter = MetalEmitter(my_potential, my_solver, my_supply, my_band)

                initial_guess = [4, 90, 4.5, 300]   # field, fermi_level, work_function, temperature
                bounds = ([1, 89.9, 3, 100], [10, 90.1, 6, 2000])
                norm_counts = counts_data/max(counts_data)

                def model(x_energies, f,fl, wf, temp):
            
                    emitter.update_params(field=f, work_function=wf, fermi=fl, temp=temp)
                    
                    _, ted = emitter.calculate_total_energy_distribution()

                    return ted/max(ted)

                popt, pcov = curve_fit(model, energy_data, norm_counts, p0=initial_guess, bounds=bounds,maxfev=1000000)

                legend = f"Fit param: F = {popt[0]:.2f} V/nm, ef = {popt[1]:.2f} eV, phi = {popt[2]:.2f} eV, T = {popt[3]:.2f} K"
                x_label = "Energy (eV)"
                y_label = "Electron count (a.u)"

                emitter.update_params(field=popt[0], work_function=popt[2], fermi=popt[1], temp=popt[3])
                _, y_fit = emitter.calculate_total_energy_distribution()
                y_data = norm_counts
                y_fit = y_fit/max(y_fit)

            elif mode == "NED":
                energy_data = self.last_x + 90
                counts_data = self.last_y

                my_potential = SchottkyPotential()
                my_band = CustomMetal(energy_data)
                my_solver = Noumerov()
                my_supply = LogFermiDirac()

                emitter = MetalEmitter(my_potential, my_solver, my_supply, my_band)

                initial_guess = [4, 90, 4.5, 300]   # field, fermi_level, work_function, temperature
                bounds = ([1, 89.9, 3, 100], [10, 90.1, 6, 2000])
                norm_counts = counts_data/max(counts_data)

                def model(x_energies, f,fl, wf, temp):
            
                    emitter.update_params(field=f, work_function=wf, fermi=fl, temp=temp)
                    
                    _, ned = emitter.calculate_normal_energy_distribution()

                    return ned/max(ned)

                popt, pcov = curve_fit(model, energy_data, norm_counts, p0=initial_guess, bounds=bounds,maxfev=1000000)

                legend = f"Fit param: F = {popt[0]:.2f} V/nm, ef = {popt[1]:.2f} eV,\n phi = {popt[2]:.2f} eV, T = {popt[3]:.2f} K"
                x_label = "Energy (eV)"
                y_label = "Electron count (a.u)"

                emitter.update_params(field=popt[0], work_function=popt[2], fermi=popt[1], temp=popt[3])
                _, y_fit = emitter.calculate_normal_energy_distribution()
                y_data = norm_counts
                y_fit = y_fit/max(y_fit)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Fit failed: {e}")
            return

        self.last_fit = y_fit

        self.main.figure.clear()
        ax = self.main.figure.add_subplot(111)
        ax.plot(self.last_x, y_data, "o", color="tab:blue", label="data")
        ax.plot(self.last_x, y_fit, "-", color="tab:orange", label=legend)
        ax.set_xscale(self.xscale_menu.currentText())
        ax.set_yscale(self.yscale_menu.currentText())
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{mode} Fit")
        ax.legend()
        self.main.canvas.draw()

    def save_fit_data(self):
        if self.last_fit is None:
            QMessageBox.warning(self, "Error", "No fit to save")
            return

        directory = os.path.dirname(self.file_path)
        path = os.path.join(directory, "fit_results.txt")

        np.savetxt(
            path,
            np.column_stack([self.last_x, self.last_fit]),
            header="x y_fit",
            fmt="%.6f"
        )

        QMessageBox.information(self, "Saved", f"Saved data to:\n{path}")

    def save_fit_figure(self):
        if self.last_fit is None:
            QMessageBox.warning(self, "Error", "No fit figure to save")
            return

        directory = os.path.dirname(self.file_path)
        path = os.path.join(directory, "fit_plot.png")

        self.main.figure.savefig(path)
        QMessageBox.information(self, "Saved", f"Saved figure to:\n{path}")


# ---------------------------------------------------------
# DropPlotCanvas
# ---------------------------------------------------------
class DropPlotCanvas(FigureCanvas):
    def __init__(self, figure, fit_tab):
        super().__init__(figure)
        self.fit_tab = fit_tab
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = event.mimeData().urls()[0].toLocalFile()
        try:
            self.fit_tab.load_file(path)
            self.fit_tab.main.plot(self.fit_tab.last_x, self.fit_tab.last_y, "Loaded Data")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load file: {e}")


# ---------------------------------------------------------
# DropArea
# ---------------------------------------------------------
class DropArea(QLabel):
    def __init__(self, fit_tab):
        super().__init__("Drop a CSV/TXT/Excel file here")
        self.fit_tab = fit_tab
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("font-size: 20px; color: #666; border: 2px dashed #aaa;")
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = event.mimeData().urls()[0].toLocalFile()
        try:
            self.fit_tab.load_file(path)
            self.setText(f"Loaded: {os.path.basename(path)}")
            self.fit_tab.main.stack.setCurrentIndex(0)
            self.fit_tab.main.plot(self.fit_tab.last_x, self.fit_tab.last_y, "Loaded Data")
        except Exception as e:
            self.setText(f"Error: {e}")


# ---------------------------------------------------------
# MainWindow
# ---------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        profile = QWebEngineProfile.defaultProfile()
        profile.settings().setAttribute(
            QWebEngineSettings.WebAttribute.PdfViewerEnabled, True
        )

        self.setWindowTitle("GETELEC – Calculate & Fit Electron Emission")
        self.resize(1200, 800)

        # Tabs
        self.tabs = QTabWidget()
        self.calculate_tab = CalculateTab(self)
        self.fit_tab = FitTab(self)
        self.documentation_tab = DocumentationTab(self)

        self.tabs.addTab(self.calculate_tab, "Calculate")
        self.tabs.addTab(self.fit_tab, "Fit")
        self.tabs.addTab(self.documentation_tab, "Documentation")

        self.tabs.currentChanged.connect(self.switch_view)

        # Plot canvas
        self.figure = plt.figure()
        self.canvas = DropPlotCanvas(self.figure, self.fit_tab)

        # Stacked widget (used only for Calculate/Fit views)
        self.stack = QStackedWidget()
        self.drop_area = DropArea(self.fit_tab)

        self.stack.addWidget(self.canvas)     # index 0 = plot
        self.stack.addWidget(self.drop_area)  # index 1 = drag & drop zone

        # Main layout: tabs on the left, stack on the right
        layout = QHBoxLayout()
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.stack, 3)
        self.setLayout(layout)

    def switch_view(self, index):
        tab_name = self.tabs.tabText(index)

        if tab_name == "Documentation":
            self.stack.hide()
            return

        self.stack.show()

        if tab_name == "Fit":
            if self.fit_tab.last_x is None:
                self.stack.setCurrentIndex(1)
            else:
                self.stack.setCurrentIndex(0)
        else:
            self.stack.setCurrentIndex(0)


    def plot(self, x, y, title="Plot"):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x, y, marker="o")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        self.canvas.draw()


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
