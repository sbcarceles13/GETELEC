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
        self.func_menu.addItems(["IF", "IT", "TED", "NED"])
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

        if func == "IF":
            self.paramA.show()
            self.paramB.show()
            self.paramD.show()
            self.xmin_box.setPlaceholderText("F_min (V/nm)")
            self.xmax_box.setPlaceholderText("F_max (V/nm)")
            self.dx_box.setPlaceholderText("delta_F (V/nm)")

        elif func == "IT":
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
            if func == "IF":
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

            elif func == "IT":
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
            <a href="mailto:s.barranco.carceles@gmail.com">s.barranco.carceles@gmail.com</a>
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
        self.fit_menu.addItems(["exp", "log", "sin"])

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
            if path.endswith(".csv") or path.endswith(".txt"):
                df = pd.read_csv(path, header=None)
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
            raise e

    def run_fit(self):
        if self.last_x is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return

        model = self.fit_menu.currentText()

        try:
            if model == "exp":
                mask = self.last_y > 0
                x = self.last_x[mask]
                y = self.last_y[mask]

                coeffs = np.polyfit(x, np.log(y), 1)
                B = coeffs[0]
                A = np.exp(coeffs[1])
                C = 1.0

                y_fit = A * np.exp(B * self.last_x)
                legend = f"exp fit: A={A:.3g}, B={B:.3g}, C={C:.3g}"

            elif model == "log":
                mask = self.last_x > 0
                x = self.last_x[mask]
                y = self.last_y[mask]

                coeffs = np.polyfit(np.log(x), y, 1)
                B = coeffs[0]
                A = coeffs[1]

                y_fit = A + B * np.log(self.last_x)
                legend = f"log fit: A={A:.3g}, B={B:.3g}"

            elif model == "sin":
                def sin_model(x, A, B, C):
                    return A * np.sin(B * x + C)

                p0 = [1, 1, 0]
                params, _ = curve_fit(sin_model, self.last_x, self.last_y, p0=p0)
                A, B, C = params

                y_fit = A * np.sin(B * self.last_x + C)
                legend = f"sin fit: A={A:.3g}, B={B:.3g}, C={C:.3g}"

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Fit failed: {e}")
            return

        self.last_fit = y_fit

        self.main.figure.clear()
        ax = self.main.figure.add_subplot(111)
        ax.plot(self.last_x, self.last_y, "o", color="tab:blue", label="data")
        ax.plot(self.last_x, y_fit, "-", color="tab:orange", label=legend)
        ax.set_xscale(self.xscale_menu.currentText())
        ax.set_yscale(self.yscale_menu.currentText())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{model} fit")
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
