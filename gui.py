import os
import sys
import time
import traceback
from pathlib import Path
import numpy as np

from PyQt6.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QTextBrowser, QComboBox, QLineEdit, QPushButton, QLabel,
    QMessageBox, QStackedWidget, QFileDialog, QProgressBar, QCheckBox,
    QGridLayout
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import getelec
import gui_backend


def parse_number(text):
    """
    A number typed into a box, accepting a decimal comma: "7,5" is 7.5.

    French and German keyboards type the comma, and a box that rejects it is
    the first thing such a user meets.

    Raises
    ------
    ValueError
        If the text is not a number.
    """
    return float(text.strip().replace(",", "."))


class ComputeWorker(QThread):
    """
    Runs a calculation off the main thread.

    The exact solver can take seconds on a wide sweep, and doing that on the GUI
    thread freezes the window -- including the progress bar meant to show that
    something is happening. Everything heavy therefore happens here, and results
    come back through signals.
    """

    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progressed = pyqtSignal(int)

    def __init__(self, kwargs):
        super().__init__()
        self._kwargs = kwargs

    def run(self):
        try:
            result = gui_backend.calculate(
                progress=lambda f: self.progressed.emit(int(100 * f)), **self._kwargs)
            self.finished_ok.emit(result)
        except Exception as exc:                      # reported, never swallowed
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CalculateTab(QWidget):
    """Parameter entry, solver choice, and plotting for a single calculation."""

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.worker = None
        self.last_result = None

        self.material_menu = QComboBox()
        self.material_menu.addItems(gui_backend.MATERIALS)
        self.material_menu.currentTextChanged.connect(self.update_calculation_choices)

        self.func_menu = QComboBox()
        self.func_menu.addItems(
            gui_backend.get_calculations(self.material_menu.currentText()))
        self.func_menu.currentTextChanged.connect(self.update_parameter_fields)

        self.solver_menu = QComboBox()
        self.solver_menu.addItems(gui_backend.SOLVERS.keys())

        self.xmin_box, self.xmax_box, self.dx_box = QLineEdit(), QLineEdit(), QLineEdit()
        for box in (self.xmin_box, self.xmax_box, self.dx_box):
            box.setFixedHeight(28)

        # One row per parameter, built once and shown or hidden as needed. This
        # is why the field list can change with the material without rebuilding
        # the layout on every switch.
        self.param_rows = {}
        for name, (label, default) in gui_backend.PARAMETER_INFO.items():
            edit = QLineEdit(str(default))
            edit.setFixedHeight(28)
            edit.setPlaceholderText(label)
            self.param_rows[name] = (QLabel(label), edit)

        self.xscale_menu, self.yscale_menu = QComboBox(), QComboBox()
        for menu in (self.xscale_menu, self.yscale_menu):
            menu.addItems(["linear", "log"])
        self.yscale_menu.setCurrentText("log")

        self.run_button = QPushButton("Calculate")
        self.run_button.clicked.connect(self.compute_function)
        self.save_data_button = QPushButton("Save data")
        self.save_data_button.clicked.connect(self.save_data)
        self.save_fig_button = QPushButton("Save figure")
        self.save_fig_button.clicked.connect(self.save_figure)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        layout = QVBoxLayout()
        for label, widget in (("Material", self.material_menu),
                              ("Calculation", self.func_menu),
                              ("Solver", self.solver_menu)):
            layout.addWidget(QLabel(label))
            layout.addWidget(widget)

        self.range_label = QLabel("x range")
        layout.addWidget(self.range_label)
        range_row = QHBoxLayout()
        for box, tip in ((self.xmin_box, "min"), (self.xmax_box, "max"),
                         (self.dx_box, "step")):
            box.setPlaceholderText(tip)
            range_row.addWidget(box)
        layout.addLayout(range_row)

        for name, (label_widget, edit) in self.param_rows.items():
            layout.addWidget(label_widget)
            layout.addWidget(edit)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("x scale"))
        scale_row.addWidget(self.xscale_menu)
        scale_row.addWidget(QLabel("y scale"))
        scale_row.addWidget(self.yscale_menu)
        layout.addLayout(scale_row)

        layout.addWidget(self.run_button)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        button_row = QHBoxLayout()
        button_row.addWidget(self.save_data_button)
        button_row.addWidget(self.save_fig_button)
        layout.addLayout(button_row)
        layout.addStretch()
        self.setLayout(layout)

        self.update_calculation_choices()

    # -- interface state ---------------------------------------------------

    def update_calculation_choices(self):
        """
        Offer only the calculations this material has, keeping the selection.

        g(E) is semiconductor-only: it is D integrated over the effective-mass
        window, and a metal has no such window. Rather than grey it out, it is
        absent from the list until a semiconductor is chosen.
        """
        wanted = gui_backend.get_calculations(self.material_menu.currentText())
        current = tuple(self.func_menu.itemText(i)
                        for i in range(self.func_menu.count()))
        if current != wanted:
            previous = self.func_menu.currentText()
            blocked = self.func_menu.blockSignals(True)
            self.func_menu.clear()
            self.func_menu.addItems(wanted)
            if previous in wanted:
                self.func_menu.setCurrentText(previous)
            self.func_menu.blockSignals(blocked)
        self.update_parameter_fields()

    def update_parameter_fields(self):
        """Show only the parameters this calculation actually uses."""
        calculation = self.func_menu.currentText()
        material = self.material_menu.currentText()
        if calculation not in gui_backend.CALCULATIONS:
            return
        needed = gui_backend.get_parameters(calculation, material)
        for name, (label_widget, edit) in self.param_rows.items():
            visible = name in needed
            label_widget.setVisible(visible)
            edit.setVisible(visible)

        swept, x_label, _, energy_axis = gui_backend.CALCULATIONS[calculation]
        self.range_label.setText(
            "Energy range relative to E_F (eV)" if energy_axis else f"{x_label} range")
        defaults = {"field": ("3", "8", "0.5"), "temperature": ("300", "2000", "100"),
                    "energy": ("-2", "1", "0.01")}
        low, high, step = defaults["energy" if energy_axis else swept]
        for box, value in ((self.xmin_box, low), (self.xmax_box, high),
                           (self.dx_box, step)):
            box.setText(value)

        # A semiconductor Fermi level sits in a different place entirely, so
        # nudge the default rather than leaving a metal value that will not run.
        # Whatever is in the box is left alone if it is not a number yet: this
        # runs on every change of menu, and the box may be half typed.
        fermi_edit = self.param_rows["fermi_level"][1]
        try:
            fermi = parse_number(fermi_edit.text() or "0")
        except ValueError:
            return
        if material == "Semiconductor" and fermi < 10:
            fermi_edit.setText("13.0")
        elif material == "Metal" and fermi > 10:
            fermi_edit.setText("7.5")

    def _read_parameters(self):
        calculation = self.func_menu.currentText()
        material = self.material_menu.currentText()
        params = {}
        for name in gui_backend.get_parameters(calculation, material):
            text = self.param_rows[name][1].text().strip()
            label = gui_backend.PARAMETER_INFO[name][0]
            if not text:
                raise ValueError(f"{label} is empty.")
            try:
                params[name] = parse_number(text)
            except ValueError:
                raise ValueError(f"{label}: {text!r} is not a number.") from None
        return calculation, material, params

    # -- running -----------------------------------------------------------

    def compute_function(self):
        if self.worker is not None and self.worker.isRunning():
            return
        try:
            calculation, material, params = self._read_parameters()
            bounds = []
            for box, name in ((self.xmin_box, "minimum"), (self.xmax_box, "maximum"),
                              (self.dx_box, "step")):
                try:
                    bounds.append(parse_number(box.text()))
                except ValueError:
                    raise ValueError(f"Range {name} is not a number.") from None
        except ValueError as exc:
            QMessageBox.warning(self, "Check the inputs", str(exc))
            return

        self.run_button.setEnabled(False)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.status.setText("Calculating...")

        self.worker = ComputeWorker(dict(
            calculation=calculation, material=material, params=params,
            x_min=bounds[0], x_max=bounds[1], x_step=bounds[2],
            method=gui_backend.SOLVERS[self.solver_menu.currentText()]))
        self.worker.progressed.connect(self.progress.setValue)
        self.worker.finished_ok.connect(self._on_result)
        self.worker.failed.connect(self._on_failure)
        self.worker.start()

    def _on_failure(self, message):
        self.run_button.setEnabled(True)
        self.progress.setVisible(False)
        self.status.setText("")
        QMessageBox.warning(self, "Calculation failed", message)

    def _on_result(self, result):
        self.run_button.setEnabled(True)
        self.progress.setVisible(False)
        self.status.setText("Done.")
        self.last_result = result

        self.main.figure.clear()
        ax = self.main.figure.add_subplot(111)
        series = (result["series"] if result["series"]
                  else [(result["legend"], result["x"], result["y"])])

        wanted = self.yscale_menu.currentText()
        if wanted == "log" and not any(np.any(np.asarray(y, dtype=float) > 0)
                                       for _, _, y in series):
            wanted = "linear"

        for name, x, y in series:
            ax.plot(x, y, label=name)
        ax.legend()
        ax.set_xscale(self.xscale_menu.currentText())
        try:
            ax.set_yscale(wanted)
        except ValueError:
            ax.set_yscale("linear")
        ax.grid(alpha=0.3)
        ax.set_xlabel(result["x_label"])
        ax.set_ylabel(result["y_label"])
        ax.set_title(result["title"])
        self.main.figure.tight_layout()
        getelec.watermark(ax)
        self.main.canvas.draw()

    # -- saving ------------------------------------------------------------

    def save_data(self):
        if self.last_result is None:
            QMessageBox.warning(self, "Nothing to save", "Run a calculation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save data", "getelec_data.txt",
                                              "Text files (*.txt)")
        if not path:
            return
        result = self.last_result
        try:
            if result["series"]:
                with open(path, "w", encoding="utf-8") as handle:
                    for name, x, y in result["series"]:
                        handle.write(f"# {name}\n# {result['x_label']}  {result['y_label']}\n")
                        np.savetxt(handle, np.column_stack([x, y]))
                        handle.write("\n")
            else:
                np.savetxt(path, np.column_stack([result["x"], result["y"]]),
                           header=f"{result['x_label']}  {result['y_label']}")
        except OSError as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        self.status.setText(f"Saved {path}")

    def save_figure(self):
        if self.last_result is None:
            QMessageBox.warning(self, "Nothing to save", "Run a calculation first.")
            return
        path, chosen = QFileDialog.getSaveFileName(self, "Save figure", "getelec_plot.svg",
                                                   FIGURE_FILTERS)
        if not path:
            return
        try:
            path = save_figure_file(self.main.figure, path, chosen)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        self.status.setText(f"Saved {path}")


#: The formats both Save Figure buttons offer, SVG first: a vector image stays
#: sharp at any size.
FIGURE_FILTERS = "SVG (*.svg);;PNG (*.png)"


def save_figure_file(figure, path, chosen_filter=""):
    """
    Save a figure where the user asked. A name typed without an extension gets
    the one of the format chosen in the dialog: SVG unless PNG was chosen.
    """
    if not Path(path).suffix:
        path += ".png" if "png" in chosen_filter.lower() else ".svg"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    return path


def resource_dir() -> Path:
    """
    Directory holding bundled resources, whether running frozen or from source.

    PyInstaller unpacks data files into a temporary tree recorded in
    ``sys._MEIPASS``; from a checkout the same files sit next to this script.
    Resolving through ``sys.argv[0]`` -- as this used to -- breaks the moment
    the application is launched from another working directory, which is the
    normal case for a packaged app.
    """
    bundled = getattr(sys, "_MEIPASS", None)
    return Path(bundled) if bundled else Path(__file__).resolve().parent


DOCUMENTATION_HTML = """
    <div align="center"><h2>GETELEC</h2></div>
    <p><b>Authors:</b><br>
       Salvador Barranco Cárceles<br>
       Andreas Kyritsakis<br>
       Anthony Ayari</p>
    <p><b>Version:</b> {version}</p>
    <p><b>Contact:</b><br>
       s [dot] barranco [dot] carceles [at] gmail [dot] com<br>
       anthony [dot] ayari [at] univ-lyon1 [dot] fr</p>
    <p><b>How to cite:</b> the software,<br>
       S. Barranco Cárceles, A. Kyritsakis and A. Ayari, <i>GETELEC: General
       Tool for Electron Emission Calculations</i>, Zenodo,
       <a href='https://doi.org/10.5281/zenodo.23093209'>doi:10.5281/zenodo.23093209</a>,<br>
       and the papers:</p>
    <ul>
        <li>A. Kyritsakis and F. Djurabekova, Comput. Mater. Sci. <b>128</b>, 15 (2017),
            <a href='https://doi.org/10.1016/j.commatsci.2016.11.010'>doi:10.1016/j.commatsci.2016.11.010</a></li>
        <li>S. Barranco Cárceles, V. Zadin, A. Mavalankar, I. Underwood and
            A. Kyritsakis, J. Appl. Phys. <b>138</b>, 155705 (2025),
            <a href='https://doi.org/10.1063/5.0284808'>doi:10.1063/5.0284808</a></li>
        <li>S. Barranco Cárceles, A. Kyritsakis and A. Ayari, arXiv:2610.07013 (2026),
            <a href='https://doi.org/10.48550/arXiv.2610.07013'>doi:10.48550/arXiv.2610.07013</a></li>
    </ul>
    <hr>
    <p><b>Documentation</b>, opened in your web browser:</p>
    <ul>
        <li><a href='introduction.html'>Introduction</a></li>
        <li><a href='usage.html'>Usage Guide</a></li>
        <li><a href='getelec.html'>API Reference</a></li>
    </ul>
    <p style="color: #666;">{status}</p>
"""

MISSING_DOCS_TEXT = (
    "{name} is missing from {folder}. From a source checkout, restore the "
    "documentation with <code>git checkout docs</code>; it is also on "
    "<a href='https://github.com/sbcarceles13/GETELEC'>the project repository</a>.")


class DocumentationTab(QWidget):
    """
    Authors, version and contact, and links to the documentation.

    Each page opens in the system browser. The pages rely on CSS layout and
    scripts that only a browser runs, and embedding one (Qt's web engine, a
    Chromium) would more than double the size of the standalone application.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.docs_dir = resource_dir() / "docs"

        self.page = QTextBrowser()
        # The links are handled here, not followed: a QTextBrowser navigating
        # to an HTML page would show it unstyled in place of the links.
        self.page.setOpenLinks(False)
        self.page.setOpenExternalLinks(False)
        self.page.anchorClicked.connect(self.on_link_click)
        self.show_status("")

        layout = QHBoxLayout()
        layout.addWidget(self.page)
        self.setLayout(layout)

    def show_status(self, status):
        self.page.setHtml(DOCUMENTATION_HTML.format(version=getelec.__version__,
                                                    status=status))

    def on_link_click(self, url: QUrl):
        if url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)
            return
        self.load_page(url.toString())

    def get_page_path(self, name: str):
        """
        The documentation file for a link, or None if it is missing.

        Anything outside the docs directory is refused: the links are fixed,
        so a path escaping it means something is wrong rather than that the
        user wants a file elsewhere.
        """
        name = name.split("/")[-1] or "introduction.html"
        target = (self.docs_dir / name).resolve()
        if not target.is_relative_to(self.docs_dir.resolve()) or not target.is_file():
            return None
        return target

    def load_page(self, name: str):
        """Open a documentation page in the system browser, or say why not."""
        target = self.get_page_path(name)
        if target is None:
            self.show_status(MISSING_DOCS_TEXT.format(name=name.split("/")[-1],
                                                      folder=self.docs_dir))
            return
        if QDesktopServices.openUrl(QUrl.fromLocalFile(str(target))):
            self.show_status(f"Opened {target.name} in your web browser.")
        else:
            self.show_status(f"No web browser could be started. The page is {target}")


class FitTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.file_path = None

        self.fit_menu = QComboBox()
        self.fit_menu.addItems(list(gui_backend.FIT_MODELS))
        self.fit_menu.currentTextChanged.connect(self.build_parameter_rows)

        # Fitting is metals-only on purpose. A semiconductor emitter carries the
        # band gap, the valence band edge and two effective masses on top of the
        # metal parameters, and an I-V curve does not contain enough independent
        # information to determine them -- the fit converges to whatever it was
        # started from. Rather than offer a control that produces confident
        # nonsense, the tab says so.
        material_note = QLabel(
            "Fitting is available for metals only. Semiconductor emitters have "
            "more free parameters than an I-V curve can constrain; use the "
            "Calculate tab to compare a semiconductor model against data by eye."
        )
        material_note.setWordWrap(True)
        material_note.setStyleSheet("color: #555; font-size: 11px;")
        self.material_note = material_note

        # One row per parameter: tick "fit" to fit it, starting from the value
        # in the box; untick to hold it at that value. The defaults free only
        # what the data can determine.
        self.parameter_grid = QGridLayout()
        self.parameter_rows = {}
        parameter_note = QLabel(
            "Ticked parameters are fitted, starting from the value shown; the "
            "others are held at their value. A correlation near ±1 means the "
            "data fix a combination of two parameters, not each one."
        )
        parameter_note.setWordWrap(True)
        parameter_note.setStyleSheet("color: #555; font-size: 11px;")

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
        layout.addWidget(self.material_note)

        layout.addWidget(QLabel("Parameters:"))
        layout.addLayout(self.parameter_grid)
        layout.addWidget(parameter_note)

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
        self.last_summary = []
        self.build_parameter_rows(self.fit_menu.currentText())

    def build_parameter_rows(self, model):
        """Replace the parameter rows with those of ``model``, at their defaults."""
        while self.parameter_grid.count():
            widget = self.parameter_grid.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()
        self.parameter_rows = {}
        for row, (name, label, default, _, _, free) in enumerate(
                gui_backend.FIT_MODELS[model][2]):
            check = QCheckBox("fit")
            check.setChecked(free)
            value = QLineEdit(f"{default:g}")
            value.setMaximumWidth(110)
            self.parameter_grid.addWidget(check, row, 0)
            self.parameter_grid.addWidget(QLabel(label), row, 1)
            self.parameter_grid.addWidget(value, row, 2)
            self.parameter_rows[name] = (label, check, value)

    def load_file(self, path):
        # Errors propagate: both drop targets report them to the user.
        self.last_x, self.last_y = gui_backend.read_two_columns(path)
        self.file_path = path

    def run_fit(self):
        if self.last_x is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return

        mode = self.fit_menu.currentText()
        values, free = {}, []
        for name, (label, check, box) in self.parameter_rows.items():
            try:
                values[name] = parse_number(box.text())
            except ValueError:
                QMessageBox.warning(self, "Error", f"{label}: '{box.text()}' is not a number")
                return
            if check.isChecked():
                free.append(name)

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = gui_backend.fit(mode, self.last_x, self.last_y, values, free)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Fit failed: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        # Show the fitted values, so the next fit starts from them.
        for name in result["free"]:
            self.parameter_rows[name][2].setText(f"{result['values'][name]:.6g}")

        self.last_fit = (result["x"], result["data"], result["fit"])
        self.last_summary = [f"{mode} fit"] + result["summary"]

        self.main.figure.clear()
        ax = self.main.figure.add_subplot(111)
        ax.plot(result["x"], result["data"], "o", ms=4, color="tab:blue", label="data")
        ax.plot(result["x"], result["fit"], "-", color="tab:orange", label="fit")
        ax.set_xscale(self.xscale_menu.currentText())
        ax.set_yscale(self.yscale_menu.currentText())
        ax.set_xlabel(result["x_label"])
        ax.set_ylabel(result["y_label"])
        ax.set_title(f"{mode} Fit")
        # One compact box: the parameters, one per line, above the data/fit key.
        # Top left stays clear of the curves and of the watermark.
        ax.legend(loc="upper left", title="\n".join(result["summary"]),
                  alignment="left", fontsize=9, title_fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3)
        getelec.watermark(self.main.figure.gca())
        self.main.canvas.draw()

    def _ask_save_path(self, title, name, filters):
        """A save dialog opening next to the loaded data file: (path, filter chosen)."""
        start = os.path.join(os.path.dirname(self.file_path or ""), name)
        return QFileDialog.getSaveFileName(self, title, start, filters)

    def save_fit_data(self):
        if self.last_fit is None:
            QMessageBox.warning(self, "Error", "No fit to save")
            return
        path, _ = self._ask_save_path("Save fit data", "fit_results.txt",
                                      "Text files (*.txt)")
        if not path:
            return
        # The fitted parameters go in the header, so the numbers and the fit
        # that produced them cannot be separated.
        header = "\n".join(self.last_summary + ["", "x  y_data  y_fit"])
        try:
            np.savetxt(path, np.column_stack(self.last_fit), header=header,
                       fmt="%.6e", encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Saved data to:\n{path}")

    def save_fit_figure(self):
        if self.last_fit is None:
            QMessageBox.warning(self, "Error", "No fit figure to save")
            return
        path, chosen = self._ask_save_path("Save figure", "fit_plot.svg", FIGURE_FILTERS)
        if not path:
            return
        try:
            path = save_figure_file(self.main.figure, path, chosen)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
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

        # Plot canvas. A bare Figure, not pyplot's: the window owns it, and
        # pyplot would also keep it in its own registry for the whole session.
        self.figure = Figure()
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
        ax.set_xlabel("column 1 of the loaded file")
        ax.set_ylabel("column 2 of the loaded file")
        getelec.watermark(ax)
        self.canvas.draw()


# ---------------------------------------------------------
# Unexpected errors
# ---------------------------------------------------------
def show_unexpected_error(exc_type, exc, tb):
    """
    Report an exception nothing else caught, and keep the window open.

    PyQt ends the whole application when an exception escapes a slot. In the
    standalone build, which has no console, that looks like the program simply
    vanishing. Installed as ``sys.excepthook``, this shows the message instead,
    with the traceback one click away ("Show Details") to copy into a report.
    """
    details = "".join(traceback.format_exception(exc_type, exc, tb))
    if sys.stderr is not None:              # None in a windowed build
        sys.stderr.write(details)
    box = QMessageBox(QMessageBox.Icon.Critical, "Unexpected error",
                      f"{exc_type.__name__}: {exc}\n\nThe window is still usable. "
                      f"If this keeps happening, please report it with the "
                      f"details below.")
    box.setDetailedText(details)
    box.exec()


# ---------------------------------------------------------
# Self-test of an installation or a build
# ---------------------------------------------------------
#: Pinned by the test suite: metal, phi = 4.5 eV, E_F = 7.5 eV, 300 K, 5 V/nm.
_REFERENCE_J = 4.087404e+05
#: Silicon with the semiconductor_emitter defaults, 5 V/nm.
_REFERENCE_J_SILICON = 1.180146e+04


def run_self_test(report_path):
    """
    Check that everything the application needs is present and working.

    ``GETELEC --self-test report.txt`` (or ``python gui.py --self-test
    report.txt``) runs every part of the application once and exits: the
    solvers against pinned numbers, both shipped networks, the data-file
    readers, a fit, the main window and the documentation view. It is how
    compile.py checks a build before it is shipped. A frozen application
    otherwise finds out about a missing data file, hidden import or Qt plugin
    only when a user reaches the feature that needs it.

    Writes one line per check to ``report_path``, since a windowed build has no
    console to print to.

    Returns
    -------
    int
        The exit code: 0 if every check passed.
    """
    import tempfile
    import warnings

    lines = [f"GETELEC {getelec.__version__} self-test, "
             f"{'frozen' if getattr(sys, 'frozen', False) else 'from source'}, "
             f"Python {sys.version.split()[0]}"]
    failed = []

    def check(name, function):
        start = time.perf_counter()
        try:
            detail = function()
            lines.append(f"PASS  {name}: {detail}  [{time.perf_counter() - start:.2f} s]")
        except Exception as exc:
            failed.append(name)
            lines.append(f"FAIL  {name}: {type(exc).__name__}: {exc}")

    def close_to(value, reference, tolerance):
        deviation = abs(value / reference - 1)
        if not deviation <= tolerance:
            raise ValueError(f"{value:.6e}, {deviation:.1e} from {reference:.6e}")
        return f"{value:.6e} ({deviation:.1e} from {reference:.6e})"

    def docs():
        folder = resource_dir() / "docs"
        pages = ["introduction.html", "usage.html", "getelec.html", "search.js",
                 "getelec/electron_emitter.html"]
        missing = [page for page in pages if not (folder / page).is_file()]
        if missing:
            raise FileNotFoundError(f"{missing} in {folder}")
        return f"{len(pages)} pages in {folder}"

    def learned(barrier, **kwargs):
        exact = getelec.current_density(field=5.0, barrier=barrier, **kwargs)
        return close_to(getelec.current_density(field=5.0, barrier=barrier,
                                                method="ml", **kwargs), exact, 0.01)

    def semiconductor_curves():
        params = {name: default for name, (_, default)
                  in gui_backend.PARAMETER_INFO.items()}
        params["fermi_level"] = 13.0
        result = gui_backend.calculate("TED  (total energy dist.)", "Semiconductor",
                                       params, -2.0, 1.0, 0.01)
        return " and ".join(f"{name} ({x.size} points)"
                            for name, x, _ in result["series"])

    def data_files():
        import pandas as pd
        import xlrd  # noqa: F401  -- .xls files; nothing to write one with here
        table = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        with tempfile.TemporaryDirectory() as folder:
            for suffix in (".csv", ".txt", ".xlsx"):
                path = os.path.join(folder, "data" + suffix)
                if suffix == ".xlsx":
                    table.to_excel(path, header=False, index=False)
                else:
                    table.to_csv(path, header=False, index=False,
                                 sep="," if suffix == ".csv" else " ")
                x, y = gui_backend.read_two_columns(path)
                if not (np.allclose(x, [1, 2, 3]) and np.allclose(y, [4, 5, 6])):
                    raise ValueError(f"{suffix} read back as {x}, {y}")
        return ".csv, .txt and .xlsx read back"

    def fit():
        volts = np.linspace(600.0, 1000.0, 9)
        values = {"gamma": 0.005, "area": 1000.0, "fermi_level": 7.5,
                  "work_function": 4.5, "temperature": 300.0}
        current = (getelec.current_density(field=volts * 0.005, fast=True)
                   * 1000.0 * 1e-14 * 1e9)
        start = dict(values, gamma=0.0045, area=300.0)
        result = gui_backend.fit("I-V", volts, current, start, ["gamma", "area"])
        return "gamma " + close_to(result["values"]["gamma"], 0.005, 1e-3)

    def figures():
        # Matplotlib loads the writer of each file format only when saving, so
        # a frozen build can lack one that works from source.
        import tempfile
        figure = Figure()
        figure.add_subplot().plot([0.0, 1.0], [0.0, 1.0])
        with tempfile.TemporaryDirectory() as folder:
            svg = save_figure_file(figure, os.path.join(folder, "check"), "SVG (*.svg)")
            png = save_figure_file(figure, os.path.join(folder, "check"), "PNG (*.png)")
            if b"<svg" not in Path(svg).read_bytes()[:2000]:
                raise ValueError(f"{svg} is not an SVG file")
            if not Path(png).read_bytes().startswith(b"\x89PNG"):
                raise ValueError(f"{png} is not a PNG file")
        return "check.svg and check.png written and read back"

    def window():
        app = QApplication.instance() or QApplication(sys.argv)
        main = MainWindow()
        # A decimal comma in a box, then a change of material: this ended the
        # whole application before the menus learned to leave such text alone.
        tab = main.calculate_tab
        tab.param_rows["fermi_level"][1].setText("7,5")
        tab.material_menu.setCurrentText("Semiconductor")
        tab.material_menu.setCurrentText("Metal")
        tab.param_rows["fermi_level"][1].setText("7,5")
        if tab._read_parameters()[2]["fermi_level"] != 7.5:
            raise ValueError("a decimal comma is not read as a decimal point")
        # Opening a browser is left to the user; that the pages it would open
        # are there is what can be checked.
        documentation = main.documentation_tab
        pages = [documentation.get_page_path(name) for name in
                 ("introduction.html", "usage.html", "getelec.html")]
        main.close()
        app.processEvents()
        if None in pages:
            raise FileNotFoundError("a documentation page the tab links to is missing")
        return "main window built, a decimal comma read, the documentation found"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check("documentation files", docs)
        check("Noumerov, metal J at 5 V/nm",
              lambda: close_to(getelec.current_density(field=5.0), _REFERENCE_J, 1e-5))
        check("Noumerov, silicon J at 5 V/nm",
              lambda: close_to(getelec.semiconductor_emitter(field=5.0)
                               .calculate_current_density(), _REFERENCE_J_SILICON, 1e-5))
        check("network, planar barrier", lambda: learned("schottky"))
        check("network, sharp tip", lambda: learned("sharp_tip", radius=50.0))
        check("GUI calculation, semiconductor TED", semiconductor_curves)
        check("data files", data_files)
        check("I-V fit", fit)
        check("figures saved", figures)
        check("window", window)

    lines.append("ALL PASSED" if not failed else f"FAILED: {', '.join(failed)}")
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    if "--self-test" in sys.argv:
        position = sys.argv.index("--self-test")
        report = (sys.argv[position + 1] if position + 1 < len(sys.argv)
                  else "getelec_self_test.txt")
        sys.exit(run_self_test(report))

    app = QApplication(sys.argv)
    sys.excepthook = show_unexpected_error
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
