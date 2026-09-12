"""
Build the standalone GETELEC application, and test the build.

Produces an application that runs on a machine with no Python, no virtual
environment and no packages installed.

    python compile.py                 # build for this platform, then test it
    python compile.py --onefile       # one file instead of a folder
    python compile.py --console       # keep a console window, to see tracebacks
    python compile.py --clean         # empty PyInstaller's cache first
    python compile.py --no-check      # skip the test of the finished build
    python compile.py --wipe          # delete everything this script built

Everything lands in ``app/``, which git ignores: the application in
``app/dist/``, PyInstaller's work files in ``app/build/``, its spec file and
the test report in ``app/``. Deliberately not ``dist/``, which holds the wheel
and source archive uploaded to PyPI, and not the repository root, where a spec
file would be picked up by ``git add``. PyInstaller cannot cross-compile, so a
Windows application has to be built on Windows, a macOS one on macOS, and so
on.

**Folder or single file.** Ship the folder (the default), zipped or inside an
installer. ``--onefile`` unpacks itself into a new temporary folder on every
launch, so it starts slower, and Numba's cache of compiled code is keyed to
that folder, so the solver is recompiled on every launch as well.

**No embedded browser.** The Documentation tab opens the pages in the system
browser. Qt's web engine, an embedded Chromium, would show them inside the
window, but with the Qt Quick libraries it pulls in it is more than half of an
application that has it (822 MB against 380 MB, 320 against 155 MB zipped), so
it is excluded even where PyQt6-WebEngine happens to be installed.

**The test.** A frozen application finds out that a data file, a hidden import
or a Qt plugin is missing only when a user reaches the feature that needs it.
So once built, the application is run with ``--self-test``
(``gui.run_self_test``): it checks the solvers against pinned numbers, both
shipped networks, the data-file readers, a fit, saving a figure as SVG and
PNG, the main window and the documentation view, writes a report and exits. A build that fails it is not
fit to ship.

Three things need care, and are handled below.

**Numba.** The Noumerov kernels are compiled at run time, so the build has to
carry Numba's runtime and llvmlite's shared library. PyInstaller's hooks find
most of it; the hidden imports below cover the rest.

**Data files.** The trained networks (``getelec/data``) and the documentation
pages are read from disk at run time. The build stops if any is missing,
rather than producing an application without them. The documentation is
bundled as it is: run ``docs/regenerate.py`` first if ``GUIDE.md`` or a
docstring has changed.

**Paths.** A frozen application has no source tree. The package reads its
networks through ``importlib.resources``, which works frozen or not; the GUI
finds the documentation through ``sys._MEIPASS`` (``gui.resource_dir``).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "app"
APP_NAME = "GETELEC"

#: Modules the application imports, with the pip package that provides each.
REQUIREMENTS = {
    "PyInstaller": "pyinstaller", "PyQt6": "PyQt6",
    "matplotlib": "matplotlib",
    "pandas": "pandas", "openpyxl": "openpyxl", "xlrd": "xlrd",
    "numpy": "numpy", "scipy": "scipy", "numba": "numba",
}

#: Files that travel with the application, as (source, destination) pairs,
#: relative to the repository and to the application's resource folder.
DATA_FILES = [
    ("getelec/data", "getelec/data"),        # the trained networks
    ("docs/index.html", "docs"),
    ("docs/introduction.html", "docs"),
    ("docs/usage.html", "docs"),
    ("docs/getelec.html", "docs"),
    ("docs/search.js", "docs"),
    ("docs/getelec", "docs/getelec"),        # the API reference, one page per module
]

#: Imports PyInstaller's static analysis cannot see. Numba and llvmlite pull
#: their runtime in dynamically; pandas imports its Excel readers by name only
#: when a file is opened; matplotlib imports the SVG writer only when a figure
#: is saved as SVG (without it, a frozen build cannot save SVG at all).
HIDDEN_IMPORTS = [
    "numba", "numba.core.typing.builtins", "llvmlite", "llvmlite.binding",
    "scipy.special.cython_special", "scipy._lib.messagestream",
    "matplotlib.backends.backend_qtagg", "matplotlib.backends.backend_svg",
    "openpyxl", "xlrd",
]

#: Qt's web engine and the Qt Quick and QML libraries that come with it, which
#: the application does not use. Excluded explicitly, so that a development
#: environment that still has PyQt6-WebEngine installed builds the same thing.
WEB_ENGINE_MODULES = [
    "PyQt6.QtWebEngineWidgets", "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineQuick",
    "PyQt6.QtWebChannel", "PyQt6.QtPositioning", "PyQt6.QtQml", "PyQt6.QtQuick",
    "PyQt6.QtQuickWidgets",
]

#: Packages the application never imports, which PyInstaller would otherwise
#: ship because something in the environment can reach them. scikit-learn is
#: imported inside getelec.training, which trains networks and has no place in
#: the application.
EXCLUDES = [
    "tkinter", "pytest", "IPython", "ipykernel", "jupyter_client", "jupyter_core",
    "notebook", "nbformat", "nbconvert", "sphinx", "pdoc", "sklearn",
    "PyQt5", "PySide2", "PySide6",
]


def get_version():
    """The package version, read from its source without importing it."""
    text = (ROOT / "getelec" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M).group(1)


def check_requirements():
    """Fail early and clearly, rather than part way through a long build."""
    import importlib.util

    missing = sorted({package for module, package in REQUIREMENTS.items()
                      if importlib.util.find_spec(module.split(".")[0]) is None
                      or importlib.util.find_spec(module) is None})
    if missing:
        joined = " ".join(missing)
        raise SystemExit(f"Missing build requirements: {joined}\n"
                         f"Install them with:\n    pip install {joined}\n"
                         f"or everything at once with:  pip install -e \".[dev]\"")
    absent = [source for source, _ in DATA_FILES if not (ROOT / source).exists()]
    if absent:
        raise SystemExit("Files the application needs are missing: "
                         + ", ".join(absent)
                         + "\nThe documentation is rebuilt with docs/regenerate.py.")


def write_version_file(version):
    """
    Windows version information, shown under the executable's Properties.

    Lets a user see which version they are running without starting it, and
    is part of what code signing presents. Other platforms ignore it.
    """
    numbers = tuple(int(n) for n in re.findall(r"\d+", version)[:3])
    numbers += (0,) * (4 - len(numbers))
    text = f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={numbers}, prodvers={numbers}, mask=0x3f, flags=0x0,
                    OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('FileDescription', 'GETELEC - General Tool for Electron Emission Calculations'),
      StringStruct('FileVersion', '{version}'),
      StringStruct('InternalName', '{APP_NAME}'),
      StringStruct('OriginalFilename', '{APP_NAME}.exe'),
      StringStruct('ProductName', 'GETELEC'),
      StringStruct('ProductVersion', '{version}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
    path = OUTPUT / "version_info.txt"
    path.write_text(text, encoding="utf-8")
    return path


def get_executable(name, onefile, console):
    """Where PyInstaller puts the program itself, for each platform and layout."""
    dist = OUTPUT / "dist"
    suffix = ".exe" if sys.platform.startswith("win") else ""
    if sys.platform == "darwin" and not console:
        return dist / f"{name}.app" / "Contents" / "MacOS" / name
    if onefile:
        return dist / f"{name}{suffix}"
    return dist / name / f"{name}{suffix}"


def build(onefile=False, console=False, clean=False, name=APP_NAME):
    """Run PyInstaller with the settings this project needs."""
    OUTPUT.mkdir(exist_ok=True)
    separator = ";" if sys.platform.startswith("win") else ":"
    version = get_version()
    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", name,
        "--onefile" if onefile else "--onedir",
        "--console" if console else "--windowed",
        "--distpath", str(OUTPUT / "dist"),
        "--workpath", str(OUTPUT / "build"),
        "--specpath", str(OUTPUT),
        "--paths", str(ROOT),
        # The package is imported by name at run time in a few places, so it
        # is collected wholesale rather than through the import graph.
        "--collect-submodules", "getelec",
    ]
    if clean:
        command.append("--clean")
    for module in HIDDEN_IMPORTS:
        command += ["--hidden-import", module]
    for module in EXCLUDES + WEB_ENGINE_MODULES:
        command += ["--exclude-module", module]
    # Absolute paths: PyInstaller resolves relative ones against the spec
    # file's folder, which is app/, not the repository.
    for source, destination in DATA_FILES:
        command += ["--add-data", f"{ROOT / source}{separator}{destination}"]
    if sys.platform.startswith("win"):
        command += ["--version-file", str(write_version_file(version))]
    elif sys.platform == "darwin":
        command += ["--osx-bundle-identifier", "io.github.sbcarceles13.getelec"]
    icon = ROOT / "docs" / ("icon.icns" if sys.platform == "darwin" else "icon.ico")
    if icon.exists():
        command += ["--icon", str(icon)]
    command.append(str(ROOT / "gui.py"))

    print(f"Building {name} {version} ({'one file' if onefile else 'folder'}, "
          f"{'console' if console else 'windowed'}) into {OUTPUT / 'dist'} ...")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit("PyInstaller failed; its messages above say why.")
    executable = get_executable(name, onefile, console)
    if not executable.exists():
        raise SystemExit(f"PyInstaller finished, but there is no {executable}.")
    return executable


def run_self_test(executable, timeout=900):
    """
    Run the built application's self-test and print its report.

    The first run is the slowest part of a user's first launch as well: Numba
    compiles the solver kernels, which takes tens of seconds.

    Returns
    -------
    bool
        Whether every check passed.
    """
    report = OUTPUT / "self_test.txt"
    report.unlink(missing_ok=True)
    print(f"\nTesting the build: {executable.name} --self-test")
    try:
        result = subprocess.run([str(executable), "--self-test", str(report)],
                                timeout=timeout)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = None
    if report.exists():
        print(report.read_text(encoding="utf-8"))
    if code is None:
        print(f"The self-test did not finish within {timeout} s.")
    elif not report.exists():
        print(f"The application exited with code {code} before writing a report. "
              f"Build with --console to see why.")
    return code == 0


def get_size(path):
    """Total size of a file or folder, in MB."""
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    return sum(p.stat().st_size for p in files) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--onefile", action="store_true",
                        help="one executable instead of a folder (slower to start: "
                             "it unpacks, and Numba recompiles, on every launch)")
    parser.add_argument("--console", action="store_true",
                        help="keep a console window, so tracebacks are visible")
    parser.add_argument("--clean", action="store_true",
                        help="clear PyInstaller's cache before building")
    parser.add_argument("--no-check", action="store_true",
                        help="do not run the self-test of the finished build")
    parser.add_argument("--name", default=APP_NAME, help="application name")
    parser.add_argument("--wipe", action="store_true",
                        help=f"delete {OUTPUT.name}/, everything this script builds, "
                             f"and exit")
    args = parser.parse_args()

    if args.wipe:
        shutil.rmtree(OUTPUT, ignore_errors=True)
        print(f"Removed {OUTPUT}.")
        return

    check_requirements()
    executable = build(onefile=args.onefile, console=args.console,
                       clean=args.clean, name=args.name)
    # What a user receives: the executable itself, or the folder (or macOS
    # .app bundle) around it.
    dist = OUTPUT / "dist"
    shipped = next((p for p in (executable, *executable.parents) if p.parent == dist),
                   executable)
    print(f"\nBuilt {shipped} ({get_size(shipped):.0f} MB).")

    if not args.no_check and not run_self_test(executable):
        raise SystemExit("\nThe build FAILED its self-test: do not ship it.")
    if not args.onefile:
        print("Ship the whole folder, zipped or in an installer: the program "
              "does not run without the files next to it.")


if __name__ == "__main__":
    main()
