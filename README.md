# GETELEC

**General Tool for Electron Emission Calculations** - A computational tool for calculating thermal-field electron emission current density and Nottingham effect heat for metallic and semiconducting emitters.

For details, see the associated publications:
* [http://dx.doi.org/10.1016/j.commatsci.2016.11.010](http://dx.doi.org/10.1016/j.commatsci.2016.11.010)
* [https://doi.org/10.1063/5.0284808](https://doi.org/10.1063/5.0284808)
* Publication 3: Extended abstract can be consulted in the proceedings of the 2026 IVNC

If you use this code, please cite these papers.

---

## Table of Contents
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)
G
---

## Installation & Setup

Follow these standard steps to set up the environment and ensure everything is correctly installed:

1. **Clone or Download the Repository:**
    Download the repository to your local machine and place it in your preferred directory.
    ```bash
    git clone [https://github.com/yourusername/getelec.git](https://github.com/yourusername/getelec.git)
    cd getelec
    ```

2. **Create a Virtual Environment:**
    Create a Python virtual environment named gt_venv.
    ```bash
    python -m venv gt_venv
    ```

3. **Activate the Virtual Environment:**
    ```bash
    .\gt_venv\Scripts\Activate.ps1
    ```
4. **Install Dependencies:**
    Install the required external libraries using the provided requirements.txt file:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## How to Run

To ensure the environment and package installations are working correctly, run the built-in test suite:
```bash
python tests.py
```

Verify that all tests return a PASS status before proceeding to execution.

## Usage

There are two primary ways to interact with and utilize **GETELEC**:

### 1. Graphical User Interface (GUI)
Run the dedicated GUI script to visually interact with basic simulations, perform data fitting, and access the comprehensive internal documentation interface.

```bash
python gui.py
```

### 2. Scripts and Native Documentation
For direct execution, custom configurations, or a deeper programmatic deep-dive, look into the package folders:

* **Examples:** Explore sample implementations and templates in the `examples/` directory.
* **Documentation:** Open the pre-built interactive HTML documentation located in the `docs/` directory using any standard web browser.

## Contributing
We welcome community contributions! Please read our CONTRIBUTING.md guide for details on our code of conduct and the submission process.

## License
This project is licensed under the Creative Commons Attribution (CC BY) License. See the LICENSE.md file for full details.

## Acknowledgements
If you use this software in your research or technical work, please acknowledge the original authors by citing the foundational publications:

* Publication 1: http://dx.doi.org/10.1016/j.commatsci.2016.11.010
* Publication 2: https://doi.org/10.1063/5.0284808
* Publication 3: Extended abstract can be consulted in the proceedings of the 2026 IVNC

## Contact
If you encounter issues, have questions, or would like to request features, please open an issue in the repository or contact:

* Email: s [dot] barranco [dot] carceles [at] gmail [dot] com