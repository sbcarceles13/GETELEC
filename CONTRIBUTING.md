# Contributing to GETELEC

Thank you for your interest in contributing to **GETELEC**! Community input helps make this tool more robust and useful for scientific research.

## How Can I Contribute?

### Reporting Bugs
* Check the current open issues to make sure the bug hasn't already been reported.
* Open a new issue with a clear title, descriptive summary of the problem, and steps to reproduce the error (along with error logs if applicable).

### Feature Requests
* Open an issue outlining the proposed enhancement, why it is valuable, and how it aligns with thermal-field electron emission calculations.

### Code Submissions (Pull Requests)
1. Fork the repository and create your branch from `main`.
2. Ensure any new features are accompanied by appropriate test coverage.
3. Run `pytest` from the repository root; every test must pass before submitting.
4. Issue a Pull Request (PR) with a detailed explanation of your updates.

## Code Style & Standards
* Follow standard Python styling guidelines (PEP 8).
* Follow the naming convention below.
* Document new functions and classes with clean, informative docstrings.

### Naming convention

A function or method that returns a computed result starts with one of three verbs:

* `generate` when creating grids, arrays or meshes, e.g. `generate_band_structure`.
* `get` when the task is light (can be used 10,000 times without slowdown), e.g. `get_potential`, `get_occupancy`.
* `calculate` when the task is computationally heavy, e.g. `calculate_transmission`, `calculate_current_density`.

Names that are not of that kind follow ordinary Python usage instead:

* the one-line functions at the top of the package, named after the quantity they return: `current_density`, `nottingham_heat`, `transmission_coefficient`, `supply_function`;
* factories and conversions: `metal_emitter`, `semiconductor_emitter`, `Noumerov.fast`, `Noumerov.reference`, `training.make_schottky`, `training.to_model`;
* the usual machine-learning verbs: `fit`, `predict`, `validate`, `save`, `load`;
* yes/no checks: `barrier_in_domain`, `in_validated_band`;
* setters, registration, file reading, plotting and the actions of the graphical interface: `update_params`, `register_features`, `read_two_columns`, `watermark`;
* private modules and helpers, whose names start with `_`.

## Questions & Contact
For direct inquiries or support regarding contributions, please reach out via email to **s [dot] barranco [dot] carceles [at] gmail [dot] com** or **anthony [dot] ayari [at] univ-lyon1 [dot] fr**.
