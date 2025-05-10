# Molecular Hydrogen Analysis App

![App Screenshot](https://github.com/agiani99/OxMet/blob/main/screenshot.png)

## Overview

The Molecular Hydrogen Analysis App is a computational chemistry tool for analyzing the most energetically favorable hydrogen removal positions in organic molecules. This application helps predict potential sites of oxidative metabolism, which is crucial in drug design and metabolic pathway analysis. it a beta version as it lacks a correction for water accessibility of the hydrogen which I would like to add asap. A Pareto best between maximal water accessibility AND minimal energy delta between ground state and radical state should do the magic. At the moment only small molecules provide reliable predictions only with energy contributions. 

## Features

- **3D Structure Generation**: Automatic generation and optimization of 3D molecular structures from SMILES input
- **Energy Calculation**: Simulates semi-empirical and ab initio quantum chemistry calculations to determine energetics
- **Radical Analysis**: Properly accounts for the radical nature of hydrogen-abstracted species
- **Visualization**: Interactive 2D and 3D molecular visualizations with highlighted reactive sites
- **Data Analysis**: Charts and tables of energy differences across all possible hydrogen abstraction sites
- **Scientific Interpretation**: Detailed explanations and recommendations based on computational results

## Installation

### Requirements

- Python 3.7+
- RDKit
- Streamlit
- Pandas
- Matplotlib
- NumPy

### Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install streamlit rdkit pandas matplotlib numpy
```

## Usage

Run the application with:

```bash
streamlit run hydrogen_analyzer_app.py
```

Then open your browser to the displayed URL (typically http://localhost:8501).

### Steps to Analyze a Molecule

1. Enter a SMILES string or select a sample molecule
2. Choose a computational method (MNDO, AM1, PM3, HF/STO-3G, B3LYP/STO-3G)
3. Select visualization preferences
4. Click "Run Analysis"
5. Review the results, which include:
   - Highlighted reactive positions
   - Energy differences for each C-H bond
   - Interactive 3D visualization
   - Scientific interpretation and recommendations

## Scientific Background

### Hydrogen Abstraction in Metabolism

Oxidative metabolism often begins with the abstraction of a hydrogen atom from a carbon atom, creating a carbon-centered radical. The energetics of this process can predict the most likely sites of metabolism in drugs and other compounds.

### Radical Correction

This application properly accounts for the radical nature of hydrogen-abstracted species, providing a more accurate energy comparison than simple molecular mechanics. The simulation includes:

- Accounting for radical stabilization effects (primary, secondary, tertiary)
- Estimating resonance stabilization (benzylic and allylic positions)
- Incorporating electronic effects of neighboring groups

## Advanced Settings

- **Force Field Type**: Choose between UFF, MMFF94, and MMFF94s
- **Minimization Steps**: Adjust the number of steps for energy minimization
- **Visualization Style**: Select between Stick, Ball and Stick, and Space-filling representations

## Example Molecules

The application includes several sample molecules:
- Ethanol
- Amphetamine
- Caffeine
- Ibuprofen
- Aspirin
- Acetaminophen

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RDKit for molecular manipulation and visualization
- Streamlit for the web interface
- PySCF (in the original version) for quantum chemistry calculations
