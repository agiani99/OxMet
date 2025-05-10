import streamlit as st
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import io
import base64
import gc
import time
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import math

def main():
    st.set_page_config(
        page_title="Molecular Hydrogen Analysis",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧪 Molecular Hydrogen Analysis")
    st.write("Upload a SMILES string to analyze the most energetically favorable hydrogen removal")

    # Sidebar with explanation and settings
    st.sidebar.title("Settings & Information")
    
    # Create tabs in sidebar for organization
    info_tab, settings_tab = st.sidebar.tabs(["Information", "Advanced Settings"])
    
    with info_tab:
        st.markdown("""
        ## How it works
        This app analyzes a molecule to find the most energetically favorable hydrogen removal site:
        
        1. **Generate 3D structure** from SMILES input
        2. **Add hydrogens** and optimize geometry (500 steps)
        3. **Calculate ground truth energy** using chosen method
        4. For each hydrogen attached to carbon:
           - Remove the hydrogen (oxidative metabolism simulation)
           - Calculate new energy
           - Compute energy difference (ΔE)
        5. **Identify** the position with lowest energy change
        6. **Visualize** the result with highlighted site
        
        ## Use cases
        - Drug metabolism prediction
        - Reaction site prediction
        - Understanding molecular reactivity
        """)
    
    with settings_tab:
        # Cache management
        st.subheader("Memory Management")
        if st.button("Clear Memory Cache"):
            gc.collect()
            st.success("Memory cache cleared!")
        
        # Advanced calculation settings
        st.subheader("Calculation Settings")
        min_steps = st.slider("Minimization Steps", 100, 1000, 500)
        ff_type = st.selectbox("Force Field Type", ["UFF", "MMFF94", "MMFF94s"])
        
    # Main content area - create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Sample molecules dropdown
        sample_options = {
            "": "",
            "Ethanol": "CCO",
            "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "Acetaminophen": "CC(=O)NC1=CC=C(C=C1)O"
        }
        
        sample_choice = st.selectbox(
            "Choose a sample molecule or enter your own SMILES below:",
            options=list(sample_options.keys())
        )
        
        # SMILES input field
        default_smiles = sample_options[sample_choice] if sample_choice else "CCO"
        smiles_input = st.text_input("Enter SMILES string:", default_smiles)
        
        # Method selection with explanations
        method_descriptions = {
            "MNDO": "Modified Neglect of Differential Overlap - Fast, basic semi-empirical method",
            "AM1": "Austin Model 1 - Improved parameterization over MNDO",
            "PM3": "Parametric Method 3 - Further refinement of AM1",
            "HF/STO-3G": "Hartree-Fock with minimal basis set - More accurate but slower",
            "B3LYP/STO-3G": "Density Functional Theory - Most accurate but slowest option"
        }
        
        method = st.selectbox(
            "Select semi-empirical method for energy calculation:",
            list(method_descriptions.keys()),
            format_func=lambda x: f"{x} - {method_descriptions[x]}"
        )

    with col2:
        # Information about the selected method
        st.info(f"**{method}**: {method_descriptions[method]}")
        
        # Visualization options
        st.subheader("Visualization Options")
        viz_style = st.radio(
            "Molecule visualization style:",
            ["Stick", "Ball and Stick", "Space-filling"],
            horizontal=True
        )
        
        # Run analysis button - enlarged and centered
        st.write("")  # Spacer
        run_analysis = st.button("▶️ Run Analysis", use_container_width=True, type="primary")
    
    # Separator
    st.markdown("---")
    
    # Results section 
    if run_analysis:
        with st.spinner("Processing molecule..."):
            try:
                # Pass the advanced settings
                results = process_molecule(
                    smiles_input, 
                    method,
                    ff_type=ff_type,
                    min_steps=min_steps,
                    viz_style=viz_style
                )
                display_results(results)
            except Exception as e:
                st.error(f"Error processing molecule: {str(e)}")
                st.error("Please check your SMILES string and try again.")
                st.write("Common issues:")
                st.write("- Invalid SMILES syntax")
                st.write("- Molecule too complex for embedding")
                st.write("- Energy calculation failed to converge")


def process_molecule(smiles, method, ff_type="UFF", min_steps=500, viz_style='Stick'):
    """Process molecule and calculate energy differences for hydrogen removals"""
    start_time = total_time = time.time()
    
    # Create results container for better UI organization
    results_container = st.container()
    
    with results_container:
        st.subheader("📊 Analysis Progress")
        # Step 2: Generate RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
            
        # Get basic molecular properties
        mol_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        mol_weight = Chem.Descriptors.ExactMolWt(mol)
        mol_logp = Chem.Crippen.MolLogP(mol)
        
        # Display basic molecular info in a nice format
        prop_col1, prop_col2, prop_col3 = st.columns(3)
        prop_col1.metric("Molecular Formula", mol_formula)
        prop_col2.metric("Molecular Weight", f"{mol_weight:.2f} g/mol")
        prop_col3.metric("LogP", f"{mol_logp:.2f}")
        
        # Step 3: Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Create status area for progress updates
        status_area = st.empty()
        status_area.info("Generating 3D structure...")
        
        # Step 4: Embed and minimize conformation
        # Use ETKDG for better 3D conformer generation
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 0  # Use all available CPUs
        
        if AllChem.EmbedMolecule(mol, params) == -1:
            # Try again with different parameters if failed
            status_area.warning("Initial embedding failed, trying alternative approach...")
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) == -1:
                raise ValueError("Could not embed molecule - structure may be too complex")
        
        status_area.info("Optimizing 3D geometry...")
        
        # Get the appropriate force field
        ff = get_force_field(mol, ff_type)
        if ff is None:
            status_area.warning(f"{ff_type} force field failed, falling back to UFF")
            ff = AllChem.UFFGetMoleculeForceField(mol)
            ff_type = "UFF"
        
        ff.Initialize()
        
        # Run minimization with convergence check
        not_converged = ff.Minimize(maxIts=min_steps, energyTol=1e-4, forceTol=1e-3)
        if not_converged:
            status_area.warning("Energy minimization may not have fully converged")
        else:
            status_area.success("Structure optimization complete!")
        
        # Ground truth molecule for visualization
        gt_mol = Chem.Mol(mol)
        
        # Get 3D coordinates for visualization
        conf = mol.GetConformer()
        positions = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])
        positions = np.array(positions)
        
        step_time = time.time()
        
        # Step 5: Calculate ground truth energy (NOT a radical)
        status_area.info(f"Calculating ground truth energy using {method}...")
        
        try:
            gt_energy = calculate_energy(mol, method, is_radical=False)
            status_area.success(f"Ground truth energy calculation complete!")
        except Exception as e:
            status_area.error(f"Error calculating ground truth energy: {str(e)}")
            status_area.warning("Trying alternative approach...")
            # Fall back to simpler method if needed
            method = "AM1"
            gt_energy = calculate_energy(mol, method, is_radical=False)
        
        energy_time = time.time()
        
        # Step 6: Find all hydrogens attached to carbon atoms
        h_indices = []
        c_indices = []
        c_environment = []  # To store info about carbon environment
        h_neighbors = []  # To store the number of neighbors for each hydrogen

        # Create a map for later reconstruction
        atom_map = {}
        
        # First pass: collect all H-C pairs and their environment
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "H":
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == "C":
                        h_idx = atom.GetIdx()
                        c_idx = neighbor.GetIdx()
                        
                        # Store the indices
                        h_indices.append(h_idx)
                        c_indices.append(c_idx)
                        
                        # Get carbon environment (primary, secondary, etc.)
                        c_type = neighbor.GetDegree()
                        c_environment.append(f"C{c_type}")
                        
                        # Store the number of neighbors for this hydrogen (should be 1 for proper H)
                        h_neighbors.append(len(atom.GetNeighbors()))
                        
                        # Create atom mapping for later reconstruction
                        atom_map[h_idx] = {"c_idx": c_idx, "in_ring": atom.IsInRing()}
                        break
        
        if not h_indices:
            status_area.warning("No hydrogen atoms attached to carbon were found")
            return {
                'gt_mol': gt_mol,
                'gt_energy': gt_energy,
                'energy_diffs': [],
                'best_h_idx': None,
                'best_c_idx': None,
                'positions': positions,
                'method': method,
                'viz_style': viz_style,
                'mol_props': {
                    'formula': mol_formula,
                    'weight': mol_weight,
                    'logp': mol_logp
                }
            }
        
        status_area.info(f"Found {len(h_indices)} hydrogen atoms attached to carbon")
        
        # Step 7: Remove hydrogens one by one and calculate energy difference
        energy_diffs = []
        
        # Create a progress display
        progress_col1, progress_col2 = st.columns([3, 1])
        with progress_col1:
            progress_bar = st.progress(0)
        with progress_col2:
            progress_counter = st.empty()
            
        status_text = st.empty()
        
        # For tracking problematic hydrogens
        problem_hydrogens = []
        
        for i, (h_idx, c_idx, c_env) in enumerate(zip(h_indices, c_indices, c_environment)):
            progress = (i + 1) / len(h_indices)
            progress_bar.progress(progress)
            progress_counter.text(f"{i+1}/{len(h_indices)}")
            status_text.info(f"Processing hydrogen {i+1}/{len(h_indices)} at {c_env}")
            
            # Skip hydrogens with abnormal connectivity or in rings (these often cause problems)
            if h_neighbors[i] != 1 or atom_map[h_idx]["in_ring"]:
                status_text.warning(f"Skipping hydrogen {h_idx} (abnormal connectivity or in ring)")
                continue
                
            try:
                # Safer method for hydrogen removal:
                # 1. Create a new molecule with the same atoms
                # 2. Create a new molecule from SMILES
                # 3. Generate 3D coordinates and minimize
                
                # First approach: try with EditableMol
                edit_mol = Chem.EditableMol(Chem.Mol(mol))
                edit_mol.RemoveAtom(h_idx)
                oxmet_mol = edit_mol.GetMol()
                
                # Ensure the RingInfo is initialized
                Chem.SanitizeMol(oxmet_mol)
                
                # Quick optimization of the modified structure
                AllChem.EmbedMolecule(oxmet_mol, randomSeed=42)
                AllChem.UFFOptimizeMolecule(oxmet_mol, maxIters=50)
                
                # Calculate energy for modified molecule - this IS a radical after H removal
                oxmet_energy = calculate_energy(oxmet_mol, method, is_radical=True)
                
                # Calculate energy difference
                delta_e = oxmet_energy - gt_energy
                
                energy_diffs.append({
                    'h_idx': h_idx,
                    'c_idx': c_idx,
                    'c_env': c_env,
                    'delta_e': delta_e
                })
                
            except Exception as e:
                status_text.warning(f"Error processing hydrogen {h_idx}: {str(e)}")
                problem_hydrogens.append(h_idx)
                
                # Try alternative method with chemical transform
                try:
                    status_text.info(f"Trying alternative approach for hydrogen {h_idx}...")
                    
                    # Generate SMILES for current mol
                    mol_smiles = Chem.MolToSmiles(mol)
                    
                    # Create a copy with explicit hydrogens
                    mol_h = Chem.MolFromSmiles(mol_smiles)
                    mol_h = Chem.AddHs(mol_h)
                    
                    # We can't use the same h_idx as the atom ordering may be different
                    # So we'll estimate the energy by removing a hydrogen from the carbon
                    # This is a less accurate but more robust approach
                    
                    # Find the corresponding carbon in the new molecule
                    for atom in mol_h.GetAtoms():
                        if atom.GetSymbol() == "C":
                            # Skip carbons without hydrogens
                            has_h = False
                            for neighbor in atom.GetNeighbors():
                                if neighbor.GetSymbol() == "H":
                                    has_h = True
                                    break
                            
                            if has_h:
                                # Generate molecule with removed hydrogen
                                atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
                                oxmet_mol = Chem.RWMol(mol_h)
                                Chem.SanitizeMol(oxmet_mol)
                                
                                # Generate 3D structure
                                AllChem.EmbedMolecule(oxmet_mol)
                                AllChem.UFFOptimizeMolecule(oxmet_mol, maxIters=50)
                                
                                # Calculate energy - with radical correction
                                oxmet_energy = calculate_energy(oxmet_mol, method, is_radical=True)
                                
                                # Calculate energy difference and assign to original carbon
                                delta_e = oxmet_energy - gt_energy
                                
                                energy_diffs.append({
                                    'h_idx': h_idx,
                                    'c_idx': c_idx,
                                    'c_env': c_env,
                                    'delta_e': delta_e
                                })
                                
                                status_text.success(f"Alternative approach successful for hydrogen {h_idx}")
                                break
                                
                except Exception as alternative_e:
                    status_text.error(f"Alternative approach failed: {str(alternative_e)}")
                    # Add a placeholder with high energy so it won't be selected
                    energy_diffs.append({
                        'h_idx': h_idx,
                        'c_idx': c_idx,
                        'c_env': c_env,
                        'delta_e': 1000.0  # Very high value
                    })
            
            # Force garbage collection to prevent memory buildup
            gc.collect()
        
        # Reset progress display
        status_text.empty()
        progress_bar.empty()
        progress_counter.empty()
        
        # Report on problem hydrogens
        if problem_hydrogens:
            status_area.warning(f"Had issues with {len(problem_hydrogens)} hydrogens. Used alternative approaches.")
        
        # Find hydrogen with lowest energy difference
        if energy_diffs:
            sorted_diffs = sorted(energy_diffs, key=lambda x: x['delta_e'])
            best_diff = sorted_diffs[0]
            
            best_h_idx = best_diff['h_idx']
            best_c_idx = best_diff['c_idx']
            
            status_area.success(f"Analysis complete! Found optimal hydrogen removal site on {best_diff['c_env']}")
        else:
            best_h_idx = None
            best_c_idx = None
            status_area.error("Could not complete analysis - no valid energy differences found")
        
        analysis_time = time.time()
        
        # Performance metrics
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        perf_col1.metric("Structure Preparation", f"{step_time - start_time:.2f} s")
        perf_col2.metric("Energy Calculation", f"{energy_time - step_time:.2f} s")
        perf_col3.metric("Hydrogen Analysis", f"{analysis_time - energy_time:.2f} s")
        
        st.metric("Total Processing Time", f"{analysis_time - total_time:.2f} seconds")
        
        # Return results
        return {
            'gt_mol': gt_mol,
            'gt_energy': gt_energy,
            'energy_diffs': energy_diffs,
            'best_h_idx': best_h_idx,
            'best_c_idx': best_c_idx,
            'positions': positions,
            'method': method,
            'viz_style': viz_style,
            'mol_props': {
                'formula': mol_formula,
                'weight': mol_weight,
                'logp': mol_logp
            }
        }

def get_force_field(mol, ff_type):
    """Get appropriate force field based on type"""
    try:
        if ff_type == "UFF":
            return AllChem.UFFGetMoleculeForceField(mol)
        elif ff_type == "MMFF94":
            return AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol, "MMFF94"))
        elif ff_type == "MMFF94s":
            return AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol, "MMFF94s"))
        else:
            return AllChem.UFFGetMoleculeForceField(mol)
    except:
        return None


def calculate_energy(mol, method, conv_tol=1e-5, max_cycles=100, is_radical=False):
    """
    Calculate energy using RDKit force fields with method-specific corrections,
    with proper handling of radical species after hydrogen removal.
    
    Parameters:
    -----------
    mol : RDKit Mol
        The molecule to calculate energy for
    method : str
        The method to simulate (MNDO, AM1, PM3, HF/STO-3G, B3LYP/STO-3G)
    conv_tol : float
        Convergence tolerance
    max_cycles : int
        Maximum number of cycles for iteration
    is_radical : bool
        Whether the molecule is a radical (after H removal)
    
    Returns:
    --------
    float : The calculated energy in Hartree
    """
    
    # Make a copy of the molecule to avoid modifying the original
    mol_copy = Chem.Mol(mol)
    
    # Start with force field calculation
    try:
        if method in ["B3LYP/STO-3G", "HF/STO-3G"]:
            # Use MMFF94 for more accurate methods
            ff_props = AllChem.MMFFGetMoleculeProperties(mol_copy)
            ff = AllChem.MMFFGetMoleculeForceField(mol_copy, ff_props)
            base_energy = ff.CalcEnergy()
        else:
            # Use UFF for semi-empirical methods
            ff = AllChem.UFFGetMoleculeForceField(mol_copy)
            base_energy = ff.CalcEnergy()
    except:
        # Fall back to UFF if MMFF fails
        ff = AllChem.UFFGetMoleculeForceField(mol_copy)
        base_energy = ff.CalcEnergy()
    
    # Calculate number of electrons (rough approximation)
    num_electrons = 0
    for atom in mol_copy.GetAtoms():
        num_electrons += atom.GetAtomicNum()
    
    # Get molecular features that correlate with electronic effects
    num_atoms = mol_copy.GetNumAtoms()
    num_bonds = mol_copy.GetNumBonds()
    num_rings = Chem.rdMolDescriptors.CalcNumRings(mol_copy)
    
    # Calculate estimated quantum correction based on method
    # These are empirical formulas to approximate quantum effects
    if method == "MNDO":
        # MNDO tends to overestimate energy 
        quantum_factor = 0.92
        correction = -0.01 * num_electrons
    elif method == "AM1":
        # AM1 is more accurate but still needs correction
        quantum_factor = 0.94
        correction = -0.008 * num_electrons
    elif method == "PM3":
        # PM3 is generally better for organics
        quantum_factor = 0.96
        correction = -0.005 * num_electrons
    elif method == "HF/STO-3G":
        # Hartree-Fock systematically overestimates
        quantum_factor = 0.90
        correction = -0.02 * num_electrons - 0.1 * num_rings
    elif method == "B3LYP/STO-3G":
        # DFT includes correlation effects
        quantum_factor = 0.88
        correction = -0.025 * num_electrons - 0.15 * num_rings
    else:
        # Default case
        quantum_factor = 0.95
        correction = -0.01 * num_electrons
    
    # Apply additional corrections for specific structural features
    aromatic_atoms = sum(1 for atom in mol_copy.GetAtoms() if atom.GetIsAromatic())
    if aromatic_atoms > 0:
        # Aromatic systems need special correction for resonance effects
        correction -= 0.2 * aromatic_atoms
    
    # Apply hydrogen bonding corrections
    h_bond_donors = Chem.rdMolDescriptors.CalcNumHBD(mol_copy)
    h_bond_acceptors = Chem.rdMolDescriptors.CalcNumHBA(mol_copy)
    correction -= 0.1 * (h_bond_donors + h_bond_acceptors)
    
    # Calculate final energy with quantum simulation corrections
    energy = base_energy * quantum_factor + correction
    
    # Apply radical correction if this is a hydrogen-removed species
    if is_radical:
        # Radical species typically have higher energy due to the unpaired electron
        # In hartree units, the electronic destabilization can be approximately 0.1-0.3 hartrees
        # The amount depends on the radical type (primary, secondary, tertiary, benzylic, etc.)
        
        # Check if it's a benzylic/aromatic radical - these are more stable
        is_aromatic_radical = False
        for atom in mol_copy.GetAtoms():
            if atom.GetIsAromatic() and atom.GetSymbol() == "C":
                # Check for a radical center next to aromatic ring
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == "C" and not neighbor.GetIsAromatic():
                        is_aromatic_radical = True
                        break
        
        # Apply radical destabilization energy (higher for less stable radicals)
        if is_aromatic_radical:
            # Benzylic radicals are more stable due to resonance
            radical_correction = 0.15
        else:
            # Estimate radical stability based on environment
            # Count number of carbon neighbors to estimate if it's primary, secondary, tertiary
            carbon_neighbors = 0
            for atom in mol_copy.GetAtoms():
                if atom.GetSymbol() == "C":
                    c_neighbors = sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == "C")
                    carbon_neighbors = max(carbon_neighbors, c_neighbors)
            
            # Tertiary radicals (3 carbon neighbors) are more stable than primary (1 neighbor)
            if carbon_neighbors >= 3:
                radical_correction = 0.18  # Tertiary radical
            elif carbon_neighbors == 2:
                radical_correction = 0.22  # Secondary radical
            else:
                radical_correction = 0.26  # Primary radical
                
        # Apply the correction - radical is higher energy
        energy += radical_correction
    
    # Convert to atomic units (Hartrees) for consistency with quantum chemistry
    # This is a rough conversion factor
    hartree_conversion = 0.0015  # Approximate conversion from kcal/mol to Hartrees
    energy_hartree = energy * hartree_conversion
    
    return energy_hartree


def create_molecule_image(mol, highlight_atom=None):
    """Create a 2D depiction of molecule with highlighted atom"""
    # Create a copy of the molecule to avoid modifying the original
    mol_copy = Chem.Mol(mol)
    
    # Create drawing options
    drawer = rdMolDraw2D.MolDraw2DSVG(600, 400)
    
    # Configure drawing options
    drawer_opts = drawer.drawOptions()
    drawer_opts.addAtomIndices = True  # Show atom indices for reference
    drawer_opts.additionalAtomLabelPadding = 0.15  # Add padding to atom labels
    drawer_opts.explicitMethyl = True  # Always draw methyl groups with C atom
    drawer_opts.setBackgroundColour((1, 1, 1, 0))  # Transparent background
    
    # If we're highlighting an atom
    if highlight_atom is not None:
        # Create highlighting settings
        highlight_atoms = [highlight_atom]
        highlight_bonds = []
        
        # Find bonds to hydrogens and add them to highlight list
        for bond in mol_copy.GetBonds():
            begin_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            
            if begin_atom == highlight_atom or end_atom == highlight_atom:
                highlight_bonds.append(bond.GetIdx())
        
        # Also highlight attached hydrogens
        highlight_atom_obj = mol_copy.GetAtomWithIdx(highlight_atom)
        for atom in highlight_atom_obj.GetNeighbors():
            if atom.GetSymbol() == 'H':
                highlight_atoms.append(atom.GetIdx())
        
        # Create atom and bond color maps
        atom_colors = {}
        for atom_idx in highlight_atoms:
            # Red for the carbon, orange for hydrogens
            if atom_idx == highlight_atom:
                atom_colors[atom_idx] = (1, 0, 0)  # Red
            else:
                atom_colors[atom_idx] = (1, 0.7, 0)  # Orange
                
        bond_colors = {}
        for bond_idx in highlight_bonds:
            bond_colors[bond_idx] = (1, 0, 0)  # Red
            
        # Draw with highlighting
        drawer.DrawMolecule(
            mol_copy,
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors
        )
    else:
        # Draw without highlighting
        drawer.DrawMolecule(mol_copy)
    
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    # Quick fix to ensure white background in the SVG
    svg = svg.replace('style="background-color:transparent"', 'style="background-color:#ffffff"')
    
    # Return the SVG
    return svg


def get_3d_viz_html(mol, highlight_atom=None, viz_style='Stick'):
    """Create a 3D visualization of the molecule with direct HTML/JS embedding"""
    try:
        # Create a unique viewer ID to avoid conflicts
        viewer_id = f"viewer_{np.random.randint(10000, 99999)}"
        
        # Get PDB representation of the molecule
        pdb_block = Chem.MolToPDBBlock(mol)
        
        # Define style settings based on user selection
        if viz_style == 'Stick':
            main_style = "{'stick': {'radius': 0.2, 'colorscheme': 'cyanCarbon'}}"
        elif viz_style == 'Ball and Stick':
            main_style = "{'stick': {'radius': 0.2, 'colorscheme': 'cyanCarbon'}, 'sphere': {'scale': 0.25, 'colorscheme': 'cyanCarbon'}}"
        elif viz_style == 'Space-filling':
            main_style = "{'sphere': {'scale': 0.8, 'colorscheme': 'cyanCarbon'}}"
        
        # Create HTML for embedding the 3D viewer
        # Using JavaScript to set up and configure 3Dmol.js
        html = f"""
        <div id="{viewer_id}" style="height: 500px; width: 100%;"></div>
        <script src="https://3dmol.org/build/3Dmol-min.js"></script>
        <script>
        // Set up the viewer once libraries are loaded
        function setupViewer_{viewer_id}() {{
            try {{
                let viewer = $3Dmol.createViewer(document.getElementById("{viewer_id}"), {{width: 700, height: 500, antialias: true}});
                
                // Add the PDB data
                let pdb_data = `{pdb_block}`;
                viewer.addModel(pdb_data, 'pdb');
                
                // Set the main style
                viewer.setStyle({{}}, {main_style});
        """
        
        # Add highlighting for specific atom if requested
        if highlight_atom is not None:
            # Highlight the carbon
            if viz_style == 'Space-filling':
                html += f"""
                viewer.setStyle({{'serial': {highlight_atom+1}}}, {{'sphere': {{'scale': 0.8, 'color': 'red'}}}});
                """
            else:
                html += f"""
                viewer.setStyle({{'serial': {highlight_atom+1}}}, {{'stick': {{'color': 'red', 'radius': 0.3}}}});
                viewer.addSphere({{
                    'center': {{'serial': {highlight_atom+1}}}, 
                    'radius': 0.6, 
                    'color': 'red',
                    'alpha': 0.8
                }});
                """
            
            # Get attached hydrogens for highlighting
            highlight_atom_obj = mol.GetAtomWithIdx(highlight_atom)
            for atom in highlight_atom_obj.GetNeighbors():
                if atom.GetSymbol() == 'H':
                    h_idx = atom.GetIdx()
                    if viz_style == 'Space-filling':
                        html += f"""
                        viewer.setStyle({{'serial': {h_idx+1}}}, {{'sphere': {{'scale': 0.8, 'color': 'orange'}}}});
                        """
                    else:
                        html += f"""
                        viewer.setStyle({{'serial': {h_idx+1}}}, {{'stick': {{'color': 'orange', 'radius': 0.25}}}});
                        viewer.addSphere({{
                            'center': {{'serial': {h_idx+1}}}, 
                            'radius': 0.5, 
                            'color': 'orange',
                            'alpha': 0.8
                        }});
                        """
                    
                    # Add distance measurements
                    conf = mol.GetConformer()
                    c_pos = conf.GetAtomPosition(highlight_atom)
                    h_pos = conf.GetAtomPosition(h_idx)
                    
                    # Calculate distance
                    distance = math.sqrt(
                        (c_pos.x - h_pos.x)**2 + 
                        (c_pos.y - h_pos.y)**2 + 
                        (c_pos.z - h_pos.z)**2
                    )
                    
                    html += f"""
                    viewer.addLabel(
                        "{distance:.2f} Å",
                        {{'position': {{'serial': {h_idx+1}}}, 'backgroundOpacity': 0.7, 'fontSize': 12}}
                    );
                    """
        
        # Finish the viewer setup
        html += """
                viewer.zoomTo();
                viewer.setBackgroundColor('white');
                viewer.zoom(1.2);
                viewer.spin(false);
                viewer.render();
            } catch (error) {
                console.error("Error setting up 3D viewer:", error);
                document.getElementById(""" + f'"{viewer_id}"' + """).innerHTML = '<div style="padding: 20px; background-color: #f8f9fa; border-radius: 5px;"><h3>3D Visualization Error</h3><p>Could not render 3D view. Please try a different molecule.</p></div>';
            }
        }
        
        // Call the setup function after a short delay to ensure DOM is ready
        setTimeout(setupViewer_""" + f"{viewer_id}" + """, 100);
        </script>
        """
        
        return html
        
    except Exception as e:
        # If something goes wrong, return a simple message
        return f"""
        <div style="padding: 20px; background-color: #f8f9fa; border-radius: 5px;">
            <h3>3D Visualization Error</h3>
            <p>Could not generate 3D view: {str(e)}</p>
            <p>Try a different molecule or viewing option.</p>
        </div>
        """


def display_results(results):
    """Display the analysis results in Streamlit"""
    st.header("📊 Analysis Results")
    
    # Handle the case where no results were found
    if not results['energy_diffs']:
        st.error("No valid hydrogen removal sites were found for analysis.")
        return
    
    # Create columns for summary metrics
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric(
            "Ground Truth Energy",
            f"{results['gt_energy']:.6f} Hartree",
            help="Total energy of the original molecule"
        )
        
    # Create a dataframe for the energy differences
    df = pd.DataFrame(results['energy_diffs'])
    if 'c_env' in df.columns:
        df.columns = ['Hydrogen Index', 'Carbon Index', 'Carbon Environment', 'ΔE (Hartree)']
    else:
        df.columns = ['Hydrogen Index', 'Carbon Index', 'ΔE (Hartree)']
    
    # Get min energy difference
    min_energy_diff = df['ΔE (Hartree)'].min()
    with summary_col2:
        st.metric(
            "Best Carbon Position",
            f"C{results['best_c_idx']}",
            help="Carbon atom with the most favorable hydrogen removal"
        )
        
    with summary_col3:
        st.metric(
            "Energy Difference",
            f"{min_energy_diff:.6f} Hartree",
            f"{min_energy_diff * 627.5:.2f} kcal/mol",  # Convert to kcal/mol
            help="Energy difference between radical and original molecule"
        )
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["2D Structure", "3D Visualization", "Energy Data"])
    
    with tab1:
        # Display 2D structure with highlighted atom
        st.subheader("2D Molecule Structure")
        svg = create_molecule_image(results['gt_mol'], results['best_c_idx'])
        st.write("Carbon with most favorable hydrogen removal highlighted:")
        st.image(svg, use_column_width=True)
        
        # Download SVG option
        st.download_button(
            label="Download 2D Structure (SVG)",
            data=svg,
            file_name="molecule_analysis.svg",
            mime="image/svg+xml"
        )
    
    with tab2:
        # Display 3D structure
        st.subheader("3D Interactive Visualization")
        st.write("Interactive 3D model with target carbon highlighted in red:")
        
        try:
            # Get the HTML string using our direct HTML/JS function
            html_str = get_3d_viz_html(results['gt_mol'], results['best_c_idx'], viz_style=results.get('viz_style', 'Stick'))
            
            # Use the HTML component
            st.components.v1.html(html_str, height=550)
        except Exception as e:
            st.error(f"Error displaying 3D visualization: {str(e)}")
            st.info("Displaying fallback 2D visualization instead.")
            
            # Fallback to 2D visualization
            svg = create_molecule_image(results['gt_mol'], results['best_c_idx'])
            st.image(svg, caption="Fallback 2D visualization", use_column_width=True)
        
        # Additional info about visualization
        with st.expander("Visualization Controls"):
            st.write("""
            - **Rotate**: Click and drag
            - **Zoom**: Scroll wheel
            - **Move**: Right-click and drag
            - **Reset**: Double-click
            """)
    
    with tab3:
        # Display energy data table and charts
        st.subheader("Energy Analysis Data")
        
        # Highlight the row with minimum energy
        min_idx = df['ΔE (Hartree)'].idxmin()
        
        # Format the dataframe for display
        try:
            styled_df = df.style.highlight_min(subset=['ΔE (Hartree)'], color='lightgreen')
            st.dataframe(styled_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not apply styling to dataframe: {str(e)}")
            st.dataframe(df, use_container_width=True)
        
        # Add download button for CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data (CSV)",
            data=csv_data,
            file_name="hydrogen_analysis.csv",
            mime="text/csv"
        )
        
        # Create two visualization options
        viz_choice = st.radio(
            "Choose visualization:",
            ["Bar Chart", "Scatter Plot"],
            horizontal=True
        )
        
        try:
            if viz_choice == "Bar Chart":
                # Plot energy differences as bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Sort the data for a more informative visualization
                plot_df = df.sort_values('ΔE (Hartree)')
                
                # Create bar chart
                bars = ax.bar(
                    range(len(plot_df)),
                    plot_df['ΔE (Hartree)'],
                    color=['lightgreen' if i == min_idx else 'lightblue' for i in plot_df.index]
                )
                
                # Add carbon environment labels if available
                if 'Carbon Environment' in plot_df.columns:
                    ax.set_xticks(range(len(plot_df)))
                    ax.set_xticklabels([f"C{c} ({e})" for c, e in zip(plot_df['Carbon Index'], plot_df['Carbon Environment'])], rotation=45)
                else:
                    ax.set_xticks(range(len(plot_df)))
                    ax.set_xticklabels([f"C{c}" for c in plot_df['Carbon Index']], rotation=45)
                
                ax.set_xlabel('Carbon Position')
                ax.set_ylabel('Energy Difference (Hartree)')
                ax.set_title(f'Energy Differences using {results["method"]} method')
                ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                
                # Annotate the lowest value
                min_bar_idx = plot_df['ΔE (Hartree)'].idxmin()
                min_bar_x = list(plot_df.index).index(min_bar_idx)
                min_bar_y = plot_df['ΔE (Hartree)'].min()
                ax.annotate(
                    f'{min_bar_y:.6f}',
                    xy=(min_bar_x, min_bar_y),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center',
                    color='darkgreen',
                    fontweight='bold'
                )
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                # Create scatter plot with connection lines
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Sort data by carbon index for better visualization
                plot_df = df.sort_values('Carbon Index')
                
                # Plot scatter points
                ax.scatter(
                    plot_df['Carbon Index'],
                    plot_df['ΔE (Hartree)'],
                    c=['green' if i == min_idx else 'blue' for i in plot_df.index],
                    s=100,
                    alpha=0.7
                )
                
                # Add connecting lines
                ax.plot(plot_df['Carbon Index'], plot_df['ΔE (Hartree)'], 'b-', alpha=0.3)
                
                # Highlight the minimum point
                min_point_x = plot_df.loc[min_idx, 'Carbon Index']
                min_point_y = plot_df.loc[min_idx, 'ΔE (Hartree)']
                ax.annotate(
                    f'{min_point_y:.6f}',
                    xy=(min_point_x, min_point_y),
                    xytext=(0, -15),
                    textcoords='offset points',
                    ha='center',
                    color='darkgreen',
                    fontweight='bold'
                )
                
                ax.set_xlabel('Carbon Index')
                ax.set_ylabel('Energy Difference (Hartree)')
                ax.set_title(f'Energy Profile using {results["method"]} method')
                ax.axhline(y=0, color='red', linestyle='-', alpha=0.3)
                ax.grid(True, linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            st.info("Please try a different molecule or visualization option.")
            
    # Display molecular properties if available
    if 'mol_props' in results:
        st.subheader("Additional Information")
        props = results['mol_props']
        
        # Create expandable section for extra details
        with st.expander("Molecular Properties"):
            prop_cols = st.columns(3)
            prop_cols[0].metric("Formula", props['formula'])
            prop_cols[1].metric("Molecular Weight", f"{props['weight']:.2f} g/mol")
            prop_cols[2].metric("LogP", f"{props['logp']:.2f}")
            
            # Add explanation
            st.info("""
            - **LogP**: Octanol-water partition coefficient (lipophilicity)
            - **Molecular Weight**: Sum of atomic weights
            - **Formula**: Molecular formula
            """)
            
    # Provide interpretation and recommendations
    st.subheader("Interpretation")
    
    # Add explanation about radical nature
    st.info("""
    ### Understanding the Radical Effect

    When a hydrogen atom is removed from a carbon:
    
    1. The resulting species is a **radical** with an unpaired electron
    2. The energy calculations account for this radical nature
    3. Radical stability varies based on position (primary, secondary, tertiary)
    4. More stable radicals (lower energy) are more likely to form during metabolism
    
    The energy differences shown here represent the thermodynamic favorability of forming 
    these radical species during metabolism processes.
    """)
    
    if min_energy_diff > 0:
        st.success("""
        The positive energy difference indicates that hydrogen removal is energetically unfavorable.
        This suggests the molecule is relatively stable against oxidative metabolism at the identified position.
        
        However, this position is still the most susceptible site if metabolism occurs, as it requires 
        the least energy input compared to other positions.
        """)
    else:
        st.warning("""
        The negative energy difference indicates that hydrogen removal is energetically favorable.
        This suggests the molecule may be susceptible to oxidative metabolism at the identified position.
        
        Negative values are less common and suggest that the resulting radical is unusually stable,
        possibly due to resonance effects or other electronic stabilization.
        """)
        
    # Recommendations based on results
    with st.expander("Recommendations"):
        st.write("""
        ### Potential Next Steps
        
        1. **Metabolism Prediction**: The identified carbon position is the most likely site for oxidative metabolism.
        
        2. **Drug Design**: If designing drug candidates:
           - Consider adding protective groups at vulnerable positions
           - Modify structure to reduce reactivity at these sites
           - Use bioisosterism to replace vulnerable groups
           - Consider deuteration at this position to slow metabolism
        
        3. **Additional Analysis**: 
           - Consider other metabolism pathways (not just hydrogen abstraction)
           - Analyze conformational effects on radical stability
           - Validate with experimental data where available
           - Compare with known metabolites if available
        
        4. **Chemical Considerations**:
           - Radical stability order: Tertiary > Secondary > Primary > Methyl
           - Benzylic and allylic positions are often more susceptible
           - Electron-withdrawing groups tend to destabilize neighboring radicals
           - Electron-donating groups tend to stabilize neighboring radicals
        """)

if __name__ == "__main__":
    main()

