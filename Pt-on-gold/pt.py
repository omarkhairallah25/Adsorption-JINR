import ase
from ase import Atoms
from ase.build import fcc111, add_adsorbate
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.visualize import view
import matplotlib.pyplot as plt
import numpy as np

#First: Building Gold Slab and Relaxing
#params and building the slab
ss=(5,5,5) #slab size
vac= 15.0 #vacuum

slab= fcc111('Au', size=ss, vacuum=vac)
slab.calc= EMT()

# Relaxing and printing the output
BFGS(slab, logfile=None).run(fmax=0.05)
Eslab= slab.get_potential_energy() #slab energy
print(f"Clean Au slab energy: {Eslab:.4f} eV")

#Second: Isolated Pt atom energy
pt = Atoms('Pt', positions=[(0, 0, 0)], cell=[10, 10, 10])
pt.calc = EMT()
Ept= pt.get_potential_energy()

print(f"Isolated Pt atom energy: {Ept:.4f} eV")

# Different adsorption sites for Pt
sites = ['ontop', 'bridge', 'fcc', 'hcp']
results = {}

for site in sites:
    
    spt = fcc111('Au', size=ss, vacuum=vac) #new slab for each site 
    fixed = list(range(len(spt)) )
    spt.constraints = [FixAtoms(indices=fixed)] #Fixing gold atoms only to let Pt relax
    add_adsorbate(spt, 'Pt', height=2.0, position=site)
    spt.calc = EMT()
    # Relaxing
    opt = BFGS(spt, logfile=None)
    opt.run(fmax=0.05)
    
    Etot = spt.get_potential_energy()
    Eads = Etot - Eslab - Ept
    
    results[site] = Eads
    print(f"  Total energy: {Etot:.4f} eV")
    print(f"  Adsorption energy at {site}: {Eads:.4f} eV")
# Printing all energies
print(f"  Slab size: {ss}")
print(f"  Number of Au atoms: {len(slab)}")
print(f"  Au slab energy:     {Eslab:.4f} eV")
print(f"  Pt atom energy:     {Ept:.4f} eV")
print(f"  {'Site':<10} {'Eads (eV)':<15} ")
for site, Eads in results.items():
    print(f"  {site:<10} {Eads:<15.4f} ")
view(spt)

