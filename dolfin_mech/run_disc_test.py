import dolfin
import math
import numpy
import sys
import os
from pathlib import Path
local_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(local_path))

import dolfin_mech as dmech


dmech.run_Disc_Hyperelasticity(
    incomp=0,  # 0 = compressible, 1 = incompressible
    mesh_params={
        "X0": 0.5,
        "Y0": 0.5,
        "R": 0.3,
        "l": 0.03,  # characteristic length for meshing
        "mesh_filebasename": "results/mesh"
    },
    mat_params={
        "model": "NeoHookean",
        "parameters": {
            "mu": 10.0,
            "lmbda": 10.0
        }
    },
    step_params={
        "Deltat": 1.0,
        "dt_ini": 0.1,
        "dt_min": 0.01
    },
    load_params={
        "type": "disp",  # or "pres"
        "dR": 0.05  # radial displacement
    },
    res_basename="results/run_Disc_Hyperelasticity",
    write_vtus_with_preserved_connectivity=True,
    verbose=1
)