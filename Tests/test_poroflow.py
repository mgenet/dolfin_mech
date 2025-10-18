import sys

#import myPythonLibrary as mypy
import dolfin_mech as dmech
from run_Poroflow import run_PoroDisc_Coupled

####################################################################### test ###

res_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=res_folder,
    perform_tests=1,
    stop_at_failure=1,
    clean_after_tests=1)

# Only test the compressible case for now (0 = compressible)
incomp_lst = [0]

for incomp in incomp_lst:

    mat_model = "CGNH" if not incomp else "NH"

    mat_params = {
        "skel": {"parameters": {"mu": 10.0, "lmbda": 10.0}, "scaling": 1.0},
        "bulk": {"parameters": {"mu": 5.0, "lmbda": 5.0}, "scaling": 1.0},
        "pore": {"parameters": {"kappa": 1.0}, "scaling": 1.0}
    }

    load_lst = ["disp"]  # Add other cases if needed

    for load in load_lst:

        print("incomp =", incomp)
        print("load =", load)

        res_basename = sys.argv[0][:-3]
        res_basename += f"-incomp={incomp}"
        res_basename += f"-load={load}"

        run_PoroDisc_Coupled(
            mat_params=mat_params,
            mesh_params={
                "X0": 0.5, "Y0": 0.5, "R": 0.3,
                "l": 0.03,
                "mesh_filebasename": res_folder + "/mesh"
            },
            step_params={"dt_ini": 1 / 10, "dt_min": 1 / 100},
            load_params={"dR": 0.05},  # For radial displacement
            res_basename=res_folder + "/" + res_basename,
            verbose=0
        )

        test.test(res_basename)