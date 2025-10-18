#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

#################################################################### imports ###

import sys

import myPythonLibrary as mypy
import sys
res_folder = sys.argv[0][:-3]
import os
from pathlib import Path
local_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(local_path))

import dolfin_mech as dmech
sys.path.remove(str(local_path))
################################################################# parameters ###

mat_params = {
    "alpha":0.16,
    "gamma":0.5,
    "c1":0.2,
    "c2":0.4,
    "kappa":1e2,
    "eta":1e-5}

####################################################################### test ###

res_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=res_folder,
    perform_tests=1,
    stop_at_failure=1,
    clean_after_tests=1)

dim_lst  = [ ]
dim_lst += [2]
# dim_lst += [3]
for dim in dim_lst:

    inverse_lst  = [ ]
    inverse_lst += [0]
    inverse_lst += [1]
    for inverse in inverse_lst:

        porosity_lst  = [                        ]
        porosity_lst += ["constant"              ]
        porosity_lst += ["mesh_function_constant"]
        porosity_lst += ["mesh_function_xml"     ]
        porosity_lst += ["function_constant"     ]
        porosity_lst += ["function_xml"          ]
        for porosity in porosity_lst:

            scaling_lst  = [        ]
            scaling_lst += ["no"    ]
            scaling_lst += ["linear"]
            for scaling in scaling_lst:

                load_lst  = [           ]
                load_lst += ["internal" ]
                load_lst += ["external0"] if (inverse) else ["external"]
                if (dim==3): load_lst += ["p_boundary_condition0"] if (inverse) else ["p_boundary_condition"]
                for load in load_lst:

                    print("dim ="     , dim     )
                    print("inverse =" , inverse )
                    print("porosity =", porosity)
                    print("scaling =" , scaling )
                    print("load ="    , load    )

                    res_basename  = sys.argv[0][:-3]
                    res_basename += "-dim="+str(dim)
                    res_basename += "-inverse="+str(inverse)
                    res_basename += "-porosity="+str(porosity)
                    res_basename += "-scaling="+str(scaling)
                    res_basename += "-load="+str(load)
                    print(mat_params)
                    dmech.run_RivlinCube_PoroHyperelasticity(
                        dim             = dim                                                 ,
                        inverse         = inverse                                             ,
                        porosity_params = {"type":porosity}                                   ,
                        cube_params     = {"l":0.1, "mesh_filebasename":res_folder+"/"+"mesh"},
                        mat_params      = {"scaling":scaling, "parameters":mat_params}        ,
                        step_params     = {"dt_min":1e-4}                                     ,
                        load_params     = {"type":load}                                       ,
                        res_basename    = res_folder+"/"+res_basename                         ,
                        plot_curves     = 0                                                   ,
                        verbose         = 0                                                   )

                    test.test(res_basename)
