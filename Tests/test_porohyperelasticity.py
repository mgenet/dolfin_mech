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
import dolfin_mech     as dmech

################################################################# parameters ###

mat_params = {
    "alpha":0.16,
    "gamma":0.5,
    "c1":0.2,
    "c2":0.4,
    "kappa":1e+2,
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

        known_porosity_lst  = [     ]
        # known_porosity_lst += ["Phis0"] if (inverse == 0) else ["phis"]
        known_porosity_lst += ["Phis0"]
        known_porosity_lst += ["phis"]
        for known_porosity in known_porosity_lst:

            init_porosity_lst  = [                        ]
            init_porosity_lst += ["constant"              ]
            init_porosity_lst += ["mesh_function_constant"]
            init_porosity_lst += ["mesh_function_xml"     ]
            init_porosity_lst += ["function_constant"     ]
            init_porosity_lst += ["function_xml"          ]
            for init_porosity in init_porosity_lst:

                scaling_lst  = [        ]
                scaling_lst += ["no"    ]
                scaling_lst += ["linear"]
                for scaling in scaling_lst:

                    load_lst  = [           ]
                    load_lst += ["internal" ]
                    load_lst += ["external0"] if (inverse) else ["external"]
                    if (dim==3): load_lst += ["p_boundary_condition0"] if (inverse) else ["p_boundary_condition"]
                    for load in load_lst:

                        print("dim ="           , dim           )
                        print("inverse ="       , inverse       )
                        print("known_porosity =", known_porosity)
                        print("init_porosity =" , init_porosity )
                        print("scaling ="       , scaling       )
                        print("load ="          , load          )

                        res_basename  = sys.argv[0][:-3]
                        res_basename += "-dim="+str(dim)
                        res_basename += "-inverse="+str(inverse)
                        res_basename += "-known_porosity="+str(known_porosity)
                        res_basename += "-init_porosity="+str(init_porosity)
                        res_basename += "-scaling="+str(scaling)
                        res_basename += "-load="+str(load)

                        dmech.run_RivlinCube_PoroHyperelasticity(
                            dim             = dim                                                 ,
                            inverse         = inverse                                             ,
                            porosity_params = {"known":known_porosity, "type":init_porosity}      ,
                            cube_params     = {"l":0.1, "mesh_filebasename":res_folder+"/"+"mesh"},
                            mat_params      = {"scaling":scaling, "parameters":mat_params}        ,
                            step_params     = {"dt_min":1e-4}                                     ,
                            load_params     = {"type":load}                                       ,
                            res_basename    = res_folder+"/"+res_basename                         ,
                            plot_curves     = 0                                                   ,
                            verbose         = 1                                                   )

                        test.test(res_basename)
