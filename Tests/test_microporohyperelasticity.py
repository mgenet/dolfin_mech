#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Mahdi Manoochehrtayebi, 2020-2024                                    ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

#################################################################### imports ###

import sys

import myPythonLibrary as mypy
import dolfin_mech     as dmech

####################################################################### test ###

res_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=res_folder,
    perform_tests=1,
    stop_at_failure=1,
    clean_after_tests=1,
    tester_numpy_tolerance=1e-2)

dim_lst  = []
dim_lst += [2]
dim_lst += [3]
for dim in dim_lst:

    bcs_lst  = []
    bcs_lst += ["kubc"]
    bcs_lst += ["pbc"]
    for bcs in bcs_lst:

        load_lst  = []
        load_lst += ["internal_pressure"]
        load_lst += ["macroscopic_stretch"]
        load_lst += ["macroscopic_stress"]
        for load in load_lst:

            print("dim =",dim)
            print("bcs =",bcs)
            print("load =",load)

            res_basename  = sys.argv[0][:-3]
            res_basename += "-dim="+str(dim)
            res_basename += "-bcs="+str(bcs)
            res_basename += "-load="+str(load)

            mesh_params = {}
            mesh_params["mesh_filebasename"] = res_folder+"/mesh"
            mesh_params["dim"] = dim
            mesh_params["xmin"] = 0.
            mesh_params["ymin"] = 0.
            mesh_params["zmin"] = 0.
            mesh_params["xmax"] = 1.
            mesh_params["ymax"] = 1.
            mesh_params["zmax"] = 1.
            mesh_params["xshift"] = -0.3
            mesh_params["yshift"] = -0.3
            mesh_params["zshift"] = -0.3
            mesh_params["r0"] = 0.2
            mesh_params["l"] = 0.1

            load_params = {}
            if (load == "internal_pressure"):
                load_params["pf"] = +0.2
                for i in range(dim):
                 for j in range (dim):
                    load_params["sigma_bar_"+str(i)+str(j)] = 0.
            elif (load == "macroscopic_stretch"):
                load_params["pf"] = 0.
                load_params["U_bar_00"] = 0.5
                for i in range(dim):
                 for j in range (dim):
                  if ((i != 0) or (j != 0)):
                    load_params["sigma_bar_"+str(i)+str(j)] = 0.
            elif (load == "macroscopic_stress"):
                load_params["pf"] = 0.
                for i in range(dim):
                 for j in range (dim):
                    load_params["sigma_bar_"+str(i)+str(j)] = 0.
                load_params["sigma_bar_00"] = 0.5

            dmech.run_HollowBox_MicroPoroHyperelasticity(
                dim=dim,
                mesh_params=mesh_params,
                mat_params={"model":"CGNHMR", "parameters":{"E":1.0, "nu":0.3}},
                bcs=bcs,
                step_params={"dt_ini":1e-1, "dt_min":1e-3},
                load_params=load_params,
                res_basename=res_folder+"/"+res_basename,
                verbose=0)

            test.test(res_basename)
