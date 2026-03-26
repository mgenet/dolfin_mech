################################################################################
###                                                                          ###
### Created by Haotian XIAO, 2024-2027                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
###                                                                          ###
### And Martin Genet, 2018-2025                                              ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

#################################################################### imports ###

import sys
import dolfin
import numpy
import myPythonLibrary as mypy
import dolfin_mech as dmech

####################################################################### test ###

res_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=res_folder,
    perform_tests=1,
    stop_at_failure=1,
    clean_after_tests=1,
    tester_numpy_tolerance=1e-2)

mat_params = {
    "alpha":0.16,
    "gamma":0.5,
    "c1":0.2,
    "c2":0.4,
    "kappa":1,
    "eta":1e-5}

dim_lst = [2]
# dim_lst += [3]

for dim in dim_lst:

    bcs_lst = []
    #bcs_lst += ["kubc"]
    bcs_lst += ["pbc"]
    for bcs in bcs_lst:

        load_lst = []
        load_lst += ["internal_air_pressure"]
        load_lst += ["liquid_pressure_gradient"]
        load_lst += ["macroscopic_stretch_with_flow"]
        
        for load in load_lst:

            print("dim =", dim)
            print("bcs =", bcs)
            print("load =", load)

            res_basename  = sys.argv[0][:-3]
            res_basename += "-dim=" + str(dim)
            res_basename += "-bcs=" + str(bcs)
            res_basename += "-load=" + str(load)

            load_params = {
                "solid": {},
                "liquid": {},
                "air": {},
            }

            # ---------------- solid loading ----------------
            for i in range(dim):
                for j in range(dim):
                    load_params["solid"]["sigma_bar_" + str(i) + str(j)] = 0.0

            # ---------------- liquid loading ----------------
            n_steps = 2
            load_params["liquid"]["pl_bar_ini_lst"] = [0.0] * n_steps
            load_params["liquid"]["pl_bar_fin_lst"] = [0.0] * n_steps

            load_params["liquid"]["grad_p_bar_x_ini_lst"] = [0.0] * n_steps
            load_params["liquid"]["grad_p_bar_x_fin_lst"] = [0.0] * n_steps
            load_params["liquid"]["grad_p_bar_y_ini_lst"] = [0.0] * n_steps
            load_params["liquid"]["grad_p_bar_y_fin_lst"] = [0.0] * n_steps

            load_params["liquid"]["Theta_in_ini_lst"]  = [0.0] * n_steps
            load_params["liquid"]["Theta_in_fin_lst"]  = [0.0] * n_steps
            load_params["liquid"]["Theta_out_ini_lst"] = [0.0] * n_steps
            load_params["liquid"]["Theta_out_fin_lst"] = [0.0] * n_steps

            # ---------------- air loading ----------------
            load_params["air"]["pf"] = 0.0

            # ==========================================================
            # Cases
            # ==========================================================
            if load == "internal_air_pressure":
                load_params["air"]["pf"] = 0.2

            elif load == "macroscopic_stretch_with_flow":
                load_params["solid"]["U_bar_00"] = 0.2
                load_params["air"]["pf"] = 0.0
                load_params["liquid"]["grad_p_bar_x_fin_lst"] = [0.0, 0.05]

            elif load == "liquid_pressure_gradient":
                load_params["air"]["pf"] = 0.0
                load_params["liquid"]["grad_p_bar_x_fin_lst"] = [0.1, 0.1]

            dmech.run_HollowBox_MicroPoroflow(
                dim=dim,
                mesh_params={
                    "dim": dim,
                    "xmin": 0., "ymin": 0.,
                    "xmax": 1., "ymax": numpy.sqrt(3.0),
                    "r0": 0.3,
                    "l": 0.1,
                    "hole_shape": "hex",
                    "add_center_hole": True,
                    "mesh_filebasename": res_folder + "/" + "mesh"
                },
                mat_params={
                        "skel": {"parameters": mat_params, "scaling": "no"},
                        "bulk": {"parameters": mat_params, "scaling": "no"},
                        "pore": {"parameters": mat_params, "scaling": "no"}
                    },
                flow_params={
                    "k_l": dolfin.Constant(((1e-6, 0.0),
                                            (0.0, 1e-6))),
                    "use_kozeny_carman": False,
                },
                porosity_params={
                    "type": "constant",
                    "val": 0.3,
                },
                bcs=bcs,
                step_params={
                    "n_steps": 2,
                    "Deltat_lst": [1e-2, 1e-1],
                    "dt_ini_lst": [1e-3, 1e-3],
                    "dt_min_lst": [1e-4, 1e-4],
                    "dt_max_lst": [5e-3, 5e-3],
                },
                load_params=load_params,
                res_basename=res_folder + "/" + res_basename,
                verbose=0)

            test.test(res_basename)