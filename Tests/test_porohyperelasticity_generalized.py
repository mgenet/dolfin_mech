#coding=utf8

################################################################################
###                                                                          ###
### Created by Felipe Álvarez, 2025                                          ###
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
import numpy as np

import myPythonLibrary as mypy
import dolfin_mech     as dmech

################################################################# parameters ###

mat_params = {
    "alpha":0.16,
    "gamma":0.5,
    "c1":0.6,
    "c2":0.,
    "kappa":1e2,
    "eta":1e-5}

####################################################################### test ###

res_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=res_folder,
    perform_tests=1,
    stop_at_failure=1,
    clean_after_tests=1)

dim     = 3
scaling = "linear"

res_basename  = sys.argv[0][:-3]

np.random.seed(0)
n_cells                = 10**3 * 6
Phis0_unloaded_imposed = [np.random.uniform(low=0.4, high=0.6) for i in range(n_cells)]

# Fixed point algorithm
phis_loaded_old = [np.random.uniform(low=0.4, high=0.6) for i in range(n_cells)]
error = 1.0
print ("Fixed point algorithm...")
while error > 1e-2:
    U_loaded_to_unloaded, Phis0_unloaded, dV_loaded = dmech.run_RivlinCube_PoroHyperelasticity(
        dim             = dim,
        inverse         = 1,
        cube_params     = {"l":0.1, "mesh_filebasename":res_folder+"/"+"mesh"},
        porosity_params = {"type":"function_xml_from_array", "val":phis_loaded_old},
        mat_params      = {"scaling":scaling, "parameters":mat_params},
        load_params     = {"type":"p_boundary_condition0"},
        step_params     = {"dt_min":1e-4},
        res_basename    = res_folder+"/"+res_basename+"-fixed-point-inverse",
        plot_curves     = 0,
        get_results     = 1,
        verbose         = 1)
    
    U_unloaded_to_loaded, phis_loaded, dV_unloaded = dmech.run_RivlinCube_PoroHyperelasticity(
        dim             = dim,
        inverse         = 0,
        cube_params     = {"l":0.1, "mesh_filebasename":res_folder+"/"+"mesh"},
        move_params     = {"move":True, "U":U_loaded_to_unloaded},
        porosity_params = {"type":"function_xml_from_array", "val":Phis0_unloaded_imposed},
        mat_params      = {"scaling":scaling, "parameters":mat_params},
        load_params     = {"type":"p_boundary_condition"},
        step_params     = {"dt_min":1e-4},
        res_basename    = res_folder+"/"+res_basename+"-fixed-point-direct",
        plot_curves     = 0,
        get_results     = 1,
        verbose         = 1)

    error = np.linalg.norm(phis_loaded - phis_loaded_old)
    print (f"Error in phis_loaded: {error:.4f}")

    phis_loaded_old = phis_loaded

test.test(res_basename+"-fixed-point-inverse")
test.test(res_basename+"-fixed-point-direct")


# Generalized PoroHyperelasticity
U_loaded_to_unloaded_GenPoro, phis_loaded_GenPoro, dV_exhal_GenPoro = dmech.run_RivlinCube_PoroHyperelasticity(
        dim             = dim,
        inverse         = 1,
        cube_params     = {"l":0.1, "mesh_filebasename":res_folder+"/"+"mesh"},
        porosity_params = {"known":"Phis0", "type":"function_xml_from_array", "val":Phis0_unloaded_imposed},
        mat_params      = {"scaling":scaling, "parameters":mat_params},
        load_params     = {"type":"p_boundary_condition0"},
        step_params     = {"dt_min":1e-4},
        res_basename    = res_folder+"/"+res_basename+"-generalized-inverse",
        plot_curves     = 0,
        get_results     = 1,
        verbose         = 1)

test.test(res_basename+"-generalized-inverse")


print("Comparing fixed point and generalized porohyperelasticity...")
phis_loaded_diff = np.linalg.norm(phis_loaded_GenPoro - phis_loaded)
print (f"Difference in phis_loaded: {phis_loaded_diff:.2e}")

U_loaded_to_unloaded_diff = np.linalg.norm(U_loaded_to_unloaded_GenPoro.vector()[:] - U_loaded_to_unloaded.vector()[:])
print (f"Difference in U_loaded_to_unloaded: {U_loaded_to_unloaded_diff:.2e}")


assert (phis_loaded_diff < 1e-2),\
    "Warning! Generalized porohyperelasticity did not match fixed point algorithm for phis_loaded."
assert (U_loaded_to_unloaded_diff < 1e-2),\
    "Warning! Generalized porohyperelasticity did not match fixed point algorithm for U_loaded_to_unloaded."
