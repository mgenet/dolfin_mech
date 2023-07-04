#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2022                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################
import dolfin_mech     as dmech
import fire
import shutil
import numpy
import os
import dolfin

################################################################################

def compute_disp(gravity_plus=None, dirpath=None) :

    alpha, gamma, c1, c2, pe, pi, p  = 0.13164026924222222, 0.5, 0.2, 0.4, -0.35, -0.55, -0.45

    
    mat_params = {
        "alpha": alpha,
        "gamma":gamma,
        "c1":c1,
        "c2":c2,
        "kappa":1e2,
        "eta":1e-5}
    
    if gravity_plus:
        coef = 1.
    else:
        coef = -1.

    # print("trying for parameters = ", alpha, gamma, c1, c2, pe, pi )

    load_params_inverse = {
        "type":"pgra0", "f":coef*9.81e-3*0.5, "DP": coef*9.81e-3*0.5, "P0" : pe/2, "X0":[50.,50.,50.]}

    # rand = numpy.random.normal(loc=0.0, scale=100, size=1)
    new_directory = "ref"+str(gravity_plus)
    res_basename = os.path.join(dirpath, new_directory)

    try:
        shutil.rmtree(res_basename)
    except OSError:
        pass

    # print("creating folder", res_basename)
       
    os.mkdir(res_basename)
    
    cube_params = {"X1":100, "Y1":100, "Z1":100, "l": 10}
    # print("inverse problem")
    U1, phis1, phi1, V1 = dmech.RivlinCube_PoroHyperelasticity(
                       dim=3,
                       inverse=1,
                    #    porosity_params={"type": "mesh_function_xml", "val":0.5},
                       cube_params=cube_params,
                       mat_params={"scaling":"linear", "parameters":mat_params},
                       step_params={ "dt_min":0.1},
                       load_params=load_params_inverse,
                       res_basename = res_basename+"/inverse",
                       plot_curves=0,
                       verbose=1)


    load_params_direct = {
        "type":"pgra", "f":coef*9.81e-3*phi1, "F0":coef*9.81e-3*phi1, "P0" : pi/2, "X0":[50.,50.,50.]}

# #     # print("direct problem")
# # ###### reference --> inspi with the porosity computed for the direct problem + ALE.move(mesh)
    U2, phis2, phi2, V2 = dmech.RivlinCube_PoroHyperelasticity(
                        dim=3,
                        inverse=0,
                        porosity_params= {"type": "mesh_function_xml_custom", "val":phis1},
                        cube_params=cube_params,
                        mat_params={"scaling":"linear", "parameters":mat_params},
                        step_params={"dt_min":1e-4},
                        load_params=load_params_direct,
                        res_basename = res_basename+"/direct",
                        move = {"move":True, "U":U1},
                        plot_curves=0,
                        verbose=1)
    
    U = U1.copy(deepcopy=True)
    U.vector().set_local(U1.vector().get_local()[:]+U2.vector().get_local()[:])


    # return(U1, V1)

if (__name__ == "__main__"):
    fire.Fire(compute_disp)