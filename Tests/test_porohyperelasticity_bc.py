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

def compute_disp(gravity_plus=None, parameters_to_identify={}, noise=None, dirpath=None, iteration=None) :

    alpha, gamma, c1, c2, pe, pi = 0.16, 0.5, 0.2, 0.4, -0.35, -0.55

    for key, value in parameters_to_identify.items():
        if key == "alpha":
            alpha = float(value)
        elif key == "gamma":
            gamma = float(value)
        elif key == "c1":
            c1 = float(value)
        elif key == "c2":
            c2 = float(value)
        elif key == "pe":
            pe = float(value)
            pe_or_pi = True
        elif key == "pi":
            pi = float(value)
            pe_or_pi = True
    
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
    if noise=='ref' or noise == 'refSum':
        new_directory = "ref"+str(gravity_plus)
    else:
        new_directory = "alpha"+str(alpha)+"gamma"+str(gamma)+"c1"+str(c1)+"c2"+str(c2)+"pe"+str(pe)+"pi"+str(pi)+"noise"+str(noise)+"i="+str(iteration)+"gravity_plus"+str(gravity_plus)
    res_basename = os.path.join(dirpath, new_directory)


    try:
        shutil.rmtree(res_basename)
    except OSError:
        pass

    # print("creating folder", res_basename)
       
    os.mkdir(res_basename)
    
    cube_params = {"X1":100, "Y1":100, "Z1":100, "l": 2}

    # print("inverse problem")
    U11, phis11, phi11, V11, mesh_old11, mesh11 = dmech.RivlinCube_PoroHyperelasticity(
                       dim=3,
                       inverse=1,
                    #    porosity_params={"type": "mesh_function_xml", "val":0.5},
                       cube_params=cube_params,
                       mat_params={"scaling":"linear", "parameters":mat_params},
                       step_params={ "dt_min":0.1},
                       load_params=load_params_inverse,
                       res_basename = res_basename+"/inverse",
                       plot_curves=0,
                       BC = 1,
                       verbose=1)

    U21, phis21, phi21, V21, mesh_old21, mesh21 = dmech.RivlinCube_PoroHyperelasticity(
                       dim=3,
                       inverse=1,
                    #    porosity_params={"type": "mesh_function_xml", "val":0.5},
                       cube_params=cube_params,
                       mat_params={"scaling":"linear", "parameters":mat_params},
                       step_params={ "dt_min":0.1},
                       load_params=load_params_inverse,
                       res_basename = res_basename+"/inverse",
                       plot_curves=0,
                       BC = 2,
                       verbose=1)

    load_params_direct1 = {
        "type":"pgra", "f":coef*9.81e-3*phi11, "F0":coef*9.81e-3*phi11, "P0" : pi/2, "X0":[50.,50.,50.]}
    load_params_direct2 = {
        "type":"pgra", "f":coef*9.81e-3*phi21, "F0":coef*9.81e-3*phi21, "P0" : pi/2, "X0":[50.,50.,50.]}


    # print("direct problem")
###### reference --> inspi with the porosity computed for the direct problem + ALE.move(mesh)
    U12, phis12, phi12, V12, mesh_old12, mesh12 = dmech.RivlinCube_PoroHyperelasticity(
                        dim=3,
                        inverse=0,
                        porosity_params= {"type": "mesh_function_xml_custom", "val":phis11},
                        cube_params=cube_params,
                        mat_params={"scaling":"linear", "parameters":mat_params},
                        step_params={"dt_min":1e-4},
                        load_params=load_params_direct1,
                        res_basename = res_basename+"/direct",
                        # move = {"move":True, "U":U11},
                        plot_curves=0,
                        BC=1,
                        verbose=1)

    U22, phis22, phi22, V22, mesh_old22, mesh22 = dmech.RivlinCube_PoroHyperelasticity(
                    dim=3,
                    inverse=0,
                    porosity_params= {"type": "mesh_function_xml_custom", "val":phis12},
                    cube_params=cube_params,
                    mat_params={"scaling":"linear", "parameters":mat_params},
                    step_params={"dt_min":1e-4},
                    load_params=load_params_direct2,
                    res_basename = res_basename+"/direct",
                    # move = {"move":True, "U":U21},
                    plot_curves=0,
                    BC=2,
                    verbose=1)
    
    # U = U11.copy(deepcopy=True)
    # U.vector().set_local(U1.vector().get_local()[:]+U2.vector().get_local()[:])
    
    # if noise != 'ref' and noise != 'refSum':
    #     shutil.rmtree(res_basename)
    # else:
    #     mesh = dolfin.BoxMesh(
    #         dolfin.Point(0., 0., 0.), dolfin.Point(cube_params["X1"], cube_params["Y1"], cube_params["Z1"]),
    #         int(cube_params["X1"]/cube_params["l"]), int(cube_params["Y1"]/cube_params["l"]), int(cube_params["Z1"]/cube_params["l"]) )
    #     if noise == 'ref':
    #         try:
    #             os.mkdir("./"+str(gravity_plus))
    #         except OSError:
    #             pass
    #         xdmf_file_mesh = dolfin.XDMFFile("./"+str(gravity_plus)+"/mesh.xdmf")
    #         xdmf_file_mesh.write(mesh)
    #         xdmf_file_mesh.close()
    #     else:
    #         try:
    #             os.mkdir("./"+"Sum/")
    #         except OSError:
    #             pass
    #         xdmf_file_mesh = dolfin.XDMFFile("./"+"Sum/"+"mesh.xdmf")
    #         xdmf_file_mesh.write(mesh)
    #         xdmf_file_mesh.close()

    # print(mesh1.coordinates())
    # # print(mesh2.coordinates())
    # for i in range(0, len(mesh22.coordinates())):
    #     print(mesh22.coordinates()[i][0], mesh21.coordinates()[i][0])
    max_diff = 0
    for i in range(len(mesh12.coordinates())):
        if abs(mesh22.coordinates()[i][0]-mesh22.coordinates()[0][0] - mesh12.coordinates()[i][0]) > max_diff:
            max_diff = mesh22.coordinates()[i][0]-mesh22.coordinates()[0][0] - mesh12.coordinates()[i][0]
        # if abs(mesh22.coordinates()[i][0]-mesh22.coordinates()[0][0] - mesh12.coordinates()[i][0]) > 1e-2:
        #     print("X", mesh22.coordinates()[i][0]-mesh22.coordinates()[0][0], mesh12.coordinates()[i][0])
        # if abs(mesh22.coordinates()[i][1]-mesh22.coordinates()[0][1] - mesh12.coordinates()[i][1]) > 1e-2:
        #     print("Y", mesh22.coordinates()[i][1]-mesh22.coordinates()[0][1], mesh12.coordinates()[i][1])
        # if abs(mesh22.coordinates()[i][2]-mesh22.coordinates()[0][2]- mesh12.coordinates()[i][2]) > 1e-2:
        #     print("Z", mesh22.coordinates()[i][2]-mesh22.coordinates()[0][2], mesh12.coordinates()[i][2])
    print(max_diff)


    # return(U, V1)

if (__name__ == "__main__"):
    fire.Fire(compute_disp)
