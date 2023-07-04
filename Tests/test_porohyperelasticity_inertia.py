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

    exponent_lst = numpy.linspace(6, 14, 9)
    # exponent_lst = numpy.append(exponent_lst  , 30)
    # exponent_lst = [9]
    disp_lst = []
    inertia_lst = []

    print("exponent list", exponent_lst)

    
    
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

    for exponent in exponent_lst:
        inertia_val = 1*10**(-exponent)
        inertia_lst.append(inertia_val)
        print("inertia value is", inertia_val)

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
    
        cube_params = {"X1":100, "Y1":100, "Z1":100, "l": 10}

        print("inverse problem")
        U1, phis1, phi1, V1, mesh_old1, mesh1 = dmech.RivlinCube_PoroHyperelasticity(
                        dim=3,
                        inverse=1,
                        #    porosity_params={"type": "mesh_function_xml", "val":0.5},
                        cube_params=cube_params,
                        mat_params={"scaling":"linear", "parameters":mat_params},
                        step_params={ "dt_min":1e-2},
                        load_params=load_params_inverse,
                        res_basename = res_basename+"/inverse",
                        plot_curves=0,
                        inertia_val=inertia_val,
                        BC=1,
                        verbose=1)

        load_params_direct = {
            "type":"pgra", "f":coef*9.81e-3*phi1, "F0":coef*9.81e-3*phi1, "P0" : pi/2, "X0":[50.,50.,50.]}


        print("direct problem")
###### reference --> inspi with the porosity computed for the direct problem + ALE.move(mesh)
        U2, phis2, phi2, V2, mesh_old2, mesh2 = dmech.RivlinCube_PoroHyperelasticity(
                            dim=3,
                            inverse=0,
                            porosity_params= {"type": "mesh_function_xml_custom", "val":phis1},
                            cube_params=cube_params,
                            mat_params={"scaling":"linear", "parameters":mat_params},
                            step_params={"dt_min":1e-4},
                            load_params=load_params_direct,
                            res_basename = res_basename+"/direct",
                            # move = {"move":True, "U":U1},
                            plot_curves=0,
                            inertia_val=inertia_val,
                            BC=1,
                            verbose=1)

    
        U = U1.copy(deepcopy=True)
        U.vector().set_local(U1.vector().get_local()[:]+U2.vector().get_local()[:])
    
        if noise != 'ref' and noise != 'refSum':
            shutil.rmtree(res_basename)
        else:
            mesh = dolfin.BoxMesh(
                dolfin.Point(0., 0., 0.), dolfin.Point(cube_params["X1"], cube_params["Y1"], cube_params["Z1"]),
                int(cube_params["X1"]/cube_params["l"]), int(cube_params["Y1"]/cube_params["l"]), int(cube_params["Z1"]/cube_params["l"]) )
            if noise == 'ref':
                try:
                    os.mkdir("./"+str(gravity_plus))
                except OSError:
                    pass
                xdmf_file_mesh = dolfin.XDMFFile("./"+str(gravity_plus)+"/mesh.xdmf")
                xdmf_file_mesh.write(mesh)
                xdmf_file_mesh.close()
            else:
                try:
                    os.mkdir("./"+"Sum/")
                except OSError:
                    pass
                xdmf_file_mesh = dolfin.XDMFFile("./"+"Sum/"+"mesh.xdmf")
                xdmf_file_mesh.write(mesh)
                xdmf_file_mesh.close()

        disp_lst.append( (dolfin.assemble(dolfin.inner(U,U)*V1)/2/dolfin.assemble(dolfin.Constant(1)*V1))**2)
    
    # print(dolfin.assemble(dolfin.inner(U,U)*V1))
    print("inertia_lst", inertia_lst)
    print("disp_lst", disp_lst)

    # return(U, V1)

if (__name__ == "__main__"):
    fire.Fire(compute_disp)
