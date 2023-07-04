#coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochhertaybei, 2020-2023                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin_mech as dmech
import myPythonLibrary as mypy

################################################################################

def HollowBox_Homogenization(
        res_basename,
        verbose,
        mesh_params={},
        mat_params={}):


    dim  = mesh_params.get("dim")

    ################################################################### Mesh ###

    mesh = dmech.HollowBox_Mesh(
            res_basename=res_basename,
            mesh_params=mesh_params)
    

    coord = mesh.coordinates()
    xmax = max(coord[:,0]); xmin = min(coord[:,0])
    ymax = max(coord[:,1]); ymin = min(coord[:,1])
    if (dim==3): zmax = max(coord[:,2]); zmin = min(coord[:,2])

    if (dim==2):    
        vol = (xmax - xmin)*(ymax - ymin)
        bbox = [xmin, xmax, ymin, ymax]

    if (dim==3):    
        vol = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)
        bbox = [xmin, xmax, ymin, ymax, zmin, zmax]

    ################################################################ Problem ###


    homo = dmech.HomogenizationProblem(dim=dim,
                            mesh=mesh,
                            mat_params=mat_params,
                            vol=vol,
                            bbox=bbox)
    [mu_, lmbda_] = homo.get_lambda_and_mu()
    kappa_ = homo.get_kappa()


    E_ = mu_*(3*lmbda_ + 2*mu_)/(lmbda_ + mu_)
    nu_ = lmbda_/(lmbda_ + mu_)/2

    if verbose:
        qoi_printer = mypy.DataPrinter(
            names=["E_s", "nu_s", "E_hom", "nu_hom", "kappa_hom"],
            filename=res_basename+"-qois.dat",
            limited_precision=False)
            
        qoi_printer.write_line([mat_params["E"], mat_params["nu"], E_, nu_, kappa_])
        qoi_printer.write_line([mat_params["E"], mat_params["nu"], E_, nu_, kappa_])



    return 

    ########################################## Boundary conditions & Loading ###
