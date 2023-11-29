#coding=utf8

################################################################################
###                                                                          ###
### Created by Mahdi Manoochehrtayebi, 2020-2023                             ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin_mech as dmech
import myPythonLibrary as mypy

################################################################################

def run_HollowBox_Homogenization(
        dim,
        mesh = None,
        mesh_params = None,
        mat_params={},
        res_basename="run_HollowBox_Homogenization",
        write_results_to_file=1,
        verbose=0):

    ################################################################### Mesh ###

    assert ((mesh is not None) ^ (mesh_params is not None))
    if (mesh is None):
        mesh = dmech.run_HollowBox_Mesh(
            params=mesh_params)
    
    coord = mesh.coordinates()
    xmax = max(coord[:,0]); xmin = min(coord[:,0])
    ymax = max(coord[:,1]); ymin = min(coord[:,1])
    if (dim==2):    
        vol = (xmax - xmin)*(ymax - ymin)
        bbox = [xmin, xmax, ymin, ymax]
    elif (dim==3):
        zmax = max(coord[:,2]); zmin = min(coord[:,2])
        vol = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)
        bbox = [xmin, xmax, ymin, ymax, zmin, zmax]

    ################################################################ Problem ###

    homogenization_problem = dmech.HomogenizationProblem(
        dim=dim,
        mesh=mesh,
        mat_params=mat_params,
        vol=vol,
        bbox=bbox)
    [mu_, lmbda_] = homogenization_problem.get_lambda_and_mu()
    kappa_ = homogenization_problem.get_kappa()

    E_ = mu_*(3*lmbda_ + 2*mu_)/(lmbda_ + mu_)
    nu_ = lmbda_/(lmbda_ + mu_)/2

    if (write_results_to_file):
        qoi_printer = mypy.DataPrinter(
            names=["E_s", "nu_s", "E_hom", "nu_hom", "kappa_hom"],
            filename=res_basename+"-qois.dat",
            limited_precision=False)
            
        qoi_printer.write_line([mat_params["E"], mat_params["nu"], E_, nu_, kappa_])
        qoi_printer.write_line([mat_params["E"], mat_params["nu"], E_, nu_, kappa_]) # MG20231124: Need to write twice for some postprocessing issue

    return mat_params["E"], mat_params["nu"], E_, nu_, kappa_
