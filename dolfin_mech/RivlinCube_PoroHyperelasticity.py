#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################
# from copy import deepcopy
import dolfin
import matplotlib.pyplot as mpl
import pandas 
import numpy
import sympy

import dolfin_mech as dmech
import dolfin_warp as dwarp
import subprocess

################################################################################

def RivlinCube_PoroHyperelasticity(
        dim=3,
        inverse=0,
        cube_params={},
        porosity_params={},
        move={},
        mat_params={},
        step_params={},
        load_params={},
        initialisation_estimation={},
        res_basename="RivlinCube_PoroHyperelasticity",
        plot_curves=False,
        verbose=0,
        inertia_val = 1e-8,
        multimaterial = 0,
        get_results_fields = 0,
        mesh_from_file=0,
        get_J =False,
        estimation_gap=False,
        get_invariants = None,
        BC=1):

    ################################################################### Mesh ###




    if mesh_from_file:
        mesh = dolfin.Mesh()
        dolfin.XDMFFile("/Users/peyrault/Documents/Gravity/Tests/Meshes_and_poro_files/Zygot.xdmf").read(mesh)
        # mesh = dolfin.refine(mesh)
        # mesh = dolfin.refine(mesh)
        boundaries_mf = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # MG20180418: size_t looks like unisgned int, but more robust wrt architecture and os
        boundaries_mf.set_all(0)
    else:
        if   (dim==2):
            mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id = dmech.RivlinCube_Mesh(dim=dim, params=cube_params)
        elif (dim==3):
            mesh, boundaries_mf, xmin_id, xmax_id, ymin_id, ymax_id, zmin_id, zmax_id = dmech.RivlinCube_Mesh(dim=dim, params=cube_params)


    if move.get("move", False) == True :
        Umove = move.get("U")
        dolfin.ALE.move(mesh, Umove)

    get_subsdomains = True

    
    # coord_ref = mesh.coordinates()[0][0]
    # for coord in mesh.coordinates():
    #     if coord[0] < coord_ref:
    #         coord_ref = coord[0]
    #         print(coord)
    # print("coord xmin", coord_ref)

    if get_subsdomains:
        domains_zones = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim())
        domains_zones.set_all(10)
        tol =1e-14
        # print("mesh.coordinates()[:].min()", mesh.coordinates())
        # xmin = mesh.coordinates()[:, 0].min()
        # xmax = mesh.coordinates()[:, 0].max()
        zmin = mesh.coordinates()[:, 2].min()
        zmax = mesh.coordinates()[:, 2].max()
        # print("zmin, zmax", zmin, zmax)
        # print("zmin, zmax", zmin, zmax)
        zones = 10
        delta_z = (zmax-zmin)/(zones+1)
        subdomain_lst = []
        # xmin += delta_x/2
        # subdomain_lst.append(dolfin.CompiledSubDomain("x[0] <= x1 + tol",  x1=xmin+delta_x, tol=tol))
        subdomain_lst.append(dolfin.CompiledSubDomain("x[2] <= z1 + tol",  z1=zmin+delta_z, tol=tol))
        subdomain_lst[0].mark(domains_zones, 10)
        for zone_ in range(0, zones-1):
            # subdomain_lst.append(dolfin.CompiledSubDomain(" x[0] >= x1 - tol",  x1=xmin+delta_x*(zone_+1), tol=tol))
            subdomain_lst.append(dolfin.CompiledSubDomain(" x[2] >= z1 - tol",  z1=zmin+delta_z*(zone_+1), tol=tol))
            # print("x1=xmin+delta_x*(zone_+1)", xmin+delta_x*(zone_+1))
            subdomain_lst[zone_+1].mark(domains_zones, 10-(zone_+1))
            # print("id", zone_+1)
        # for i in range(0, zones): 
            # marked_cells = dolfin.SubsetIterator(domains_mf, i)
            # compteur = 0
            # for cell in marked_cells:     
                # compteur += 1
            # print("compteur", compteur, "for i=", i)
        boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Tests/Article/boundaries.pvd") 
        boundary_file << domains_zones



    if mesh_from_file:
        domains_mf=None
        if multimaterial :
            ymin = mesh.coordinates()[:, 1].min()
            ymax = mesh.coordinates()[:, 1].max()
            delta_y = ymax - ymin
            domains_mf = None
            tol = 1E-14
            number_zones = len(mat_params)
            length_zone = delta_y*0.1
            domains_mf = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim())
            domains_mf.set_all(0)
            subdomain_lst = []
            subdomain_lst.append(dolfin.CompiledSubDomain("x[1] <= y1 + tol",  y1=ymin+length_zone, tol=tol))
            subdomain_lst[0].mark(domains_mf, 0)
            for mat_id in range(0, number_zones):
                subdomain_lst.append(dolfin.CompiledSubDomain(" x[1] >= y1 - tol",  y1=ymin+length_zone*(mat_id+1), tol=tol))
                subdomain_lst[mat_id].mark(domains_mf, mat_id)
                mat_params[mat_id]["subdomain_id"] = mat_id
            # boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Gravity_cluster/Tests/boundaries.pvd") 
            # boundary_file << domains_mf
    else:
        domains_mf=None
        if multimaterial :
            domains_mf = None
            # print(len(mat_params))
            if len(mat_params)>=2 :
                tol = 1E-14
                number_zones = len(mat_params)
                mid_point = abs(cube_params.get("Y1", 1.)-cube_params.get("Y0", 0.))/number_zones
                domains_mf = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim())
                if number_zones == 2:
                    domains_mf.set_all(0)
                    subdomain_0 = dolfin.CompiledSubDomain("x[1] <= y1 + tol",  y1=mid_point, tol=tol)
                    subdomain_1 = dolfin.CompiledSubDomain(" x[1] >= y1 - tol",  y1=mid_point, tol=tol)
                    subdomain_0.mark(domains_mf, 0)
                    mat_params[0]["subdomain_id"] = 0
                    subdomain_1.mark(domains_mf, 1)
                    mat_params[1]["subdomain_id"] = 1
                    # boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Gravity_cluster/Tests/boundaries.pvd") 
                    # boundary_file << domains_mf
                elif number_zones == 3 :
                    domains_mf.set_all(0)
                    # print("creating 3 subdomains...")
                    y1 = mid_point
                    y2 = 2*mid_point
                    # print("y1, y2", y1, y2)
                    subdomain_0 = dolfin.CompiledSubDomain("x[1] <= y1 + tol",  y1=y1, tol=tol)
                    subdomain_1 = dolfin.CompiledSubDomain(" x[1] >= y1 - tol",  y1=y1, tol=tol)
                    subdomain_2 = dolfin.CompiledSubDomain("x[1] >= y2 - tol",  y2=y2, tol=tol)
                    subdomain_0.mark(domains_mf, 0)
                    mat_params[0]["subdomain_id"] = 0
                    subdomain_1.mark(domains_mf, 1)
                    mat_params[1]["subdomain_id"] = 1
                    subdomain_2.mark(domains_mf, 2)
                    mat_params[2]["subdomain_id"] = 2
                    # boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Gravity_cluster/Tests/boundaries.pvd") 
                    # boundary_file << domains_mf
                else:
                    y1 = mid_point
                    domains_mf.set_all(0)
                    subdomain_lst = []
                    subdomain_lst.append(dolfin.CompiledSubDomain("x[1] <= y1 + tol",  y1=y1, tol=tol))
                    subdomain_lst[0].mark(domains_mf, 0)
                    for mat_id in range(0, number_zones):
                        subdomain_lst.append(dolfin.CompiledSubDomain(" x[1] >= y1 - tol",  y1=(mat_id+1)*y1, tol=tol))
                        subdomain_lst[mat_id].mark(domains_mf, mat_id)
                        mat_params[mat_id]["subdomain_id"] = mat_id
                    # boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Gravity_cluster/Tests/boundaries.pvd") 
                    # boundary_file << domains_mf

  
    
    # mesh.init()
    # bmesh = dolfin.BoundaryMesh(mesh, 'exterior')
    # dolfin.XDMFFile("/Users/peyrault/Documents/Gravity/Tests/Article/boundary_mesh.xdmf").write(bmesh)
    # facet_domains = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    # facet_domains.set_all(0)
    # for f in dolfin.facets(mesh):
    #     if f.exterior():
    #     # if any(ff.exterior() for ff in dolfin.facets(f)):
    #         facet_domains[f] = 1

    vertex_coords = []
    facet_domains = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    facet_domains.set_all(0)
    # f_on_boundary = dolfin.CompiledSubDomain("on_boundary ")
    for f in dolfin.facets(mesh):
        if f.exterior():
            facet_domains[f] = 1
            for vertex in dolfin.vertices(f):
                vertex_coords.append(list(vertex.point().array()))

    for f in dolfin.facets(mesh):
        # for vertex in dolfin.vertices(f):
        #     print(list(vertex.point().array()) in vertex_coords)
            # print(list(vertex.point().array()))
        if any(list(vertex.point().array()) in vertex_coords for vertex in dolfin.vertices(f)):
            # print("marking facet")
            facet_domains[f] = 1

        # for v in dolfin.vertices(f):
        # if any(ff.exterior() for ff in dolfin.facets(f)):
                # facet_domains[f] = 1

    # print("vertex_coords", vertex_coords)
    

    # boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Tests/Article/boundaries_facets.pvd") 
    # boundary_file << facet_domains

    

    # cells_domains = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
    # boundary_adjacent_cells = [c for c in dolfin.cells(mesh) 
    #                          if any(f.exterior()  for f in dolfin.facets(c))]
    
    # for c in boundary_adjacent_cells:
    #     cells_domains[c] = 1

    # boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Tests/Article/boundaries_cells.pvd") 
    # boundary_file << cells_domains
    


    domains_dirichlet = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    domains_dirichlet.set_all(0)
    # external = dolfin.CompiledSubDomain("x[0] >= 100.0 - DOLFIN_EPS or x[0] < DOLFIN_EPS or x[1] < DOLFIN_EPS or x[2] < DOLFIN_EPS or x[1] >= 100.0 - DOLFIN_EPS or x[2] >= 100.0 - DOLFIN_EPS")
    # internal = dolfin.CompiledSubDomain(" 100.0 > x[0]  && x[0]>0. &&  100.0 > x[1]  && x[1]> 0. && 100.0 > x[2]  && x[2]> 0.")  
    internal = dolfin.CompiledSubDomain("not on_boundary ")
    external = dolfin.CompiledSubDomain("on_boundary ")
    external_id = 1
    external.mark(domains_dirichlet, external_id)
    # internal_id = 1
    # internal.mark(domains_dirichlet, internal_id)
    # internal.mark(boundaries_mf, internal_id)
    # boundary_file = dolfin.File("/Users/peyrault/Documents/Gravity/Tests/Article/boundaries.pvd") 
    # boundary_file << domains_dirichlet

   

    


    
    ################################################################ Porosity ###

    porosity_type = porosity_params.get("type", "constant")
    porosity_val  = porosity_params.get("val", 0.5)

    if (porosity_type=="from_file"):
        porosity_filename = porosity_val
        porosity_mf = dolfin.MeshFunction(
            "double",
            mesh,
            porosity_filename)
        porosity_expr = dolfin.CompiledExpression(getattr(dolfin.compile_cpp_code(dmech.get_ExprMeshFunction_cpp_pybind()), "MeshExpr")(), mf=porosity_mf, degree=0)
        porosity_fs = dolfin.FunctionSpace(mesh, 'DG', 0)
        porosity_fun = dolfin.interpolate(porosity_expr, porosity_fs)
        porosity_val = None
    if (porosity_type == "constant"):
        porosity_fun = None
        # print("constant")
    elif (porosity_type.startswith("mesh_function")):
        # print("mesh_function")
        if (porosity_type == "mesh_function_constant"):
            # print("mesh_function constant")
            porosity_mf = dolfin.MeshFunction(
                value_type="double",
                mesh=mesh,
                dim=dim,
                value=porosity_val)
        elif (porosity_type == "mesh_function_xml"):
            # print("mesh_function xml")
            porosity_filename = res_basename+"-poro.xml"
            n_cells = len(mesh.cells())
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <mesh_function type="double" dim="'+str(dim)+'" size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    file.write('    <entity index="'+str(k_cell)+'" value="'+str(porosity_val)+'"/>\n')
                file.write('  </mesh_function>\n')
                file.write('</dolfin>\n')
                file.close()
            porosity_mf = dolfin.MeshFunction(
                "double",
                mesh,
                porosity_filename)
        elif (porosity_type == "mesh_function_random_xml"):
            # print("mesh_function xml")
            porosity_filename = res_basename+"-poro.xml"
            n_cells = len(mesh.cells())
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <mesh_function type="double" dim="'+str(dim)+'" size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    value = numpy.random.uniform(low=0.4, high=0.6)
                    # positive_value = False
                    # while not positive_value:
                        # value = float(numpy.random.normal(loc=0.5, scale=0.13, size=1)[0])
                        # if value >0 and value<1:
                            # positive_value = True
                    file.write('    <entity index="'+str(k_cell)+'" value="'+str(value)+'"/>\n')
                file.write('  </mesh_function>\n')
                file.write('</dolfin>\n')
                file.close()
            porosity_mf = dolfin.MeshFunction(
                "double",
                mesh,
                porosity_filename)
        elif (porosity_type == "mesh_function_xml_custom"):
            # print("mesh_function xml custom")
            porosity_filename = res_basename+"-poro.xml"
            n_cells = len(mesh.cells())
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <mesh_function type="double" dim="'+str(dim)+'" size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    file.write('    <entity index="'+str(k_cell)+'" value="'+str(porosity_val[k_cell])+'"/>\n')
                file.write('  </mesh_function>\n')
                file.write('</dolfin>\n')
                file.close()
            porosity_mf = dolfin.MeshFunction(
                "double",
                mesh,
                porosity_filename)
        porosity_expr = dolfin.CompiledExpression(getattr(dolfin.compile_cpp_code(dmech.get_ExprMeshFunction_cpp_pybind()), "MeshExpr")(), mf=porosity_mf, degree=0)
        porosity_fs = dolfin.FunctionSpace(mesh, 'DG', 0)
        porosity_fun = dolfin.interpolate(porosity_expr, porosity_fs)
        porosity_val = None
    elif (porosity_type.startswith("function")):
        # print("function")
        porosity_fs = dolfin.FunctionSpace(mesh, 'DG', 0)
        if (porosity_type == "function_constant"):
            # print("function constant")
            porosity_fun = dolfin.Function(porosity_fs)
            porosity_fun.vector()[:] = porosity_val
        elif (porosity_type == "function_xml"):
            # print("function xml")
            porosity_filename = res_basename+"-poro.xml"
            n_cells = len(mesh.cells())
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <function_data size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    file.write('    <dof index="'+str(k_cell)+'" value="'+str(porosity_val)+'" cell_index="'+str(k_cell)+'" cell_dof_index="0"/>\n')
                file.write('  </function_data>\n')
                file.write('</dolfin>\n')
                file.close()
            porosity_fun = dolfin.Function(
                porosity_fs,
                porosity_filename)
        elif (porosity_type == "function_xml_from_array"):
            # print("function xml")
            porosity_filename = res_basename+"-poro.xml"
            # print("porosity_filename=", porosity_filename)
            n_cells = len(mesh.cells())
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <function_data size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    # print("kcell=", k_cell)
                    # print("porosity for kcell", porosity_val[k_cell])
                    file.write('    <dof index="'+str(k_cell)+'" value="'+str(porosity_val[k_cell])+'" cell_index="'+str(k_cell)+'" cell_dof_index="0"/>\n')
                file.write('  </function_data>\n')
                file.write('</dolfin>\n')
                file.close()
            porosity_fun = dolfin.Function(
                porosity_fs,
                porosity_filename)
        porosity_val = None

    ################################################################ Problem ###
    load_type = load_params.get("type", "internal")

    if load_type=="p_boundary_condition0":
        gradient_operators="inverse"
    elif load_type=="p_boundary_condition":
        gradient_operators="direct"
    else:
        gradient_operators=None

    if (inverse):
        problem = dmech.InversePoroHyperelasticityProblem(
            mesh=mesh,
            define_facet_normals=1,
            boundaries_mf=boundaries_mf,
            displacement_degree=1,
            quadrature_degree=1,
            porosity_init_val=porosity_val,
            porosity_init_fun=porosity_fun,
            skel_behavior=mat_params,
            bulk_behavior=mat_params,
            pore_behavior=mat_params,
            gradient_operators=gradient_operators)
    else:
        problem = dmech.PoroHyperelasticityProblem(
            mesh=mesh,
            define_facet_normals=1,
            boundaries_mf=boundaries_mf,
            displacement_degree=1,
            quadrature_degree=1,
            porosity_init_val=porosity_val,
            porosity_init_fun=porosity_fun,
            skel_behavior=mat_params,
            bulk_behavior=mat_params,
            pore_behavior=mat_params,
            gradient_operators=gradient_operators)


    ########################################## Boundary conditions & Loading ###
    # problem.add_constraint(V=problem.get_displacement_function_space().sub(0), sub_domains=boundaries_mf, sub_domain_id=xmin_id, val=0.)
    # problem.add_constraint(V=problem.get_displacement_function_space().sub(1), sub_domains=boundaries_mf, sub_domain_id=ymin_id, val=0.)
    # if (dim==3):
    # problem.add_constraint(V=problem.get_displacement_function_space(), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=[0.,0.,0.])
    # problem.add_constraint(V=problem.get_displacement_function_space().sub(2), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=0.)
    #     problem.add_constraint(V=problem.get_displacement_function_space().sub(1), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=0.)
    #     problem.add_constraint(V=problem.get_displacement_function_space().sub(2), sub_domains=boundaries_mf, sub_domain_id=zmin_id, val=0.)
    
    
    # if not inverse:
    #     print("Phis0", problem.Phis0.vector()[:])
    #     print("rho0", dolfin.assemble(problem.Phis0*V)/dolfin.assemble(1*V))


    BC=0
    if BC == 1:
        pinpoint_sd = dmech.PinpointSubDomain(coords=[0.,0.,0.], tol=1e-4)
        problem.add_constraint(
                V=problem.get_displacement_function_space(), 
                val=[0.] * dim,
                sub_domain=pinpoint_sd,
                method='pointwise')
        if move.get("move", False) == False:
            pinpoint_sd = dmech.PinpointSubDomain(coords=[cube_params["X1"],0.,0.], tol=1e-4)
        else :
            pinpoint_sd = dmech.PinpointSubDomain(coords=[cube_params["X1"] + Umove([cube_params["X1"], 0., 0.])[0], Umove([cube_params["X1"], 0., 0.])[1], Umove([cube_params["X1"], 0., 0.])[2]], tol=1e-4)
        problem.add_constraint(
                V=problem.get_displacement_function_space().sub(1), 
                val=0.,
                sub_domain=pinpoint_sd,
                method='pointwise')
        problem.add_constraint(
                V=problem.get_displacement_function_space().sub(2), 
                val=0.,
                sub_domain=pinpoint_sd,
                method='pointwise')
        if move.get("move", False) == False:
            pinpoint_sd = dmech.PinpointSubDomain(coords=[0.,cube_params["Y1"],0.], tol=1e-4)
        else :
            pinpoint_sd = dmech.PinpointSubDomain(coords=[0. + Umove([0., cube_params["Y1"], 0.])[0],cube_params["Y1"] + Umove([0., cube_params["Y1"], 0. ])[1], Umove([cube_params["X1"], 0., 0.])[2]], tol=1e-3)
        problem.add_constraint(
                V=problem.get_displacement_function_space().sub(2), 
                val=0.,
                sub_domain=pinpoint_sd,
                method='pointwise')
    elif BC ==2:
        pinpoint_sd = dmech.PinpointSubDomain(coords=[0., 0., cube_params["Z1"]], tol=1e-3)
        problem.add_constraint(
            V=problem.get_displacement_function_space(), 
            val=[0.]*dim,
            sub_domain=pinpoint_sd,
            method='pointwise')
        if move.get("move", False) == False:
            pinpoint_sd = dmech.PinpointSubDomain(coords=[cube_params["X1"], 0., cube_params["Z1"]], tol=1e-3)
        else :
            pinpoint_sd = dmech.PinpointSubDomain(coords=[cube_params["X1"] + Umove([cube_params["X1"], 0., cube_params["Z1"]])[0], Umove([cube_params["X1"], 0., cube_params["Z1"]])[1], cube_params["Z1"]+Umove([cube_params["X1"], 0., cube_params["Z1"]])[2]], tol=1e-4)
        problem.add_constraint(
            V=problem.get_displacement_function_space().sub(1), 
            val=0.,
            sub_domain=pinpoint_sd,
            method='pointwise')
        problem.add_constraint(
            V=problem.get_displacement_function_space().sub(2), 
            val=0.,
            sub_domain=pinpoint_sd,
            method='pointwise')
        if move.get("move", False) == False:
            pinpoint_sd = dmech.PinpointSubDomain(coords=[0., cube_params["Y1"], cube_params["Z1"]], tol=1e-3)
        else :
            # print("mesh moved")
            pinpoint_sd = dmech.PinpointSubDomain(coords=[ Umove([0., cube_params["Y1"], cube_params["Z1"]])[0], cube_params["Y1"] + Umove([0., cube_params["Y1"], cube_params["Z1"]])[1], cube_params["Z1"]+Umove([0., cube_params["Y1"], cube_params["Z1"]])[2]], tol=1e-4)
        problem.add_constraint(
            V=problem.get_displacement_function_space().sub(2), 
            val=0.,
            sub_domain=pinpoint_sd,
            method='pointwise')
    

    Deltat = step_params.get("Deltat", 1.)
    dt_ini = step_params.get("dt_ini", 1.)
    dt_min = step_params.get("dt_min", 1.)
    dt_max = step_params.get("dt_max", 1.)
    k_step = problem.add_step(
        Deltat=Deltat,
        dt_ini=dt_ini,
        dt_min=dt_min,
        dt_max=dt_max)

    if multimaterial:
        rho_solid = mat_params[0].get("parameters").get("rho_solid", 1.06e-6)
    else:
        rho_solid = mat_params.get("parameters").get("rho_solid", 1.06e-6)

    load_type = load_params.get("type", "internal")

    problem.add_inertia_operator(
        measure=problem.dV,
        rho_val=float(inertia_val),
        # rho_val=0,
        k_step=k_step)
    
    surface_forces = []
    volume_forces = []
    boundary_conditions = []

    if (load_type == "internal"):
        pf = load_params.get("pf", +0.5)
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0., pf_fin=pf,
            k_step=k_step)
    elif (load_type == "external"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0., pf_fin=0.,
            k_step=k_step)
        P = load_params.get("P", -0.5)
        problem.add_surface_pressure_loading_operator(
            measure=problem.dS(xmax_id),
            P_ini=0., P_fin=P,
            k_step=k_step)
        problem.add_surface_pressure_loading_operator(
            measure=problem.dS(ymax_id),
            P_ini=0., P_fin=P,
            k_step=k_step)
        if (dim==3): problem.add_surface_pressure_loading_operator(
            measure=problem.dS(zmax_id),
            P_ini=0., P_fin=P,
            k_step=k_step)
    elif (load_type == "external0"):
        # print("adding external 0 op")
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0., pf_fin=0.,
            k_step=k_step)
        P = load_params.get("P", -0.5)
        problem.add_surface_pressure0_loading_operator(
            # measure=problem.dS(xmax_id),
            measure=problem.dS,
            P_ini=0., P_fin=P,
            k_step=k_step)
        surface_forces.append([P, problem.dS])
        # problem.add_surface_pressure0_loading_operator(
        #     measure=problem.dS(ymax_id),
        #     P_ini=0., P_fin=P,
        #     k_step=k_step)
        # if (dim==3): problem.add_surface_pressure0_loading_operator(
        #     measure=problem.dS(zmax_id),
        #     P_ini=0., P_fin=P,
        #     k_step=k_step)
    elif (load_type == "volu0"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        f = load_params.get("f", 5e-4)
        problem.add_volume_force0_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim,
            F_fin = [0.]*(dim-1) + [f],
            k_step=k_step)
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        # P = load_params.get("P", -0.5)
        # problem.add_surface_pressure0_loading_operator(
            # measure=problem.dS(xmax_id),
            # P_ini=0,
            # P_fin=P,
            # k_step=k_step)
        # problem.add_surface_pressure0_loading_operator(
            # measure=problem.dS(ymax_id),
            # P_ini=0,
            # P_fin=P,
            # k_step=k_step)
        # if (dim==3): problem.add_surface_pressure0_loading_operator(
            # measure=problem.dS(zmax_id),
            # P_ini=0,
            # P_fin=P,
            # k_step=k_step)
    elif (load_type == "volu"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        f = load_params.get("f", 5e-4)
        problem.add_volume_force_loading_operator(
            measure=problem.dV,
            F_ini=[0.]*dim,
            F_fin=[0.]*(dim-1) + [f],
            k_step=k_step)
    elif(load_type=="surface_pressure"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        P = load_params.get("P", -0.5)
        problem.add_surface_pressure_loading_operator(
            measure=problem.dS,
            P_ini=0,
            P_fin=P,
            k_step=k_step)
        # problem.add_surface_pressure_loading_operator(
        #     measure=problem.dS(ymax_id),
        #     P_ini=0,
        #     P_fin=P,
        #     k_step=k_step)
        # if (dim==3): problem.add_surface_pressure_loading_operator(
        #     measure=problem.dS(zmax_id),
        #     P_ini=0,
        #     P_fin=P,
        #     k_step=k_step)
    elif (load_type == "pgra0"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        f = load_params.get("f", 1e4)
        P0 = load_params.get("P0", -0.5)
        problem.add_surface_pressure_gradient0_loading_operator(
            dV=problem.dV,
            dS=problem.dS,
            f_ini=[0.]*dim,
            # f_fin=[0., f, 0.],
            f_fin=[0., 0., f],
            rho_solid=rho_solid,
            phis=problem.phis,
            P0_ini=0.,
            P0_fin=P0,
            k_step=k_step)
        volume_forces.append([[0., 0.,rho_solid*f], problem.dV])
    elif (load_type == "pgra"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        f = load_params.get("f", 1e4)
        P0 = load_params.get("P0", -0.5)
        problem.add_surface_pressure_gradient_loading_operator(
            dV=problem.dV,
            dS=problem.dS,
            f_ini=[0.]*dim,
            f_fin=[0., 0., f],
            # f_fin=[0., f, 0.],
            rho_solid=rho_solid,
            P0_ini=0.,
            P0_fin=P0,
            k_step=k_step)
    elif (load_type == "p_boundary_condition0"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        f = load_params.get("f", 1e4)
        P0 = load_params.get("P0", -0.5)
        rho_solid = mat_params.get("parameters").get("rho_solid", 1e-6)
        problem.add_pressure_balancing_gravity0_loading_operator(
            dV=problem.dV,
            dS=problem.dS,
            f_ini=[0.]*dim,
            f_fin=[0., 0., f],
            rho_solid=rho_solid,
            phis=problem.phis,
            P0_ini=0.,
            P0_fin=P0,
            k_step=k_step)
        volume_forces.append([[0., 0.,rho_solid*f], problem.dV])
    elif (load_type == "p_boundary_condition"):
        problem.add_pf_operator(
            measure=problem.dV,
            pf_ini=0.,
            pf_fin=0.,
            k_step=k_step)
        f = load_params.get("f", 1e4)
        P0 = load_params.get("P0", -0.5)
        print("P0", P0)
        rho_solid = mat_params.get("parameters").get("rho_solid", 1e-6)
        problem.add_pressure_balancing_gravity_loading_operator(
            dV=problem.dV,
            dS=problem.dS,
            f_ini=[0.]*dim,
            f_fin=[0, 0., f],
            rho_solid=rho_solid,
            P0_ini=0.,
            P0_fin=P0,
            k_step=k_step)
        volume_forces.append([[0., 0.,rho_solid*f], problem.dV])

    ################################################# Quantities of Interest ###

    problem.add_deformed_volume_qoi()
    problem.add_global_strain_qois()
    problem.add_global_stress_qois()
    problem.add_global_porosity_qois()
    problem.add_global_fluid_pressure_qoi()

    ################################################################# Solver ###

    solver = dmech.NonlinearSolver(
        problem=problem,
        parameters={
            "sol_tol":[1e-6]*len(problem.subsols),
            "n_iter_max":32},
        # relax_type="aitken",
        # relax_type="constant",
        relax_type = "backtracking",
        write_iter=0)

    integrator = dmech.TimeIntegrator(
        problem=problem,
        solver=solver,
        parameters={
            "n_iter_for_accel":4,
            "n_iter_for_decel":16,
            "accel_coeff":2,
            "decel_coeff":2},
        print_out=res_basename*verbose,
        print_sta=res_basename*verbose,
        write_qois=res_basename+"-qois",
        write_qois_limited_precision=1,
        write_sol=res_basename*verbose)

    success = integrator.integrate()
    assert (success),\
        "Integration failed. Aborting."

    integrator.close()

    ################################################################## Plots ###
    # print("mesh coordinates are", mesh.coordinates())

    if (plot_curves):
        qois_data = pandas.read_csv(
            res_basename+"-qois.dat",
            delim_whitespace=True,
            comment="#",
            names=open(res_basename+"-qois.dat").readline()[1:].split())

        qois_fig, qois_axes = mpl.subplots()
        all_strains = ["E_XX", "E_YY"]
        if (dim == 3): all_strains += ["E_ZZ"]
        all_strains += ["E_XY"]
        if (dim == 3): all_strains += ["E_YZ", "E_ZX"]
        qois_data.plot(x="t", y=all_strains, ax=qois_axes, ylabel="Green-Lagrange strain")
        qois_fig.savefig(res_basename+"-strains-vs-time.pdf")

        for comp in ["skel", "bulk", "tot"]:
            qois_fig, qois_axes = mpl.subplots()
            all_stresses = ["s_"+comp+"_XX", "s_"+comp+"_YY"]
            if (dim == 3): all_stresses += ["s_"+comp+"_ZZ"]
            all_stresses += ["s_"+comp+"_XY"]
            if (dim == 3): all_stresses += ["s_"+comp+"_YZ", "s_"+comp+"_ZX"]
            qois_data.plot(x="t", y=all_stresses, ax=qois_axes, ylabel="Cauchy stress")
            qois_fig.savefig(res_basename+"-stresses-"+comp+"-vs-time.pdf")

        qois_fig, qois_axes = mpl.subplots()
        all_porosities = []
        if (inverse):
            all_porosities += ["phis0", "phif0", "Phis0", "Phif0"]
        else:
            all_porosities += ["Phis", "Phif", "phis", "phif"]
        qois_data.plot(x="t", y=all_porosities, ax=qois_axes, ylim=[0,1], ylabel="porosity")
        qois_fig.savefig(res_basename+"-porosities-vs-time.pdf")

        qois_fig, qois_axes = mpl.subplots()
        qois_data.plot(x="pf", y=all_porosities, ax=qois_axes, ylim=[0,1], ylabel="porosity")
        qois_fig.savefig(res_basename+"-porosities-vs-pressure.pdf")


    for foi in problem.fois:
        # print(foi.name)
        if foi.name == "Phis0":
            phi = foi.func.vector().get_local()
            Phis0 = foi.func.vector().get_local()
            Phis0_func = foi.func
        if foi.name == "phis":
            phis = foi.func.vector().get_local()
            phis_func = foi.func
        if foi.name == "E":
            green_lagrange_tensor = foi.func

    # print("C=", C)
    # print("len(C)", len(C))
    # print("J=", J)
    # print("lenJ", len(J))
    # print("IC1", len(IC))
    # print("IIC", len(IIC))

    v=0
    for qoi in problem.qois:
        # print(qoi.name)
        if qoi.name == "v":
            v = qoi.value
    
    # print("deformed mesh is", mesh.coordinates())
    # for i in range(0, len( mesh.coordinates())):
        # print(mesh_old[i]- mesh.coordinates()[i])
    
    # V = problem.dV

    # print ("green-lagrange tensor", (dolfin.assemble(dolfin.inner(green_lagrange_tensor,green_lagrange_tensor)*V)/2/dolfin.assemble(dolfin.Constant(1)*V)))

    # print (problem.get_lbda0_subsol().func.vector().get_local())
    if not inverse:

        # print(problem.get_x0_direct_subsol().func.vector().get_local())
        # print(problem.get_porosity_subsol().func.vector().get_local())
        # print("Phis0", dolfin.assemble(problem.Phis0*problem.dV)/dolfin.assemble(problem.kinematics.J*problem.dV))
        # print("x0=", dolfin.assemble(problem.Phis0 * (problem.X[0] + problem.get_displacement_subsol().func[0]) * problem.dV) / dolfin.assemble(problem.Phis0 * problem.dV) )

        # print("x0=", dolfin.assemble(problem.Phis0 * (problem.X[1] + problem.get_displacement_subsol().func[1])* problem.dV) / dolfin.assemble(problem.Phis0 * problem.dV) )

        # print("x0=", dolfin.assemble(problem.Phis0 * (problem.X[2] + problem.get_displacement_subsol().func[2])* problem.dV) / dolfin.assemble(problem.Phis0 * problem.dV) )
        # print("x0z=", dolfin.assemble(problem.get_porosity_subsol().func * (problem.X[2] + problem.get_displacement_subsol().func[2])* problem.kinematics.J* problem.dV) / dolfin.assemble(problem.get_porosity_subsol().subfunc * problem.kinematics.J * problem.dV) )
        # # print("x0z=", dolfin.assemble(problem.Phis0 * (problem.X[2] + problem.get_displacement_subsol().func[2]) * problem.dV) / dolfin.assemble(problem.get_porosity_subsol().subfunc * problem.kinematics.J * problem.dV) )
        # print("Phis0/phis", dolfin.assemble(problem.Phis0*problem.dV), dolfin.assemble(problem.get_porosity_subsol().subfunc*problem.kinematics.J*problem.dV))
        # print("phisx", dolfin.assemble(problem.get_porosity_subsol().func * ( problem.get_displacement_subsol().func[2])* problem.dV) )
        # print("Phis0x", dolfin.assemble(problem.Phis0 *  (problem.get_displacement_subsol().func[2])* problem.dV) )
        pass
    else:
        # print("phis", dolfin.assemble(problem.phis*problem.dV)/problem.mesh_V0)
        pass

    
    if estimation_gap:
        if inverse:
            print("phis", problem.phis)
            kinematics = dmech.InverseKinematics(u=problem.get_displacement_subsol().func)
            surface_forces.append([problem.get_balanced_gravity_boundary_pressure_subsol().subfunc, problem.dS])
        else:
            # print("Phis0", problem.Phis0)
            kinematics = dmech.Kinematics(U=problem.get_displacement_subsol().func)
            surface_forces.append([problem.get_balanced_gravity_boundary_pressure_subsol().subfunc, problem.dS])
            # surface_forces.append([-0.5, problem.dS])

        dmech.EquilibriumGap(problem=problem, kinematics=kinematics, material_model=None, material_parameters=mat_params["parameters"], initialisation_estimation=initialisation_estimation, surface_forces=surface_forces, volume_forces=volume_forces, boundary_conditions=boundary_conditions, inverse=inverse, U=problem.get_displacement_subsol().func)
        

    if get_results_fields:
        # phis0 = problem.get_porosity_subsol().func.vector().get_local()
        porosity_filename = res_basename+"-poro_final.xml"
        n_cells = len(mesh.cells())
        if inverse:
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <mesh_function type="double" dim="'+str(dim)+'" size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    file.write('    <entity index="'+str(k_cell)+'" value="'+str(Phis0[k_cell])+'"/>\n')
                file.write('  </mesh_function>\n')
                file.write('</dolfin>\n')
                file.close()
            # equilibrium_rhog= dolfin.assemble(1e-6*f*problem.phis*problem.dV)
            # equilibrium_p_x = dolfin.assemble(problem.get_p_subsol().func*problem.mesh_normals[0]*problem.dS)
            # # equilibrium_p_y = dolfin.assemble(problem.get_p_subsol().func*problem.mesh_normals[1]*problem.dS)
            # # equilibrium_p_z = dolfin.assemble(problem.get_p_subsol().func*problem.mesh_normals[2]*problem.dS)
            # equilibrium = equilibrium_rhog - equilibrium_p_z
        else:
            with open(porosity_filename, "w") as file:
                file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                file.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
                file.write('  <mesh_function type="double" dim="'+str(dim)+'" size="'+str(n_cells)+'">\n')
                for k_cell in range(n_cells):
                    file.write('    <entity index="'+str(k_cell)+'" value="'+str(phis[k_cell])+'"/>\n')
                file.write('  </mesh_function>\n')
                file.write('</dolfin>\n')
                file.close()
            kinematics = dmech.Kinematics(U=problem.get_displacement_subsol().func)
            nf = dolfin.dot(problem.mesh_normals, dolfin.inv(kinematics.F))
            nf_norm = dolfin.sqrt(dolfin.dot(nf,nf))
            n = nf/nf_norm
            # equilibrium_rhog= dolfin.assemble(1e-6*f*problem.Phis0*problem.dV)
            # p = problem.get_p_subsol().func
            # equilibrium_p_x = dolfin.assemble(p * n[0] * nf_norm * kinematics.J * problem.dS)
            # equilibrium_p_y = dolfin.assemble(p * n[1] * nf_norm * kinematics.J * problem.dS)
            # equilibrium_p_z = dolfin.assemble(p * n[2] * nf_norm * kinematics.J * problem.dS)
            # equilibrium = equilibrium_rhog - equilibrium_p_z
            # lbda = problem.get_lbda_subsol().func
            # mu =  problem.get_mu_subsol().func
            # gamma =  problem.get_gamma_subsol().func
            # x0 = problem.get_x0_direct_subsol().func
            # print("x0", x0.vector()[:])
            # print("x0 computed", dolfin.assemble(problem.get_porosity_subsol().func*(problem.X[2]+problem.get_displacement_subsol().func[2])*kinematics.J*problem.dV)/dolfin.assemble(problem.get_porosity_subsol().func*kinematics.J*problem.dV))
        # # print("equilibrium_rhog",dolfin.assemble(1e-6*f*problem.get_porosity_subsol().func*problem.dV))
        # print("equilibrium_rhog_end",equilibrium_rhog)
        # print("equilibrium_p", equilibrium_p_x)
        # print("equilibrium_p", equilibrium_p_y)
        # print("equilibrium_p", equilibrium_p_z)
        # print("equilibrium=", equilibrium)
        if get_invariants:

            U_inspi = problem.get_displacement_subsol().func
            U_expi = get_invariants["Uexpi"]
            U_tot = U_inspi.copy(deepcopy=True)
            U_tot.vector().set_local(U_inspi.vector()[:] - U_expi.vector()[:])
            # print("U_inspi", U_inspi.vector()[:])
            # print("U_expi", U_expi.vector()[:])
            # print("U_tot", U_tot.vector()[:])

            

            sfoi_fe = dolfin.FiniteElement(
                family="DG",
                cell=mesh.ufl_cell(),
                degree=0)
            sfoi_fs = dolfin.FunctionSpace(
                mesh,
                sfoi_fe)

            
            dolfin.ALE.move(mesh, U_expi)

            fe_u = dolfin.VectorElement(
                family="CG",
                cell=mesh.ufl_cell(),
                degree=1)
            
            U_fs = dolfin.FunctionSpace(mesh, fe_u)
            U_tot_expi=dolfin.Function(U_fs)
            
            U_tot_expi.vector().set_local(U_tot.vector()[:])

            kinematics_new = dmech.Kinematics(U=U_tot_expi, U_old=None, Q_expr=None)

            material =  dmech.material_factory(kinematics_new, "CGNHMR_poro", mat_params["parameters"], problem)
            stresses = material.Sigma
            stresses_iso = 1/3.*dolfin.tr(stresses) * dolfin.Identity(3)
            stresses_devia= stresses - stresses_iso
            von_Mises = (3./2*dolfin.inner(stresses_devia, stresses_devia))**(1/2)
            

            

            # sfoi_fe = dolfin.TensorElement(
            #     family="DG",
            #     cell=mesh.ufl_cell(),
            #     degree=0)
            # sfoi_fs = dolfin.FunctionSpace(
            #     mesh,
            #     sfoi_fe)

            # xdmf_file_mesh_vtk_0 = dolfin.XDMFFile("./mesh_tot_0.xdmf")
            # xdmf_file_mesh_vtk_0.write(mesh)
            # xdmf_file_mesh_vtk_0.close()

            
            # dolfin.ALE.move(mesh, U_inspi)
            
            # dolfin.ALE.move(mesh, U_tot)
            
            sfoi_fe = dolfin.FiniteElement(
                family="DG",
                cell=mesh.ufl_cell(),
                degree=0)
            sfoi_fs = dolfin.FunctionSpace(
                mesh,
                sfoi_fe)
            

            # mfoi_fe = dolfin.TensorElement(
            #     family="Quadrature",
            #     cell=mesh.ufl_cell(),
            #     degree=0,
            #     quad_scheme="default")
            
            mfoi_fe = dolfin.TensorElement(
                family="DG",
                cell=mesh.ufl_cell(),
                degree=0)


            sfoi_fs_t = dolfin.FunctionSpace(
                mesh,
                mfoi_fe)
            

            # xdmf_file_mesh = dolfin.XDMFFile("./mesh_tot.xdmf")
            # xdmf_file_mesh.write(mesh)
            # xdmf_file_mesh_vtk_1 = dolfin.XDMFFile("./mesh_tot_1.xdmf")
            # xdmf_file_mesh_vtk_1.write(mesh)
            # xdmf_file_mesh_vtk_1.close()

            # process=subprocess.Popen(["meshio", "convert", "mesh_tot_1.xdmf", "mesh_tot_1.vtk"])
            

            C_tot_field = kinematics_new.C
            # print("C_tot_field", C_tot_field, type(C_tot_field))
            C_tot_proj = dolfin.project(C_tot_field, sfoi_fs_t)
            C_tot = C_tot_proj.vector().get_local()
            



            mu_I1_test, sigma_I1_test, mu_I2_test, sigma_I2_test = 0, 0, 0, 0

            I1_test_lst = []
            I2_test_lst = []
            for iteration_c in range(len(C_tot)//9):
                C_mat = [[C_tot[9*iteration_c], C_tot[9*iteration_c+1], C_tot[9*iteration_c+2]], [C_tot[9*iteration_c+3], C_tot[9*iteration_c+4], C_tot[9*iteration_c+5]], [C_tot[9*iteration_c+6], C_tot[9*iteration_c+7], C_tot[9*iteration_c+8]]]
                C_mat = sympy.Matrix(C_mat)
                # print("C_mat", C_mat)
                eigen_values_C = list(C_mat.eigenvals().keys())
                # print("eigen_values_C", eigen_values_C)
                l1, l2, l3 = eigen_values_C[0]**(1/2), eigen_values_C[1]**(1/2), eigen_values_C[2]**(1/2)
                # print("l1,l2,l3", l1, l2, l3)
                I1_test = float((l1+l2+l3)/3)
                I2_test = float((l1*l2+l2*l3+l3*l1)/3)
                # print("I1_test", I1_test)
                # print("I2_test", I2_test)
                I1_test_lst.append(I1_test)
                I2_test_lst.append(I2_test)
                mu_I1_test +=numpy.log(I1_test)
                mu_I2_test +=numpy.log(I2_test)
            

            J_tot_field = kinematics_new.J
            J_tot_proj = dolfin.project(J_tot_field, sfoi_fs)
            # xdmf_file_mesh.write(J_tot_proj, 1)
            # xdmf_file_mesh.write(U_tot, 1)
            # xdmf_file_mesh.close()

            J_tot = J_tot_proj.vector().get_local()

            print("len J", len(J_tot))

            # Ic_tot_field = kinematics_new.IC
            Ic_tot_field = kinematics_new.I1
            Ic_tot_proj = dolfin.project(Ic_tot_field, sfoi_fs)
            Ic_tot = Ic_tot_proj.vector().get_local()


            # IIc_tot_field = kinematics_new.IIC
            IIc_tot_field = kinematics_new.I2
            IIc_tot_proj = dolfin.project(IIc_tot_field, sfoi_fs)
            IIc_tot = IIc_tot_proj.vector().get_local()

            U_tot_expi.rename("U","")
            J_tot_proj.rename("J", "")
            Ic_tot_proj.rename("I1", "")
            IIc_tot_proj.rename("I2", "")

          
            dmech.write_VTU_file("mesh_disp", U_tot_expi, 0)



            # print("len", len(J_tot_proj), len(Ic_tot_proj), len(IIc_tot_proj))

            mu_J_tot, mu_Ic_tot, mu_IIc_tot = 0, 0, 0
            sigma_J_tot, sigma_Ic_tot, sigma_IIc_tot = 0, 0, 0
            number_cells = 0



            for cell in range(mesh.num_cells()):
                number_cells += 1
                mu_J_tot += numpy.log(J_tot[cell])
                mu_Ic_tot += numpy.log(Ic_tot[cell])
                mu_IIc_tot += numpy.log(IIc_tot[cell])


            print("number cells", number_cells)

            mu_J_tot /= number_cells
            mu_Ic_tot /= number_cells
            mu_IIc_tot /= number_cells
            mu_I1_test /= number_cells
            mu_I2_test /= number_cells

            compteur = 0

            for cell in range(mesh.num_cells()):
                sigma_J_tot += (numpy.log(J_tot[cell])-mu_J_tot)*(numpy.log(J_tot[cell])-mu_J_tot)
                sigma_Ic_tot += (numpy.log(Ic_tot[cell])-mu_Ic_tot)*(numpy.log(Ic_tot[cell])-mu_Ic_tot)
                sigma_IIc_tot += (numpy.log(IIc_tot[cell])-mu_IIc_tot)*(numpy.log(IIc_tot[cell])-mu_IIc_tot)
                sigma_I1_test += (numpy.log(I1_test_lst[cell])-mu_I1_test)*(numpy.log(I1_test_lst[compteur])-mu_I1_test)
                sigma_I2_test += (numpy.log(I2_test_lst[cell])-mu_I2_test)*(numpy.log(I2_test_lst[compteur])-mu_I2_test)
                compteur += 1
            print("compteur", compteur)

            sigma_J_tot /= number_cells
            sigma_J_tot = sigma_J_tot**(1/2)
            sigma_Ic_tot /= number_cells
            sigma_Ic_tot = sigma_Ic_tot**(1/2)
            sigma_IIc_tot /= number_cells
            sigma_IIc_tot = sigma_IIc_tot**(1/2)
            sigma_I1_test /= number_cells
            sigma_I1_test = sigma_I1_test**(1/2)
            sigma_I2_test /= number_cells
            sigma_I2_test = sigma_I2_test**(1/2)

            print(" J_tot",  J_tot)
            print("IC_tot", Ic_tot)
            print("IIc_tot", IIc_tot)
            print("sigma, mu J", sigma_J_tot, mu_J_tot)
            print("sigma, mu Ic", sigma_Ic_tot, mu_Ic_tot)
            print("sigma, mu IIc", sigma_IIc_tot, mu_IIc_tot)

            

            results_zones = {}
            results_zones["zone"] = []
            # results_zones["average_phi"] = []
            # results_zones["std+_phi"] = []
            # results_zones["std-_phi"] = []
            

            results_zones["average_J"] = []
            results_zones["std+_J"] = []
            results_zones["std-_J"] = []
            results_zones["J^"] = []
            results_zones["average_I1"] = []
            results_zones["std+_I1"] = []
            results_zones["std-_I1"] = []
            results_zones["I1^"] = []
            results_zones["average_I2"] = []
            results_zones["std+_I2"] = []
            results_zones["std-_I2"] = []
            results_zones["I2^"] = []
            results_zones["I1^test"] = []
            results_zones["I2^test"] = []
            c_cells=0
            I1_chapeau_whole=[]
            I2_chapeau_whole=[]

            J_chapeau_write = J_tot_proj.copy(deepcopy=True)
            I1_chapeau_write = Ic_tot_proj.copy(deepcopy=True)
            I2_chapeau_write = IIc_tot_proj.copy(deepcopy=True)

            for i in range(1, zones+1):
                # print("zone", i)
                results_zones["zone"].append(i)
                marked_cells = dolfin.SubsetIterator(domains_zones, i)
                J_lst = []
                I1_lst = []
                I2_lst = []
                I1_test_lst2 = []
                I2_test_lst2 = []
                # print("phis=", phis)
                for cell in marked_cells:
                    c_cells+=1
                    # print("cell index", cell.index())
                    # print("type(cell index)", cell.index())
                    # print("J(cell index)", Jfun[cell.index])
                    # cell_index = int(cell.index())
                    # phi_lst.append(phis[cell.index()])
                    J_lst.append(J_tot[cell.index()])
                    J_chapeau_write.vector()[cell.index()] = (numpy.log(J_tot[cell.index()])-mu_J_tot)/sigma_J_tot
                    I1_chapeau_write.vector()[cell.index()] = (numpy.log(Ic_tot[cell.index()])-mu_Ic_tot)/sigma_Ic_tot
                    I2_chapeau_write.vector()[cell.index()] = (numpy.log(IIc_tot[cell.index()])-mu_IIc_tot)/sigma_IIc_tot
                    I1_lst.append(Ic_tot[cell.index()])
                    I2_lst.append(IIc_tot[cell.index()])
                    I1_test_lst2.append(I1_test_lst[cell.index()])
                    I2_test_lst2.append(I2_test_lst[cell.index()])
                # phi_average = numpy.average(phi_lst)
                # phi_std =numpy.std(phi_lst)
                # results_zones["average_phi"].append(phi_average)
                # results_zones["std+_phi"].append(phi_average + phi_std)
                # results_zones["std-_phi"].append(phi_average - phi_std)
                J_average = numpy.average(J_lst)
                # print("J_average", J_average)
                J_chapeau = ((numpy.log(J_average)-mu_J_tot)/sigma_J_tot)
                J_std =numpy.std(J_lst)
                results_zones["average_J"].append(J_average)
                results_zones["std+_J"].append(J_average + J_std)
                results_zones["std-_J"].append(J_average - J_std)
                I1_average = numpy.average(I1_lst)
                I1_chapeau = ((numpy.log(I1_average)-mu_Ic_tot)/sigma_Ic_tot)
                I1_average_test = numpy.average(I1_test_lst2)
                I1_chapeau_test = ((numpy.log(I1_average_test)-mu_I1_test)/sigma_I1_test)
                I1_std =numpy.std(I1_lst)
                results_zones["average_I1"].append(I1_average)
                results_zones["std+_I1"].append(I1_average + I1_std)
                results_zones["std-_I1"].append(I1_average - I1_std)
                I2_average = numpy.average(I2_lst)
                I2_chapeau = ((numpy.log(I2_average)-mu_IIc_tot)/sigma_IIc_tot)
                I2_std =numpy.std(I2_lst)
                I2_average_test = numpy.average(I2_test_lst2)
                I2_chapeau_test = ((numpy.log(I2_average_test)-mu_I2_test)/sigma_I2_test)
                # print("I1_test_lst2", I1_test_lst2)
                # print("numpy.log(I2_average_test)", numpy.log(I2_average_test))
                # print("mu_I2_test", mu_I2_test)
                # print("sigma_I2_test", sigma_I2_test)
                results_zones["I1^test"].append(I1_chapeau_test)
                results_zones["I2^test"].append(I2_chapeau_test)

                results_zones["average_I2"].append(I2_average)
                results_zones["std+_I2"].append(I2_average + I2_std)
                results_zones["std-_I2"].append(I2_average - I2_std)
                results_zones["J^"].append(J_chapeau)
                results_zones["I1^"].append(I1_chapeau)
                results_zones["I2^"].append(I2_chapeau)
            print("nb cells", c_cells)
            print("results", results_zones)


            J_chapeau_write.rename("J^", "")
            I1_chapeau_write.rename("I1^", "")
            I2_chapeau_write.rename("I2^", "")

            # VM_space = dolfin.FunctionSpace(mesh, 'P', 1)
            von_Mises = dolfin.project(von_Mises, sfoi_fs)
            stresses_iso = dolfin.project(stresses_iso, sfoi_fs_t)
            stresses = dolfin.project(stresses, sfoi_fs_t)

            von_Mises.rename("VM", "")
            stresses_iso.rename("iso", "")
            stresses.rename("stresses", "")

            with dolfin.XDMFFile(res_basename+'final_fields2.xdmf') as file:
                file.parameters.update(
                {
                    "functions_share_mesh": True,
                    "rewrite_function_mesh": False
                })
                file.write(J_chapeau_write, 0.)
                file.write(I1_chapeau_write, 0.)
                file.write(I2_chapeau_write, 0.)
                file.write(stresses_iso, 0.)
                file.write(von_Mises, 0.)
                file.write(stresses, 0.)

                file.write(U_tot_expi, 0.)
                file.write(J_tot_proj, 0.)
                file.write(Ic_tot_proj, 0.)
                file.write(IIc_tot_proj, 0.)

            print("fields are written")

            df = pandas.DataFrame(results_zones)
            myfile= open(res_basename+str(load_params)+".dat", 'w')
            myfile.write(df.to_string(index=False))
            myfile.close()


            dwarp.compute_strains(working_folder="/Users/peyrault/Documents/Gravity/Tests", working_basename="mesh_disp", working_ext="vtu",  verbose=1)

            # print("get_J", get_J)
        if get_J:
            if inverse:
                return(problem.get_displacement_subsol().func,  phi, problem.mesh_V0, 1)
            else:
                vol =  problem.get_deformed_volume_subsol().func.vector().get_local()[0]
                return(problem.get_displacement_subsol().func,  phi, problem.mesh_V0, vol)
        else:
            # print("in this function")
            if inverse:
                return(problem.get_displacement_subsol().func,  phi, problem.dV)
            else:
                return(problem.get_displacement_subsol().func,  phis, problem.dV)
    else:
        return
    

# def get_real_n_ext(mesh):

    
#     # mesh = UnitSquareMesh(10, 10)

#     ufc_element = UFCTetrahedron() if mesh.ufl_cell() == dolfin.tetrahedron else UFCTriangle()

#     tdim = mesh.topology().dim()
#     fdim = tdim - 1
#     top = mesh.topology()
#     mesh.init(tdim, fdim)
#     mesh.init(fdim, tdim)
#     f_to_c = top(fdim, tdim)

#     bndry_facets = []
#     for facet in dolfin.facets(mesh):
#         f_index = facet.index()
#         cells = f_to_c(f_index)
#         if len(cells) == 1:
#             bndry_facets.append(f_index)

#     # As we are using simplices, we can compute the Jacobian at an arbitrary point
#     fiat_el = Lagrange(ufc_simplex(tdim), 1)  
#     dphi_ = fiat_el.tabulate(1, numpy.array([0,]*tdim))
#     dphi = numpy.zeros((tdim, dolfin.Cell(mesh, 0).num_entities(0)), dtype=numpy.float64)
#     for i in range(tdim):
#         t = tuple()
#         for j in range(tdim):
#             if i == j:
#                 t += (1, )
#             else:
#                 t += (0, )
#         dphi[i] = dphi_[t]

#     test = []
#     lbda = [0.,1.,0.]

    # c_to_f = top(tdim, fdim)
    # for facet in bndry_facets:
    #     cell = f_to_c(facet)[0]
    #     facets = c_to_f(cell)
    #     local_index = numpy.flatnonzero(facets==facet)[0]
    #     normal = ufc_element.compute_reference_normal(fdim, local_index)
    #     coord_dofs = mesh.cells()[cell]
    #     coords = mesh.coordinates()[coord_dofs]
    #     J = numpy.dot(dphi, coords).T
    #     Jinv = numpy.linalg.inv(J)
    #     print("J=", J)
    #     n_glob = numpy.dot(Jinv.T, normal)
    #     n_glob = n_glob/numpy.linalg.norm(n_glob)
    #     test.append(dolfin.dot(dolfin.Constant(lbda), dolfin.Constant(n_glob)))
    # return(n_glob)
