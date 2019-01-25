#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2019                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import collections
import dolfin
import numpy
import operator

import dolfin_cm as dcm

################################################################################

class Problem():



    def __init__(self):

        self.subsols = collections.OrderedDict()

        self.inelastic_behaviors_mixed    = []
        self.inelastic_behaviors_internal = []

        self.constraints = []
        self.penalties   = []

        self.surface0_loadings  = []
        self.pressure0_loadings = []
        self.volume0_loadings   = []
        self.surface_loadings   = []
        self.pressure_loadings  = []
        self.volume_loadings    = []

        self.steps = []

        self.fois = []
        self.qois = []

        self.form_compiler_parameters = {}



    def set_mesh(self,
            mesh,
            compute_normals=False,
            compute_local_cylindrical_basis=False):

        self.dim = mesh.ufl_domain().geometric_dimension()

        self.mesh = mesh
        self.dV = dolfin.Measure(
            "dx",
            domain=self.mesh)
        self.mesh_V0 = dolfin.assemble(dolfin.Constant(1) * self.dV)

        if (compute_normals):
            self.mesh_normals = dolfin.FacetNormal(mesh)

        if (compute_local_cylindrical_basis):
            self.local_basis_fe = dolfin.VectorElement(
                family="DG",
                cell=mesh.ufl_cell(),
                degree=1)

            self.eR_expr = dolfin.Expression(
                cppcode=("+x[0]/sqrt(pow(x[0],2)+pow(x[1],2))", "+x[1]/sqrt(pow(x[0],2)+pow(x[1],2))"),
                element=self.local_basis_fe)
            self.eT_expr = dolfin.Expression(
                cppcode=("-x[1]/sqrt(pow(x[0],2)+pow(x[1],2))", "+x[0]/sqrt(pow(x[0],2)+pow(x[1],2))"),
                element=self.local_basis_fe)

            self.Q_expr = dolfin.as_matrix([[self.eR_expr[0], self.eR_expr[1]],
                                            [self.eT_expr[0], self.eT_expr[1]]])

            self.local_basis_fs = dolfin.FunctionSpace(
                mesh,
                self.local_basis_fe) # MG: element keyword don't work here…

            self.eR_func = dolfin.interpolate(
                v=self.eR_expr,
                V=self.local_basis_fs)
            self.eR_func.rename("eR", "eR")

            self.eT_func = dolfin.interpolate(
                v=self.eT_expr,
                V=self.local_basis_fs)
            self.eT_func.rename("eT", "eT")
        else:
            self.Q_expr = None



    def set_measures(self,
            domains=None,
            boundaries=None):

        if (domains is not None):
            self.dV = dolfin.Measure(
                "dx",
                domain=self.mesh,
                subdomain_data=domains)
        else:
            self.dV = dolfin.Measure(
                "dx",
                domain=self.mesh)

        if (boundaries is not None):
            self.dS = dolfin.Measure(
                "ds",
                domain=self.mesh,
                subdomain_data=boundaries)
        else:
            self.dS = dolfin.Measure(
                "ds",
                domain=self.mesh)



    def add_subsol(self,
            name,
            *args,
            **kwargs):

        subsol = dcm.SubSol(
            name=name,
            *args,
            **kwargs)
        self.subsols[name] = subsol
        return subsol



    def add_scalar_subsol(self,
            name,
            family="CG",
            degree=1,
            init_val=None):

        fe = dolfin.FiniteElement(
            family=family,
            cell=self.mesh.ufl_cell(),
            degree=degree)

        self.add_subsol(
            name=name,
            fe=fe,
            init_val=init_val)



    def add_vector_subsol(self,
            name,
            family="CG",
            degree=1,
            init_val=None):

        fe = dolfin.VectorElement(
            family=family,
            cell=self.mesh.ufl_cell(),
            degree=degree)

        self.add_subsol(
            name=name,
            fe=fe,
            init_val=init_val)



    def add_tensor_subsol(self,
            name,
            family="CG",
            degree=1,
            init_val=None):

        fe = dolfin.TensorElement(
            family=family,
            cell=self.mesh.ufl_cell(),
            degree=degree)

        self.add_subsol(
            name=name,
            fe=fe,
            init_val=init_val)



    def get_subsol_function_space(self,
            name):

        index = self.subsols.keys().index(name)
        # print str(name)+" index = "+str(index)
        return self.sol_fs.sub(index)



    def get_displacement_function_space(self):

        if (len(self.subsols) == 1):
            return self.sol_fs
        else:
            return self.get_subsol_function_space(name="U")



    def get_unloaded_displacement_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="Up")



    def get_pressure_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="P")



    def get_unloaded_pressure_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="Pp")



    def get_growth_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="thetag")
        #return self.get_subsol_function_space(name="Fg")



    def get_relaxation_function_space(self):

        assert (len(self.subsols) > 1)
        return self.get_subsol_function_space(name="Fr")



    def set_solution_finite_element(self):

        if (len(self.subsols) == 1):
            self.sol_fe = self.subsols.values()[0].fe
        else:
            self.sol_fe = dolfin.MixedElement([subsol.fe for subsol in self.subsols.itervalues()])
        #print self.sol_fe



    def set_solution_function_space(self):

        self.sol_fs = dolfin.FunctionSpace(
            self.mesh,
            self.sol_fe) # MG: element keyword don't work here…
        #print self.sol_fs



    def set_solution_functions(self):

        self.sol_func     = dolfin.Function(self.sol_fs)
        self.sol_old_func = dolfin.Function(self.sol_fs)
        self.dsol_func    = dolfin.Function(self.sol_fs)
        self.dsol_test    = dolfin.TestFunction(self.sol_fs)
        self.dsol_tria    = dolfin.TrialFunction(self.sol_fs)

        if (len(self.subsols) == 1):
            subfuncs  = (self.sol_func,)
            dsubtests = (self.dsol_test,)
            dsubtrias = (self.dsol_tria,)
            funcs     = (self.sol_func,)
            funcs_old = (self.sol_old_func,)
            dfuncs    = (self.dsol_func,)
        else:
            subfuncs  = dolfin.split(self.sol_func)
            dsubtests = dolfin.split(self.dsol_test)
            dsubtrias = dolfin.split(self.dsol_tria)
            funcs     = dolfin.Function(self.sol_fs).split(deepcopy=1)
            funcs_old = dolfin.Function(self.sol_fs).split(deepcopy=1)
            dfuncs    = dolfin.Function(self.sol_fs).split(deepcopy=1)

        for k_subsol in xrange(len(self.subsols)):
            subsol = self.subsols.values()[k_subsol]

            subsol.subfunc  = subfuncs[k_subsol]
            subsol.dsubtest = dsubtests[k_subsol]
            subsol.dsubtria = dsubtrias[k_subsol]


            subsol.func = funcs[k_subsol]
            subsol.func.rename(subsol.name, subsol.name)
            subsol.func_old = funcs_old[k_subsol]
            subsol.func_old.rename(subsol.name+"_old", subsol.name+"_old")
            subsol.dfunc = dfuncs[k_subsol]
            subsol.dfunc.rename("d"+subsol.name, "d"+subsol.name)

        init_val = [str(val) for val in numpy.concatenate([subsol.init_val.flatten() for subsol in self.subsols.itervalues()])]
        self.sol_func.interpolate(dolfin.Expression(
            init_val,
            element=self.sol_fe))
        self.sol_old_func.interpolate(dolfin.Expression(
            init_val,
            element=self.sol_fe))
        if (len(self.subsols) > 1):
            dolfin.assign(
                self.get_subsols_func_lst(),
                self.sol_func)
            dolfin.assign(
                self.get_subsols_func_old_lst(),
                self.sol_old_func)



    def get_subsols_func_lst(self):

        return [subsol.func for subsol in self.subsols.itervalues()]



    def get_subsols_func_old_lst(self):

        return [subsol.func_old for subsol in self.subsols.itervalues()]



    def get_subsols_dfunc_lst(self):

        return [subsol.dfunc for subsol in self.subsols.itervalues()]



    def set_quadrature_degree(self,
            quadrature_degree):

        self.form_compiler_parameters["quadrature_degree"] = quadrature_degree



    def set_foi_finite_elements_DG(self,
            degree=0): # MG20180420: DG elements are simpler to manage than quadrature elements, since quadrature elements must be compatible with the expression's degree, which is not always trivial (e.g., for J…)

        self.sfoi_fe = dolfin.FiniteElement(
            family="DG",
            cell=self.mesh.ufl_cell(),
            degree=degree)

        self.vfoi_fe = dolfin.VectorElement(
            family="DG",
            cell=self.mesh.ufl_cell(),
            degree=degree)

        self.mfoi_fe = dolfin.TensorElement(
            family="DG",
            cell=self.mesh.ufl_cell(),
            degree=degree)



    def set_foi_finite_elements_Quad(self,
            degree=0): # MG20180420: DG elements are simpler to manage than quadrature elements, since quadrature elements must be compatible with the expression's degree, which is not always trivial (e.g., for J…)

        self.sfoi_fe = dolfin.FiniteElement(
            family="Quadrature",
            cell=self.mesh.ufl_cell(),
            degree=degree,
            quad_scheme="default")
        self.sfoi_fe._quad_scheme = "default"           # MG20180406: is that even needed?
        for sub_element in self.sfoi_fe.sub_elements(): # MG20180406: is that even needed?
            sub_element._quad_scheme = "default"        # MG20180406: is that even needed?

        self.vfoi_fe = dolfin.VectorElement(
            family="Quadrature",
            cell=self.mesh.ufl_cell(),
            degree=degree,
            quad_scheme="default")
        self.vfoi_fe._quad_scheme = "default"           # MG20180406: is that even needed?
        for sub_element in self.vfoi_fe.sub_elements(): # MG20180406: is that even needed?
            sub_element._quad_scheme = "default"        # MG20180406: is that even needed?

        self.mfoi_fe = dolfin.TensorElement(
            family="Quadrature",
            cell=self.mesh.ufl_cell(),
            degree=degree,
            quad_scheme="default")
        self.mfoi_fe._quad_scheme = "default"           # MG20180406: is that still needed?
        for sub_element in self.mfoi_fe.sub_elements(): # MG20180406: is that still needed?
            sub_element._quad_scheme = "default"        # MG20180406: is that still needed?



    def set_foi_function_spaces(self):

        self.sfoi_fs = dolfin.FunctionSpace(
            self.mesh,
            self.sfoi_fe) # MG: element keyword don't work here…

        self.vfoi_fs = dolfin.FunctionSpace(
            self.mesh,
            self.vfoi_fe) # MG: element keyword don't work here…

        self.mfoi_fs = dolfin.FunctionSpace(
            self.mesh,
            self.mfoi_fe) # MG: element keyword don't work here…



    def add_foi(self,
            *args,
            **kwargs):

        foi = dcm.FOI(
            *args,
            form_compiler_parameters=self.form_compiler_parameters,
            **kwargs)
        self.fois += [foi]
        return foi



    def update_fois(self):

        for foi in self.fois:
            foi.update()



    def get_fois_func_lst(self):

        return [foi.func for foi in self.fois]



    def add_constraint(self,
            *args,
            **kwargs):

        constraint = dcm.Constraint(
            *args,
            **kwargs)
        self.constraints += [constraint]
        return constraint



    def add_penalty(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.penalties += [loading]
        return loading



    def add_surface0_loading(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.surface0_loadings += [loading]
        return loading



    def add_pressure0_loading(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.pressure0_loadings += [loading]
        return loading



    def add_volume0_loading(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.volume0_loadings += [loading]
        return loading



    def add_surface_loading(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.surface_loadings += [loading]
        return loading



    def add_pressure_loading(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.pressure_loadings += [loading]
        return loading



    def add_volume_loading(self,
            *args,
            **kwargs):

        loading = dcm.Loading(
            *args,
            **kwargs)
        self.volume_loadings += [loading]
        return loading



    def add_step(self,
            Deltat=1.,
            *args,
            **kwargs):

        if len(self.steps) == 0:
            t_ini = 0.
            t_fin = Deltat
        else:
            t_ini = self.steps[-1].t_fin
            t_fin = t_ini + Deltat
        step = dcm.Step(
            t_ini=t_ini,
            t_fin=t_fin,
            *args,
            **kwargs)
        self.steps += [step]
        return step



    def add_qoi(self,
            *args,
            **kwargs):

        qoi = dcm.QOI(
            *args,
            form_compiler_parameters=self.form_compiler_parameters,
            **kwargs)
        self.qois += [qoi]
        return qoi



    def add_strain_qois(self,
            strain_type="elastic",
            configuration_type="loaded"):

        if (configuration_type == "loaded"):
            kin = self.kinematics
        elif (configuration_type == "unloaded"):
            kin = self.unloaded_kinematics

        if (strain_type == "elastic"):
            basename = "E^e_"
            strain = kin.Ee
        elif (strain_type == "total"):
            basename = "E^t_"
            strain = kin.Et

        self.add_qoi(
            name=basename+"XX",
            expr=strain[0,0] * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"YY",
                expr=strain[1,1] * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"ZZ",
                    expr=strain[2,2] * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"XY",
                expr=strain[0,1] * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"YZ",
                    expr=strain[1,2] * self.dV)
                self.add_qoi(
                    name=basename+"ZX",
                    expr=strain[2,0] * self.dV)



    def add_stress_qois(self,
            stress_type="cauchy"):

        if (stress_type in ("cauchy", "sigma")):
            basename = "s_"
            stress = self.sigma
        elif (stress_type in ("piola", "PK2", "Sigma")):
            basename = "S_"
            stress = self.Sigma
        elif (stress_type in ("PK1", "P")):
            basename = "P_"
            stress = self.PK1

        self.add_qoi(
            name=basename+"XX",
            expr=stress[0,0] * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"YY",
                expr=stress[1,1] * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"ZZ",
                    expr=stress[2,2] * self.dV)
        if (self.dim >= 2):
            self.add_qoi(
                name=basename+"XY",
                expr=stress[0,1] * self.dV)
            if (self.dim >= 3):
                self.add_qoi(
                    name=basename+"YZ",
                    expr=stress[1,2] * self.dV)
                self.add_qoi(
                    name=basename+"ZX",
                    expr=stress[2,0] * self.dV)



    def update_qois(self):

        for qoi in self.qois:
            qoi.update()