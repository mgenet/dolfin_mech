#coding=utf8

##############################################################################
###                                                                        ###
### Created by Mahdi Manoocherhtayebi 2020-2024                            ###
###                                                                        ###  
### Inspired by https://comet-fenics.readthedocs.io/en/latest/demo/        ###
###                         periodic_homog_elas/periodic_homog_elas.html   ###
###                                                                        ###
### École Polytechnique, Palaiseau, France                                 ###
###                                                                        ###
##############################################################################

import dolfin
import numpy

import dolfin_mech as dmech

#############################################################################

class HomogenizationProblem():



    def __init__(self,
            dim,
            mesh,
            mat_params,
            vol,
            bbox,
            vertices=None):

        self.mesh = mesh
        self.E_s = mat_params["E"]
        self.nu_s = mat_params["nu"]
        self.dim = dim
        self.vertices = vertices
        self.vol = vol

        self.bbox = bbox
        # print("self.vol:", self.vol)
    
        self.material_parameters = [(self.E_s, self.nu_s)]



    def eps(self, v):

        return dolfin.sym(dolfin.grad(v))



    def sigma(self, v, i, Eps):

        E, nu = self.material_parameters[i]
        lmbda = dolfin.Constant(E*nu/(1+nu)/(1-2*nu))
        mu = dolfin.Constant(E/2/(1+nu))
        return lmbda*dolfin.tr(self.eps(v)+Eps)*dolfin.Identity(self.dim) + 2*mu*(self.eps(v)+Eps)



    def Voigt2strain(self, s):

        if (self.dim==2):
            strain_tensor = numpy.array([[s[0]   , s[2]/2.],
                                         [s[2]/2., s[1]   ]])
        if (self.dim==3):
            strain_tensor = numpy.array([[s[0]   , s[5]/2.,  s[4]/2],
                                         [s[5]/2., s[1]   ,  s[3]/2],
                                         [s[4]/2 , s[3]/2 ,  s[2]  ]])
        return strain_tensor



    def get_macro_strain(self, i):
        """returns the macroscopic strain for the 3 elementary load cases"""

        if (self.dim==2):
            Eps_Voigt = numpy.zeros(3)
        if (self.dim==3):
            Eps_Voigt = numpy.zeros(6)
        Eps_Voigt[i] = 1
        return self.Voigt2strain(Eps_Voigt)



    def stress2Voigt(self, s):

        if (self.dim==2):
            stress_vector = dolfin.as_vector([s[0,0], s[1,1], s[0,1]])
        if (self.dim==3):
            stress_vector = dolfin.as_vector([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]])
        return stress_vector



    def get_lambda_and_mu(self):

        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 2)
        Re = dolfin.VectorElement("R", self.mesh.ufl_cell(), 0)
        W = dolfin.FunctionSpace(self.mesh, dolfin.MixedElement([Ve, Re]), constrained_domain=dmech.PeriodicSubDomain(self.dim, self.bbox, self.vertices))
        V = dolfin.FunctionSpace(self.mesh, Ve)

        v_, lamb_ = dolfin.TestFunctions(W)
        dv, dlamb = dolfin.TrialFunctions(W)
        w = dolfin.Function(W)
        dx = dolfin.Measure('dx')(domain=self.mesh)

        if (self.dim==2): Eps = dolfin.Constant(((0, 0), (0, 0)))
        if (self.dim==3): Eps = dolfin.Constant(((0, 0, 0), (0, 0, 0), (0, 0, 0)))

        F = sum([dolfin.inner(self.sigma(dv, 0, Eps), self.eps(v_))*dx])

        a, L = dolfin.lhs(F), dolfin.rhs(F)
        a += dolfin.dot(lamb_,dv)*dx + dolfin.dot(dlamb,v_)*dx

        if (self.dim==2): Chom = numpy.zeros((3, 3))
        if (self.dim==3): Chom = numpy.zeros((6, 6))

        if (self.dim==2): Enum=enumerate(["Exx", "Eyy", "Exy"])
        if (self.dim==3): Enum=enumerate(["Exx", "Eyy", "Ezz", "Eyz", "Exz", "Exy"])

        for (j, case) in Enum:
            # print("Solving {} case...".format(case))
            macro_strain = self.get_macro_strain(j)
            Eps.assign(dolfin.Constant(macro_strain))
            if (self.dim ==3):
                dolfin.solve(a == L, w, [], solver_parameters={"linear_solver": "cg"})
            else:
                dolfin.solve(a == L, w, [], solver_parameters={"linear_solver": "lu"})
            (v, lamb) = dolfin.split(w)
            # xdmf_file_per.write(w, float(j))

            if (self.dim==2): Sigma_len = 3
            if (self.dim==3): Sigma_len = 6

            Sigma = numpy.zeros((Sigma_len,))
            
            # self.vol = dolfin.assemble(dolfin.Constant(1) * dx)
            for k in range(Sigma_len):
                Sigma[k] = dolfin.assemble(self.stress2Voigt(self.sigma(v, 0, Eps))[k]*dx)/self.vol
            Chom[j, :] = Sigma
        
        # print(Chom)
        
        lmbda_hom = Chom[0, 1]
        if (self.dim==2): mu_hom = Chom[2, 2]
        if (self.dim==3): mu_hom = Chom[4, 4]

        # print(Chom[0, 0], lmbda_hom + 2*mu_hom)
        # print('Chom:' +str(Chom))

        # print("lmbda_hom: " +str(lmbda_hom))
        # print("mu_hom: " +str(mu_hom))

        E_hom = mu_hom*(3*lmbda_hom + 2*mu_hom)/(lmbda_hom + mu_hom)
        nu_hom = lmbda_hom/(lmbda_hom + mu_hom)/2

        # print("E_hom: " +str(E_hom))
        # print("nu_hom: " +str(nu_hom))
        # print(Chom[0, 0], lmbda_hom + 2*mu_hom)
        # print("Isotropy = " +str ((lmbda_hom + 2*mu_hom - abs(lmbda_hom + 2*mu_hom - Chom[0, 0]))/(lmbda_hom + 2*mu_hom) * 100) +"%")

        self.E_hom = E_hom
        self.nu_hom = nu_hom 
        return lmbda_hom, mu_hom



    def get_kappa(self):

        p_f = 1

        coord = self.mesh.coordinates()
        xmax = max(coord[:,0]); xmin = min(coord[:,0])
        ymax = max(coord[:,1]); ymin = min(coord[:,1])
        if (self.dim==3): zmax = max(coord[:,2]); zmin = min(coord[:,2])
        if (self.dim==2):    
            vol = (xmax - xmin)*(ymax - ymin)

        if (self.dim==3):    
            vol = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)

        vertices = numpy.array([[xmin, ymin],
                                [xmax, ymin],
                                [xmax, ymax],
                                [xmin, ymax]])
            
        tol = 1E-8
        vv = vertices
        a1 = vv[1,:]-vv[0,:] # first vector generating periodicity
        a2 = vv[3,:]-vv[0,:] # second vector generating periodicity
        # check if UC vertices form indeed a parallelogram
        assert numpy.linalg.norm(vv[2, :]-vv[3, :] - a1) <= tol
        assert numpy.linalg.norm(vv[2, :]-vv[1, :] - a2) <= tol

        ################################################## Subdomains & Measures ###

        class BoundaryX0(dolfin.SubDomain):
            def inside(self,x,on_boundary):
                # return on_boundary and dolfin.near(x[0],xmin,tol)
                return on_boundary and dolfin.near(x[0],vv[0,0] + x[1]*a2[0]/vv[3,1],tol)

        class BoundaryY0(dolfin.SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and dolfin.near(x[1],ymin,tol)

        class BoundaryX1(dolfin.SubDomain):
            def inside(self,x,on_boundary):
                # return on_boundary and dolfin.near(x[0],xmax,tol)
                return on_boundary and dolfin.near(x[0],vv[1,0] + x[1]*a2[0]/vv[3,1],tol)

        class BoundaryY1(dolfin.SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and dolfin.near(x[1],ymax,tol)            

        if (self.dim ==3):
            class BoundaryZ0(dolfin.SubDomain):
                def inside(self,x,on_boundary):
                    # return on_boundary and dolfin.near(x[0],xmax,tol)
                    return on_boundary and dolfin.near(x[2],zmin,tol)

            class BoundaryZ1(dolfin.SubDomain):
                def inside(self,x,on_boundary):
                    return on_boundary and dolfin.near(x[2],zmax,tol)

        boundaries_mf = dolfin.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        boundaries_mf.set_all(0)

        bX0 = BoundaryX0()
        bY0 = BoundaryY0()
        bX1 = BoundaryX1()
        bY1 = BoundaryY1()
        if (self.dim==3):
            bZ0 = BoundaryZ0()
            bZ1 = BoundaryZ1()

        bX0.mark(boundaries_mf, 1)
        bY0.mark(boundaries_mf, 2)
        bX1.mark(boundaries_mf, 3)
        bY1.mark(boundaries_mf, 4)
        if (self.dim==3):
            bZ0.mark(boundaries_mf, 5)
            bZ1.mark(boundaries_mf, 6)

        dV = dolfin.Measure(
            "dx",
            domain=self.mesh)
        dS = dolfin.Measure(
            "exterior_facet",
            domain=self.mesh,
            subdomain_data=boundaries_mf)

        ############################################################# Functions #######

        def eps(v_bar):
            return dolfin.sym(dolfin.grad(v_bar))

        def sigma(v_bar, i, Eps):
            E, nu = self.material_parameters[i]
            lmbda = E*nu/(1+nu)/(1-2*nu)
            mu = E/2/(1+nu)
            return lmbda*dolfin.tr(eps(v_bar) + Eps)*dolfin.Identity(self.dim) + 2*mu*(eps(v_bar)+Eps)

        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 2)
        Re = dolfin.VectorElement("R", self.mesh.ufl_cell(), 0)
        W = dolfin.FunctionSpace(self.mesh, dolfin.MixedElement([Ve, Re]), constrained_domain=dmech.PeriodicSubDomain(self.dim, self.bbox, vertices))
        V = dolfin.FunctionSpace(self.mesh, Ve)

        v_, lamb_ = dolfin.TestFunctions(W)
        dv, dlamb = dolfin.TrialFunctions(W)
        w = dolfin.Function(W)

        (v, lamb) = dolfin.split(w)       

        mesh_V0 = dolfin.assemble(dolfin.Constant(1) * dV)

        ##################################################################################
        
        macro_strain = numpy.zeros((self.dim, self.dim))

        if (self.dim==2): Eps = dolfin.Constant(((0, 0), (0, 0)))
        if (self.dim==3): Eps = dolfin.Constant(((0, 0, 0), (0, 0, 0), (0, 0, 0)))
        Eps.assign(dolfin.Constant(macro_strain))

        X = dolfin.SpatialCoordinate(self.mesh)
        X_0 = [0.]*self.dim
        for k_dim in range(self.dim):
            X_0[k_dim] = dolfin.assemble(X[k_dim] * dV)/mesh_V0
        X_0 = dolfin.Constant(X_0)

        u_bar   = dolfin.dot(Eps, X-X_0)
        u_tot = u_bar + v

        ########################################################### Solver ###############

        pf = dolfin.Constant(p_f)
        N = dolfin.FacetNormal(self.mesh)

        F = dolfin.inner(sigma(dv, 0, Eps), eps(v_))*dV - dolfin.inner(-pf * N, v_) * dS(0)

        a, L = dolfin.lhs(F), dolfin.rhs(F)
        a += dolfin.dot(lamb_,dv)*dV + dolfin.dot(dlamb,v_)*dV

        if (self.dim==3):
            dolfin.solve(a == L, w, [], solver_parameters={"linear_solver": "cg"}) 
        else:
            dolfin.solve(a == L, w, [], solver_parameters={"linear_solver": "lu"})  #MMT20230616: "cg" solver doesn't work when material Poisson ratio nu=0.499, and "lu" doesn't work for 3D geometries

        Vs0 = mesh_V0
        vs = dolfin.assemble((1 + dolfin.tr(eps(v)))*dV)
        V0 = vol
        Phi_s0 = Vs0/V0
        Phi_s = vs/V0
        kappa_tilde = Phi_s0**2 * p_f/(Phi_s0 - Phi_s)/2

        return kappa_tilde
