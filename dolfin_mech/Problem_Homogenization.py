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

        self.dim = dim
        if   (self.dim==2):
            self.n_Voigt = 3
        elif (self.dim==3):
            self.n_Voigt = 6

        self.mesh = mesh
        self.dV = dolfin.Measure(
            "dx",
            domain=self.mesh)
        self.mesh_V0 = dolfin.assemble(dolfin.Constant(1) * self.dV)

        self.E_s = mat_params["E"]
        self.nu_s = mat_params["nu"]
        self.lmbda_s = dolfin.Constant(self.E_s*self.nu_s/(1+self.nu_s)/(1-2*self.nu_s))
        self.mu_s = dolfin.Constant(self.E_s/2/(1+self.nu_s))

        self.vertices = vertices
        self.vol = vol
        self.bbox = bbox



    def eps(self, v):

        return dolfin.sym(dolfin.grad(v))



    def sigma(self, v, eps):

        return self.lmbda_s * dolfin.tr(eps + self.eps(v)) * dolfin.Identity(self.dim) + 2*self.mu_s * (eps + self.eps(v))



    def Voigt2strain(self, s):

        if (self.dim==2):
            return numpy.array([[s[0]   , s[2]/2.],
                                [s[2]/2., s[1]   ]])
        if (self.dim==3):
            return numpy.array([[s[0]   , s[5]/2.,  s[4]/2],
                                [s[5]/2., s[1]   ,  s[3]/2],
                                [s[4]/2 , s[3]/2 ,  s[2]  ]])



    def stress2Voigt(self, s):

        if (self.dim==2):
            return dolfin.as_vector([s[0,0], s[1,1], s[0,1]])
        if (self.dim==3):
            return dolfin.as_vector([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]])



    def get_macro_strain(self, i):

        if (self.dim==2):
            Eps_Voigt = numpy.zeros(3)
        if (self.dim==3):
            Eps_Voigt = numpy.zeros(6)
        Eps_Voigt[i] = 1
        return self.Voigt2strain(Eps_Voigt)



    def get_lambda_and_mu(self):

        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 2)
        Re = dolfin.VectorElement("R", self.mesh.ufl_cell(), 0)
        W = dolfin.FunctionSpace(self.mesh, dolfin.MixedElement([Ve, Re]), constrained_domain=dmech.PeriodicSubDomain(self.dim, self.bbox, self.vertices))

        v_test, lmbda_test = dolfin.TestFunctions(W)
        v_tria, lmbda_tria = dolfin.TrialFunctions(W)

        macro_strain = dolfin.Constant(numpy.zeros((self.dim, self.dim)))
        F = dolfin.inner(self.sigma(v_tria, macro_strain), self.eps(v_test)) * self.dV
        a, b = dolfin.lhs(F), dolfin.rhs(F)
        a += dolfin.inner(lmbda_test, v_tria) * self.dV
        a += dolfin.inner(lmbda_tria, v_test) * self.dV

        w = dolfin.Function(W)
        (v, lmbda) = dolfin.split(w)

        C_hom = numpy.zeros((self.n_Voigt, self.n_Voigt))

        for j in range(self.n_Voigt):
            macro_strain.assign(dolfin.Constant(self.get_macro_strain(j)))
            dolfin.solve(a == b, w, solver_parameters={"linear_solver": "mumps"})
            # if (self.dim == 3):
            #     dolfin.solve(a == b, w, [], solver_parameters={"linear_solver": "cg"})
            # else:
            #     dolfin.solve(a == b, w, [], solver_parameters={"linear_solver": "lu"}) # MMT20230616: "cg" solver doesn't work when material Poisson ratio nu=0.499, and "lu" doesn't work for 3D geometries
            # xdmf_file_per.write(w, float(j))

            for k in range(self.n_Voigt):
                C_hom[j,k]  = dolfin.assemble(self.stress2Voigt(self.sigma(v, macro_strain))[k] * self.dV)
                C_hom[j,k] /= self.vol
        # print("C_hom:" + str(C_hom))
        
        lmbda_hom = C_hom[0,1]
        if (self.dim==2): mu_hom = C_hom[2,2]
        if (self.dim==3): mu_hom = C_hom[4,4]

        # print("lmbda_hom: " + str(lmbda_hom))
        # print("mu_hom: " + str(mu_hom))
        # print("Isotropy = " +str ((lmbda_hom + 2*mu_hom - abs(lmbda_hom + 2*mu_hom - C_hom[0,0]))/(lmbda_hom + 2*mu_hom) * 100) +"%")

        # E_hom = mu_hom*(3*lmbda_hom + 2*mu_hom)/(lmbda_hom + mu_hom)
        # nu_hom = lmbda_hom/(lmbda_hom + mu_hom)/2
        # print("E_hom: " + str(E_hom))
        # print("nu_hom: " + str(nu_hom))

        return lmbda_hom, mu_hom



    def get_kappa(self):

        coord = self.mesh.coordinates()
        xmax = max(coord[:,0]); xmin = min(coord[:,0])
        ymax = max(coord[:,1]); ymin = min(coord[:,1])
        if (self.dim==3): zmax = max(coord[:,2]); zmin = min(coord[:,2])

        if (self.dim==2):    
            vol = (xmax - xmin)*(ymax - ymin)
        elif (self.dim==3):    
            vol = (xmax - xmin)*(ymax - ymin)*(zmax - zmin)
        d = vol**(1./self.dim)
        tol = 1e-3 * d

        vertices = numpy.array([[xmin, ymin],
                                [xmax, ymin],
                                [xmax, ymax],
                                [xmin, ymax]])

        vv = vertices
        a1 = vv[1,:] - vv[0,:] # first vector generating periodicity
        a2 = vv[3,:] - vv[0,:] # second vector generating periodicity
        assert numpy.linalg.norm(vv[2,:] - vv[3,:] - a1) <= tol # check if UC vertices form indeed a parallelogram
        assert numpy.linalg.norm(vv[2,:] - vv[1,:] - a2) <= tol # check if UC vertices form indeed a parallelogram

        ################################################## Subdomains & Measures ###

        class BoundaryX0(dolfin.SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and dolfin.near(x[0], vv[0,0]+x[1]*a2[0]/vv[3,1], tol)

        class BoundaryY0(dolfin.SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and dolfin.near(x[1], ymin, tol)

        class BoundaryX1(dolfin.SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and dolfin.near(x[0], vv[1,0]+x[1]*a2[0]/vv[3,1], tol)

        class BoundaryY1(dolfin.SubDomain):
            def inside(self,x,on_boundary):
                return on_boundary and dolfin.near(x[1], ymax, tol)            

        if (self.dim==3):
            class BoundaryZ0(dolfin.SubDomain):
                def inside(self,x,on_boundary):
                    return on_boundary and dolfin.near(x[2], zmin, tol)

            class BoundaryZ1(dolfin.SubDomain):
                def inside(self,x,on_boundary):
                    return on_boundary and dolfin.near(x[2], zmax, tol)

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

        dS = dolfin.Measure(
            "exterior_facet",
            domain=self.mesh,
            subdomain_data=boundaries_mf)

        ############################################################# Functions #######

        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), 2)
        Re = dolfin.VectorElement("R", self.mesh.ufl_cell(), 0)
        W = dolfin.FunctionSpace(self.mesh, dolfin.MixedElement([Ve, Re]), constrained_domain=dmech.PeriodicSubDomain(self.dim, self.bbox, vertices))

        v_test, lmbda_test = dolfin.TestFunctions(W)
        v_tria, lmbda_tria = dolfin.TrialFunctions(W)

        w = dolfin.Function(W)
        (v, lmbda) = dolfin.split(w)

        ##################################################################################
        
        macro_strain = dolfin.Constant(numpy.zeros((self.dim, self.dim)))

        X = dolfin.SpatialCoordinate(self.mesh)
        X_0 = numpy.zeros(self.dim)
        for k_dim in range(self.dim):
            X_0[k_dim]  = dolfin.assemble(X[k_dim] * self.dV)
            X_0[k_dim] /= self.mesh_V0
        X_0 = dolfin.Constant(X_0)

        u_bar = dolfin.dot(macro_strain, X-X_0)
        u_tot = u_bar + v

        ########################################################### Solver ###############

        p_f = 1.
        pf = dolfin.Constant(p_f)
        N = dolfin.FacetNormal(self.mesh)

        F  = dolfin.inner(self.sigma(v_tria, macro_strain), self.eps(v_test)) * self.dV
        F -= dolfin.inner(-pf * N, v_test) * dS(0)
        a, b = dolfin.lhs(F), dolfin.rhs(F)
        a += dolfin.dot(lmbda_test,v_tria) * self.dV
        a += dolfin.dot(lmbda_tria,v_test) * self.dV

        dolfin.solve(a == b, w, solver_parameters={"linear_solver": "mumps"})
        # if (self.dim==3):
        #     dolfin.solve(a == L, w, [], solver_parameters={"linear_solver": "cg"}) 
        # else:
        #     dolfin.solve(a == L, w, [], solver_parameters={"linear_solver": "lu"}) # MMT20230616: "cg" solver doesn't work when material Poisson ratio nu=0.499, and "lu" doesn't work for 3D geometries

        V_0 = vol
        V_s0 = self.mesh_V0
        Phi_s0 = V_s0/V_0
        v_s = dolfin.assemble((1 + dolfin.tr(self.eps(v))) * self.dV)
        Phi_s = v_s/V_0

        kappa_tilde = Phi_s0**2 * p_f/(Phi_s0 - Phi_s)/2

        return kappa_tilde
