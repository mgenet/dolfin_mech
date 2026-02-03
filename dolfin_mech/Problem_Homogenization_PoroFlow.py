

import dolfin
import numpy
import sys
import os
from pathlib import Path
local_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(local_path))
import dolfin_mech as dmech
import myPythonLibrary as mypy
#############################################################################

class MicroPoroHomogenizationProblem:

    def __init__(self,
        dim,
        skel_params,      # {"E":..., "nu":...} or {"lambda":..., "mu":...}
        bulk_params,      # {"M":..., "Phi_s0":...}
        flow_params,      # {"rho_l":..., "k_l":..., "pl_bar":...}
        mesh_params=None,
        mesh=None,
        vol=None,
        bbox=None,
        vertices=None):

        # ---------------- mesh ----------------
        # if mesh is None:
        #     if mesh_params is None:
        #         raise ValueError("Provide either mesh or mesh_params.")
        #     mesh = dmech.run_HollowBox_Mesh(params=mesh_params)

        mesh = dolfin.Mesh()
        with dolfin.XDMFFile("./mesh/voronoi_2D_RVE.xdmf") as infile:
            infile.read(mesh)

        self.mesh = mesh
        self.dim = dim
        if self.dim == 2:
            self.n_Voigt = 3
        elif self.dim == 3:
            self.n_Voigt = 6
        else:
            raise ValueError("dim must be 2 or 3")

        # ---------------- measures ----------------
        self.dV = dolfin.Measure("dx", domain=self.mesh)
        self.mesh_V0 = dolfin.assemble(dolfin.Constant(1.0) * self.dV)

        # ---------------- bbox / vertices / vol (auto if not provided) ----------------
        coord = self.mesh.coordinates()
        xmin, xmax = coord[:, 0].min(), coord[:, 0].max()
        ymin, ymax = coord[:, 1].min(), coord[:, 1].max()
        if self.dim == 2:
            bbox_auto = [xmin, xmax, ymin, ymax]
            vertices_auto = numpy.array([[xmin, ymin],
                                         [xmax, ymin],
                                         [xmax, ymax],
                                         [xmin, ymax]])
            vol_auto = (xmax - xmin) * (ymax - ymin)
        else:
            zmin, zmax = coord[:, 2].min(), coord[:, 2].max()
            bbox_auto = [xmin, xmax, ymin, ymax, zmin, zmax]
            # NOTE: vertices for 3D depends on how your PeriodicSubDomain expects them.
            # If your dmech.PeriodicSubDomain can work without vertices in 3D, keep None.
            vertices_auto = vertices  # leave as provided
            vol_auto = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)

        self.bbox = bbox if (bbox is not None) else bbox_auto
        self.vertices = vertices if (vertices is not None) else vertices_auto
        self.vol = float(vol) if (vol is not None) else float(vol_auto)

        # ============================================================
        # Skeleton (linearized W_skel): either (E, nu) or (lambda, mu)
        # ============================================================
        if ("lambda" in skel_params) and ("mu" in skel_params):
            self.lmbda_s = dolfin.Constant(float(skel_params["lambda"]))
            self.mu_s    = dolfin.Constant(float(skel_params["mu"]))
            self.E_s  = skel_params.get("E", None)
            self.nu_s = skel_params.get("nu", None)

        elif ("E" in skel_params) and ("nu" in skel_params):
            self.E_s  = float(skel_params["E"])
            self.nu_s = float(skel_params["nu"])
            self.lmbda_s = dolfin.Constant(self.E_s*self.nu_s/(1+self.nu_s)/(1-2*self.nu_s))
            self.mu_s    = dolfin.Constant(self.E_s/2/(1+self.nu_s))
        else:
            raise ValueError("skel_params must contain either (E, nu) or (lambda, mu)")

        # ============================================================
        # Bulk/storage (linearized W_bulk)
        # ============================================================
        if ("M" not in bulk_params) or ("Phi_s0" not in bulk_params):
            raise ValueError("bulk_params must contain 'M' and 'Phi_s0'")

        self.M = dolfin.Constant(float(bulk_params["M"]))
        self.Phi_s0 = dolfin.Constant(float(bulk_params["Phi_s0"]))

        # ============================================================
        # Flow parameters for Darcy homogenization
        # ============================================================
        self.rho_l = dolfin.Constant(float(flow_params.get("rho_l", 1.0)))
        self.pl_bar = dolfin.Constant(float(flow_params.get("pl_bar", 0.0)))

        if "k_l" not in flow_params:
            raise ValueError("flow_params must contain 'k_l' (scalar or tensor)")

        k0 = flow_params["k_l"]
        # Normalize to a dim×dim UFL tensor
        if isinstance(k0, (int, float)):
            self.k_l0 = dolfin.Constant(float(k0)) * dolfin.Identity(self.dim)
        else:
            # could already be dolfin.Constant(((..,..),..)) or numpy array
            self.k_l0 = k0


    # ---------- geometry / spaces ----------
    def build_space_elasticity(self, degree_u=2):
        """
        Build periodic mixed space for elasticity cell problems:
            W_u = [V_u, R]
        where V_u is vector CG space for periodic displacement fluctuation u_hat,
        and R is a global Lagrange multiplier used to remove rigid-body/mean modes.
        """
        Ve = dolfin.VectorElement("CG", self.mesh.ufl_cell(), degree_u)
        Re = dolfin.VectorElement("R",  self.mesh.ufl_cell(), 0)

        pbc = dmech.PeriodicSubDomain(self.dim, self.bbox, self.vertices)

        W_u = dolfin.FunctionSpace(
            self.mesh,
            dolfin.MixedElement([Ve, Re]),
            constrained_domain=pbc)

        return W_u


    def build_space_pressure(self, degree_p=1):
        """
        Build periodic mixed space for Darcy pressure corrector cell problems:
            W_p = [Q, R]
        where Q is scalar CG space for periodic pressure fluctuation p_hat,
        and R is a global Lagrange multiplier enforcing zero-mean pressure:
            ∫ p_hat dV = 0
        """
        Qe = dolfin.FiniteElement("CG", self.mesh.ufl_cell(), degree_p)
        Re = dolfin.FiniteElement("R",  self.mesh.ufl_cell(), 0)

        pbc = dmech.PeriodicSubDomain(self.dim, self.bbox, self.vertices)

        W_p = dolfin.FunctionSpace(
            self.mesh,
            dolfin.MixedElement([Qe, Re]),
            constrained_domain=pbc)

        return W_p


    # ---------- linear kinematics ----------
    def eps(self, v):
        """Small-strain tensor: eps(v) = sym(grad(v))."""
        return dolfin.sym(dolfin.grad(v))
    
    def sigma_skel(self, u_hat, E_macro):
        """
        Linearized skeleton stress (small strain):
            Sigma = lambda tr(E) I + 2 mu E
        where E = E_macro + eps(u_hat)
        """
        E = E_macro + self.eps(u_hat)
        return self.lmbda_s * dolfin.tr(E) * dolfin.Identity(self.dim) + 2.0 * self.mu_s * E
    

    def solve_cell_elasticity(self, E_macro, degree_u=2, linear_solver="mumps"):
        """
        Solve periodic elasticity cell problem for a given macro strain E_macro.
        Unknowns: (u_hat, lambda_vec) in W_u = [V_u, R^dim].
        Constraint via Lagrange multiplier removes rigid/mean mode:
            ∫ u_hat dV = 0
        """
        W_u = self.build_space_elasticity(degree_u=degree_u)

        v_test, lam_test = dolfin.TestFunctions(W_u)
        u_tria, lam_tria = dolfin.TrialFunctions(W_u)

        # weak form: ∫ Sigma(u_tria,E_macro) : eps(v_test) dV = 0
        F = dolfin.inner(self.sigma_skel(u_tria, E_macro), self.eps(v_test)) * self.dV
        a, b = dolfin.lhs(F), dolfin.rhs(F)

        # zero-mean constraint for u_hat
        a += dolfin.inner(lam_test, u_tria) * self.dV
        a += dolfin.inner(lam_tria, v_test) * self.dV

        w = dolfin.Function(W_u)
        dolfin.solve(a == b, w, solver_parameters={"linear_solver": linear_solver})

        u_hat, _ = w.split(deepcopy=True)
        return u_hat
    
    def compute_C_hom(self, degree_u=2, linear_solver="mumps"):
        """
        Compute homogenized stiffness C_hom (Voigt form) by strain-controlled cell problems.
        Returns:
            C_hom: numpy (n_Voigt, n_Voigt)
        """
        C_hom = numpy.zeros((self.n_Voigt, self.n_Voigt), dtype=float)

        for j in range(self.n_Voigt):
            E_macro = dolfin.Constant(self.get_macro_strain(j))

            # solve cell for u_hat^(j)
            u_hat = self.solve_cell_elasticity(E_macro, degree_u=degree_u, linear_solver=linear_solver)

            # compute averaged stress in Voigt components
            Sigma = self.sigma_skel(u_hat, E_macro)
            Sigma_V = self.stress2Voigt(Sigma)

            for k in range(self.n_Voigt):
                C_hom[j, k] = dolfin.assemble(Sigma_V[k] * self.dV) / self.vol

        return C_hom
    
    def get_macro_strain(self, i):
        """
        Return the i-th unit macro strain tensor in Voigt basis.

        Voigt ordering (consistent with your Voigt2strain):
        2D: [xx, yy, xy]
        3D: [xx, yy, zz, yz, xz, xy]

        The Voigt2strain() function converts engineering shear components
        by dividing by 2 in the off-diagonals.
        """
        if self.dim == 2:
            Eps_Voigt = numpy.zeros(3)
        elif self.dim == 3:
            Eps_Voigt = numpy.zeros(6)
        else:
            raise ValueError("dim must be 2 or 3")

        Eps_Voigt[i] = 1.0
        return self.Voigt2strain(Eps_Voigt)
    
    def Voigt2strain(self, s):
        if self.dim == 2:
            return numpy.array([[s[0],    s[2]/2.0],
                                [s[2]/2.0, s[1]   ]])
        elif self.dim == 3:
            return numpy.array([[s[0],     s[5]/2.0, s[4]/2.0],
                                [s[5]/2.0, s[1],     s[3]/2.0],
                                [s[4]/2.0, s[3]/2.0, s[2]    ]])
        else:
            raise ValueError("dim must be 2 or 3")
        
    def stress2Voigt(self, S):
        """
        Map a 2nd-order stress tensor to Voigt vector.

        Voigt ordering:
        2D: [S_xx, S_yy, S_xy]
        3D: [S_xx, S_yy, S_zz, S_yz, S_xz, S_xy]
        """
        if self.dim == 2:
            return dolfin.as_vector([S[0, 0], S[1, 1], S[0, 1]])
        elif self.dim == 3:
            return dolfin.as_vector([S[0, 0], S[1, 1], S[2, 2],
                                    S[1, 2], S[0, 2], S[0, 1]])
        else:
            raise ValueError("dim must be 2 or 3")

    def extract_isotropic_lame(self, C_hom, tol=1e-6):
        """
        Extract isotropic Lamé parameters (lambda_hom, mu_hom) from a homogenized
        stiffness matrix in Voigt form.

        Assumes Voigt ordering consistent with:
        2D: [xx, yy, xy]
        3D: [xx, yy, zz, yz, xz, xy]

        For an isotropic material:
        lambda ≈ C_xy coupling terms (e.g., C01, and also C02/C12 in 3D)
        mu     ≈ shear diagonal term (2D: C22, 3D: C33=C44=C55 ideally)
        """
        C = numpy.array(C_hom, dtype=float)

        if self.dim == 2:
            # isotropic 2D (with your Voigt convention)
            lam = C[0, 1]
            mu  = C[2, 2]

            # optional consistency checks
            # C00 should be lam + 2mu, C11 should be lam + 2mu
            if abs(C[0, 0] - (lam + 2.0*mu)) > tol*max(1.0, abs(C[0,0])):
                pass
            if abs(C[1, 1] - (lam + 2.0*mu)) > tol*max(1.0, abs(C[1,1])):
                pass

            return float(lam), float(mu)

        elif self.dim == 3:
            # In 3D isotropy:
            # lambda = C01 = C02 = C12
            lam_candidates = [C[0, 1], C[0, 2], C[1, 2]]
            lam = float(sum(lam_candidates) / len(lam_candidates))

            # mu = C33 = C44 = C55 (shear terms yz, xz, xy)
            mu_candidates = [C[3, 3], C[4, 4], C[5, 5]]
            mu = float(sum(mu_candidates) / len(mu_candidates))

            return lam, mu

        else:
            raise ValueError("dim must be 2 or 3")


        

    def M_from_wbulk(kappa, Phis0, scaling="no"):
        if scaling == "no":
            return float(kappa) / (float(Phis0)**2)
        elif scaling == "linear":
            return float(kappa) / float(Phis0)
        else:
            raise ValueError('scaling must be "no" or "linear"')

    def Phi_s_from_pl(self, p_l):
        """
        Closure for solid fraction (linearized):
            phi := δPhi_s
            p_l + M * phi = 0  ->  Phi_s = Phi_s0 + phi = Phi_s0 - p_l/M
        """
        return self.Phi_s0 - p_l / self.M
    

    def solve_cell_pressure_corrector(self, i, degree_p=1, linear_solver="mumps"):
        """
        Periodic Darcy corrector cell problem for imposed macro gradient grad(p_bar)=e_i.
        Unknowns: (p_hat, eta) in W_p = [Q, R]
        Constraint: ∫ p_hat dV = 0
        """

        W_p = self.build_space_pressure(degree_p=degree_p)
        q_test, eta_test = dolfin.TestFunctions(W_p)
        p_tria, eta_tria = dolfin.TrialFunctions(W_p)

        e = numpy.zeros(self.dim); e[i] = 1.0
        grad_p_bar = dolfin.Constant(e)

        # Weak form: ∫ k0 (grad p_hat + grad_p_bar) · grad(q) dV = 0
        F = dolfin.inner(self.k_l0 * (dolfin.grad(p_tria) + grad_p_bar), dolfin.grad(q_test)) * self.dV
        a, b = dolfin.lhs(F), dolfin.rhs(F)

        # zero-mean constraint for p_hat
        a += eta_test * p_tria * self.dV
        a += eta_tria * q_test * self.dV

        w = dolfin.Function(W_p)
        dolfin.solve(a == b, w, solver_parameters={"linear_solver": linear_solver})

        p_hat, _ = w.split(deepcopy=True)
        return p_hat
    
    def compute_k_hom(self, degree_p=1, linear_solver="mumps"):
        """
        Compute homogenized permeability-like tensor k_hom such that:
            <q> = - k_hom * grad(p_macro)
        where micro flux is q = -k_l0*(grad(p_hat)+grad_p_bar).
        Returns:
            k_hom: numpy (dim, dim)
        """
        k_hom = numpy.zeros((self.dim, self.dim), dtype=float)

        for i in range(self.dim):
            p_hat = self.solve_cell_pressure_corrector(i, degree_p=degree_p, linear_solver=linear_solver)

            e = numpy.zeros(self.dim); e[i] = 1.0
            grad_p_bar = dolfin.Constant(e)

            q = - self.k_l0 * (grad_p_bar + dolfin.grad(p_hat))

            q_avg = numpy.zeros(self.dim, dtype=float)
            for a in range(self.dim):
                q_avg[a] = dolfin.assemble(q[a] * self.dV) / self.vol

            # column i of k_hom equals -<q>
            for a in range(self.dim):
                k_hom[a, i] = -q_avg[a]

        return k_hom
    
    

def M_from_wbulk(kappa, Phis0, scaling="no"):
    if scaling == "no":
        return float(kappa) / (float(Phis0)**2)
    elif scaling == "linear":
        return float(kappa) / float(Phis0)
    else:
        raise ValueError('scaling must be "no" or "linear"')

def Phi_s_from_pl(self, p_l):
    """
    Closure for solid fraction (linearized):
        phi := δPhi_s
        p_l + M * phi = 0  ->  Phi_s = Phi_s0 + phi = Phi_s0 - p_l/M
    """
    return self.Phi_s0 - p_l / self.M

def fit_lame_from_wskel(dim, material_parameters, eps0=1e-7):

    if dim == 2:
        mesh = dolfin.UnitSquareMesh(1, 1)
    elif dim == 3:
        mesh = dolfin.UnitCubeMesh(1, 1, 1)
    else:
        raise ValueError("dim must be 2 or 3")

    V = dolfin.VectorFunctionSpace(mesh, "CG", 1)
    X = dolfin.SpatialCoordinate(mesh)
    dx = dolfin.Measure("dx", domain=mesh)
    vol = dolfin.assemble(dolfin.Constant(1.0) * dx)

    def set_affine_u(A):
        A = numpy.asarray(A, dtype=float)
        u_ufl = dolfin.as_vector([
            sum(dolfin.Constant(A[i, j]) * X[j] for j in range(dim))
            for i in range(dim)
        ])
        return dolfin.project(u_ufl, V)

    def avg_S(u):
        kin = dmech.Kinematics(U=u) 
        solid = dmech.WskelLungElasticMaterial(kinematics=kin, parameters=dict(material_parameters))
        S = solid.Sigma
        S_avg = numpy.zeros((dim, dim), dtype=float)
        for i in range(dim):
            for j in range(dim):
                S_avg[i, j] = dolfin.assemble(S[i, j] * dx) / vol
        return S_avg

    e = float(eps0)

    # symmetric shear -> mu
    g = e
    A_sh = numpy.zeros((dim, dim))
    A_sh[0, 1] = 0.5 * g
    A_sh[1, 0] = 0.5 * g
    S_sh = avg_S(set_affine_u(A_sh))
    mu = S_sh[0, 1] / g

    # uniaxial strain -> lambda
    A_u = numpy.zeros((dim, dim))
    A_u[0, 0] = e
    S_u = avg_S(set_affine_u(A_u))
    lam = S_u[1, 1] / e

    # checks
    Sxx_pred = (lam + 2.0 * mu) * e
    Sxx_err = float(S_u[0, 0] - Sxx_pred)
    Sxy_err = float(S_sh[0, 1] - mu * g)

    return float(lam), float(mu), {
        "S_uniax_strain": S_u,
        "S_sym_shear": S_sh,
        "Sxx_err": Sxx_err,
        "Sxy_err": Sxy_err,
        "eps0": e,
    }




res_folder = sys.argv[0][:-3]
test = mypy.Test(
    res_folder=res_folder,
    perform_tests=0,
    stop_at_failure=1,
    clean_after_tests=0,
    tester_numpy_tolerance=1e-2)

mesh_params={"dim":2, "xmin":0., "ymin":0., "zmin":0., "xmax":1., "ymax":1., "zmax":1., "xshift":-0.5, "yshift":-0.5, "zshift":-0.5, "r0":0.2, "l":0.05, "mesh_filebasename":res_folder+"/"+"mesh"}
mat_params = {
    "alpha": 0.16,
    "gamma": 0.5,
    "c1": 0.2,
    "c2": 0.4,
    "kappa": 1e2,
    "eta": 1e-5,
}
kappa =1 
Phis0 = 0.3
dim = 2
# --- mesh ---
# mesh = dmech.run_HollowBox_Mesh(params=mesh_params)
# dim = mesh.geometry().dim()

# --- skeleton: already fitted from wskel ---
lam, mu, info = fit_lame_from_wskel(dim, mat_params, eps0=1e-7)
skel_params = {"lambda": lam, "mu": mu}

# --- bulk: already analytic ---
M = M_from_wbulk(kappa, Phis0, scaling="no")   # or "linear"
bulk_params = {"M": M, "Phi_s0": Phis0}

# --- flow ---
#K reference value: 1.0 for test
flow_params = {
    "rho_l": 1.0,
    "k_l": dolfin.Constant(((1, 0.0),
                            (0.0, 1))),
    "pl_bar": 0.3
}

# --- periodic cell geometry (if your init auto-computes bbox/vertices/vol, you can omit them) ---
hp = MicroPoroHomogenizationProblem(
    dim=dim,
    mesh=None,              # if your init supports mesh input
    skel_params=skel_params,
    bulk_params=bulk_params,
    flow_params=flow_params,
    mesh_params=mesh_params, # only if your init builds mesh internally
    vol=None, bbox=None, vertices=None
)

C_hom = hp.compute_C_hom(degree_u=2, linear_solver="mumps")
print("C_hom =\n", C_hom)

lam_hom, mu_hom = hp.extract_isotropic_lame(C_hom)
print("lambda_hom =", lam_hom)
print("mu_hom     =", mu_hom)

k_hom = hp.compute_k_hom(degree_p=1, linear_solver="mumps")
print("k_hom =\n", k_hom)

k_hom_rho = float(flow_params["rho_l"]) * k_hom
print("rho*k_hom =\n", k_hom_rho)

#Phi_s_expr = hp.Phi_s_from_pl(p_l)   # p_l 可以是 dolfin Function

