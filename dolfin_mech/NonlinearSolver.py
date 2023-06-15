#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import decimal
import dolfin
import glob
import math
import numpy
import os
import petsc4py
import petsc4py.PETSc
import sys
import time

import myPythonLibrary as mypy

import dolfin_mech as dmech

################################################################################

class NonlinearSolver():



    def __init__(self,
            problem,
            parameters,
            relax_type="constant",
            relax_parameters={},
            print_out=True,
            write_iter=False):

        self.problem = problem

        self.default_linear_solver_type = "petsc"
        #self.default_linear_solver_type = "dolfin"

        self.linear_solver_type = parameters.get("linear_solver_type", self.default_linear_solver_type)

        if (self.linear_solver_type == "petsc"):

            self.res_vec = dolfin.PETScVector()
            self.jac_mat = dolfin.PETScMatrix()

            self.linear_solver = dolfin.PETScKrylovSolver()

            self.default_linear_solver_name = "mumps"

            self.linear_solver_name = parameters.get("linear_solver_name", self.default_linear_solver_name)

            if (self.linear_solver_name == "mumps"):
                if (int(dolfin.__version__.split('.')[0]) >= 2018):
                    options = petsc4py.PETSc.Options()
                    options["ksp_type"] = "preonly"
                    options["pc_type"] = "lu"
                    options["pc_factor_mat_solver_type"] = "mumps"
                    options["mat_mumps_icntl_33"] = 0
                else:
                    options = dolfin.PETScOptions()
                    options.set("ksp_type", "preonly")
                    options.set("pc_type", "lu")
                    options.set("pc_factor_mat_solver_package", "mumps")
                    options.set("mat_mumps_icntl_33", 0)

            self.linear_solver.ksp().setFromOptions()
            self.linear_solver.ksp().setOperators(A=self.jac_mat.mat())

        elif (self.linear_solver_type == "dolfin"):

            self.res_vec = dolfin.Vector()
            self.jac_mat = dolfin.Matrix()

            # self.default_linear_solver_name = "default"
            self.default_linear_solver_name = "mumps"
            # self.default_linear_solver_name = "petsc"
            # self.default_linear_solver_name = "superlu"
            # self.default_linear_solver_name = "umfpack"

            self.linear_solver_name = parameters.get("linear_solver_name", self.default_linear_solver_name)

            self.linear_solver = dolfin.LUSolver(
                self.jac_mat,
                self.linear_solver_name)
            # self.linear_solver.parameters['report']               = bool(0)
            # self.linear_solver.parameters['reuse_factorization']  = bool(0)
            # self.linear_solver.parameters['same_nonzero_pattern'] = bool(1)
            # self.linear_solver.parameters['symmetric']            = bool(1)
            # self.linear_solver.parameters['verbose']              = bool(1)

        if (relax_type == "constant"):
            self.compute_relax = self.compute_relax_constant
            self.relax_val = relax_parameters.get("relax", 1.)
        elif (relax_type == "aitken"):
            self.compute_relax = self.compute_relax_aitken
        elif (relax_type == "gss"):
            self.compute_relax = self.compute_relax_gss
            self.relax_n_iter_max = relax_parameters.get("relax_n_iter_max", 9)
        elif (relax_type == "backtracking"):
            self.compute_relax = self.compute_relax_backtracking
            self.relax_backtracking_factor = parameters["relax_backtracking_factor"] if ("relax_backtracking_factor" in parameters) and (parameters["relax_backtracking_factor"] is not None) else 2.
            self.relax_n_iter_max          = parameters["relax_n_iter_max"]          if ("relax_n_iter_max"          in parameters) and (parameters["relax_n_iter_max"]          is not None) else 8

        self.sol_tol    = parameters.get("sol_tol"   , [1e-6]*len(self.problem.subsols))
        self.n_iter_max = parameters.get("n_iter_max", 32)

        if (type(print_out) is str):
            if (print_out=="stdout"):
                self.printer_filename = None
            elif (print_out=="argv"):
                self.printer_filename = sys.argv[0][:-3]+".out"
            else:
                self.printer_filename = print_out+".out"
        else:
            self.printer_filename = None
        self.printer = mypy.Printer(
            filename=self.printer_filename,
            silent=not(print_out))

        self.write_iter = bool(write_iter)
        if (self.write_iter):
            for filename in glob.glob(sys.argv[0][:-3]+"-sol-k_step=*-k_t=*.*"):
                os.remove(filename)

            self.functions_to_write = []
            self.functions_to_write += self.problem.get_subsols_func_lst()
            self.functions_to_write += self.problem.get_subsols_func_old_lst()
            self.functions_to_write += self.problem.get_fois_func_lst()



    def solve(self,
            k_step=None,
            k_t=None,
            dt=None,
            t=None):

        # write
        if (self.write_iter):
            xdmf_file_iter = dmech.XDMFFile(
                filename=sys.argv[0][:-3]+"-sol-k_step="+str(k_step)+"-k_t="+str(k_t)+".xdmf",
                functions=self.functions_to_write)
            self.problem.update_fois()
            xdmf_file_iter.write(0.)

        self.k_iter = 0
        self.success = False
        self.printer.inc()
        while (True):
            self.k_iter += 1
            self.printer.print_var("k_iter",self.k_iter,-1)

            # linear problem
            linear_success = self.linear_solve(
                k_step=k_step,
                k_t=k_t)
            if not (linear_success):
                break
            self.compute_dsol_norm()

            # constraints update
            if (self.k_iter == 1):
                for constraint in self.constraints:
                    constraint.homogenize()

            # solution update
            self.compute_relax()
            self.update_sol()
            self.compute_sol_norm()

            # internal variables update
            for inelastic_behavior in self.problem.inelastic_behaviors_internal:
                inelastic_behavior.update_internal_variables_after_solve(
                    dt, t)

            # write
            if (self.write_iter):
                self.problem.update_fois()
                xdmf_file_iter.write(self.k_iter)

            # error
            self.compute_sol_err()

            # exit test
            self.exit_test()

            if (self.success):
                self.printer.print_str("Nonlinear solver converged…")
                break

            if (self.k_iter == self.n_iter_max):
                self.printer.print_str("Warning! Nonlinear solver failed to converge!")
                break

        self.printer.dec()

        # write
        if (self.write_iter):
            xdmf_file_iter.close()

        return self.success, self.k_iter



    def linear_solve(self,
            k_step=None,
            k_t=None):

        assemble_linear_system = self.assemble_linear_system()   

        if assemble_linear_system==False:
            return False 
        
        # eigen problem
        if (k_step == 1) and (k_t == 1) and (self.k_iter == 1) and (0):
            self.eigen_solve()

        # linear system: solve
        try:
            self.printer.print_str("Solve…",newline=False)
            timer = time.time()
            self.linear_solver.solve(
                self.problem.dsol_func.vector(),
                self.res_vec)
            timer = time.time() - timer
            self.printer.print_str(" "+str(timer)+" s",tab=False)
            #self.printer.print_var("dsol_func",self.problem.dsol_func.vector().get_local())
        except:
            self.printer.print_str("Warning! Linear solver failed!",tab=False)
            return False

        if not (numpy.isfinite(self.problem.dsol_func.vector()).all()):
            # self.problem.dsol_func.vector().zero()

            self.printer.print_str("Warning! Solution increment is NaN!")
            return False

        if (len(self.problem.subsols) > 1):
            dolfin.assign(
                self.problem.get_subsols_dfunc_lst(),
                self.problem.dsol_func)
            # for subsol_name,subsol in self.problem.subsols.items():
            #     self.printer.print_var("d"+subsol_name+"_func",subsol.dfunc.vector().get_local())

        if (0):
            rinfo12 = self.linear_solver.ksp().getPC().getFactorMatrix().getMumpsRinfog(12)
            #self.printer.print_sci("rinfo12",rinfo12)
            rinfo12 = decimal.Decimal(rinfo12)
            #self.printer.print_sci("rinfo12",rinfo12)
            infog34 = self.linear_solver.ksp().getPC().getFactorMatrix().getMumpsInfog(34)
            #self.printer.print_sci("infog34",infog34)
            infog34 = decimal.Decimal(infog34)
            #self.printer.print_sci("infog34",infog34)
            self.jac_det = rinfo12*(decimal.Decimal(2.)**infog34)
            self.printer.print_sci("jac_det",self.jac_det)

        return True

    def assemble_linear_system(self):

        # res_old
        if (self.k_iter > 1):
            if (hasattr(self, "res_old_vec")):
                self.res_old_vec[:] = self.res_vec[:]
            else:
                self.res_old_vec = self.res_vec.copy()
            self.res_old_norm = self.res_norm

        # linear system: Assembly
        if any([(operator.measure.integral_type() == "vertex") for operator in self.problem.operators]): # MG20190513: Cannot use point integral within assemble_system
            self.printer.print_str("Assembly (without vertex integrals)…",newline=False)
            timer = time.time()
            dolfin.assemble_system(
                self.problem.jac_form,
               -self.problem.res_form,
                bcs=[constraint.bc for constraint in self.constraints],
                A_tensor=self.jac_mat,
                b_tensor=self.res_vec,
                add_values=False,
                finalize_tensor=False,
                form_compiler_parameters=self.problem.form_compiler_parameters)
            timer = time.time() - timer
            self.printer.print_str(" "+str(timer)+" s",tab=False)
            # self.printer.print_var("res_vec",self.res_vec.get_local())
            # self.printer.print_var("jac_mat",self.jac_mat.array())

            for operator in self.problem.operators:
                if (operator.measure.integral_type() == "vertex"):
                    self.printer.print_str("Assembly (vertex integrals)…",newline=False)
                    timer = time.time()
                    dolfin.assemble( # MG20190513: However, vertex integrals only work if solution only has dofs on vertices…
                       -operator.res_form,
                        tensor=self.res_vec,
                        add_values=True,
                        finalize_tensor=True,
                        form_compiler_parameters=self.problem.form_compiler_parameters)
                    operator.jac_form = dolfin.derivative(
                        operator.res_form,
                        self.problem.sol_func,
                        self.problem.dsol_tria)
                    dolfin.assemble(
                        operator.jac_form,
                        tensor=self.jac_mat,
                        add_values=True,
                        finalize_tensor=True,
                        form_compiler_parameters=self.problem.form_compiler_parameters)
                    timer = time.time() - timer
                    self.printer.print_str(" "+str(timer)+" s",tab=False)
                    # self.printer.print_var("res_vec",self.res_vec.get_local())
                    # self.printer.print_var("jac_mat",self.jac_mat.array())
        else:
            self.printer.print_str("Assembly…",newline=False)
            timer = time.time()
            dolfin.assemble_system(
                self.problem.jac_form,
               -self.problem.res_form,
                bcs=[constraint.bc for constraint in self.constraints],
                A_tensor=self.jac_mat,
                b_tensor=self.res_vec,
                add_values=False,
                finalize_tensor=True,
                form_compiler_parameters=self.problem.form_compiler_parameters)
            timer = time.time() - timer
            self.printer.print_str(" "+str(timer)+" s",tab=False)
            # self.printer.print_var("res_vec",self.res_vec.get_local())
            # self.printer.print_var("jac_mat",self.jac_mat.array())

        if not (numpy.isfinite(self.res_vec).all()):
            self.printer.print_str("Warning! Residual is NaN!")
            return False

        # res_norm
        self.res_norm = self.res_vec.norm("l2")
        self.printer.print_sci("res_norm",self.res_norm)

        if (self.res_norm > 1e9):
            self.printer.print_str("Warning! Residual is too large!")
            return False

        # res_err
        if (self.k_iter == 1):
            self.res_norm0 = self.res_norm
        else:
            self.res_err = dmech.compute_error(
                val=self.res_norm,
                ref=self.res_norm0)
            self.printer.print_sci("res_err",self.res_err)

            if (self.res_err > 1e3):
                self.printer.print_str("Warning! Residual is increasing too much!")
                return False

        # dres
        if (self.k_iter > 1):
            if (hasattr(self, "dres_vec")):
                self.dres_vec[:] = self.res_vec[:] - self.res_old_vec[:]
            else:
                self.dres_vec = self.res_vec - self.res_old_vec
            self.dres_norm = self.dres_vec.norm("l2")
            self.printer.print_sci("dres_norm",self.dres_norm)

        # res_err_rel
        if (self.k_iter > 1):
            self.res_err_rel = dmech.compute_error(
                val=self.dres_norm,
                ref=self.res_old_norm)
            self.printer.print_sci("res_err_rel",self.res_err_rel)


    def eigen_solve():

        jac_eigensolver = dolfin.SLEPcEigenSolver(
            dolfin.as_backend_type(self.jac_mat))

        # jac_eigensolver.parameters["problem_type"] = "non_hermitian"
        jac_eigensolver.parameters["problem_type"] = "hermitian"

        jac_eigensolver.parameters["solver"] = "krylov-schur"
        # jac_eigensolver.parameters["solver"] = "power"
        # jac_eigensolver.parameters["solver"] = "subspace"
        # jac_eigensolver.parameters["solver"] = "arnoldi"
        # jac_eigensolver.parameters["solver"] = "lanczos"

        # jac_eigensolver.parameters["tolerance"] = 1e-1
        # jac_eigensolver.parameters["maximum_iterations"] = 100

        jac_eigensolver.parameters["verbose"] = True

        mode_func = dolfin.Function(self.problem.sol_fs)

        n_modes = 10
        spectrums  = []
        # spectrums += ["largest"]
        spectrums += ["smallest"]

        for spectrum in spectrums:
            jac_eigensolver.parameters["spectrum"] = spectrum+" magnitude"

            self.printer.print_str("Eigenproblem solve…",newline=False)
            timer = time.time()
            jac_eigensolver.solve(n_modes)
            timer = time.time() - timer
            self.printer.print_str(" "+str(timer)+" s",tab=False,newline=False)

            n_converged = jac_eigensolver.get_number_converged()
            self.printer.print_str(" ("+str(n_converged)+" converged modes)",tab=False)

            xdmf_file_modes = dmech.XDMFFile(
                filename=sys.argv[0][:-3]+"-eigenmodes-"+spectrum+".xdmf",
                functions=[mode_func])
            for k_mode in range(n_converged):
                # print(k_mode+1)
                val_r, val_c, vec_r, vec_c = jac_eigensolver.get_eigenpair(k_mode)
                # print(val_r)
                mode_func.vector()[:] = vec_r[:]
                xdmf_file_modes.write(k_mode)
            xdmf_file_modes.close()



    def compute_dsol_norm(self):

        self.dsubsol_norm_lst = [subsol.dfunc.vector().norm("l2") for subsol in self.problem.subsols.values()]
        for (k_subsol,subsol) in enumerate(self.problem.subsols.values()):
            self.printer.print_sci("d"+subsol.name+"_norm",self.dsubsol_norm_lst[k_subsol])



    def compute_relax_constant(self):

        if (self.k_iter == 1):
            self.relax = 1. # MG20180505: Otherwise Dirichlet boundary conditions are not correctly enforced
        else:
            self.relax = self.relax_val
            self.printer.print_sci("relax",self.relax)



    def compute_relax_aitken(self):

        if (self.k_iter == 1):
            self.relax = 1. # MG20180505: Otherwise Dirichlet boundary conditions are not correctly enforced
        else:
            self.relax *= (-1.) * self.res_old_vec.inner(self.dres_vec) / self.dres_norm**2
        self.printer.print_sci("relax",self.relax)



    def compute_relax_gss(self):

        if (self.k_iter == 1):
            self.relax = 1. # MG20180505: Otherwise Dirichlet boundary conditions are not correctly enforced
        else:
            phi = (1+math.sqrt(5))/2
            a = (1-phi)/(2-phi)
            b = 1./(2-phi)
            need_update_c = True
            need_update_d = True
            cur = 0.
            relax_list = []
            relax_vals = []
            self.printer.inc()
            relax_k = 0
            while (True):
                self.printer.print_var("relax_k",relax_k)
                self.printer.print_sci("a",a)
                self.printer.print_sci("b",b)
                if (need_update_c):
                    c = b - (b - a) / phi
                    relax_list.append(c)
                    self.printer.print_sci("c",c)
                    self.problem.sol_func.vector().axpy(
                        c-cur,
                        self.problem.dsol_func.vector())
                    if (len(self.problem.subsols) > 1):
                        dolfin.assign(
                            self.problem.get_subsols_func_lst(),
                            self.problem.sol_func)
                    cur = c
                    relax_fc  = dolfin.assemble(
                        self.problem.Pi_expr,
                        form_compiler_parameters=self.problem.form_compiler_parameters)
                    #self.printer.print_sci("relax_fc",relax_fc)
                    if (numpy.isnan(relax_fc)):
                        relax_fc = float("+inf")
                        #self.printer.print_sci("relax_fc",relax_fc)
                    self.printer.print_sci("relax_fc",relax_fc)
                    relax_vals.append(relax_fc)
                    #self.printer.print_var("relax_list",relax_list)
                    #self.printer.print_var("relax_vals",relax_vals)
                if (need_update_d):
                    d = a + (b - a) / phi
                    relax_list.append(d)
                    self.printer.print_sci("d",d)
                    self.problem.sol_func.vector().axpy(
                        d-cur,
                        self.problem.dsol_func.vector())
                    if (len(self.problem.subsols) > 1):
                        dolfin.assign(
                            self.problem.get_subsols_func_lst(),
                            self.problem.sol_func)
                    cur = d
                    relax_fd  = dolfin.assemble(
                        self.problem.Pi_expr,
                        form_compiler_parameters=self.problem.form_compiler_parameters)
                    if (numpy.isnan(relax_fd)):
                        relax_fd = float("+inf")
                        #self.printer.print_sci("relax_fd",relax_fd)
                    self.printer.print_sci("relax_fd",relax_fd)
                    relax_vals.append(relax_fd)
                    #self.printer.print_var("relax_list",relax_list)
                    #self.printer.print_var("relax_vals",relax_vals)
                #if ((relax_fc < 1e-12) and (relax_fd < 1e-12)):
                    #break
                if (relax_fc < relax_fd):
                    b = d
                    d = c
                    relax_fd = relax_fc
                    need_update_c = True
                    need_update_d = False
                elif (relax_fc >= relax_fd):
                    a = c
                    c = d
                    relax_fc = relax_fd
                    need_update_c = False
                    need_update_d = True
                else: assert(0)
                if (relax_k >= self.relax_n_iter_max):
                #if (relax_k >= 9) and (numpy.argmin(relax_vals) > 0):
                    break
                relax_k += 1
            self.printer.dec()
            self.problem.sol_func.vector().axpy(
                -cur,
                self.problem.dsol_func.vector())
            if (len(self.problem.subsols) > 1):
                dolfin.assign(
                    self.problem.get_subsols_func_lst(),
                    self.problem.sol_func)
            #self.printer.print_var("relax_vals",relax_vals)

            self.relax = relax_list[numpy.argmin(relax_vals)]
            self.printer.print_sci("relax",self.relax)
            if (self.relax == 0.):
                self.printer.print_str("Warning! Optimal relaxation is null…")

    def compute_relax_backtracking(self):
        k_relax = 1
        self.printer.inc()
        while (True):
            relax = 1./self.relax_backtracking_factor**(k_relax-1)
            self.problem.sol_func.vector().axpy(relax, self.problem.dsol_func.vector())
            self.assemble_linear_system()
            res_is_finite = numpy.isfinite(self.res_vec).all()
            # print("numpy.isfinite(self.res_vec).all()", res_is_finite)
            self.problem.sol_func.vector().axpy(-relax, self.problem.dsol_func.vector())
            if (res_is_finite):
                self.relax = relax
                break
            if (k_relax == self.relax_n_iter_max):
                self.relax = 0.
                self.printer.print_str("Warning! Optimal relaxation is null…")
                break
            k_relax += 1
        self.printer.dec()


    def update_sol(self):

        # for constraint in self.problem.constraints+self.problem.steps[k_step-1].constraints:
        #     print(constraint.bc.get_boundary_values())
        self.problem.sol_func.vector().axpy(
            self.relax,
            self.problem.dsol_func.vector())
        # for constraint in self.problem.constraints+self.problem.steps[k_step-1].constraints:
        #     print(constraint.bc.get_boundary_values())
        # self.printer.print_var("sol_func",self.problem.sol_func.vector().get_local())

        if (len(self.problem.subsols) > 1):
            dolfin.assign(
                self.problem.get_subsols_func_lst(),
                self.problem.sol_func)
            # for subsol_name,subsol in self.problem.subsols.items()):
            #     self.printer.print_var(subsol_name+"_func",subsol.func.vector().get_local())



    def compute_sol_norm(self):

        self.subsol_norm_lst = [subsol.func.vector().norm("l2") for subsol in self.problem.subsols.values()]
        self.subsol_norm_old_lst = [subsol.func_old.vector().norm("l2") for subsol in self.problem.subsols.values()]
        for (k_subsol,subsol) in enumerate(self.problem.subsols.values()):
            self.printer.print_sci(subsol.name+"_norm"    ,self.subsol_norm_lst[k_subsol]    )
            self.printer.print_sci(subsol.name+"_norm_old",self.subsol_norm_old_lst[k_subsol])



    def compute_sol_err(self):

        self.subsol_err_lst = [dmech.compute_error(
            val=self.dsubsol_norm_lst[k_subsol],
            ref=max(
                self.subsol_norm_lst[k_subsol],
                self.subsol_norm_old_lst[k_subsol])) for k_subsol in range(len(self.problem.subsols))]
        for (k_subsol,subsol) in enumerate(self.problem.subsols.values()):
            self.printer.print_sci(subsol.name+"_err",self.subsol_err_lst[k_subsol])



    def exit_test(self):

        self.success = all([self.subsol_err_lst[k_subsol]<self.sol_tol[k_subsol] for k_subsol in range(len(self.problem.subsols)) if self.sol_tol[k_subsol] is not None])
