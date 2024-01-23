#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import sys

import myPythonLibrary as mypy

import dolfin_mech as dmech

################################################################################

class TimeIntegrator():

    def __init__(self,
            problem,
            solver,
            parameters,
            print_out=True,
            print_sta=True,
            write_qois=True,
            write_qois_limited_precision=False,
            write_sol=True,
            write_vtus=False,
            write_vtus_with_preserved_connectivity=False,
            write_xmls=False):

        self.problem = problem

        self.solver = solver

        self.n_iter_for_accel = parameters.get("n_iter_for_accel",  4)
        self.n_iter_for_decel = parameters.get("n_iter_for_decel", 16)
        self.accel_coeff      = parameters.get("accel_coeff"     ,  2)
        self.decel_coeff      = parameters.get("decel_coeff"     ,  2)

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
        self.solver.printer = self.printer

        if (type(print_sta) is str):
            if (print_sta=="stdout"):
                self.table_printer_filename = None
            elif (print_sta=="argv"):
                self.table_printer_filename = sys.argv[0][:-3]+".sta"
            else:
                self.table_printer_filename = print_sta+".sta"
        else:
            self.table_printer_filename = sys.argv[0][:-3]+".sta"
        self.table_printer = mypy.TablePrinter(
            titles=["k_step", "k_t", "dt", "t", "t_step", "n_iter", "success"],
            filename=self.table_printer_filename,
            silent=not(print_sta))

        self.write_qois = bool(write_qois) and (len(self.problem.qois)>0)
        if (self.write_qois):
            self.write_qois_filebasename = write_qois if (type(write_qois) is str) else sys.argv[0][:-3]+"-qois"

            self.qoi_printer = mypy.DataPrinter(
                names=["t"]+[qoi.name for qoi in self.problem.qois],
                filename=self.write_qois_filebasename+".dat",
                limited_precision=write_qois_limited_precision)

            self.problem.update_qois(dt=1)
            self.qoi_printer.write_line([0.]+[qoi.value for qoi in self.problem.qois])

        self.problem.update_fois()
        self.write_sol = bool(write_sol)
        if (self.write_sol):
            self.write_sol_filebasename = write_sol if (type(write_sol) is str) else sys.argv[0][:-3]+"-sol"

            self.functions_to_write  = []
            self.functions_to_write += self.problem.get_subsols_func_lst()
            self.functions_to_write += self.problem.get_subsols_func_old_lst()
            self.functions_to_write += self.problem.get_fois_func_lst()

            self.xdmf_file_sol = dmech.XDMFFile(
                filename=self.write_sol_filebasename+".xdmf",
                functions=self.functions_to_write)
            self.xdmf_file_sol.write(0.)

            self.write_vtus                             = bool(write_vtus)
            self.write_vtus_with_preserved_connectivity = bool(write_vtus_with_preserved_connectivity)
            if (self.write_vtus):
                dmech.write_VTU_file(
                    filebasename=self.write_sol_filebasename,
                    function=self.problem.get_displacement_subsol().subfunc,
                    time=0,
                    preserve_connectivity=self.write_vtus_with_preserved_connectivity)

            self.write_xmls = bool(write_xmls)
            if (self.write_xmls):
                dolfin.File(self.write_sol_filebasename+"_"+str(0).zfill(3)+".xml") << self.problem.get_displacement_subsol().subfunc



    def close(self):

        self.printer.close()
        self.table_printer.close()

        if (self.write_qois):
            self.qoi_printer.close()

        if (self.write_sol):
            self.xdmf_file_sol.close()



    def integrate(self):

        k_t_tot = 0
        n_iter_tot = 0
        self.printer.inc()
        for k_step in range(1,len(self.problem.steps)+1):
            self.printer.print_var("k_step",k_step,-1)

            self.step = self.problem.steps[k_step-1]

            t = self.step.t_ini
            dt = self.step.dt_ini

            self.problem.set_variational_formulation(
                k_step=k_step-1)

            self.solver.constraints  = []
            self.solver.constraints += self.problem.constraints
            self.solver.constraints += self.step.constraints

            k_t = 0
            self.printer.inc()
            while (True):
                k_t += 1
                k_t_tot += 1
                self.printer.print_var("k_t",k_t,-1)

                if (t+dt > self.step.t_fin):
                    dt = self.step.t_fin - t
                self.printer.print_var("dt",dt)

                # self.problem.set_variational_formulation(
                #     k_step=k_step-1,
                #     dt=dt)

                t += dt
                self.printer.print_var("t",t)

                t_step = (t - self.step.t_ini)/(self.step.t_fin - self.step.t_ini)
                self.printer.print_var("t_step",t_step)

                for operator in self.step.operators:
                    operator.set_value_at_t_step(t_step)
                    operator.set_dt(dt)

                for constraint in self.step.constraints:
                    constraint.set_value_at_t_step(t_step)

                for inelastic_behavior in self.problem.inelastic_behaviors_internal:
                    inelastic_behavior.update_internal_variables_at_t(t)

                self.problem.sol_old_func.vector()[:] = self.problem.sol_func.vector()[:]
                if (len(self.problem.subsols) > 1):
                    dolfin.assign(
                        self.problem.get_subsols_func_old_lst(),
                        self.problem.sol_old_func)
                solver_success, n_iter = self.solver.solve(k_step, k_t, dt, t)

                self.table_printer.write_line([k_step, k_t, dt, t, t_step, n_iter, solver_success])

                if (solver_success):
                    n_iter_tot += n_iter

                    self.problem.update_fois()
                    if (self.write_sol):
                        self.xdmf_file_sol.write(t)

                        if (self.write_vtus):
                            dmech.write_VTU_file(
                                filebasename=self.write_sol_filebasename,
                                function=self.problem.get_displacement_subsol().subfunc,
                                time=k_t_tot,
                                preserve_connectivity=self.write_vtus_with_preserved_connectivity)

                        if (self.write_xmls):
                            dolfin.File(self.write_sol_filebasename+"_"+str(k_t_tot).zfill(3)+".xml") << self.problem.get_displacement_subsol().subfunc

                    if (self.write_qois):
                        self.problem.update_qois(dt)
                        self.qoi_printer.write_line([t]+[qoi.value for qoi in self.problem.qois])

                    if dolfin.near(t, self.step.t_fin, eps=1e-9):
                        self.success = True
                        break
                    else:
                        if (n_iter <= self.n_iter_for_accel):
                            dt *= self.accel_coeff
                            if (dt > self.step.dt_max):
                                dt = self.step.dt_max
                        elif (n_iter >= self.n_iter_for_decel):
                            dt /= self.decel_coeff
                            if (dt < self.step.dt_min):
                                dt = self.step.dt_min
                else:
                    self.problem.sol_func.vector()[:] = self.problem.sol_old_func.vector()[:]
                    if (len(self.problem.subsols) > 1):
                        dolfin.assign(
                            self.problem.get_subsols_func_lst(),
                            self.problem.sol_func)

                    for inelastic_behavior in self.problem.inelastic_behaviors_internal:
                        inelastic_behavior.restore_old_value()

                    for constraint in self.step.constraints:
                        constraint.restore_old_value()

                    k_t -= 1
                    k_t_tot -= 1
                    t -= dt

                    dt /= self.decel_coeff
                    if (dt < self.step.dt_min):
                        self.printer.print_str("Warning! Time integrator failed to move forward!")
                        self.success = False
                        break

            self.printer.dec()

            if not (self.success):
                break

        self.printer.dec()

        return self.success
