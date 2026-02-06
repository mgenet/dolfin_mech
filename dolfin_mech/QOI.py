#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2025                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech

################################################################################

class QOI():



    def __init__(self,
            name,
            expr=None,
            expr_lst=None,
            norm=1.,
            constant=0.,
            divide_by_dt=False,
            form_compiler_parameters={},
            point=None,
            update_type="assembly"):

        self.name                     = name
        self.expr                     = expr
        self.expr_lst                 = expr_lst
        self.norm                     = norm
        self.constant                 = constant
        self.divide_by_dt             = divide_by_dt
        self.form_compiler_parameters = form_compiler_parameters
        self.point                    = point

        if (update_type == "assembly"):
            self.update = self.update_assembly
        elif (update_type == "direct"):
            self.update = self.update_direct



    def update_assembly(self, dt=None, k_step=None, expr=None):

        # print(self.name)
        # print(self.expr)
        # print(self.form_compiler_parameters)
        if (self.expr is not None):
            self.value = dolfin.assemble(
                self.expr,
                form_compiler_parameters=self.form_compiler_parameters)
        else:
            if (k_step is None):
                self.value = dolfin.assemble(
                    self.expr_lst[0],
                    form_compiler_parameters=self.form_compiler_parameters)
            else:
                self.value = dolfin.assemble(
                    self.expr_lst[k_step - 1],
                    form_compiler_parameters=self.form_compiler_parameters)


        self.value += self.constant
        self.value /= self.norm

        if (self.divide_by_dt):
            assert (dt != 0),\
                "dt (="+str(dt)+") should be non zero. Aborting."
            self.value /= dt



    def update_direct(self, dt=None, k_step=None):
        """
        Docstring for MPI SAFE update_direct
        """
        # MPI safe version
        try:
            local_value = self.expr(self.point)
            found = 1.0
        except RuntimeError:
            # Point not in this rank's subdomain in MPIrun
            local_value = 0.0
            found = 0.0

        # Sum across all ranks to get the true value on all processes (can be shared)
        comm = dolfin.MPI.comm_world
        global_value = dolfin.MPI.sum(comm, local_value)
        global_found = dolfin.MPI.sum(comm, found)

        if global_found > 0:
            self.value = global_value / global_found
        else:
            # This happens if the point is truly outside the global domain
            self.value = 0.0

        self.value = dolfin.MPI.sum(dolfin.MPI.comm_world, local_value)

        self.value += self.constant
        self.value /= self.norm
        
        if (self.divide_by_dt) and (dt is not None):
            self.value /= dt
