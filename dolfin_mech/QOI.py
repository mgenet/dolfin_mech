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
        
        self.value = self.expr(self.point)

        self.value += self.constant
        self.value /= self.norm

        if (self.divide_by_dt) and (dt is not None):
            self.value /= dt
