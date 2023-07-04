#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2023                                       ###
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
            expr,
            norm=1.,
            constant=0.,
            divide_by_dt=False,
            form_compiler_parameters={},
            point=None,
            update_type="assembly"):

        self.name                     = name
        self.expr                     = expr
        self.norm                     = norm
        self.constant                 = constant
        self.divide_by_dt             = divide_by_dt
        self.form_compiler_parameters = form_compiler_parameters
        self.point                    = point

        if (update_type == "assembly"):
            self.update = self.update_assembly
        elif (update_type == "direct"):
            self.update = self.update_direct



    def update_assembly(self, dt=None):

        # print(self.name)
        # print(self.expr)
        # print(self.form_compiler_parameters)

        self.value = dolfin.assemble(
            self.expr,
            form_compiler_parameters=self.form_compiler_parameters)

        self.value += self.constant
        self.value /= self.norm

        if (self.divide_by_dt):
            assert (dt != 0),\
                "dt (="+str(dt)+") should be non zero. Aborting."
            self.value /= dt



    def update_direct(self, dt=None):
        
        self.value = self.expr(self.point)

        self.value += self.constant
        self.value /= self.norm

        if (self.divide_by_dt) and (dt is not None):
            self.value /= dt
