#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

import dolfin_mech as dmech

################################################################################

class Constraint():



    def __init__(self,
            V,
            sub_domain=None,
            sub_domains=None,
            sub_domain_id=None,
            val=None, val_ini=None, val_fin=None,
            method=None): # topological, geometric, pointwise

        if (val is not None) and (val_ini is None) and (val_fin is None):
            self.tv_val = dmech.TimeVaryingConstant(
                val_ini=val, val_fin=val)
        elif (val is None) and (val_ini is not None) and (val_fin is not None):
            self.tv_val = dmech.TimeVaryingConstant(
                val_ini=val_ini, val_fin=val_fin)

        if (sub_domain is not None) and (sub_domains is None) and (sub_domain_id is None):
            if (method is None):
                self.bc = dolfin.DirichletBC(
                    V,
                    self.tv_val.val,
                    sub_domain)
            else:
                self.bc = dolfin.DirichletBC(
                    V,
                    self.tv_val.val,
                    sub_domain,
                    method)
        elif (sub_domain is None) and (sub_domains is not None) and (sub_domain_id is not None):
            if (method is None):
                self.bc = dolfin.DirichletBC(
                    V,
                    self.tv_val.val,
                    sub_domains,
                    sub_domain_id)
            else:
                self.bc = dolfin.DirichletBC(
                    V,
                    self.tv_val.val,
                    sub_domains,
                    sub_domain_id,
                    method)



    def set_value_at_t_step(self,
            t_step):

        self.tv_val.set_dvalue_at_t_step(t_step)



    def restore_old_value(self):

        self.tv_val.restore_old_value()



    def homogenize(self):

        self.tv_val.homogenize()
        # self.bc.homogenize() # MG20180508: seems to be changing the constant…
