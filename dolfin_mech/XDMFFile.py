#coding=utf8

################################################################################
###                                                                          ###
### Created by Martin Genet, 2018-2024                                       ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin

################################################################################

class XDMFFile():

    def __init__(self,
            filename,
            functions):

        self.xdmf_file = dolfin.XDMFFile(filename)
        self.xdmf_file.parameters["flush_output"] = True
        self.xdmf_file.parameters["functions_share_mesh"] = True
        #self.xdmf_file.parameters["rewrite_function_mesh"] = False

        self.functions = functions



    def close(self):

        self.xdmf_file.close()



    def write(self,
            time=0):

        for function in self.functions:
            self.xdmf_file.write(function, float(time))
