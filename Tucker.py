# -*- coding: utf-8 -*-
"""
Created on Wed May  2 09:25:40 2018

@author: Diego Britez
"""
from tensor_descriptor_class import TensorDescriptor
from high_order_decomposition_method_functions import multilinear_multiplication
from full_format_class import FullFormat
import pickle
class Tucker(TensorDescriptor):
    """
    This class is created to storage a decomposed Tensor in the Tucker
    Format, this is code is based in ttensor code from pytensor code package.\n 
    **Attributes**\n
         **shape**: array like, with the numbers of elements that each 1-rank 
        tensor is going to discretize each subspace of the full tensor. \n
        **dim**: integer type, number that represent the n-rank tensor that is 
        going to be represented. The value of dim must be coherent with the 
        size of _tshape parameter. \n
        
        **core:** List type, in this list will be storage the core of the 
        decomposed tensor.\n
        
        **u:**List of the projection matrices in  each subspace.\n
       
    **Tucker Format Definition**\n
    """
    core = None;
    u = None;
#-----------------------------------------------------------------------------    
    def __init__(self, core, uIn):
        
        
        #Handle if the uIn is not a list
        if(uIn.__class__ != list):
            uIn=[x for x in uIn]
            #newuIn = [];
            #for x in uIn:
            #    newuIn.extend([x]);
            #uIn = newuIn;
           
        #newuIn = []; 
        #for i in range(0, len(uIn)):
        #    newuIn.extend([uIn[i].copy()]);
        #uIn = newuIn;
        
        # check that each U is indeed a matrix
        for i in range(0,len(uIn)):
            if (len(uIn[i].shape) != 2):
                raise ValueError("{0} is not a 2-D matrix!".format(uIn[i]));
        
        # Size error checking
        k = core.shape;
        a="""Number of dims of Core and the number of matrices are different"""
        b="""{0} th dimension of Core is different from the number
            of columns of uIn[i]"""
        if (len(k) != len(uIn)):
            raise ValueError(a);
        
        for i in range(0,len(uIn)):
            if (k[i] != len((uIn[i])[0])):
                raise ValueError(b.format(i));
                   
         
        self.core = core.copy();
        self.u = uIn;
        
        #save the shape of the ttensor
        shape = [];
        for i in range(0, len(self.u)):
            shape.extend([len(self.u[i])]);
        self.shape = tuple(shape);
        # constructor end #
#-----------------------------------------------------------------------------
    def size(self):
        ret = 1;
        for i in range(0, len(self.shape)):
            ret = ret * self.shape[i];
        return ret;
#-----------------------------------------------------------------------------   
    def dimsize(self):
        return len(self.u)
#-----------------------------------------------------------------------------    
    def copy(self):
        return Tucker(self.core, self.u);
#-----------------------------------------------------------------------------        
    def destructor(self):
        self.u=[]
        self.core=0
        self.shape=[]
#-----------------------------------------------------------------------------    
    def reconstruction(self):
        """returns a FullFormat object that is represented by the 
        tucker tensor"""
    
        dim=len(self.u)
        Fresult=multilinear_multiplication(self.u,self.core,dim)
        tshape=Fresult.shape
        Result=FullFormat(Fresult,tshape,dim)
    
        return Result
#-----------------------------------------------------------------------------
    def save(self, file_name):
        """
        This method has the function to save a Tucker class object as a 
        binary file. \n
        **Parameters**:\n
            Object= A Tucker class object.\n
            file_name= String type. Name of the file that is going to be 
            storage. \n
        **Returns**:\n
            File= Binary file that will reproduce the object class when 
            reloaded.
        """
        if type(file_name)!=str:
            raise ValueError('Variable file_name must be a string')
        pickle_out=open(file_name,"wb")
        pickle.dump(self,pickle_out)
        pickle_out.close()
#-----------------------------------------------------------------------------           
    def __str__(self):
        ret = "ttensor of size {0}\n".format(self.shape);
        ret += "Core = {0} \n".format(self.core.__str__());
        for i in range(0, len(self.u)):
            ret += "u[{0}] =\n{1}\n".format(i, self.u[i]);
        
        return ret; 
    