"""
Created on 28/06/2018

Function that might be useful and do not belong to any particular family
"""
import numpy as np


def quick_gif(im_list,path,output_name):
    """
        This is a quick and dirty way to create a gif from a list of images

        im_list         a list of images names
        path            input/output directory
        output_name     Name of the output
    """
    import imageio
    print("Assembling gif "+path+output_name)
    k=0
    with imageio.get_writer(path+output_name, mode='I',duration=0.3) as writer:
        for filename in im_list:
            k+=1
            # print(str((100.*k)/len(im_list))+'%') #print percentage of gif build
            image = imageio.imread(filename)
            writer.append_data(image)
    print(path+output_name+" has been saved")
    return

def list_transpose(phi):
    # @Diego Overspecialized. What does it work on ?
    # Please move it accordingly. If its used in several context
    """
    Parameter:
        phi: list with n-array matrices as elements.

    Return:
        phit: list with n-array transposed matrices of input parameter.
    """
    phit=[i.T for i in phi]
    return phit

def integration_1dtrap(f,x):
    """This function will integrate the 'f' vector discretised in the x domain
    f= nd array represetation of the function
    x= nd array representation of the grid of the domain
    """
    nx=x.size
    """
    if (nx!=f.size):

        print('Integration-1dtrap fonction and domaine dont have same dimentions','\n',
              'dimention of the fonction', f.size,'\n', 'dimention of the discretitation grid',
              x.size)
     """

    w=np.zeros(nx)
    w[1:-1]=(x[2:]-x[0:-2])/2
    w[0]=(x[1]-x[0])/2
    w[-1]=(x[-1]-x[-2])/2
    Int =np.dot(f,w)


    return Int


def test(x):
    return x**1

#------------------------------------------------------------------------------

if __name__=="__main__":
    nx=51
    x=np.linspace(0,1,nx)     #Sert a créer une grille entre 0 et 1 avec nx éléments
    print(x)
    f=test(x)
    print(f)
    F=integration_1dtrap(f,x)
    print(F)
    print(F-1./2)
