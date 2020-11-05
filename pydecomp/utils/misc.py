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

def rank_sampling(maxrank,sampling="default"):
    """Returns a sampling of ranks for approximation error plots"""
    print("Sampling parameter set to "+sampling)
    if maxrank<10:
        rank_sampling=[i+1 for i in range(maxrank)]
        print("overriding sampling method due to very small rank: {}".format(rank_sampling))
        return rank_sampling

    if maxrank<20:
        rank_sampling=[i for i in range(1,maxrank+1,2)]
        print("overriding sampling method due to very small rank: {}".format(rank_sampling))
        return rank_sampling
    

    if sampling=="sparse":
        if maxrank>25:
            rank_sampling=[i for i in np.arange(1,11,2)] +[15,20,30,45]\
                        +[i for i in range(60,min(maxrank,100),20)]\
                        +[i for i in range(100,min(maxrank,300),50)]\
                        +[i for i in range(300,min(maxrank,1000),100)]\
                        +[i for i in range(1000,maxrank,200)]\
                        +[maxrank]
        else:
            rank_sampling=[i for i in range(1,maxrank,2)]
    elif sampling=="super_sparse":
        if rank >40:
            rank_sampling=[1,2,3,5,7,11,15,25,40]\
                        +[i for i in range(60,min(maxrank,200),40)]\
                        +[i for i in range(200,min(maxrank,500),100)]\
                        +[i for i in range(500,min(maxrank,1000),250)]\
                        +[i for i in range(1000,maxrank,1000)]\
                        +[maxrank]
    elif sampling=="exponential":
        rank_sampling=[1,2]
        i=2
        while maxrank>2**i:
            i+=1
            rank_sampling.append(min(int(2**i),maxrank))
    elif sampling=="exponential_fine":
        rank_sampling=[1,2]
        i=2
        while maxrank>2**i:
            i+=0.5
            rank_sampling.append(min(int(2**i),maxrank))
    elif sampling=="quadratic":
        rank_sampling=[]
        if maxrank<50:
            i=1
            step=1
        else:
            i=4
            step=3
        rank=0
        while rank<maxrank:
            i+=step
            rank=min(int(i**2),maxrank)
            rank_sampling.append(rank)
            print(rank_sampling)
    elif sampling=="linear10":
        rank_sampling=[ int(i) for i in (np.linspace(10,maxrank,num=10,endpoint=True))]
    else:
        if maxrank>25:
            rank_sampling=[i for i in np.arange(1,11)] +[15,20,25,30,35,40]\
                        +[i for i in range(50,min(maxrank,100),10)]\
                        +[i for i in range(100,min(maxrank,300),20)]\
                        +[i for i in range(300,min(maxrank,1000),50)]\
                        +[i for i in range(1000,maxrank,100)]\
                        +[maxrank]
        else:
            rank_sampling=[i for i in range(1,maxrank)]
    print("sampling {}",rank_sampling)
    return rank_sampling


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
