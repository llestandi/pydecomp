import adios as ad
from evtk.hl import gridToVTK
import numpy as np



def vtk_known_2dfunction_test(nx,ny):
    nx, ny, nz = 8, 5, 1
    lx, ly, lz = 1.0, 1.0, 1.0
    dx, dy, dz = lx/nx, ly/ny, lz/nz

    ncells = nx * ny * nz
    npoints = (nx + 1) * (ny + 1) * (nz + 1)

    # Coordinates
    x = np.arange(0, lx + 0.1*dx, dx, dtype='float64')
    print('x:',x)
    y = np.arange(0, ly + 0.1*dy, dy, dtype='float64')
    print('y:',y)
    z = np.asarray([0])#np.arange(0, lz + 0.1*dz, dz, dtype='float64')
    print('z:',z)
    f= lambda x: x[0]**2+x[1]
    X=f(np.meshgrid(x,y,indexing='ij'))

    print(X.shape)
    print(nz)
    X=np.reshape(X,(nx+1,ny+1,1))
    print(X)
    gridToVTK("output/rectilinear",x,y,z, pointData = {'f':X})

    return

def load_bp(file,dir,var_list):
    source=ad.File(dir+filename)
    X=source['X'].read()
    Y=source['Y'].read()
    Z=np.asarray([0.])
    var_dic={}
    for var in var_list:
        var_dic[var]=source[var].read()
    # pressure=source['pressure'].read()
    # u=source['velocity_u'].read()
    # v=source['velocity_v'].read()
    # rho=source['density'].read()
    # vort=source['vorticity'].read()
    # dx=X[1]-X[0]
    # dy=Y[1]-Y[0]
    # x=X[1:]-dx/2
    # y=Y[1:]-dy/2
    return X,Y,Z, var_dic

def prepare_dic_for_vtk(var_dic,nx,ny):
    """ This function reorders data inside arrays for vtk output """
    for field in var_dic:
        var_dic[field]=np.reshape(var_dic[field].T,(nx,ny,1),order='F')
    return

if __name__=="__main__":
    # vtk_known_2dfunction_test(9,3)

    out_dir="output/"
    in_dir="data_notus_wave/"
    var_list=["pressure","density",'velocity_u','velocity_v','vorticity']
    filename='Lucas_HL009_dL010_000500.bp'
    x,y,z,var_dic=load_bp(filename,in_dir,var_list)
    nxC=x.size-1
    nyC=y.size-1
    prepare_dic_for_vtk(var_dic,nxC,nyC)


    gridToVTK(out_dir+"bpIOtest",x,y,z, cellData = var_dic)
