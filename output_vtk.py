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

def load_bp(file,dir):
    source=ad.File(filename)
    X=source['X'].read()
    Y=source['Y'].read()
    pressure=source['pressure'].read()
    u=source['velocity_u'].read()
    v=source['velocity_v'].read()
    rho=source['density'].read()
    vort=source['vorticity'].read()
    nx=X.size-1
    ny=Y.size-1
    nz=1
    z=np.asarray([0.])
    dx=X[1]-X[0]
    dy=Y[1]-Y[0]
    x=X[1:]-dx/2
    y=Y[1:]-dy/2
    return nx,ny,nz, x,y,z, pressure,u,v,rho,vort


if __name__=="__main__":
    vtk_known_2dfunction_test(9,3)

    out_dir="output/"
    in_dir="data_notus_wave/"
    filename=in_dir+'Lucas_HL0095_dL010_002700.bp'
    nx,ny,nz,x,y,z,pressure, U,V, density,vort=load_bp(filename,out_dir)

    p=np.reshape(pressure.T ,(nx,ny,1),order='F')
    print(np.isfortran(p))
    u=np.reshape(U.T,(nx,ny,1),order='F')
    v=np.reshape(V.T,(nx,ny,1),order='F')
    rho=np.reshape(density.T,(nx,ny,1),order='F')
    vort=np.reshape(vort.T,(nx,ny,1),order='F')

    var_dic={"pressure" : p,"velocity_u" : u,"velocity_v" : v,'density' : rho,'vorticity':vort}

    gridToVTK(out_dir+"bpIOtest",x,y,z, cellData = var_dic)
