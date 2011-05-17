from LLE import *
import pylab
from matplotlib import ticker, axes3d
import numpy
import sys

def S(theta):
    """
    returns x,y
      a 2-dimensional S-shaped function
      for theta ranging from 0 to 1
    """
    t = 3*numpy.pi * (theta-0.5)
    x = numpy.sin(t)
    y = numpy.sign(t)*(numpy.cos(t)-1)
    return x,y

def rand_on_S(N,sig=0):
    t = numpy.random.random(N)
    x,z = S(t)
    y = numpy.random.random(N)*5.0
    if sig:
        x += numpy.random.normal(scale=sig,size=N)
        y += numpy.random.normal(scale=sig,size=N)
        z += numpy.random.normal(scale=sig,size=N)
    return x,y,z,t

def rand_on_S_hole(N):
    t = numpy.random.random(N)
    t = numpy.random.random(N)
    x,z = S(t)
    y = numpy.random.random(N)*5.0

    #indices = numpy.where( ((0.3>t) | (0.7<t)) | ((1.0>y) | (4.0<y)) )
    indices = numpy.where( (0.3>t) | ((1.0>y) | (4.0<y)) )
    return x[indices],y[indices],z[indices],t[indices]
    

def S_scatter(x,y,z,t=None,cmap=pylab.cm.jet):
    fig = pylab.figure()
    ax = axes3d.Axes3D(fig)

    if t==None:
        ax.scatter3D(x,y,z)
    else:
        ax.scatter3D(x,y,z,c=t,cmap=cmap)
    
    ax.set_xlim(-2,2)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # elev, az
    ax.view_init(10, -80)

def Hem(theta,phi):
    x = numpy.sin(phi)*numpy.cos(theta)
    y = numpy.sin(phi)*numpy.sin(theta)
    z = numpy.cos(phi)
    return x,y,z

def rand_on_Hem(N):
    theta = numpy.random.random(N) * 2* numpy.pi
    phi = numpy.random.random(N) * numpy.pi/2

    x,y,z = Hem(theta,phi)

    return x,y,z,phi

def Hem_scatter(x,y,z,t=None,cmap=pylab.cm.jet):
    fig = pylab.figure()
    ax = axes3d.Axes3D(fig)

    if t==None:
        ax.scatter3D(x,y,z)
    else:
        ax.scatter3D(x,y,z,c=t,cmap=cmap)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_zlim(0,2)
    
    # elev, az
    ax.view_init(10, -80)

def scatter_2D(x,y,t=None,cmap=pylab.cm.jet):
    if t==None:
        pylab.scatter(x,y)
    else:
        pylab.scatter(x,y,c=t,cmap=cmap)

    pylab.xlabel('x')
    pylab.ylabel('y')



#------------------------------------------------

if __name__ == '__main__':
    x,y,z,t = rand_on_S(500,0.05)
    #x,y,z,t = rand_on_Hem(1000)
    M3 = numpy.array([x,y,z])

    #S_scatter(x,y,z,t)
    #pylab.savefig('S.eps')
    #pylab.close()
    #pylab.show()
    
    #for k in range(2,20):
    #    m= dimensionality(M3,k)
    #    print k,numpy.median(m),numpy.std(m)
    #
    #exit()

    k = 16
    
    M2 = HLLE(M3,k,2)
    
    scatter_2D(M2[0],M2[1],t)
    pylab.show()
    exit()

    x,y,z,t2 = rand_on_S(400,0.1)

    x3 = numpy.array([x,y,z])

    x2 = new_LLE_pts(M3,M2,k,x3)
    
    scatter_2D(M2[0],M2[1],t)
    pylab.savefig('LLE_700.eps')
    
    scatter_2D(x2[0],x2[1],t2)
    pylab.savefig('LLE+500.eps')
    
    #pylab.savefig('S_LLE.eps')
    #pylab.close()
    pylab.show()
        
    #M2 = HLLE(M3,10,2)
    #scatter_2D(M2[0],M2[1],t)
    #pylab.savefig('S_HLLE.eps')
    #pylab.close()
    
    
