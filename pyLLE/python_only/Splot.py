#!/astro/apps/pkg/python/bin/python

import pylab
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D
import numpy

def S(theta):
    x = numpy.sin(theta)
    y = numpy.sign(theta)*(numpy.cos(theta)-1)
    return x,y

def rand_on_S(N):
    t = 3*numpy.pi * (numpy.random.random(N) - 0.5)
    x,z = S(t)
    y = numpy.random.random(N)*2.0
    return x,y,z,t

def Hem(theta,phi):
    x = numpy.sin(phi)*numpy.cos(theta)
    y = numpy.sin(phi)*numpy.sin(theta)
    z = numpy.cos(phi)
    return x,y,z

def Hem_scatter(N):
    theta = numpy.random.random(N) * 2* numpy.pi
    phi = numpy.random.random(N) * numpy.pi/2

    x,y,z = Hem(theta,phi)
    
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter3D(x,y,z,c=phi,cmap=pylab.cm.jet)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_zlim3d(0,2)
    
    # elev, az
    ax.view_init(10, -80)


def S_plot(N,style='surface'):
    if style in ['surface','wire']:
        num_t = int(3*numpy.sqrt(N))
        num_y = int(numpy.sqrt(N)/3)
        theta = numpy.linspace(-1.5*numpy.pi,1.5*numpy.pi,num_t)
        x,z = S(theta)
        y = numpy.linspace(0.0,2.0,num_y)
        
        x_plot  = numpy.multiply.outer(x, numpy.ones(len(y))) 
        z_plot  = numpy.multiply.outer(z, numpy.ones(len(y))) 
        y_plot  = numpy.multiply.outer(y, numpy.ones(len(x))).T
        
    elif style=='scatter':
        x_plot,y_plot,z_plot,theta = rand_on_S(N)

    else:
        raise ValueError, 'unrecognized plot style: %s' % style
        
        
    fig = pylab.figure()
    ax = Axes3D(fig)

    if style=='wire':
        ax.plot_wireframe(x_plot,y_plot,z_plot,color='k')
    if style=='surface':
        ax.plot_surface(x_plot,y_plot,z_plot)
    if style=='scatter':
        ax.scatter3D(x_plot,y_plot,z_plot,c=theta,cmap=pylab.cm.jet)

    ax.set_xlim3d(-2,2)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # elev, az
    ax.view_init(10, -80)

if __name__ == '__main__':
    Hem_scatter(1000)
    S_plot(1000,'scatter')
    pylab.show()
