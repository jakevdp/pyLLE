import numpy

def Scurve(Npts):
    S = lambda t: (numpy.sin(t),numpy.sign(t)*(numpy.cos(t)-1))

    theta = 3*numpy.pi * (numpy.random.random(Npts) - 0.5)
    x,z = S(theta)
    y = numpy.random.random(Npts)*2.0

    return numpy.concatenate([[x],[y],[z]]), theta
