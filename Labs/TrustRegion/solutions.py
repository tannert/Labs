#solutions.py
#for the Trust Regions lab

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la
from scipy import optimize as op

def trustRegion(f,grad,hess,subprob,x0,r0,rmax,eta,gtol=1e-5):
    """
    Minimize a function using the trust-region algorithm.

    Parameters
    ----------
    f : callable function object
        The objective function to minimize
    g : callable function object
        The gradient (or approximate gradient) of the objective function
    hess : callable function object
        The hessian (or approximate hessian) of the objective function
    subprob: callable function object
        Returns the step p_k
    x0 : numpy array of shape (n,)
        The initial point
    r0 : float
        The initial trust-region radius
    rmax : float
        The max value for trust-region radii
    eta : float in [0,0.25)
        Acceptance threshold
    gtol : float
        Convergence threshold

    Returns
    -------
    x : the minimizer of f

    Notes
    -----
    The functions f, g, and hess should all take a single parameter.
    The function subprob takes as parameters a gradient vector, hessian matrix, and radius.
    """
    while la.norm(grad(x0)) > gtol:
        p = subprob(grad(x0), hess(x0), r0)
        m = f(x0) + p.T.dot(grad(x0)) + p.T.dot(hess(x0)).dot(p)/2.
        rho = (f(x0) - f(x0+p))/(f(x0) - m)
        if rho < .25:
            r0 /= 4.
        elif rho > .75 and np.allclose(la.norm(p),r0):
            r0 = min(2*r0,rmax)
        
        if rho > eta:
            x0 += p
    return x
        
def dogleg(gk,Hk,rk):
    """
    Calculate the dogleg minimizer of the quadratic model function.

    Parameters
    ----------
    gk : ndarray of shape (n,)
        The current gradient of the objective function
    Hk : ndarray of shape (n,n)
        The current (or approximate) hessian
    rk : float
        The current trust region radius

    Returns
    -------
    pk : ndarray of shape (n,)
        The dogleg minimizer of the model function.
    """
    
    
    pH = np.linalg.solve(-1*Hk, gk)
    pD = -gk.T.dot(gk)/(gk.T.dot(Hk).dot(gk)) * gk
    
    if np.linalg.norm(pH) <= rk:
        return pH
        
    if np.linalg.norm(pD) >= rk:
        return rk/np.linalg.norm(pD)*pD
        
    DD = pD.dot(pD)
    HD = pH.dot(pD)
    HH = pH.dot(pH)
    
    a = HH - 2*HD + DD
    b = 2*HD - 2*DD
    c = DD - rk**2
    
    t = 1 + (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        
    return pD + (t-1)*(pH-pD)
        
def test_rosen():        

    x = np.array([10.,10.])
    rhat = 2.
    r = .25
    eta = 1./16
    tol = 1e-5
    opts = {'initial_trust_radius':r, 'max_trust_radius':rhat, 'eta':eta, 'gtol':tol}
    
    sol1 = op.minimize(op.rosen, x, method='dogleg', jac=op.rosen_der, hess=op.rosen_hess, options=opts)
    
    sol2 = trustRegion(op.rosen, op.rosen_der, op.rosen_hess, dogleg, x, r, rhat, eta, gtol=tol)
    
    print np.allclose(sol1.x, sol2)
        
        
        
if __name__ == '__main__':
    test_rosen():
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        