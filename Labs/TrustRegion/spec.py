#name this file solutions.py
#Trust Regions lab

# Problem 1: Implement this function

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
    raise NotImplementedError("Problem 1 Incomplete")
    
    
# Problem 2: Implement this function    
    
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
    
    raise NotImplementedError("Problem 2 Incomplete")
        
def test_rosen():
    """
    A test function for problem 2.
    Tests whether your trust region implementation works on the Rosenbrock function by comparing it to scipy.optimize.minimize.
    """

    x = np.array([10.,10.])
    rhat = 2.
    r = .25
    eta = 1./16
    tol = 1e-5
    opts = {'initial_trust_radius':r, 'max_trust_radius':rhat, 'eta':eta, 'gtol':tol}
    
    sol1 = op.minimize(op.rosen, x, method='dogleg', jac=op.rosen_der, hess=op.rosen_hess, options=opts)
    
    sol2 = trustRegion(op.rosen, op.rosen_der, op.rosen_hess, dogleg, x, r, rhat, eta, gtol=tol)
    
    print np.allclose(sol1.x, sol2)
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        