###########################
### Trust Region Method ###
###########################

# Authors: Jonas Kohler and Aurelien Lucchi, 2017

from math import sqrt, ceil, log
import numpy as np
from datetime import datetime


def Trust_Region(w, loss, gradient, Hv=None, hessian=None, X=None, Y=None, opt=None,**kwargs):

    """
    Minimize a continous, unconstrained function using the Trust Region method.

    References
    ----------
    Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Trust region methods. Society for Industrial and Applied Mathematics.

    Parameters
    ----------
    loss : callable f(x,**kwargs)
        Objective function to be minimized.
    
    grad : callable f'(x,**kwargs), optional
        Gradient of f.
    Hv: callable Hv(x,**kwargs), optional
        Matrix-vector-product of Hessian of f and arbitrary vector v
    **kwargs : dict, optional 
        Extra arguments passed to loss, gradient and Hessian-vector-product computation, e.g. regularization constant or number of classes for softmax regression.
    opt : dictionary, optional
        optional arguments passed to ARC
    """

    print ('--- Trust Region ---\n')
    
    if X is None:
        n=1 
        d=1
    else:
        n = X.shape[0] 
        d = X.shape[1] 

    # Basics
    n_iterations = opt.get('n_iterations',100)   
    eta_1 = opt.get('successful_treshold',0.1)
    eta_2 = opt.get('very_successful_treshold',0.9)
    gamma_1 = opt.get('penalty_increase_multiplier',2.)
    gamma_2= opt.get('penalty_derease_multiplier',2.)
    assert (gamma_1 >= 1 and gamma_2 >= 1), "penalty update parameters must be greater or equal to 1"

    tr_radius = opt.get('initial_tr_radius',1.)  # intial tr radius
    max_tr_radius = opt.get('max_trust_radius',1e4)  # max tr radius
    assert (tr_radius > 0 and max_tr_radius > 0), "Trust region radius must be positive"
    
    grad_tol = opt.get('grad_tol',1e-6)

    # Subproblem
    subproblem_solver= opt.get('subproblem_solver_TR','cauchy_point')
    krylov_tol=opt.get('krylov_tol',1e-1)
    exact_tol=opt.get('exact_tol',1e-1)

    # Sampling
    Hessian_sampling_flag=opt.get('Hessian_sampling', False)
    gradient_sampling_flag=opt.get('gradient_sampling', False)

    if gradient_sampling_flag==True or Hessian_sampling_flag==True:
        assert (X is not None and Y is not None), "Subsampling is only possible if data is passsed, i.e. X and Y may not be none"

    initial_sample_size_Hessian=opt.get('initial_sample_size_Hessian',0.05)
    initial_sample_size_gradient=opt.get('initial_sample_size_gradient',0.05)
    unsuccessful_sample_scaling=opt.get('unsuccessful_sample_scaling',1.25)
    sampling_scheme=opt.get('sampling_scheme', 'exponential')
    if Hessian_sampling_flag==False and gradient_sampling_flag ==False:
        sampling_scheme=None
   
    print ("- Subproblem_solver:", subproblem_solver)
    print("- Hessian_sampling:" , Hessian_sampling_flag)
    print("- Gradient_sampling:", gradient_sampling_flag)
    print("- Sampling_scheme:" , sampling_scheme,"\n")

    successful_flag=False
    k = 0
    lambda_k=0
    n_samples_seen = 0

    grad = gradient(w, X, Y,**kwargs)  
    grad_norm=np.linalg.norm(grad)

    loss_collector=[]
    timings_collector=[]
    samples_collector=[]
    
    _loss = loss(w, X, Y, **kwargs) 
    loss_collector.append(_loss)
    timings_collector.append(0)
    samples_collector.append(0)

    start = datetime.now()
    timing=0

    if sampling_scheme=='exponential':
       exp_growth_constant=((1-initial_sample_size_Hessian)*n)**(1/n_iterations)

    for i in range(n_iterations):
        #### I: Subsampling #####
        ## a) determine batchsize ##
        if sampling_scheme=='exponential':
            sample_size_Hessian = Hessian_sampling_flag*(int(min(n, n*initial_sample_size_Hessian + exp_growth_constant**(i+1)))+1) + (1-Hessian_sampling_flag)*n
            sample_size_gradient= gradient_sampling_flag*(int(min(n, n*initial_sample_size_gradient + exp_growth_constant**(i+1)))+1) + (1-gradient_sampling_flag)*n
        elif sampling_scheme=='linear':
            sample_size_Hessian = Hessian_sampling_flag*int(min(n, max(n*initial_sample_size_Hessian, n/n_iterations*(i+1))))+(1-Hessian_sampling_flag)*n
            sample_size_gradient= gradient_sampling_flag*int(min(n, max(n*initial_sample_size_gradient, n/n_iterations*(i+1))))+(1-gradient_sampling_flag)*n

        else:
            sample_size_Hessian=n
            sample_size_gradient=n

        ## b) draw batches ##
        if sample_size_Hessian <n:
            int_idx_Hessian=np.random.randint(0, high=n, size=sample_size_Hessian)        

            bool_idx_Hessian = np.zeros(n,dtype=bool)
            bool_idx_Hessian[int_idx_Hessian]=True
            _X=np.zeros((sample_size_Hessian,d))
            _X=np.compress(bool_idx_Hessian,X,axis=0)
            _Y=np.compress(bool_idx_Hessian,Y,axis=0)

        else: 
            _X=X
            _Y=Y

        if sample_size_gradient < n:
            int_idx-gradient=np.random.randint(0, high=n, size=sample_size_gradient)        

            bool_idx_gradient = np.zeros(n,dtype=bool)
            bool_idx_gradient[int_idx_gradient]=True
            _X2=np.zeros((sample_size_gradient,d))
            _X2=np.compress(bool_idx_gradient,X,axis=0)
            _Y2=np.compress(bool_idx_gradient,Y,axis=0)

        else:
            _X2=X
            _Y2=Y

        n_samples_per_step=sample_size_Hessian+sample_size_gradient        

        #### II: Step computation #####
        

        # b) call subproblem solver
        (s,lambda_k) = solve_TR_subproblem(grad,Hv,hessian,tr_radius,_X, _Y, w, successful_flag, lambda_k,subproblem_solver,
            exact_tol,krylov_tol,**kwargs)

        sn=np.linalg.norm(s)

        #### III: Regularization Update ####
        previous_f = loss(w, X, Y, **kwargs)
        current_f = loss(w + s, X, Y,**kwargs)

        function_decrease = previous_f - current_f

        Hs=Hv( w, _X, _Y, s, **kwargs)

        model_decrease = -(np.dot(grad, s) + 0.5 * np.dot(s, Hs))
        rho = function_decrease / model_decrease

        assert (model_decrease >=0), 'negative model decrease. This should not have happend'


        # Update w if step s is successful
        if rho >= eta_1:
            w = w + s 
            _loss=current_f
            successful_flag=True
        else:
            _loss=previous_f   

        
        # Update trust region radius
        _tr_radius=tr_radius
        if rho < eta_1:
            tr_radius *= 1/gamma_1
            print ('unscuccesful iteration')
            successful_flag=False
        elif rho > eta_2 and (np.linalg.norm(s) - tr_radius < 1e-10):
                tr_radius = min(gamma_2 * tr_radius, max_tr_radius)

        # recompute gradient either because of accepted step or because of re-sampling
        if gradient_sampling_flag==True or successful_flag==True:
            grad = gradient(w,_X2, _Y2, **kwargs)  
            grad_norm =np.linalg.norm(grad)  
            if grad_norm < grad_tol:
                break

        n_samples_seen += n_samples_per_step

        ### IV: Save Iteration Information  ###
        
        _timing=timing
        timing=(datetime.now() - start).total_seconds()

        print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(np.linalg.norm(grad)),
          'time= ', timing-_timing, 'tr_radius=',_tr_radius, 'stepnorm=', sn, 'Samples Hessian=', sample_size_Hessian,'samples Gradient=', sample_size_gradient,"\n")
    
    
        
       
        timings_collector.append(timing)
        samples_collector.append(n_samples_seen)

        loss_collector.append(_loss)
        k += 1

    return w,timings_collector,loss_collector, samples_collector


def solve_TR_subproblem(grad,Hv, hessian ,tr_radius, X, Y, w, successful_flag,lambda_k,subproblem_solver,exact_tol,krylov_tol,**kwargs):
    from scipy import linalg    

    if (subproblem_solver == 'cauchy_point'):
        Hg=Hv(w, X, Y, grad,**kwargs)
        gBg = np.dot(grad, Hg)
        tau = 1
        if gBg > 0:  # if model is convex quadratic the unconstrained minimizer may be inside the TR
            tau = min(np.linalg.norm(grad) ** 3 / (tr_radius * gBg), 1)
        pc = - tau * tr_radius * np.divide(grad, np.linalg.norm(grad))
        return (pc,0)

    elif (subproblem_solver == 'dog_leg'):
        H =hessian(w, X, Y, **kwargs)
        gBg = np.dot(grad, np.dot(H, grad))
        if gBg <= 0:
            raise ValueError(
                'dog_leg requires H to be positive definite in all steps!') 

        ## Compute the Newton Point and return it if inside the TR
        cholesky_B = linalg.cho_factor(H)
        pn = -linalg.cho_solve(cholesky_B, grad)
        if (np.linalg.norm(pn) < tr_radius):
            return (pn,0)


        # Compute the 'unconstrained Cauchy Point'
        pc = -(np.dot(grad, grad) / gBg) * grad
        pc_norm = np.linalg.norm(pc)

        # if it is outside the TR, return the point where the path intersects the boundary
        if pc_norm >= tr_radius:
            p_boundary = pc * (tr_radius / pc_norm)
            return (p_boundary,0)


        # else, give intersection of path from pc to pn with tr_radius.
        t_lower, t_upper = solve_quadratic_equation(pc, pn, tr_radius)
        p_boundary = pc + t_upper * (pn - pc)
        return (p_boundary,0)


    elif subproblem_solver == 'cg': 
        grad_norm = np.linalg.norm(grad)
        p_start = np.zeros_like(grad)

        if grad_norm < min(sqrt(linalg.norm(grad)) * linalg.norm(grad),krylov_tol):
            return (p_start,0)

        # initialize
        z = p_start
        r = grad
        d = -r
        k = 0
          
        while True:
            Bd=Hv(w, X, Y, d, **kwargs)
            dBd = np.dot(d, Bd)
            # terminate when encountering a direction of negative curvature with lowest boundary point along current search direction
            if dBd <= 0:
                t_lower, t_upper = solve_quadratic_equation(z, d, tr_radius)
                p_low = z + t_lower * d
                p_up = z + t_upper * d
                m_p_low = loss(w + p_lowX, Y, **kwargs  ) + np.dot(grad, p_low) + 0.5 * np.dot(p_low, np.dot(H, p_low))
                m_p_up = loss(w + p_up, X, Y, **kwargs) + np.dot(grad, p_up) + 0.5 * np.dot(p_up, np.dot(H, p_up))
                if m_p_low < m_p_up:
                    return (p_low,0)

                else:
                    return (p_up,0)


            alpha = np.dot(r, r) / dBd
            z_next = z + alpha * d
            # terminate if z_next violates TR bound
            if np.linalg.norm(z_next) >= tr_radius:
                # return intersect of current search direction w/ boud
                t_lower, t_upper = solve_quadratic_equation(z, d, tr_radius)
                return z + t_upper * d
            r_next = r + alpha * Bd
            if np.linalg.norm(r_next) < min(sqrt(linalg.norm(grad)) * linalg.norm(grad),krylov_tol):
                return (z_next,0)


            beta_next = np.dot(r_next, r_next) / np.dot(r, r)
            d_next = -r_next + beta_next * d
            # update iterates
            z = z_next
            r = r_next
            d = d_next
            k = k + 1


    elif subproblem_solver == 'GLTR':

        g_norm = np.linalg.norm(grad)
        s = np.zeros_like(grad)

        if g_norm == 0:
            # escape along the direction of the leftmost eigenvector as far as tr_radius permits
            print ('zero gradient encountered')
            H =hessian(w, X, Y, w, **kwargs)
            (s,lambda_k) = exact_TR_suproblem_solver(grad, H, tr_radius, exact_tol,successful_flag,lambda_k)

        else:
            # initialize
            g = grad
            p = -g
            gamma = g_norm
            T = np.zeros((1, 1))
            alpha_k = []
            beta_k = []
            interior_flag = True
            k = 0
      
            while True:
                Hp=Hv(w, X, Y, p, **kwargs)
                pHp = np.dot(p, Hp)
                alpha = np.dot(g, g) / pHp
               
                alpha_k.append(alpha)

                ###Lanczos Step 1: Build up subspace 
                # a) Create g_lanczos = gamma*e_1
                e_1 = np.zeros(k + 1)
                e_1[0] = 1.0
                g_lanczos = gamma * e_1
                # b) Create T for Lanczos Model 
                T_new = np.zeros((k + 1, k + 1))
                if k == 0:
                    T[k, k] = 1. / alpha
                    T_new[0:k,0:k]=T 
                else:
                    T_new[0:k,0:k]=T
                    T_new[k, k] = 1. / alpha + beta/ alpha_k[k - 1]
                    T_new[k - 1, k] = sqrt(beta) / abs(alpha_k[k - 1])
                    T_new[k, k - 1] = sqrt(beta) / abs(alpha_k[k - 1])
                    T = T_new

                if (interior_flag == True and alpha < 0) or np.linalg.norm(s + alpha * p) >= tr_radius:
                    interior_flag = False


                if interior_flag == True:
                    s = s + alpha * p
                else:
                    ###Lanczos Step 2: solve problem in subspace
                    (h,lambda_k) = exact_TR_suproblem_solver(g_lanczos, T, tr_radius, exact_tol,successful_flag,lambda_k)

                g_next = g + alpha * Hp

                # test for convergence
                e_k = np.zeros(k + 1)
                e_k[k] = 1.0
                
                if interior_flag == True and np.linalg.norm(g_next) < min(sqrt(linalg.norm(grad)) * linalg.norm(grad),krylov_tol) :
                    break
                if interior_flag == False and np.linalg.norm(g_next) * abs(np.dot(h, e_k)) < min(sqrt(linalg.norm(grad)) * linalg.norm(grad),krylov_tol): 
                    break

                if k==X.shape[1]:
                    print ('Krylov dimensionality reach full space! Breaking out..')
                    break
                beta= np.dot(g_next, g_next) / np.dot(g, g)
                beta_k.append(beta)
                p = -g_next + beta* p
                g = g_next
                k = k + 1

            if interior_flag == False:
                #### Recover Q by building up the lanczos space, TBD: keep storable Qs in memory
                n = np.size(grad)
                Q1 = np.zeros((n, k + 1))

                g = grad
                p = -g
                for j in range(0, k + 1):
                    gn = np.linalg.norm(g)
                    if j == 0:
                        sigma = 1
                    else:
                        sigma = -np.sign(alpha_k[j - 1]) * sigma
                    Q1[:, j] = sigma * g / gn  

                    if not j == k:
                        Hp=Hv(w, X, Y, p, **kwargs)
                        g= g + alpha_k[j] * Hp
                        p = -g + beta_k[j] * p

                # compute final step in R^n
                s = np.dot(Q1, np.transpose(h))
        return (s,lambda_k)


    elif (subproblem_solver == 'exact'):
        H =hessian(w, X, Y, **kwargs)
        (s,lambda_k) = exact_TR_suproblem_solver(grad, H, tr_radius, exact_tol,successful_flag,lambda_k)
        return (s,lambda_k)
    else:
        raise ValueError('solver unknown')


def exact_TR_suproblem_solver(grad, H, tr_radius, exact_tol,successful_flag,lambda_k):
    from scipy import linalg
    s = np.zeros_like(grad)
    ## Step 0: initialize safeguards
    H_ii_min = min(np.diagonal(H))
    H_max_norm = sqrt(H.shape[0] ** 2) * np.absolute(H).max()
    H_fro_norm = np.linalg.norm(H, 'fro')
    gerschgorin_l = max([H[i, i] + (np.sum(np.abs(H[i, :])) - np.abs(H[i, i])) for i in range(len(H))])
    gerschgorin_u = max([-H[i, i] + (np.sum(np.abs(H[i, :])) - np.abs(H[i, i])) for i in range(len(H))])

    lambda_lower = max(0, -H_ii_min, np.linalg.norm(grad) / tr_radius - min(H_fro_norm, H_max_norm, gerschgorin_l))
    lambda_upper = max(0, np.linalg.norm(grad) / tr_radius + min(H_fro_norm, H_max_norm, gerschgorin_u))

    if successful_flag==False and lambda_lower <= lambda_k <= lambda_upper: #reinitialize at previous lambda in case of unscuccesful iterations
        lambda_j=lambda_k
    elif lambda_lower == 0:  # allow for fast convergence in case of inner solution
        lambda_j = lambda_lower
    else:
        lambda_j=np.random.uniform(lambda_lower, lambda_upper)

    i=0
    # Root Finding
    while True:
        i+=1
        lambda_in_N = False
        lambda_plus_in_N = False
        B = H + lambda_j * np.eye(H.shape[0], H.shape[1])
        try:
            # 1 Factorize B
            L = np.linalg.cholesky(B)
            # 2 Solve LL^Ts=-g
            Li = np.linalg.inv(L)
            s = - np.dot(np.dot(Li.T, Li), grad)
            sn = np.linalg.norm(s)
            ## 2.1 Termination: Lambda in F, if q(s(lamda))<eps_opt q(s*) and sn<eps_tr tr_radius -> stop. By Conn: Lemma 7.3.5:
            phi_lambda = 1. / sn - 1. / tr_radius
            #if (abs(sn - tr_radius) <= exact_tol * tr_radius):
            if (abs(phi_lambda)<=exact_tol): #
                break;

            # 3 Solve Lw=s
            w = np.dot(Li, s)
            wn = np.linalg.norm(w)

            
            ##Step 1: Lambda in L
            if lambda_j > 0 and (phi_lambda) < 0:
                # print ('lambda: ',lambda_j, ' in L')
                lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * (sn ** 2 / wn ** 2)
                lambda_j = lambda_plus


            ##Step 2: Lambda in G    (sn<tr_radius)
            elif (phi_lambda) > 0 and lambda_j > 0 and np.any(grad != 0): #TBD: remove grad
                # print ('lambda: ',lambda_j, ' in G')
                lambda_upper = lambda_j
                lambda_plus = lambda_j + ((sn - tr_radius) / tr_radius) * (sn ** 2 / wn ** 2)

                ##Step 2a: If factorization succeeds: lambda_plus in L
                if lambda_plus > 0:
                    try:
                        # 1 Factorize B
                        B_plus = H + lambda_plus * np.eye(H.shape[0], H.shape[1])
                        L = np.linalg.cholesky(B_plus)
                        lambda_j = lambda_plus
                        # print ('lambda+', lambda_plus, 'in L')


                    except np.linalg.LinAlgError:
                        lambda_plus_in_N = True

                ##Step 2b/c: If not: Lambda_plus in N
                if lambda_plus <= 0 or lambda_plus_in_N == True:
                    # 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)
                    try:
                        U = np.linalg.cholesky(H)
                        H_pd = True
                    except np.linalg.LinAlgError:
                        H_pd = False

                    if lambda_lower == 0 and H_pd == True and phi_lambda >= 0: #cannot happen in ARC!
                        lambda_j = 0
                        #print ('inner solution found')
                        break
                    # 2. Else, choose a lambda within the safeguard interval
                    else:
                        # print ('lambda_plus', lambda_plus, 'in N')
                        lambda_lower = max(lambda_lower, lambda_plus)  # reset lower safeguard
                        lambda_j = max(sqrt(lambda_lower * lambda_upper),
                                       lambda_lower + 0.01 * (lambda_upper - lambda_lower))

                        lambda_upper = np.float32(
                            lambda_upper) 
                        
                        if lambda_lower == lambda_upper:
                            lambda_j = lambda_lower
                            ## Hard case
                            ew, ev = linalg.eigh(H, eigvals=(0, 0))
                            d = ev[:, 0]
                            dn = np.linalg.norm(d)
                            assert (ew == -lambda_j), "Ackward: in hard case but lambda_j != -lambda_1"
                            tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-tr_radius**2)
                            s=s + tao_lower * d   
                            print ('hard case resolved inside')

                            return s

            elif (phi_lambda) == 0: 
                break
            else:      #TBD:  move into if lambda+ column #this only happens for Hg=0 -> s=(0,..,0)->phi=inf -> lambda_plus=nan -> hard case (e.g. at saddle) 
                lambda_in_N = True
        ##Step 3: Lambda in N
        except np.linalg.LinAlgError:
            lambda_in_N = True
        if lambda_in_N == True:
            # print ('lambda: ',lambda_j, ' in N')
            try:
                U = np.linalg.cholesky(H)
                H_pd = True
            except np.linalg.LinAlgError:
                H_pd = False

            # 1. Check for interior convergence (H pd, phi(lambda)>=0, lambda_l=0)
            if lambda_lower == 0 and H_pd == True and phi_lambda >= 0: 
                lambda_j = 0
                #print ('inner solution found')
                break
            # 2. Else, choose a lambda within the safeguard interval
            else:
                lambda_lower = max(lambda_lower, lambda_j)  # reset lower safeguard
                lambda_j = max(sqrt(lambda_lower * lambda_upper),
                               lambda_lower + 0.01 * (lambda_upper - lambda_lower))  # eq 7.3.14
                lambda_upper = np.float32(lambda_upper)  
                # Check for Hard Case:
                if lambda_lower == lambda_upper:
                    lambda_j = lambda_lower
                    ew, ev = linalg.eigh(H, eigvals=(0, 0))
                    d = ev[:, 0]
                    dn = np.linalg.norm(d)
                    assert (ew == -lambda_j), "Ackward: in hard case but lambda_j != -lambda_1"
                    tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-tr_radius**2)
                    s=s + tao_lower * d 

                    print ('hard case resolved outside')
                    return s




    # compute final step
    B = H + lambda_j * np.eye(H.shape[0], H.shape[1])
    # 1 Factorize B
    L = np.linalg.cholesky(B)
    # 2 Solve LL^Ts=-g
    Li = np.linalg.inv(L)
    s = - np.dot(np.dot(Li.T, Li), grad)
    #print (i,' exact solver iterations')
    return (s,lambda_j)




def solve_quadratic_equation(pc, pn, tr_radius):
    # solves ax^2+bx+c=0
    a = np.dot(pn - pc, pn - pc)
    b = 2 * np.dot(pc, pn - pc)
    c = np.dot(pc, pc) - tr_radius ** 2
    sqrt_discriminant = sqrt(b * b - 4 * a * c)
    t_lower = (-b - sqrt_discriminant) / (2 * a)
    t_upper = (-b + sqrt_discriminant) / (2 * a)
    return t_lower, t_upper
