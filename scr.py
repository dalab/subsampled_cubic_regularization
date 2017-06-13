#######################################
### Subsampled Cubic Regularization ###
#######################################

# Authors: Jonas Kohler and Aurelien Lucchi, 2017

from math import sqrt, ceil, log, isnan
from datetime import datetime
import numpy as np

def SCR(w, loss, gradient, Hv=None, hessian=None, X=None, Y=None, opt=None,**kwargs):

    """
    Minimize a continous, unconstrained function using the Adaptive Cubic Regularization method.

    References
    ----------
    Cartis, C., Gould, N. I., & Toint, P. L. (2011). Adaptive cubic regularisation methods for unconstrained optimization. Part I: motivation, convergence and numerical results. Mathematical Programming, 127(2), 245-295.
    Chicago 

    Conn, A. R., Gould, N. I., & Toint, P. L. (2000). Trust region methods. Society for Industrial and Applied Mathematics.

    Kohler, J. M., & Lucchi, A. (2017). Sub-sampled Cubic Regularization for Non-convex Optimization. arXiv preprint arXiv:1705.05933.


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
    print ('--- Subsampled Cubic Regularization ---\n')

    ### Set Parameters ###

    if X is None:
        n=1 
        d=1
    else:
        n = X.shape[0] 
        d = X.shape[1] 

    # Basics
    eta_1 = opt.get('success_treshold',0.1)
    eta_2 = opt.get('very_success_treshold',0.9)
    gamma_1 = opt.get('penalty_increase_multiplier',2.)
    gamma_2 = opt.get('penalty_derease_multiplier',2.)
    assert (gamma_1 >= 1 and gamma_2 >= 1), "penalty update parameters must be greater or equal to 1"

    sigma = opt.get('initial_penalty_parameter',1.)
    grad_tol = opt.get('grad_tol',1e-6)
    n_iterations = opt.get('n_iterations',100)   

    # Subproblem
    subproblem_solver= opt.get('subproblem_solver_SCR','cauchy_point')
    solve_each_i_th_krylov_space=opt.get('solve_each_i-th_krylov_space',1)
    krylov_tol=opt.get('krylov_tol',1e-1)
    exact_tol=opt.get('exact_tol',1e-1)
    keep_Q_matrix_in_memory=opt.get('keep_Q_matrix_in_memory',True)

    # Sampling
    Hessian_sampling_flag=opt.get('Hessian_sampling', False)
    gradient_sampling_flag=opt.get('gradient_sampling', False)

    if gradient_sampling_flag==True or Hessian_sampling_flag==True:
        assert (X is not None and Y is not None), "Subsampling is only possible if data is passsed, i.e. X and Y may not be none"

    initial_sample_size_Hessian=opt.get('initial_sample_size_Hessian',0.05)
    initial_sample_size_gradient=opt.get('initial_sample_size_gradient',0.05)
    sample_scaling_Hessian=opt.get('sample_scaling_Hessian',1)
    sample_scaling_gradient=opt.get('sample_scaling_gradient',1)
    unsuccessful_sample_scaling=opt.get('unsuccessful_sample_scaling',1.25)
    sampling_scheme=opt.get('sampling_scheme', 'adaptive')
    if Hessian_sampling_flag==False and gradient_sampling_flag ==False:
        sampling_scheme=None
   
    print ("- Subproblem_solver:", subproblem_solver)
    print("- Hessian_sampling:" , Hessian_sampling_flag)
    print("- Gradient_sampling:", gradient_sampling_flag)
    print("- Sampling_scheme:" , sampling_scheme,"\n")

    ### -> no opt call after here!!
    k=0
    n_samples_seen=0

    lambda_k=0
    successful_flag=False

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

    # compute exponential growth constant such that full sample size is reached in n_iterations
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

        elif sampling_scheme=='adaptive':
            if i==0:
                sample_size_Hessian=Hessian_sampling_flag*int(initial_sample_size_Hessian*n)+(1-Hessian_sampling_flag)*n
                sample_size_gradient=gradient_sampling_flag*int(initial_sample_size_gradient*n)+(1-gradient_sampling_flag)*n
            else:
                #adjust sampling constant c such that the first step would have given a sample size of initial_sample_size
                if i==1:
                    c_Hessian=(initial_sample_size_Hessian*n*sn**2)/log(d)
                    c_gradient=(initial_sample_size_gradient*n*sn**4)/log(d)
                if successful_flag==False:
                    sample_size_Hessian=Hessian_sampling_flag*min(n,int(sample_size_Hessian*unsuccessful_sample_scaling)) + (1-Hessian_sampling_flag)*n
                    sample_size_gradient=gradient_sampling_flag*min(n,int(sample_size_gradient*unsuccessful_sample_scaling)) +(1-gradient_sampling_flag)*n
                else:
                    sample_size_Hessian=Hessian_sampling_flag*min(n,int(max((c_Hessian*log(d)/(sn**2)*sample_scaling_Hessian),initial_sample_size_Hessian*n))) + (1-Hessian_sampling_flag)*n            
                    sample_size_gradient=gradient_sampling_flag*min(n,int(max((c_gradient*log(d)/(sn**4)*sample_scaling_gradient),initial_sample_size_gradient*n))) + (1-gradient_sampling_flag)*n
        else:
            sample_size_Hessian=n
            sample_size_gradient=n

        ## b) draw batches ##
        if sample_size_Hessian <n:
            int_idx_Hessian=np.random.permutation(n)[0:sample_size_Hessian]
            bool_idx_Hessian = np.zeros(n,dtype=bool)
            bool_idx_Hessian[int_idx_Hessian]=True
            _X=np.zeros((sample_size_Hessian,d))
            _X=np.compress(bool_idx_Hessian,X,axis=0)
            _Y=np.compress(bool_idx_Hessian,Y,axis=0)

        else: 
            _X=X
            _Y=Y

        if sample_size_gradient < n:
            int_idx_gradient=np.random.permutation(n)[0:sample_size_gradient]
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
        # a) recompute gradient either because of accepted step or because of re-sampling
        if gradient_sampling_flag==True or successful_flag==True:
            grad = gradient(w,_X2, _Y2, **kwargs)  
            grad_norm =np.linalg.norm(grad)  
            if grad_norm < grad_tol:
                break

        # b) call subproblem solver
        (s,lambda_k) = solve_ARC_subproblem(grad,Hv,hessian,sigma, _X, _Y, w, successful_flag, lambda_k,subproblem_solver,
            exact_tol,krylov_tol,solve_each_i_th_krylov_space,keep_Q_matrix_in_memory,**kwargs)
        sn=np.linalg.norm(s)

        #### III: Regularization Update #####
        previous_f = loss(w, X, Y, **kwargs)
        current_f = loss(w + s, X, Y,**kwargs)

        function_decrease = previous_f - current_f
        Hs=Hv(w, _X, _Y, s,**kwargs)
        model_decrease=-(np.dot(grad, s) + 0.5 * np.dot(s, Hs)+1/3*sigma*sn**3)
       
        rho = function_decrease / model_decrease
        assert (model_decrease >=0), 'negative model decrease. This should not have happened'

        # Update w if step s is successful
        if rho >= eta_1:
            w = w + s
            _loss=current_f
            successful_flag=True
        else:
            _loss=previous_f                        

        n_samples_seen += n_samples_per_step

        #Update penalty parameter
        if rho >= eta_2:
            sigma=max(sigma/gamma_2,1e-16)
            #alternative (Cartis et al. 2011): sigma= max(min(grad_norm,sigma),np.nextafter(0,1)) 
        
        elif rho < eta_1:
            sigma = gamma_1*sigma
            successful_flag=False   
            print ('unscuccesful iteration')

        ### IV: Save Iteration Information  ###
        _timing=timing
        timing=(datetime.now() - start).total_seconds()
        print ('Iteration ' + str(i) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(
            grad_norm), 'time= ', round(timing-_timing,3), 'penalty=', sigma, 'stepnorm=', sn, 'Samples Hessian=', sample_size_Hessian,'samples Gradient=', sample_size_gradient,"\n")

        timings_collector.append(timing)
        samples_collector.append(n_samples_seen)

        loss_collector.append(_loss)

        k += 1
    return w,timings_collector,loss_collector, samples_collector

def solve_ARC_subproblem(grad,Hv,hessian, sigma, X, Y, w, successful_flag,lambda_k,subproblem_solver,exact_tol,krylov_tol,solve_each_i_th_krylov_space, keep_Q_matrix_in_memory, **kwargs):
    

    if subproblem_solver == 'cauchy_point':
        #min m(-a*grad) leads to finding the root of a quadratic polynominal
        
        Hg=Hv(w, X, Y,grad,**kwargs)
        gHg=np.dot(grad,Hg)
        a=sigma*np.linalg.norm(grad)**3
        b=gHg
        c=-np.dot(grad,grad)
        (alpha_l,alpha_h)=mitternachtsformel(a,b,c)
        alpha=alpha_h
        s=-alpha*grad

        return (s,0)
   
    elif subproblem_solver=='exact':  
        H =hessian(w,X, Y, w, **kwargs) #would be cool to memoize this. 
        (s, lambda_k) = exact_ARC_suproblem_solver(grad, H, sigma, exact_tol,successful_flag,lambda_k)
       
        return (s,lambda_k)

    elif subproblem_solver=='lanczos':   
        y=grad
        grad_norm=np.linalg.norm(grad)
        gamma_k_next=grad_norm
        delta=[] 
        gamma=[] # save for cheaper reconstruction of Q

        dimensionality = len(w)
        if keep_Q_matrix_in_memory: 
            q_list=[]    

        k=0
        T = np.zeros((1, 1)) #Building up tri-diagonal matrix T

        while True:
            if gamma_k_next==0: #From T 7.5.16 u_k was the minimizer of m_k. But it was not accepted. Thus we have to be in the hard case.
                H =hessian(w, X, Y, **kwargs)
                (s, lambda_k) = exact_ARC_suproblem_solver(grad,H, sigma, exact_tol,successful_flag,lambda_k)
                return (s,lambda_k)

            #a) create g
            e_1=np.zeros(k+1)
            e_1[0]=1.0
            g_lanczos=grad_norm*e_1
            #b) generate H
            gamma_k = gamma_k_next
            gamma.append(gamma_k)

            if not k==0:
                q_old=q
            q=y/gamma_k
            
            if keep_Q_matrix_in_memory:
                q_list.append(q)    

            Hq=Hv(w, X, Y, q, **kwargs) #matrix free            
            delta_k=np.dot(q,Hq)
            delta.append(delta_k)
            T_new = np.zeros((k + 1, k + 1))
            if k==0:
                T[k,k]=delta_k
                y=Hq-delta_k*q
            else:
                T_new[0:k,0:k]=T
                T_new[k, k] = delta_k
                T_new[k - 1, k] = gamma_k
                T_new[k, k - 1] = gamma_k
                T = T_new
                y=Hq-delta_k*q-gamma_k*q_old

            gamma_k_next=np.linalg.norm(y)
            #### Solve Subproblem only in each i-th Krylov space          
            if k %(solve_each_i_th_krylov_space) ==0 or (k==dimensionality-1) or gamma_k_next==0:
                (u,lambda_k) = exact_ARC_suproblem_solver(g_lanczos,T, sigma, exact_tol,successful_flag,lambda_k)
                e_k=np.zeros(k+1)
                e_k[k]=1.0
                if np.linalg.norm(y)*abs(np.dot(u,e_k))< min(krylov_tol,np.linalg.norm(u)/max(1, sigma))*grad_norm:
                    break
               
            if k==dimensionality-1: 
                print ('Krylov dimensionality reach full space!')
                break      
                        
            successful_flag=False     


            k=k+1
        
        # Recover Q to compute s
        n=np.size(grad) 
        Q=np.zeros((k + 1,n))  #<--------- since numpy is ROW MAJOR its faster to fill the transpose of Q
        y=grad

        for j in range (0,k+1):
            if keep_Q_matrix_in_memory:
                Q[j,:]=q_list[j]
            else:
                if not j==0:
                    q_re_old=q_re
                q_re=y/gamma[j]
                Q[:,j]=q_re
                Hq=Hv(w, X, Y, q_re, **kwargs) #matrix free

                if j==0:
                    y=Hq-delta[j]*q_re
                elif not j==k:
                    y=Hq-delta[j]*q_re-gamma[j]*q_re_old

        s=np.dot(u,Q)
        del Q
        return (s,lambda_k)
    else: 
        raise ValueError('solver unknown')

def exact_ARC_suproblem_solver(grad,H,sigma, eps_exact,successful_flag,lambda_k):
    from scipy import linalg
    s = np.zeros_like(grad)

    #a) EV Bounds
    gershgorin_l=min([H[i, i] - np.sum(np.abs(H[i, :])) + np.abs(H[i, i]) for i in range(len(H))]) 
    gershgorin_u=max([H[i, i] + np.sum(np.abs(H[i, :])) - np.abs(H[i, i]) for i in range(len(H))]) 
    H_ii_min=min(np.diagonal(H))
    H_max_norm=sqrt(H.shape[0]**2)*np.absolute(H).max() 
    H_fro_norm=np.linalg.norm(H,'fro') 

    #b) solve quadratic equation that comes from combining rayleigh coefficients
    (lambda_l1,lambda_u1)=mitternachtsformel(1,gershgorin_l,-np.linalg.norm(grad)*sigma)
    (lambda_u2,lambda_l2)=mitternachtsformel(1,gershgorin_u,-np.linalg.norm(grad)*sigma)
    
    lambda_lower=max(0,-H_ii_min,lambda_l2)  
    lambda_upper=max(0,lambda_u1)            #0's should not be necessary


    if successful_flag==False and lambda_lower <= lambda_k <= lambda_upper: #reinitialize at previous lambda in case of unscuccesful iterations
        lambda_j=lambda_k
    else:
        lambda_j=np.random.uniform(lambda_lower, lambda_upper)

    no_of_calls=0 
    for v in range(0,50):
        no_of_calls+=1
        lambda_plus_in_N=False
        lambda_in_N=False

        B = H + lambda_j * np.eye(H.shape[0], H.shape[1])
        
        if lambda_lower==lambda_upper==0 or np.any(grad)==0:
            lambda_in_N=True
        else:
            try: # if this succeeds lambda is in L or G.
                # 1 Factorize B
                L = np.linalg.cholesky(B)
                # 2 Solve LL^Ts=-g
                Li = np.linalg.inv(L)
                s = - np.dot(np.dot(Li.T, Li), grad)
                sn = np.linalg.norm(s)
               
                ## 2.1 Terminate <- maybe more elaborated check possible as Conn L 7.3.5 ??? 
                phi_lambda=1./sn -sigma/lambda_j
                if (abs(phi_lambda)<=eps_exact): #
                    break
                # 3 Solve Lw=s
                w = np.dot(Li, s)
                wn = np.linalg.norm(w)

                
                
                ## Step 1: Lambda in L and thus lambda+ in L
                if phi_lambda < 0: 
                    #print ('lambda: ',lambda_j, ' in L')
                    c_lo,c_hi= mitternachtsformel((wn**2/sn**3),1./sn+(wn**2/sn**3)*lambda_j,1./sn*lambda_j-sigma)
                    lambda_plus=lambda_j+c_hi
                    #lambda_plus = lambda_j-(1/sn-sigma/lambda_j)/(wn**2*sn**(-3)+sigma/lambda_j**2) #ARC gives other formulation of same update, faster?

                    lambda_j = lambda_plus
    
                ## Step 2: Lambda in G, hard case possible
                elif phi_lambda>0:
                    #print ('lambda: ',lambda_j, ' in G')
                    #lambda_plus = lambda_j-(1/sn-sigma/lambda_j)/(wn**2*sn**(-3)+sigma/lambda_j**2) #ARC gives other formulation of same update, faster?
                    lambda_upper=lambda_j
                    _lo,c_hi= mitternachtsformel((wn**2/sn**3),1./sn+(wn**2/sn**3)*lambda_j,1./sn*lambda_j-sigma)
                    lambda_plus=lambda_j+c_hi
                    ##Step 2a: If lambda_plus positive factorization succeeds: lambda+ in L (right of -lambda_1 and phi(lambda+) always <0) -> hard case impossible
                    if lambda_plus >0:
                        try:
                            #1 Factorize B
                            B_plus = H + lambda_plus*np.eye(H.shape[0], H.shape[1])
                            L = np.linalg.cholesky(B_plus)
                            lambda_j=lambda_plus
                            #print ('lambda+', lambda_plus, 'in L')
                        except np.linalg.LinAlgError: 
                            lambda_plus_in_N=True
                    
                    ##Step 2b/c: else lambda+ in N, hard case possible
                    if lambda_plus <=0 or lambda_plus_in_N==True:
                        #print ('lambda_plus', lambda_plus, 'in N')
                        lambda_lower=max(lambda_lower,lambda_plus) #reset lower safeguard
                        lambda_j=max(sqrt(lambda_lower*lambda_upper),lambda_lower+0.01*(lambda_upper-lambda_lower))  

                        lambda_lower=np.float32(lambda_lower)
                        lambda_upper=np.float32(lambda_upper)
                        if lambda_lower==lambda_upper:
                                lambda_j = lambda_lower #should be redundant?
                                ew, ev = linalg.eigh(H, eigvals=(0, 0))
                                d = ev[:, 0]
                                dn = np.linalg.norm(d)
                                #note that we would have to recompute s with lambda_j but as lambda_j=-lambda_1 the cholesk facto may fall. lambda_j-1 should only be digits away!
                                tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-lambda_j**2/sigma**2)
                                s = s + tao_lower * d # both, tao_l and tao_up should give a model minimizer!
                                print ('hard case resolved') 
                                break
                    #else: #this only happens for Hg=0 -> s=(0,..,0)->phi=inf -> lambda_plus=nan -> hard case (e.g. at saddle) 
                     #   lambda_in_N = True
                ##Step 3: Lambda in N
            except np.linalg.LinAlgError:
                lambda_in_N = True
        if lambda_in_N == True:
            #print ('lambda: ',lambda_j, ' in N')
            lambda_lower = max(lambda_lower, lambda_j)  # reset lower safeguard
            lambda_j = max(sqrt(lambda_lower * lambda_upper), lambda_lower + 0.01 * (lambda_upper - lambda_lower))  # eq 7.3.1
            #Check Hardcase
            #if (lambda_upper -1e-4 <= lambda_lower <= lambda_upper +1e-4):
            lambda_lower=np.float32(lambda_lower)
            lambda_upper=np.float32(lambda_upper)

            if lambda_lower==lambda_upper:
                lambda_j = lambda_lower #should be redundant?
                ew, ev = linalg.eigh(H, eigvals=(0, 0))
                d = ev[:, 0]
                dn = np.linalg.norm(d)
                if ew >=0: #H is pd and lambda_u=lambda_l=lambda_j=0 (as g=(0,..,0)) So we are done. returns s=(0,..,0)
                    break
                #note that we would have to recompute s with lambda_j but as lambda_j=-lambda_1 the cholesk.fact. may fail. lambda_j-1 should only be digits away!
                sn= np.linalg.norm(s)
                tao_lower, tao_upper = mitternachtsformel(1, 2*np.dot(s,d), np.dot(s,s)-lambda_j**2/sigma**2)
                s = s + tao_lower * d 
                print ('hard case resolved') 
                break 

    return s,lambda_j

############################
### Auxiliary Functions ###
############################
def mitternachtsformel(a,b,c):
    sqrt_discriminant = sqrt(b * b - 4 * a * c)
    t_lower = (-b - sqrt_discriminant) / (2 * a)
    t_upper = (-b + sqrt_discriminant) / (2 * a)
    return t_lower, t_upper
