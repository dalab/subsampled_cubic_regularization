###################################
###             SAGA            ###
###################################

# Reference:
# A. Defazio, F. Bach, and S. Lacoste-Julien
# "Saga: A fast incremental gradient method with support for non-strongly convex composite objectives."
# Advances in Neural Information Processing Systems. 2014.


# Authors: Aurelien Lucchi and Jonas Kohler, 2017

from datetime import datetime
import numpy as np

def SAGA(w, loss, gradient, X=None, Y=None, opt=None, **kwargs):

    print ('--- SAGA ---')
    n = X.shape[0]
    d = X.shape[1]
    
    n_epochs = opt.get('n_epochs_saga', 100)
    eta = opt.get('learning_rate_saga',1e-1)
    batch_size = 1

    loss_collector=[]
    timings_collector=[]
    samples_collector=[]

    _loss = loss(w, X, Y, **kwargs) 
    loss_collector.append(_loss)
    timings_collector.append(0)
    samples_collector.append(0)

    start = datetime.now()
    timing=0
    k=0

    # Store past gradients in a table
    mem_gradients = {}
    nGradients = 0 # no gradient stored in mem_gradients at initialization
    avg_mg = np.zeros(d)
    
    # Fill in table
    a = 1.0/n
    bool_idx = np.zeros(n,dtype=bool)
    for i in range(n):
        bool_idx[i]=True
        _X=np.zeros((batch_size,d))
        _X=np.compress(bool_idx,X,axis=0)
        _Y=np.compress(bool_idx,Y,axis=0)
        grad = gradient(w, _X, _Y,**kwargs)
        bool_idx[i]=False

        mem_gradients[i] = grad
        #avg_mg = avg_mg + (grad*a)
        avg_mg = avg_mg + grad
    avg_mg = avg_mg/n
    nGradients = n
    
    n_samples_per_step = 1
    n_steps = int((n_epochs*n)/n_samples_per_step)
    n_samples_seen = 0  # number of samples processed so far
    k = 0

    for i in range(n_steps):

        # I: subsampling
        #int_idx=np.random.permutation(n)[0:batch_size]
        int_idx=np.random.randint(0, high=n, size=1)        
        bool_idx = np.zeros(n,dtype=bool)
        bool_idx[int_idx]=True
        idx = int_idx[0]

        _X=np.zeros((batch_size,d))
        _X=np.compress(bool_idx,X,axis=0)
        _Y=np.compress(bool_idx,Y,axis=0)


        # II: compute step
        grad = gradient(w, _X, _Y,**kwargs)
        n_samples_seen += batch_size
        
        if (n_samples_seen >= n*k)  == True:

            _timing=timing
            timing=(datetime.now() - start).total_seconds()
            
            _loss = loss(w, X, Y, **kwargs)
            
            print ('Epoch ' + str(k) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(np.linalg.norm(grad)), 'time=',round(timing-_timing,3))

            timings_collector.append(timing)
            samples_collector.append((i+1)*batch_size)
            loss_collector.append(_loss)
            k+=1

            
        # Parameter update
        if idx in mem_gradients:
            w = w - eta*(grad - mem_gradients[idx] + avg_mg) # SAGA step
        else:
            w = w - eta*grad # SGD step
                    
        # Update average gradient
        if idx in mem_gradients:
            delta_grad = grad - mem_gradients[idx]
            a = 1.0/nGradients
            avg_mg = avg_mg + (delta_grad*a)
        else:
            # Gradient for datapoint idx does not exist yet
            nGradients = nGradients + 1 # increment number of gradients
            a = 1.0/nGradients
            b = 1.0 - a
            avg_mg = (avg_mg*b) + (grad*a)
        
        # Sanity check
        #a = 1.0/n
        #avg_mg_2 = np.zeros(d)
        #for i in range(n):
        #    avg_mg_2 = avg_mg_2 + (mem_gradients[i]*a)
        #print('diff = ', np.linalg.norm(avg_mg_2-avg_mg), np.linalg.norm(avg_mg))
        
        # Update memorized gradients
        mem_gradients[idx] = grad
        
    return w, timings_collector, loss_collector, samples_collector
