###################################
### Stochastic Gradient Descent ###
###################################

# This SGD implementation assumes a constant stepsize that can be specified as opt['learning_rate_sgd']= ...

# Authors: Jonas Kohler and Aurelien Lucchi, 2017

from datetime import datetime
import numpy as np

def SGD(w, loss, gradient, X=None, Y=None, opt=None, **kwargs):
    print ('--- SGD ---')
    n = X.shape[0]
    d = X.shape[1]
    
    n_epochs = opt.get('n_epochs_sgd', 100)
    eta = opt.get('learning_rate_sgd',1e-1)
    batch_size =int(opt.get('batch_size_sgd',0.01*n))

    n_steps = int((n_epochs * n) / batch_size)
    n_samples_seen = 0  # number of samples processed so far

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

    for i in range(n_steps):

        # I: subsampling
        #int_idx=np.random.permutation(n)[0:batch_size]
        int_idx=np.random.randint(0, high=n, size=batch_size)        

        bool_idx = np.zeros(n,dtype=bool)
        bool_idx[int_idx]=True
        _X=np.zeros((batch_size,d))
        _X=np.compress(bool_idx,X,axis=0)
        _Y=np.compress(bool_idx,Y,axis=0)


        # II: compute step
        grad = gradient(w, _X, _Y,**kwargs)  

        n_samples_seen += batch_size
        
        if (n_samples_seen >= n*k)  == True:
            k+=1

            _timing=timing
            timing=(datetime.now() - start).total_seconds()
            
            _loss = loss(w, X, Y, **kwargs)
            
            print ('Epoch ' + str(k) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(np.linalg.norm(grad)), 'time=',round(timing-_timing,3))

            timings_collector.append(timing)
            samples_collector.append((i+1)*batch_size)
            loss_collector.append(_loss)

        w = w - eta * grad
    return w, timings_collector, loss_collector, samples_collector
