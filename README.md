# subsampled_cubic_regularization
Source code for "Sub-sampled Cubic Regularization for Non-convex Optimization", JM Kohler, A Lucchi, ICML 2017. https://arxiv.org/abs/1705.05933

**Description**

Python implementations of regularized Newton's methods for unconstrained continous (non- and convex) optimization. In particular we implement 

a) The **Trust Region** framework as presented in Conn et al. 2000

b) The **Adaptive Cubic Regularization (ARC)** framework as presented in Cartis et al. 2011

c) The **Subsampled Cubic Regularization (SCR)** method presented in Kohler and Lucchi, 2017


You can pass any kind of continous objective (and derivatives) to these methods and choose between different common solvers for the quadratic models that are minimized in each step. In particular, a **Krylov sub-space minimization** routine based on the Lanczos process is available. This routine allows to escape *strict* saddle points efficiently. 

<img src="https://picload.org/image/ricdogcr/intro.png" width="400"/>

Furthermore, these methods possess the best-known worst case iteration complexity bounds. Details regarding the **convergence analysis** and implementation of a sub-sampled cubic regularization method can be found in **our ICML paper**: https://arxiv.org/abs/1705.05933

<img src="https://picload.org/image/ricdodwr/table.png" width="600"/>


Finally, for empirical risk minimization and similar objective involving loss function over datapoints, we offer the possibility of **sub-sampling datapoints** in each iteration according to different schemes.


<img src="https://picload.org/image/ricdccpa/samples.png" width="600"/>
