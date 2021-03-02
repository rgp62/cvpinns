# Control Volume Physics-informed Neural Networks

This code implements the Control Volume Physics-informed Neural Networks (CVPINNs) method in [1]. `cvpinns.py` implements the `PDE` class. Objects of this class are constructed with mesh, quadrature rule, and PDE specifications, and provide the `getRES` function for computing CVPINNs residuals. `euler.ipynb` is an example script applying CVPINNs to recover the equation of state from the analytical solution to a Sod shock problem.

Python >= 3.5  
numpy  
scipy  
matplotlib  
toolz  
tensorflow >= 2.2  

SAND No: SAND2021-2386 O

[1] R. G. Patel, I. Manickam, N. A. Trask, M. A. Wood, M. Lee, I. Tomas, E. C. Cyr. Thermodynamically consistent physics-informed neural networks for hyperbolic systems. arXiv preprint arXiv:2012.05343, 2020.
