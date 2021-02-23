# newton-gd

This code implements the Control Volume Physics-informed Neural Networks (CVPINNs) method in [1]. Section 2 implements the ‘PDE‘ class. Objects of this class are constructed with mesh, quadrature rule, and PDE specifications, and provide the ‘getRES‘ function for computing CVPINNs residuals. Section 1 is an example script applying CVPINNs to recover the equation of state from the analytical solution to a Sod shock problem.

Python >= 3.5
numpy
scipy
matplotlib
toolz
tensorflow >= 2.2
tensorflow-datasets

SAND No: 

[1] R. G. Patel, N. A. Trask, M. A. Gulian, and E. C. Cyr. A block coordinate descent optimizer for classification problems exploiting convexity. arXiv preprint arXiv:2006.10123, 2020.
