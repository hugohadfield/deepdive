# deepdive
Numba CUDA parallel nonlinear optimisers

# What?
The majority of optimisers focus on parallelisation within the cost functions or perhaps in the gradient estimation steps.
deepdive takes a different approach, here we are interested in running **many** non linear optimisers in **parallel**.

# Why?
Problems requiring multiple parallel non linear solves are prevelent in sample and test bassed algorithms such as RANSAC variants.
