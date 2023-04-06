# Full-Marks-Stochastic-Control
Coursework

## Problem 1
* Part (1): The class SolveLQR, is initialised with matrices specifying the linear quadratic regulator, parameter time_grid contains the terminal time and it is also where Ricatti ODE is solved on(see self.solution). All the input and output of functions in SolveLQR is compatible with type and shape required in coursework. 
  * sol__ricatti: sovls Ricatti ODE on time grid 
  * get_value: calculates control problem values at t,x (t and x can be torch tensor with shapes required in coursework)
  * get_controller: calculates optimal control at t,x (t and x can be torch tensor with shapes required in coursework)
* Part (2): we defines a class Monte_Carlo to run the simulation, parameters to specify this class is the same as class SolveLQR. One may ignore all other functions except train_MC: in this train_MC function, the **Objective Function** will be calculated many times and average will be taken by iterations. If one sets 'visualize = True', then relative error (Averaged Objective Function v.s. Value Function in Part (i)) will be plotted. 

## Prblem 2.
* the plot for both 2.1 and 2.2 will appear if one downloads all the files and click bottom run.
* detailed explanations on report


## Problem 3.
* the plot will appear if one downloads all the files and click bottom run.
* detailed explanations on report

## Problem 4.
* detailed explanations on report

