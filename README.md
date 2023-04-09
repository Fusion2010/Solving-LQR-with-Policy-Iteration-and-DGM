# Full-Marks-Stochastic-Control
Coursework
## Plot instructions:
* Plot for exercise 1.2 will need E1_1.py and E1_2_PLOT.py, one will get the same graph as in the report if he or she runs function plot_exercise_1_2_2()    and plot_exercise_1_2_3() in E1_2_PLOT.py.
* If one wants to test the error of Monte Carlo out of his or her own interests, he could custimze in E1_2_PLOT.py 
* Plot for exercise 2.1: run E2_1.py, E1_1.py is needed
* Plot for exercise 2.2: run E2_2.py, E1_1.py is needed 
* Plot for exercise 3: run E3.py, E1_1.py and E3_MC_fix_control.py is needed 
* Plot for exercise 4: run E4.py, E1_1.py and Network.py is needed 

##Brief introdiction on codes:
## Problem 1
* plot: see plot instructions
* Part (1): The class SolveLQR, is initialised with matrices specifying the linear quadratic regulator, parameter time_grid contains the terminal time and it is also where Ricatti ODE is solved on(see self.solution). All the input and output of functions in SolveLQR is compatible with type and shape required in coursework. 
  * sol__ricatti: sovls Ricatti ODE on time grid 
  * get_value: calculates control problem values at t,x (t and x can be torch tensor with shapes required in coursework)
  * get_controller: calculates optimal control at t,x (t and x can be torch tensor with shapes required in coursework)
* Part (2): We defines a class Monte_Carlo to run the simulation, parameters to specify this class is the same as class SolveLQR. One may ignore all other functions except train_MC: in this train_MC function, the **Objective Function** will be calculated many times and average will be taken by iterations. If one sets 'visualize = True', then absolute error (Averaged Objective Function v.s. Value Function in Part (i)) will be plotted. He can also choose  relative error by setting 'relative = True'

## Prblem 2.
* plot: see plot instructions
* code is a relization of couresework question 2.1 (Supervised learning of value function) and 2.2 (Supervised learning of Markov control)


## Problem 3.
* plot: see plot instructions
* code is a relization couresework question 3 (Deep Galerkin approximation)

## Problem 4.
* plot: see plot instructions
* code is a relization couresework question (Policy iteration with DGM)

