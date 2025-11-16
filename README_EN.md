# Numerical Methods & Optimization Library

Python implementation of numerical methods and optimization algorithms.

## Languages

- [English](README.md)
- [Русский](README_RU.md)

## Results

### Optimization Methods

<div align="center">
  <img src="images/optimization_multidimensional/blind_search_isosurfaces_3d.png" width="600">
  <br>
  <em>Isosurface visualization in multidimensional space for generalized blind search algorithm</em>
</div>

<div align="center">
  <img src="images/optimization_multidimensional/blind_search_isotropic_function_from_local_extremum_to_global_extremum.png" width="600">
  <br>
  <em>Convergence to global optimum from local minimum on anisotropic function</em>
</div>

### Partial Differential Equations

<div align="center">
  <img src="images/differential_equations/pde_wave_equation.gif" width="600">
  <br>
  <em>Wave equation solution animation using finite difference method</em>
</div>

<div align="center">
  <img src="images/differential_equations/pde_one_dimension_parabolic_mixed_problem_heat_equation.png" width="600">
  <br>
  <em>1D parabolic equation solution (mixed boundary problem)</em>
</div>

<div align="center">
  <img src="images/differential_equations/runge-kutta_euler_adams-runge_adams-euler_comparison.png" width="600">
  <br>
  <em>Comparison of differential equation solving methods</em>
</div>

### Interpolation & Approximation

<div align="center">
  <img src="images/interpolation/cubic_splines.png" width="600">
  <br>
  <em>Data interpolation using cubic splines</em>
</div>

<div align="center">
  <img src="images/approximation/discrete_LSM_least_squares_method.png" width="600">
  <br>
  <em>Data approximation with discrete least squares method</em>
</div>

### Nonlinear Equations Systems

<div align="center">
  <img src="images/system_of_non_linear_equations/newton_simple_iteration.png" width="600">
  <br>
  <em>Nonlinear system solution using Newton's method</em>
</div>

## Project Structure

### Modeling

- Diffusion and transport equation solvers (Roberts, Gaussian model, Euler, UpWind)
- Explicit and implicit schemes for PDEs
- Crank-Nicolson method for semi-empirical equations

### Numerical Methods

- **Approximation**: LSM (discrete and integral)
- **ODEs/PDEs**: Cauchy problems, boundary problems (Thomas algorithm), hyperbolic and parabolic equations, Dirichlet problem for Laplace equation
- **Differentiation**: Runge 2nd order method
- **Nonlinear equations**: Newton, simple iteration, dichotomy
- **Nonlinear systems**: Newton, simple iteration
- **Interpolation**: Lagrange (equidistant/non-equidistant nodes), Newton, cubic splines
- **Integration**: rectangles, trapezoids, Simpson

### Optimization

- **Extremum search**: bisection, Fibonacci, golden section, quadratic/cubic interpolation, scanning
- **Multidimensional optimization**:
  - Gradient method
  - Conjugate gradients
  - Gauss-Seidel
  - Rosenbrock method
  - Pairwise probe (classical, stochastic)
  - Custom pairwise probe with direction batching
  - Random directions
  - Blind search
  - Random penalty search

## Features

- Generalized algorithms for N-dimensional cases
- My modification of pairwise probe: batching N samples with best direction selection
- Algorithms with usage examples + visualization
- 40+ implemented methods
- Pure Python + NumPy/SciPy + Matplotlib

## Usage

```bash
pip install -r requirements.txt
```
