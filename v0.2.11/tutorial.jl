#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/tutorial.ipynb)

# # Quick Start Tutorial

# The core of generalized Bloch model is implemented in the function [`apply_hamiltonian_gbloch!(∂m∂t, m, mfun, p, t)`](@ref), which calculates the derivative `∂m/∂t` for a given magnetization vector `m` and stores it in-place in the the variable `∂m∂t`. The function interface is written in a way that we can directly feed it into a differential equation solver of the [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) package.

# For this example, we need the following packages:
using MRIgeneralizedBloch
using DifferentialEquations
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb
 
# We simulate the dynamics of a coupled spin system with the following parameters:
m0s = 0.15
R1 = 1 # 1/s
R2f = 15 # 1/s
T2s = 10e-6 # s
Rx = 30; # 1/s

# and the thermal equilibrium of the magnetization `m = [xf; yf; zf; zs; 1]`:
m0 = [0; 0; 1-m0s; m0s; 1];

# during a rectangular RF-pulse with the flip angle and pulse duration
α = π
TRF = 100e-6; # s

# Further, we assume a perfectly calibrated, on-resonant RF-pulse:
B1 = 1
ω0 = 0; # rad/s

# as well as a [super-Lorentzian lineshape](http://dx.doi.org/10.1002/mrm.1910330404). We interpolate the corresponding Green's function to improve performance:
G = interpolate_greens_function(greens_superlorentzian, 0, TRF / T2s);

# The generalized Bloch model is a so-called integro-differential equation where the derivative ``∂m/∂t`` at the time ``t_1`` does not just depend on ``m(t_1)``, but on ``m(t)`` for ``t \in [0, t_1]``. This is solved with a [delay differential equation (DDE) solver](https://diffeq.sciml.ai/stable/tutorials/dde_example/) that stores an interpolated *history function* `mfun(p, t)`, which we use in the [`apply_hamiltonian_gbloch!`](@ref) function to evaluate the integral. This history function has to be initialized with `mfun(p, 0) = m0`. Here, we use a slightly more complicated initialization that allows us to index the history function in [`apply_hamiltonian_gbloch!`](@ref), which improves performance:
mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0;

# With this, we are ready to formulate and solve the differential equation:
param = (α/TRF, B1, ω0, m0s, R1, R2f, T2s, Rx, G) # defined by apply_hamiltonian_gbloch!
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)

# The plot function is implemented for such solution objects and we can plot the solution simply with
p = plot(sol, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ" "1"], xlabel="t [s]", ylabel="m(t)")
#md Main.HTMLPlot(p) #hide

# More details on the interface, including the linear approximation of the generalized Bloch model can found in the following scripts that replicate all simulations, data analyses, and figures of the generalized Bloch paper. 