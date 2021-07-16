# Quick Start Tutorial
## Single RF-Pulse

The core of generalized Bloch model is implemented in the function [`apply_hamiltonian_gbloch!`](@ref), which calculates the derivative `∂m/∂t` for a given magnetization vector `m` and stores it in-place in the the variable `∂m∂t`. The function interface is written in a way that we can directly feed it into the a differential equation solver of [DifferentialEquations.jl](https://diffeq.sciml.ai/stable/): 

```@example HSFP_IDE
using MRIgeneralizedBloch
using DifferentialEquations
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native) #hide

α = π
TRF = 100e-6
ω1 = α/TRF
B1 = 1
ω0 = 0
m0s = 0.15
R1 = 1
R2f = 15
T2s = 10e-6
Rx = 30
G = interpolate_greens_function(greens_superlorentzian, 0, TRF / T2s)
m0 = [0; 0; 1-m0s; m0s; 1]
m(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0
p = (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, G)
sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, m, (0, TRF), p), MethodOfSteps(DP8()))
plot(sol, labels=["xf" "yf" "zf" "zs" "1"], xlabel="t [s]", ylabel="m(t)")
savefig("m_single_pulse.html"); nothing #hide
```

```@raw html
<object type="text/html" data="m_single_pulse.html" style="width:100%;height:450px;"></object>
```

The generalized Bloch model is a so-called integro-differential equation where the derivative ``∂m/∂t`` at the time ``t_1`` does not just depend on ``m(t_1)``, but on ``m(t)`` for ``t \in [0, t_1]``. This is solved with a [delay differential equation solver](https://diffeq.sciml.ai/stable/tutorials/dde_example/) that stores an interpolated *history function* `m(p, t)` that needs to be initialized with `m(p, 0) = m0`. 

As our own work focuses on using on-resonant RF pulses to excite the free pool and simultaneously saturate the semi-solid pool, we assume the pulses to be on-resonant for the semi-solid pool, i.e., ``ω_0 \ll 1/T_2^s``. If you want to simulate off-resonant RF-pulses with the generalized Bloch model, please contact me and I can incorporate this functionality.
