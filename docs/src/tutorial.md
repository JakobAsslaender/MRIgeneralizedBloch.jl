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

## Balanced Hybrid-State Free Precession Pulse Sequence

```@example HSFP_IDE
using MAT

TR = 3.5e-3
control = matread(normpath(joinpath(pathof(MRIgeneralizedBloch), "../../examples/control_MT_v3p2_TR3p5ms_discretized.mat")))["control"]
TRF = [500e-6; control[1:end - 1,2]]
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
plot(TR*(1:length(α)), [α/π, TRF], layout=(2,1), label=:none, xlabel="t [s]", ylabel=["α/π" "TRF [s]"])
savefig("alphaTRF.html"); nothing #hide
```

```@raw html
<object type="text/html" data="alphaTRF.html" style="width:100%;height:450px;"></object>
```

```@example HSFP_IDE
m_gBloch = calculatesignal_gbloch_ide(α, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s;output=:realmagnetization)

plot( TR*(1:length(TRF)), m_gBloch[:,1] ./ (1 - m0s), label="xf / m0f", xlabel="t [s]", ylabel="m", legend=:topleft)
plot!(TR*(1:length(TRF)), m_gBloch[:,2] ./ (1 - m0s), label="yf / m0f")
plot!(TR*(1:length(TRF)), m_gBloch[:,3] ./ (1 - m0s), label="zf / m0f")
plot!(TR*(1:length(TRF)), m_gBloch[:,4] ./ m0s, label="zs / m0s")
savefig("m_hsfp_ide.html"); nothing #hide
```

```@raw html
<object type="text/html" data="m_hsfp_ide.html" style="width:100%;height:450px;"></object>
```