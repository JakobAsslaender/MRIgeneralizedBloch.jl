using MRIgeneralizedBloch
using DifferentialEquations
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

m0s = 0.15
R1 = 1 # 1/s
R2f = 15 # 1/s
T2s = 10e-6 # s
Rx = 30; # 1/s

m0 = [0; 0; 1-m0s; m0s; 1];

α = π
TRF = 100e-6; # s

B1 = 1
ω0 = 0; # rad/s

G = interpolate_greens_function(greens_superlorentzian, 0, TRF / T2s);

mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? m0[idxs] : m0;

param = (α/TRF, B1, ω0, m0s, R1, R2f, T2s, Rx, G) # defined by apply_hamiltonian_gbloch!
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)

p = plot(sol, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ" "1"], xlabel="t [s]", ylabel="m(t)")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

