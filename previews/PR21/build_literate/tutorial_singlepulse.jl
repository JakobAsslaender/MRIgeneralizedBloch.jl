using MRIgeneralizedBloch
using DifferentialEquations
using SpecialFunctions
using QuadGK
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

m0s = 0.15
R1f = 0.5 # 1/s
R2f = 13 # 1/s
R1s = 3 # 1/s
T2s = 12e-6 # s
Rx = 17; # 1/s

α = π
TRF = 200e-6; # s

B1 = 1
ω0 = 0; # rad/s

G = interpolate_greens_function(greens_superlorentzian, 0, TRF / T2s);

ω1 = α/TRF;

m0 = [m0s; 1];

mfun(p, t) = m0;

param = (ω1, B1, ω0, R1s, T2s, G) # defined by apply_hamiltonian_gbloch!
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)

p = plot(sol, xlabel="t [s]", ylabel="zˢ(t)", idxs=1, labels=:none)

m0 = [0; 0; 1-m0s; m0s; 1];

param = (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G);

prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)
p = plot(sol, xlabel="t [s]", ylabel="m(t)", idxs=1:4, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ"])

NSideLobes = 1
f_ω1(t) = sinc(2(NSideLobes+1) * t/TRF - (NSideLobes+1)) * α / (sinint((NSideLobes+1)π) * TRF/π / (NSideLobes+1));

tᵢ = 0:1e-6:TRF
p = plot(tᵢ, f_ω1.(tᵢ), xlabel="t [s]", ylabel="ω₁(t)", labels=:none)

quadgk(f_ω1, 0, TRF)[1] / α

m0 = [m0s; 1]
param = (f_ω1, B1, ω0, R1s, T2s, G)
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)
p = plot(sol, xlabel="t [s]", ylabel="zˢ(t)", idxs=1, labels=:none)

param = (f_ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, G)
m0 = [0; 0; 1-m0s; m0s; 1];
prob = DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), param)
sol = solve(prob)
p = plot(sol, xlabel="t [s]", ylabel="m(t)", idxs=1:4, labels=["xᶠ" "yᶠ" "zᶠ" "zˢ"])

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

