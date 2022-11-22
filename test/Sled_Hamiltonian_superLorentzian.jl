##
using MRIgeneralizedBloch
using DifferentialEquations
using StaticArrays
using Test

## choose random parameters
# this test should only be passed for small α and long TRF
α = rand() * π/2
TRF = 500e-6 + 1000e-6 * rand()
ω1 = α / TRF
B1 = 0.7 + 0.6 * rand()
ω0 = 0
m0s = 0.2 * rand()
m0f = 1 - m0s
R1f = 0.5 + 0.2 * rand()
R1s = 2 + rand()
R2f = 1 / (40e-3 + 100e-3 * rand())
T2s = 5e-6 + 10e-6 * rand()
R2s = 1 / T2s
Rx = 100 * rand()

ms = m0s * rand()
mf = (1 - m0s) * rand()
ϑ = rand() * π / 2
φ = rand() * 2π

## Solve Sled's solution
g = interpolate_greens_function(greens_superlorentzian, 0, TRF/T2s)
p = (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g)
m0 = [mf * sin(ϑ) * cos(φ), mf * sin(ϑ) * sin(φ), mf * cos(ϑ), ms, 1]

sol = solve(ODEProblem(MRIgeneralizedBloch.apply_hamiltonian_sled!, m0, (0, TRF), p))
u_Sled = sol[end]

## Solve generalized Bloch-McConnell with super-Lorentzian lineshape
mfun(p, t; idxs = nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5)
p = (ω1, B1, ω0, m0s, R1f, R2f, Rx, R1s, T2s, g)
sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), p))
u_gBloch = sol[end]

## Test!
@test u_gBloch ≈ u_Sled atol = 1e-2