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
ω0 = 0 # Graham's spectral model is only implemented for on-resonance pulses
m0s = 0.2 * rand()
m0f = 1 - m0s
R1 = 0.7 + 0.6 * rand()
R2f = 1 / (40e-3 + 100e-3 * rand())
T2s = 5e-6 + 10e-6 * rand()
R2s = 1 / T2s
Rx = 100 * rand()

ms = m0s * rand()
mf = (1 - m0s) * rand()
ϑ = rand() * π / 2
φ = rand() * 2π

## Solve Graham's solution
p = (ω1, B1, ω0, TRF, m0s, R1, R2f, T2s, Rx)
alg = Tsit5()
m0 = [mf * sin(ϑ) * cos(φ), mf * sin(ϑ) * sin(φ), mf * cos(ϑ), ms, 1]

sol = solve(
    ODEProblem(MRIgeneralizedBloch.apply_hamiltonian_graham_superlorentzian!, m0, (0.0, TRF), p),
    alg,
)

u_Graham = sol[end]

## Solve generalized Bloch-McConnell with super-Lorentzian lineshape
mfun(p, t; idxs = nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5)

p = (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, greens_superlorentzian)
alg = MethodOfSteps(DP8())
sol = solve(
    DDEProblem(MRIgeneralizedBloch.apply_hamiltonian_gbloch!, m0, mfun, (0.0, TRF), p),
    alg,
)

u_gBloch = sol[end]

## Test!
max_error = 1e-2
@test u_gBloch[1] ≈ u_Graham[1] atol = max_error
@test u_gBloch[2] ≈ u_Graham[2] atol = max_error
@test u_gBloch[3] ≈ u_Graham[3] atol = max_error
@test u_gBloch[4] ≈ u_Graham[4] atol = max_error
@test u_gBloch[5] ≈ u_Graham[5] atol = max_error
