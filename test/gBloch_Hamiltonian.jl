##
using MRIgeneralizedBloch
using DifferentialEquations
using StaticArrays
using Test

## choose random parameters
α = rand() * π
TRF = 100e-6 + 900e-6 * rand()
ω1 = α / TRF
B1 = 0.7 + 0.6 * rand()
ω0 = 1000 * rand() - 500
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

## Solve generalized Bloch-McConnell with Lorentzian lineshape
h(p, t; idxs = nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5)

p = (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, Greens_Lorentzian)
alg = MethodOfSteps(DP8())
u0 = [mf * sin(ϑ) * cos(φ), mf * sin(ϑ) * sin(φ), mf * cos(ϑ), ms, 1]

sol = solve(
    DDEProblem(MRIgeneralizedBloch.gBloch_Hamiltonian!, u0, h, (0.0, TRF), p),
    alg,
)

u_gBloch = sol[end]

## Solve original Bloch-McConnell with Lorentzian lineshape
H = @SMatrix [
    -R2f  -ω0      B1*ω1      0          0      0
      ω0 -R2f          0      0          0      0
  -B1*ω1    0 -R1-Rx*m0s      0     Rx*m0f R1*m0f
       0    0          0   -R2s      B1*ω1      0
       0    0     Rx*m0s -B1*ω1 -R1-Rx*m0f R1*m0s
       0    0          0      0          0      0
]

u0 = [mf * sin(ϑ) * cos(φ), mf * sin(ϑ) * sin(φ), mf * cos(ϑ), 0, ms, 1]
u_Bloch = exp(H * TRF) * u0

## Test!
max_error = 1e-3
@test u_gBloch[1] ≈ u_Bloch[1] atol = max_error
@test u_gBloch[2] ≈ u_Bloch[2] atol = max_error
@test u_gBloch[3] ≈ u_Bloch[3] atol = max_error
@test u_gBloch[4] ≈ u_Bloch[5] atol = max_error
@test u_gBloch[5] ≈ u_Bloch[6] atol = max_error