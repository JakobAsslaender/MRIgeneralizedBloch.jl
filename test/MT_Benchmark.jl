import Cubature
using DifferentialEquations
using ApproxFun
using BenchmarkTools
using Profile
using Plots
plotlyjs(ticks=:native)
theme(:lime);

include("../src/MT_Hamiltonians.jl")

## set parameters
ω1 = π / 500e-6
ω0 = 0.0
m0s = 0.15
R1 = 1.0
R2f = 1 / 65e-3
T2s = 10e-6
Rx = 30.0
TRF = 500e-6
tspan = (0.0, TRF)
h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(30)
alg = MethodOfSteps(Tsit5())
N = 10.0


## baseline IDE solution
u0 = [0.5 * (1 - m0s), 0.0, 0.5 * (1 - m0s), m0s, 1.0]
@benchmark solve(DDEProblem(gBloch_Hamiltonian!, u0, h, tspan, (ω1, ω0, m0s, R1, R2f, T2s, Rx, N)), alg)

##
g = (τ) -> quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8), 0.0, 1.0)[1]
x = Fun(identity, 0..100)
ga = g(x)
@benchmark solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h, tspan, (ω1, ω0, m0s, R1, R2f, T2s, Rx, ga)), alg)

##
@benchmark solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, 3e-3), (ω0, m0s, R1, R2f, Rx)), Tsit5())

## Benchmark gradients
dg_oT2 = (τ) -> quadgk(ct -> exp(- τ^2 * (3.0 * ct^2 - 1)^2 / 8.0) * (τ^2 * (3.0 * ct^2 - 1)^2 / 4.0), 0.0, 1.0)[1]
dg_oT2_a = dg_oT2(x)

u0 = zeros(5 * (length(grad_list)+1))
u0[1] = 0.5 * (1 - m0s)
u0[3] = 0.5 * (1 - m0s)
u0[4] = m0s
u0[5] = 1.0

@benchmark solve(DDEProblem(gBloch_Hamiltonian_Gradient_ApproxFun!, u0, h, tspan, (ω1, ω0, m0s, R1, R2f, T2s, Rx, ga, dg_oT2_a)), alg, save_everystep=false)

