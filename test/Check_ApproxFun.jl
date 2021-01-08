##
import Cubature
import HCubature
using QuadGK
using DifferentialEquations
using BenchmarkTools
using ApproxFun
using Plots
plotlyjs(ticks=:native)
theme(:lime);

include("../src/MT_Hamiltonians.jl")

##
g = (τ) -> quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8), 0.0, 1.0)[1]
τ = 0.0:0.1:100
plot(τ, g.(τ))

##
x = Fun(identity, 0..100)
ga = g(x)
# @benchmark g_a.(τ)
plot!(τ, ga.(τ))

## ApproxFun for derivative (still has to be multiplied with T2s)
dg_oT2 = (τ) -> quadgk(ct -> exp(- τ^2 * (3.0 * ct^2 - 1)^2 / 8.0) * (τ^2 * (3.0 * ct^2 - 1)^2 / 4.0), 0.0, 1.0)[1]
dg_oT2_a = dg_oT2(x)

plot(τ, dg_oT2.(τ))
plot!(τ, dg_oT2_a.(τ))

##
ω1 = π / 500e-6
ω0 = 0.0
m0s = 0.15
R1 = 1.0
R2f = 1 / 65e-3
T2s = 10e-6
Rx = 30.0
TRF = 500e-6
TE = 3.5e-3 / 2 - TRF
h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(30)
alg = MethodOfSteps(Tsit5())
N = Inf

## plot full vs ApproxFun result
u0 = [0.5 * (1 - m0s), 0.0, 0.5 * (1 - m0s), m0s, 1.0]
gBloch_sol_Full = solve(DDEProblem(gBloch_Hamiltonian!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, T2s, Rx, 4, N)), alg)

gBloch_sol_Approx = solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, T2s, Rx, 4, ga)), alg)

plot(gBloch_sol_Full)
plot!(gBloch_sol_Approx)

## cf. benchmarks
@benchmark solve(DDEProblem(gBloch_Hamiltonian!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, T2s, Rx, 4, N)), alg)
@benchmark solve(DDEProblem(gBloch_Hamiltonian!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, T2s, Rx, 4, 10.0)), alg)
@benchmark solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, T2s, Rx, 4, ga)), alg)

## Gradients: plot full vs ApproxFun result
u0 = zeros(30, 1)
u0[1] = 0.5 * (1 - m0s)
u0[3] = 0.5 * (1 - m0s)
u0[4] = m0s
u0[5] = 1.0
gBloch_sol_grad = solve(DDEProblem(gBloch_Hamiltonian_Gradient!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, T2s, Rx, 10.0)), alg)

gBloch_sol_Approx = solve(DDEProblem(gBloch_Hamiltonian_Gradient_ApproxFun!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, T2s, Rx, ga, dg_oT2_a)), alg)

##
t = 0:1e-5:TRF
dxf = similar(t)
dyf = similar(t)
dzf = similar(t)
dzs = similar(t)

dxfa = similar(t)
dyfa = similar(t)
dzfa = similar(t)
dzsa = similar(t)

ip = 5
for i = 1:length(t)
    dxf[i] = gBloch_sol_grad(t[i])[5 * ip + 1]
    dyf[i] = gBloch_sol_grad(t[i])[5 * ip + 2]
    dzf[i] = gBloch_sol_grad(t[i])[5 * ip + 3]
    dzs[i] = gBloch_sol_grad(t[i])[5 * ip + 4]

    dxfa[i] = gBloch_sol_Approx(t[i])[5 * ip + 1]
    dyfa[i] = gBloch_sol_Approx(t[i])[5 * ip + 2]
    dzfa[i] = gBloch_sol_Approx(t[i])[5 * ip + 3]
    dzsa[i] = gBloch_sol_Approx(t[i])[5 * ip + 4]
end
plot(dxf, label="xf full")
plot!(dzf, label="zf full")
plot!(dzs, label="zs full")
plot!(dxfa, label="xf a")
plot!(dzfa, label="zf a")
plot!(dzsa, label="zs a")

## cf. benchmarks
@benchmark solve(DDEProblem(gBloch_Hamiltonian_Gradient!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, T2s, Rx, 10.0)), alg)

@benchmark solve(DDEProblem(gBloch_Hamiltonian_Gradient_ApproxFun!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, T2s, Rx, ga, dg_oT2_a)), alg)
