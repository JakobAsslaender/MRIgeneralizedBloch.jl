using Revise
using QuadGK
using DifferentialEquations
using SpecialFunctions
using ApproxFun
using StaticArrays
using LinearAlgebra
using LsqFit
using Optim
using BenchmarkTools
using Printf
using Plots
plotlyjs(ticks=:native)
theme(:lime);

using MT_generalizedBloch

## define lineshapes
G(τ) = quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8),   0, sqrt(1 / 3), 1.0)[1]
G_a = G(Fun(identity, 0..100))

## Solve full IDE problem
TRF = 500e-6
ω1 = π / TRF
m0s = .1
R1 = 1.0
R2f = 15
T2s = 10e-6
Rx = 70
u0 = [0,0,1 - m0s,m0s,1] # initial z-magnetization
h(p, t; idxs=nothing) = typeof(idxs) <: Number ?   0 : zeros(5)

sol_cw_full = solve(DDEProblem(MT_generalizedBloch.gBloch_Hamiltonian_ApproxFun!, u0, h, (  0, TRF), (ω1, 1, 0, m0s, R1, R2f, T2s, Rx, G_a)), MethodOfSteps(DP8()))

##
function gBloch_Hamiltonian_ApproxFun!(du, u, h, p::NTuple{3,Any}, t)
    ωy, T2s, g = p
    du[1] = -ωy^2 * quadgk(x -> g((t - x) / T2s) * h(p, x)[1],   0, t)[1]
end

function Linear_Hamiltonian_Matrix(ωy, T, R2s)
    u = @SMatrix [
                  -R2s * T  ωy * T;
                   -ωy * T       0]
end

S = Chebyshev((0.9 * TRF / 15e-6)..(1.1 * TRF / 5e-6)) * Chebyshev((0.9 * ω1 * 5e-6)..(1.1 * ω1 * 15e-5))
_points = points(S, 10^3) 
# fDDE = (xy) -> solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, [1.0], h, (  0, xy[1]), (α, 1.0, G_a)), MethodOfSteps(DP8()))[end][1]
T = TRF
function calc_R2s(xy)
    τ = xy[1]
    α = xy[2]
    
    z = solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, [1.0], h, (0, τ), (α, 1.0, G_a)), MethodOfSteps(DP8()))[end][1]

    function f!(F, ρ)
        s = sqrt(Complex((ρ[1]*τ)^2 - 4 * (τ*α)^2))
        x = exp(-ρ[1]*τ/2) * (cosh(1/2 * s) + (ρ[1]*τ * sinh(1/2 * s)) / s)
        F[1] = real(x) - z
    end
    function j!(J, ρ)
        s = sqrt(Complex((ρ[1]*τ)^2 - 4 * (τ*α)^2))
        J[1] = (2 * exp(-ρ[1]*τ/2) * α^2 * (s * cosh(1/2 * s) - 2 * sinh(1/2 * s) )) / s^3
    end

    sol = nlsolve(f!, j!, [1.0])
    sol.zero[1]
end

calc_R2s(_points[1])
@benchmark calc_R2s($_points[1])

f_p = similar(_points, Float64)
Threads.@threds for i in eachindex(f_p)
    f_p[i] = calc_R2s(_points[i])
end    
fapprox = Fun(S, transform(S, f_p))
fapprox = calc_R2s(Fun(identity,S))

Fun((x,y)->calc_R2s([x,y]),S)
Fun(identity,0..100)    
## 
function Linear_Hamiltonian_Matrix(ωy, B1, ωz, T, m0s, R1, R2f, Rx, R2s)
    m0f = 1 - m0s
    @SMatrix [
             -R2f * T   -ωz * T           B1 * ωy * T             0                     0             0;
               ωz * T  -R2f * T                     0             0                     0             0;
         -B1 * ωy * T         0  (-R1 - Rx * m0s) * T             0          Rx * m0f * T  R1 * m0f * T;
                    0         0                     0      -R2s * T           B1 * ωy * T             0;
                    0         0          Rx * m0s * T  -B1 * ωy * T  (-R1 - Rx * m0f) * T  R1 * m0s * T;
                    0         0                     0             0                     0             0]
end
u0 = [0,0,1 - m0s,0,m0s,1] # initial z-magnetization

##
R2s = fapprox([TRF / T2s, ω1 * T2s]) / T2s
t = 0:1e-5:TRF
M_appx = zeros(length(t), 6)
u0 = [0,0,1 - m0s,0,m0s,1] # initial z-magnetization
for i in eachindex(t)
    M_appx[i,:] = exp(Linear_Hamiltonian_Matrix(ω1, 1, 0, t[i], m0s, R1, R2f, Rx, R2s)) * u0
end

plot(t, (hcat(sol_cw_full(t).u...)')[:,end-1], ticks=:native, label="full IDE solution", legend=:none)
plot!(t, M_appx[:,end-1], ticks=:native, label="linearized solution")

