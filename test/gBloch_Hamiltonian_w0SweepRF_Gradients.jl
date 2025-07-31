using DifferentialEquations
using SpecialFunctions
using QuadGK
using FiniteDifferences
using Test
using MRIgeneralizedBloch

## set parameters
TRF = 10.24e-3
γ = 267.522e6 # rad/s/T
ω₁ᵐᵃˣ = 13e-6 * γ # rad/s
μ = 5 # rad
β = 674.1 # 1/s

f_ω1 = (t) -> ω₁ᵐᵃˣ .* sech.(β * (t - TRF/2)) # rad/s
f_ω0 = (t) -> -μ * β * tanh.(β * (t - TRF/2)) # rad/s
f_φ  = (t) -> -μ * log(cosh(β * t) - sinh(β*t) * tanh(β*TRF/2))

@test ω₁ᵐᵃˣ /(√μ * β) > 1 # adiabatic condition

B1 = 0.8
m0s = 0.15
R1f = 0.3
R1s = 2
R2f = 1 / 65e-3
T2s = 10e-6
Rex = 30

m0 = [0, 0, 1 - m0s, m0s, 1]
mfun = (p, t; idxs = nothing) -> typeof(idxs) <: Number ? 0.0 : zeros(30)

G = interpolate_greens_function(greens_superlorentzian, 0, TRF/T2s)
dGdT2s = interpolate_greens_function(dG_o_dT2s_x_T2s_superlorentzian, 0, TRF/T2s)

t = 0:1e-5:TRF

## Analytical gradients (using ApproxFun)
grad_list = (grad_B1(), grad_ω0(), grad_m0s(), grad_R1f(), grad_R2f(), grad_Rex(), grad_R1s(), grad_T2s())
gBloch_sol_grad = solve(DDEProblem(apply_hamiltonian_gbloch!, [m0; zeros(5 * (length(grad_list)),1)], mfun, (0, TRF), (f_ω1, B1, f_φ, m0s, R1f, R2f, Rex, R1s, T2s, G, dGdT2s, grad_list)), MethodOfSteps(Tsit5()))

d_B1_ = [gBloch_sol_grad(t[i])[j] for j ∈ 6:9, i ∈ eachindex(t)]
d_ω0_ = [gBloch_sol_grad(t[i])[j] for j ∈ 11:14, i ∈ eachindex(t)]
d_m0s = [gBloch_sol_grad(t[i])[j] for j ∈ 16:19, i ∈ eachindex(t)]
d_R1f = [gBloch_sol_grad(t[i])[j] for j ∈ 21:24, i ∈ eachindex(t)]
d_R2f = [gBloch_sol_grad(t[i])[j] for j ∈ 26:29, i ∈ eachindex(t)]
d_Rex = [gBloch_sol_grad(t[i])[j] for j ∈ 31:34, i ∈ eachindex(t)]
d_R1s = [gBloch_sol_grad(t[i])[j] for j ∈ 36:39, i ∈ eachindex(t)]
d_T2s = [gBloch_sol_grad(t[i])[j] for j ∈ 41:44, i ∈ eachindex(t)]


## FD derivative
fd = central_fdm(5, 1; factor=1e9)

_f = function (p)
    B1, ω0, m0s, R1f, R2f, Rex, R1s, T2s = p
    f_dφ(t) = f_φ(t) + ω0 * t
    T2s = 1e-5T2s
    sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), (f_ω1, B1, f_dφ, m0s, R1f, R2f, Rex, R1s, T2s, G)), MethodOfSteps(Tsit5()))
    return reduce(hcat, [sol(t[i]) for i ∈ eachindex(t)])
end
_fd = jacobian(fd, _f, (B1, 0, m0s, R1f, R2f, Rex, R1s, 1e5T2s))[1]
_fd = reshape(_fd, 5, length(t), :)

d_B1__fd = _fd[1:4, :, 1]
d_ω0__fd = _fd[1:4, :, 2]
d_m0s_fd = _fd[1:4, :, 3]
d_R1f_fd = _fd[1:4, :, 4]
d_R2f_fd = _fd[1:4, :, 5]
d_Rex_fd = _fd[1:4, :, 6]
d_R1s_fd = _fd[1:4, :, 7]
d_T2s_fd = _fd[1:4, :, 8] .* 1e5

@test d_B1_ ≈ d_B1__fd rtol = 1e-2
@test d_ω0_ ≈ d_ω0__fd rtol = 1e-2
@test d_m0s ≈ d_m0s_fd rtol = 1e-3
@test d_R1f ≈ d_R1f_fd rtol = 1e-3
@test d_R2f ≈ d_R2f_fd rtol = 1e-3
@test d_Rex ≈ d_Rex_fd rtol = 1e-3
@test d_R1s ≈ d_R1s_fd rtol = 1e-3
@test d_T2s ≈ d_T2s_fd rtol = 1e-3


## ###################################################################
# apparent R1
######################################################################
R1a = 1
grad_list = (grad_R1a(),)
gBloch_sol_grad = solve(DDEProblem(apply_hamiltonian_gbloch!, [m0; zeros(5 * (length(grad_list)),1)], mfun, (0, TRF), (f_ω1, B1, f_φ, m0s, R1f, R2f, Rex, R1s, T2s, G, dGdT2s, grad_list)), MethodOfSteps(Tsit5()))
d_R1a = [gBloch_sol_grad(t[i])[j] for j ∈ 6:9, i ∈ eachindex(t)]

_f = function (R1a)
    sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun, (0, TRF), (f_ω1, B1, f_φ, m0s, R1a, R2f, Rex, R1a, T2s, G)), MethodOfSteps(Tsit5()))
    return reduce(hcat, [sol(t[i]) for i ∈ eachindex(t)])
end
_fd = fd(_f, R1a)
d_R1a_fd = _fd[1:4, :]

@test d_R1a ≈ d_R1a_fd rtol = 1e-2