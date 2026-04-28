using MRIgeneralizedBloch
using Test
using FiniteDifferences
using DelayDiffEq
using DifferentialEquations
using SpecialFunctions
using QuadGK

## set parameters
TRF = 500e-6
α = π

NSideLobes = 1
f_ω1 = (t) -> sinc(2(NSideLobes+1) * t/TRF - (NSideLobes+1)) * α / (sinint((NSideLobes+1)π) * TRF/π / (NSideLobes+1))
@test quadgk(f_ω1, 0, TRF)[1] ≈ α

ω0 = 200
B1 = 0.8
M0 = 0.85
m0s = 0.15
R1f = 0.3
R1s = 2
R2f = 1 / 65e-3
T2s = 10e-6
Rex = 30
m0 = [0.5 * M0 * (1 - m0s), 0, 0.5 * M0 * (1 - m0s), M0 * m0s, M0]

G = interpolate_greens_function(greens_superlorentzian, 0, 100)
dGdT2s = interpolate_greens_function(dG_o_dT2s_x_T2s_superlorentzian, 0, 100)

alg = MethodOfSteps(Tsit5())
t = 0:1e-5:TRF

## Analytical gradients
grad_list = (grad_M0(), grad_B1(), grad_ω0(), grad_m0s(), grad_R1f(), grad_R2f(), grad_Rex(), grad_R1s(), grad_T2s())
m0_grad = [m0; zeros(5 * length(grad_list))]
mfun = (p, t; idxs = nothing) -> typeof(idxs) <: Number ? 0.0 : zeros(length(m0_grad))
gBloch_sol_grad = solve(DDEProblem(apply_hamiltonian_gbloch!, m0_grad, mfun, (0.0, TRF), (f_ω1, B1, ω0, m0s, R1f, R2f, Rex, R1s, T2s, G, dGdT2s, grad_list)), alg)

d_M0_ = [gBloch_sol_grad(t[i])[j] for j ∈  6:9,  i ∈ eachindex(t)]
d_B1_ = [gBloch_sol_grad(t[i])[j] for j ∈ 11:14, i ∈ eachindex(t)]
d_ω0_ = [gBloch_sol_grad(t[i])[j] for j ∈ 16:19, i ∈ eachindex(t)]
d_m0s = [gBloch_sol_grad(t[i])[j] for j ∈ 21:24, i ∈ eachindex(t)]
d_R1f = [gBloch_sol_grad(t[i])[j] for j ∈ 26:29, i ∈ eachindex(t)]
d_R2f = [gBloch_sol_grad(t[i])[j] for j ∈ 31:34, i ∈ eachindex(t)]
d_Rex = [gBloch_sol_grad(t[i])[j] for j ∈ 36:39, i ∈ eachindex(t)]
d_R1s = [gBloch_sol_grad(t[i])[j] for j ∈ 41:44, i ∈ eachindex(t)]
d_T2s = [gBloch_sol_grad(t[i])[j] for j ∈ 46:49, i ∈ eachindex(t)]


## FD derivative
fd = central_fdm(5, 1; factor=1e9)

_f = function (p)
    M0, B1, ω0, m0s, R1f, R2f, Rex, R1s, T2s = p
    ω0 = 1e5ω0
    T2s = 1e-5T2s
    sol = solve(DDEProblem(apply_hamiltonian_gbloch!, [m0[1], m0[2], m0[3], m0[4], M0], mfun, (0, TRF), (f_ω1, B1, ω0, m0s, R1f, R2f, Rex, R1s, T2s, G)), alg)
    return reduce(hcat, [sol(t[i]) for i ∈ eachindex(t)])
end
_fd = jacobian(fd, _f, (M0, B1, 1e-5ω0, m0s, R1f, R2f, Rex, R1s, 1e5T2s))[1]
_fd = reshape(_fd, 5, length(t), :)

@test d_M0_ ≈ _fd[1:4, :, 1]           rtol = 1e-2
@test d_B1_ ≈ _fd[1:4, :, 2]           rtol = 1e-2
@test d_ω0_ ≈ _fd[1:4, :, 3] .* 1e-5   rtol = 1e-2
@test d_m0s ≈ _fd[1:4, :, 4]           rtol = 1e-3
@test d_R1f ≈ _fd[1:4, :, 5]           rtol = 1e-3
@test d_R2f ≈ _fd[1:4, :, 6]           rtol = 1e-3
@test d_Rex ≈ _fd[1:4, :, 7]           rtol = 1e-3
@test d_R1s ≈ _fd[1:4, :, 8]           rtol = 1e-3
@test d_T2s ≈ _fd[1:4, :, 9] .* 1e5    rtol = 1e-3



######################################################################
## apparent R1 (separate solve since R1f=R1s=R1a)
######################################################################
R1a = 1
grad_list = (grad_R1a(),)
mfun_R1a = (p, t; idxs = nothing) -> typeof(idxs) <: Number ? 0.0 : zeros(10)
gBloch_sol_grad = solve(DDEProblem(apply_hamiltonian_gbloch!, [m0; zeros(5)], mfun_R1a, (0.0, TRF), (f_ω1, B1, ω0, m0s, R1a, R2f, Rex, R1a, T2s, G, dGdT2s, grad_list)), alg)
d_R1a = [gBloch_sol_grad(t[i])[j] for j ∈ 6:9, i ∈ eachindex(t)]

_f = function (R1a)
    sol = solve(DDEProblem(apply_hamiltonian_gbloch!, m0, mfun_R1a, (0, TRF), (f_ω1, B1, ω0, m0s, R1a, R2f, Rex, R1a, T2s, G)), alg)
    return reduce(hcat, [sol(t[i]) for i ∈ eachindex(t)])
end
_fd = fd(_f, R1a)
d_R1a_fd = _fd[1:4, :]

@test d_R1a ≈ d_R1a_fd rtol = 1e-3