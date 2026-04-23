using DifferentialEquations
using Test
using FiniteDifferences
using MRIgeneralizedBloch
using MRIgeneralizedBloch: apply_hamiltonian_freeprecession!, apply_hamiltonian_gbloch!


## set parameters
ω0 = 200
m0s = 0.15
R1f = 0.3
R2f = 1 / 65e-3
R1s = 2
T2s = 10e-6
Rex = 30
TE = 3.5e-3
mfun = (p, t; idxs=nothing) -> typeof(idxs) <: Number ? 0.0 : zeros(30)
alg = Tsit5()

t = 0 : 1e-5 : TE

## random initialization
M0 = 0.85
ms = m0s * 0.3
mf = (1 - m0s) * 0.5
ϑ = 0.7 * π / 2
φ = 0.4 * 2π
m0 = [mf * sin(ϑ) * cos(φ), mf * sin(ϑ) * sin(φ), mf * cos(ϑ), ms, M0]


## Analytical gradients
grad_list = (grad_M0(), grad_B1(), grad_ω0(), grad_m0s(), grad_R1f(), grad_R2f(), grad_Rex(), grad_R1s(), grad_T2s())
m0_grad = [m0; zeros(5 * length(grad_list))]
FP_sol_grad = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0_grad, (0, TE), (ω0, m0s, R1f, R2f, Rex, R1s, grad_list)), alg)

d_M0_ = [FP_sol_grad(t[i])[j] for j ∈  6:9,  i ∈ eachindex(t)]
d_B1_ = [FP_sol_grad(t[i])[j] for j ∈ 11:14, i ∈ eachindex(t)]
d_ω0_ = [FP_sol_grad(t[i])[j] for j ∈ 16:19, i ∈ eachindex(t)]
d_m0s = [FP_sol_grad(t[i])[j] for j ∈ 21:24, i ∈ eachindex(t)]
d_R1f = [FP_sol_grad(t[i])[j] for j ∈ 26:29, i ∈ eachindex(t)]
d_R2f = [FP_sol_grad(t[i])[j] for j ∈ 31:34, i ∈ eachindex(t)]
d_Rex = [FP_sol_grad(t[i])[j] for j ∈ 36:39, i ∈ eachindex(t)]
d_R1s = [FP_sol_grad(t[i])[j] for j ∈ 41:44, i ∈ eachindex(t)]
d_T2s = [FP_sol_grad(t[i])[j] for j ∈ 46:49, i ∈ eachindex(t)]


## FD derivative
fd = central_fdm(5, 1; factor=1e6)

_f = function (p)
    M0, ω0, m0s, R1f, R2f, Rex, R1s = p
    sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, [m0[1], m0[2], m0[3], m0[4], M0], (0, TE), (ω0, m0s, R1f, R2f, Rex, R1s)), alg)
    return reduce(hcat, [sol(t[i]) for i ∈ eachindex(t)])
end
_fd = jacobian(fd, _f, (M0, ω0, m0s, R1f, R2f, Rex, R1s))[1]
_fd = reshape(_fd, 5, length(t), :)

@test d_M0_ ≈ _fd[1:4, :, 1] rtol = 1e-3
@test d_ω0_ ≈ _fd[1:4, :, 2] rtol = 1e-3
@test d_m0s ≈ _fd[1:4, :, 3] rtol = 1e-3
@test d_R1f ≈ _fd[1:4, :, 4] rtol = 1e-3
@test d_R2f ≈ _fd[1:4, :, 5] rtol = 1e-3
@test d_Rex ≈ _fd[1:4, :, 6] rtol = 1e-3
@test d_R1s ≈ _fd[1:4, :, 7] rtol = 1e-3
@test d_T2s ≈ zeros(size(d_T2s)) atol = 1e-6
@test d_B1_ ≈ zeros(size(d_B1_)) atol = 1e-6

## ##########################################################
## R1a (separate solve since R1f=R1s=R1a)
#############################################################
R1a = 1
grad_list = (grad_R1a(),)
FP_sol_grad = solve(ODEProblem(apply_hamiltonian_freeprecession!, [m0; zeros(5)], (0, TE), (ω0, m0s, R1a, R2f, Rex, R1a, grad_list)), alg)
d_R1a = [FP_sol_grad(t[i])[j] for j ∈ 6:9, i ∈ eachindex(t)]

_f = function (R1a)
    sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, m0[1:5], (0, TE), (ω0, m0s, R1a, R2f, Rex, R1a)), alg)
    return reduce(hcat, [sol(t[i]) for i ∈ eachindex(t)])
end
_fd = fd(_f, R1a)
d_R1a_fd = _fd[1:4, :]

@test d_R1a ≈ d_R1a_fd rtol = 1e-3
