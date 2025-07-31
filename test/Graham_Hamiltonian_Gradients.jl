using DifferentialEquations
using Test
using FiniteDifferences
using MRIgeneralizedBloch
using MRIgeneralizedBloch: apply_hamiltonian_graham_superlorentzian!, apply_hamiltonian_linear!
ode_solver = Tsit5()

## set parameters
ω1 = π / 500e-6
ω0 = 200
B1 = 1
m0s = 0.15
R1f = 0.3
R1s = 2
R2f = 1 / 65e-3
T2s = 10e-6
Rex = 30
TRF = 500e-6

m0 = [0.5 * (1 - m0s), 0, 0.5 * (1 - m0s), m0s, 1]
t = 0:1e-5:TRF

## Analytical gradients (using ApproxFun)
grad_list = (grad_B1(), grad_ω0(), grad_m0s(), grad_R1f(), grad_R2f(), grad_Rex(), grad_R1s(), grad_T2s())
Graham_sol_grad = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, [m0; zeros(5 * (length(grad_list)))], (0, TRF), (ω1, B1, ω0, TRF, m0s, R1f, R2f, Rex, R1s, T2s, grad_list)), ode_solver)

d_B1_ = [Graham_sol_grad(t[i])[j] for j ∈ 6:9, i ∈ eachindex(t)]
d_ω0_ = [Graham_sol_grad(t[i])[j] for j ∈ 11:14, i ∈ eachindex(t)]
d_m0s = [Graham_sol_grad(t[i])[j] for j ∈ 16:19, i ∈ eachindex(t)]
d_R1f = [Graham_sol_grad(t[i])[j] for j ∈ 21:24, i ∈ eachindex(t)]
d_R2f = [Graham_sol_grad(t[i])[j] for j ∈ 26:29, i ∈ eachindex(t)]
d_Rex = [Graham_sol_grad(t[i])[j] for j ∈ 31:34, i ∈ eachindex(t)]
d_R1s = [Graham_sol_grad(t[i])[j] for j ∈ 36:39, i ∈ eachindex(t)]
d_T2s = [Graham_sol_grad(t[i])[j] for j ∈ 41:44, i ∈ eachindex(t)]


## FD derivative
fd = central_fdm(5, 1; factor=1e6)

_f = function (p)
    B1, ω0, m0s, R1f, R2f, Rex, R1s, T2s = p
    T2s = 1e-5T2s
    sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, m0, (0, TRF), (ω1, B1, ω0, TRF, m0s, R1f, R2f, Rex, R1s, T2s)), ode_solver)
    return reduce(hcat, [sol(t[i]) for i ∈ eachindex(t)])
end
_fd = jacobian(fd, _f, (B1, ω0, m0s, R1f, R2f, Rex, R1s, 1e5T2s))[1]
_fd = reshape(_fd, 5, length(t), :)

d_B1__fd = _fd[1:4, :, 1]
d_ω0__fd = _fd[1:4, :, 2]
d_m0s_fd = _fd[1:4, :, 3]
d_R1f_fd = _fd[1:4, :, 4]
d_R2f_fd = _fd[1:4, :, 5]
d_Rex_fd = _fd[1:4, :, 6]
d_R1s_fd = _fd[1:4, :, 7]
d_T2s_fd = _fd[1:4, :, 8] .* 1e5

@test d_B1_ ≈ d_B1__fd rtol = 1e-3
@test d_ω0_ ≈ d_ω0__fd rtol = 1e-3
@test d_m0s ≈ d_m0s_fd rtol = 1e-3
@test d_R1f ≈ d_R1f_fd rtol = 1e-3
@test d_R2f ≈ d_R2f_fd rtol = 1e-3
@test d_Rex ≈ d_Rex_fd rtol = 1e-3
@test d_R1s ≈ d_R1s_fd rtol = 1e-3
@test d_T2s ≈ d_T2s_fd rtol = 1e-3



## ##########################################################
## R1a
#############################################################
R1a = 1
grad_list = (grad_R1a(),)
Graham_sol_grad = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, [m0; zeros(5 * (length(grad_list)))], (0, TRF), (ω1, B1, ω0, TRF, m0s, R1f, R2f, Rex, R1s, T2s, grad_list)), ode_solver)
d_R1a = [Graham_sol_grad(t[i])[j] for j ∈ 6:9, i ∈ eachindex(t)]

_f = function (R1a)
    sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, m0, (0, TRF), (ω1, B1, ω0, TRF, m0s, R1a, R2f, Rex, R1a, T2s)), ode_solver)
    return reduce(hcat, [sol(t[i]) for i ∈ eachindex(t)])
end
_fd = fd(_f, R1a)
d_R1a_fd = _fd[1:4, :]

@test d_R1a ≈ d_R1a_fd rtol = 1e-3