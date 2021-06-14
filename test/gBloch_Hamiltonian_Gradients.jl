using DifferentialEquations
using Test
using MRIgeneralizedBloch
using MRIgeneralizedBloch:
    apply_hamiltonian_gbloch_superlorentzian!, apply_hamiltonian_freeprecession!, apply_hamiltonian_gbloch!

## 
max_error = 5e-2

## set parameters
ω1 = π / 500e-6
ω0 = 200.0
B1 = 1.0
m0s = 0.15
R1 = 1.0
R2f = 1 / 65e-3
T2s = 10e-6
Rx = 30.0
TRF = 500e-6
h(p, t; idxs = nothing) = typeof(idxs) <: Number ? 0.0 : zeros(30)
alg = MethodOfSteps(DP8())
N = Inf

# ApproxFun
G = interpolate_greens_function(greens_superlorentzian, 0, 100)
dGdT2s = interpolate_greens_function(dG_o_dT2s_x_T2s_superlorentzian, 0, 100)


## baseline IDE solution
u0 = [0.5 * (1 - m0s), 0.0, 0.5 * (1 - m0s), m0s, 1.0]
gBloch_sol = solve(
    DDEProblem(
        apply_hamiltonian_gbloch_superlorentzian!,
        u0,
        h,
        (0, TRF),
        (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, N),
    ),
    alg,
)

## Analytical gradients (using ApproxFun)
grad_list = [grad_m0s(), grad_R1(), grad_R2f(), grad_Rx(), grad_T2s(), grad_ω0(), grad_B1()]
u0 = zeros(5 * (length(grad_list) + 1), 1)
u0[1] = 0.5 * (1 - m0s)
u0[3] = 0.5 * (1 - m0s)
u0[4] = m0s
u0[5] = 1.0

gBloch_sol_grad = solve(
    DDEProblem(
        apply_hamiltonian_gbloch!,
        u0,
        h,
        (0.0, TRF),
        (ω1, B1, ω0, m0s, R1, R2f, T2s, Rx, G, dGdT2s, grad_list),
    ),
    alg,
)

## FD derivative wrt. m0s
dm0s = 1e-9

u0 = u0[1:5]
gBloch_sol_dm0s = solve(
    DDEProblem(
        apply_hamiltonian_gbloch_superlorentzian!,
        u0,
        h,
        (0.0, TRF),
        (ω1, B1, ω0, (m0s + dm0s), R1, R2f, T2s, Rx, N),
    ),
    alg,
)

t = 0:1e-5:TRF
dxf = similar(t)
dyf = similar(t)
dzf = similar(t)
dzs = similar(t)
dxf_fd = similar(t)
dyf_fd = similar(t)
dzf_fd = similar(t)
dzs_fd = similar(t)

for i = 1:length(t)
    dxf_fd[i] = (gBloch_sol_dm0s(t[i])[1] - gBloch_sol(t[i])[1]) / dm0s
    dyf_fd[i] = (gBloch_sol_dm0s(t[i])[2] - gBloch_sol(t[i])[2]) / dm0s
    dzf_fd[i] = (gBloch_sol_dm0s(t[i])[3] - gBloch_sol(t[i])[3]) / dm0s
    dzs_fd[i] = (gBloch_sol_dm0s(t[i])[4] - gBloch_sol(t[i])[4]) / dm0s

    dxf[i] = gBloch_sol_grad(t[i])[6]
    dyf[i] = gBloch_sol_grad(t[i])[7]
    dzf[i] = gBloch_sol_grad(t[i])[8]
    dzs[i] = gBloch_sol_grad(t[i])[9]
end

@test dxf ≈ dxf_fd rtol = max_error
@test dyf ≈ dyf_fd rtol = max_error
@test dzf ≈ dzf_fd rtol = max_error
@test dzs ≈ dzs_fd rtol = max_error

## test derivative wrt. R1 
dR1 = 1e-9

gBloch_sol_dR1 = solve(
    DDEProblem(
        apply_hamiltonian_gbloch_superlorentzian!,
        u0,
        h,
        (0.0, TRF),
        (ω1, B1, ω0, m0s, (R1 + dR1), R2f, T2s, Rx, N),
    ),
    alg,
)

for i = 1:length(t)
    dxf_fd[i] = (gBloch_sol_dR1(t[i])[1] - gBloch_sol(t[i])[1]) / dR1
    dyf_fd[i] = (gBloch_sol_dR1(t[i])[2] - gBloch_sol(t[i])[2]) / dR1
    dzf_fd[i] = (gBloch_sol_dR1(t[i])[3] - gBloch_sol(t[i])[3]) / dR1
    dzs_fd[i] = (gBloch_sol_dR1(t[i])[4] - gBloch_sol(t[i])[4]) / dR1

    dxf[i] = gBloch_sol_grad(t[i])[11]
    dyf[i] = gBloch_sol_grad(t[i])[12]
    dzf[i] = gBloch_sol_grad(t[i])[13]
    dzs[i] = gBloch_sol_grad(t[i])[14]
end

@test dxf ≈ dxf_fd rtol = max_error
@test dyf ≈ dyf_fd rtol = max_error
@test dzf ≈ dzf_fd rtol = max_error
@test dzs ≈ dzs_fd rtol = max_error * 10

## test derivative wrt. R2f
dR2f = 1e-9

gBloch_sol_dR2f = solve(
    DDEProblem(
        apply_hamiltonian_gbloch_superlorentzian!,
        u0,
        h,
        (0.0, TRF),
        (ω1, B1, ω0, m0s, R1, (R2f + dR2f), T2s, Rx, N),
    ),
    alg,
)

for i = 1:length(t)
    dxf_fd[i] = (gBloch_sol_dR2f(t[i])[1] - gBloch_sol(t[i])[1]) / dR2f
    dyf_fd[i] = (gBloch_sol_dR2f(t[i])[2] - gBloch_sol(t[i])[2]) / dR2f
    dzf_fd[i] = (gBloch_sol_dR2f(t[i])[3] - gBloch_sol(t[i])[3]) / dR2f
    dzs_fd[i] = (gBloch_sol_dR2f(t[i])[4] - gBloch_sol(t[i])[4]) / dR2f

    dxf[i] = gBloch_sol_grad(t[i])[16]
    dyf[i] = gBloch_sol_grad(t[i])[17]
    dzf[i] = gBloch_sol_grad(t[i])[18]
    dzs[i] = gBloch_sol_grad(t[i])[19]
end

@test dxf ≈ dxf_fd rtol = max_error
@test dyf ≈ dyf_fd rtol = max_error
@test dzf ≈ dzf_fd rtol = max_error

## test derivative wrt. Rx
dRx = 1e-6

gBloch_sol_dRx = solve(
    DDEProblem(
        apply_hamiltonian_gbloch_superlorentzian!,
        u0,
        h,
        (0.0, TRF),
        (ω1, B1, ω0, m0s, R1, R2f, T2s, (Rx + dRx), N),
    ),
    alg,
)

for i = 1:length(t)
    dxf_fd[i] = (gBloch_sol_dRx(t[i])[1] - gBloch_sol(t[i])[1]) / dRx
    dyf_fd[i] = (gBloch_sol_dRx(t[i])[2] - gBloch_sol(t[i])[2]) / dRx
    dzf_fd[i] = (gBloch_sol_dRx(t[i])[3] - gBloch_sol(t[i])[3]) / dRx
    dzs_fd[i] = (gBloch_sol_dRx(t[i])[4] - gBloch_sol(t[i])[4]) / dRx

    dxf[i] = gBloch_sol_grad(t[i])[21]
    dyf[i] = gBloch_sol_grad(t[i])[22]
    dzf[i] = gBloch_sol_grad(t[i])[23]
    dzs[i] = gBloch_sol_grad(t[i])[24]
end

@test dxf ≈ dxf_fd rtol = max_error
@test dyf ≈ dyf_fd rtol = max_error
@test dzf ≈ dzf_fd rtol = max_error
@test dzs ≈ dzs_fd rtol = max_error


## test derivative wrt. T2s
dT2s = 1e-14

gBloch_sol_dT2s = solve(
    DDEProblem(
        apply_hamiltonian_gbloch_superlorentzian!,
        u0,
        h,
        (0.0, TRF),
        (ω1, B1, ω0, m0s, R1, R2f, (T2s + dT2s), Rx, N),
    ),
    alg,
)

for i = 1:length(t)
    dxf_fd[i] = (gBloch_sol_dT2s(t[i])[1] - gBloch_sol(t[i])[1]) / dT2s
    dyf_fd[i] = (gBloch_sol_dT2s(t[i])[2] - gBloch_sol(t[i])[2]) / dT2s
    dzf_fd[i] = (gBloch_sol_dT2s(t[i])[3] - gBloch_sol(t[i])[3]) / dT2s
    dzs_fd[i] = (gBloch_sol_dT2s(t[i])[4] - gBloch_sol(t[i])[4]) / dT2s

    dxf[i] = gBloch_sol_grad(t[i])[26]
    dyf[i] = gBloch_sol_grad(t[i])[27]
    dzf[i] = gBloch_sol_grad(t[i])[28]
    dzs[i] = gBloch_sol_grad(t[i])[29]
end

@test dxf ≈ dxf_fd rtol = max_error
@test dyf ≈ dyf_fd rtol = max_error
@test dzf ≈ dzf_fd rtol = max_error
@test dzs ≈ dzs_fd rtol = max_error

## test derivative wrt. ω0
dω0 = 1

gBloch_sol_dω0 = solve(
    DDEProblem(
        apply_hamiltonian_gbloch_superlorentzian!,
        u0,
        h,
        (0.0, TRF),
        (ω1, B1, (ω0 + dω0), m0s, R1, R2f, T2s, Rx, N),
    ),
    alg,
)

for i = 1:length(t)
    dxf_fd[i] = (gBloch_sol_dω0(t[i])[1] - gBloch_sol(t[i])[1]) / dω0
    dyf_fd[i] = (gBloch_sol_dω0(t[i])[2] - gBloch_sol(t[i])[2]) / dω0
    dzf_fd[i] = (gBloch_sol_dω0(t[i])[3] - gBloch_sol(t[i])[3]) / dω0
    dzs_fd[i] = (gBloch_sol_dω0(t[i])[4] - gBloch_sol(t[i])[4]) / dω0

    dxf[i] = gBloch_sol_grad(t[i])[31]
    dyf[i] = gBloch_sol_grad(t[i])[32]
    dzf[i] = gBloch_sol_grad(t[i])[33]
    dzs[i] = gBloch_sol_grad(t[i])[34]
end

@test dxf ≈ dxf_fd rtol = max_error
@test dyf ≈ dyf_fd rtol = max_error
@test dzf ≈ dzf_fd rtol = max_error

## test derivative wrt. B1
dB1 = 1e-9

gBloch_sol_dB1 = solve(
    DDEProblem(
        apply_hamiltonian_gbloch_superlorentzian!,
        u0,
        h,
        (0.0, TRF),
        (ω1, (B1 + dB1), ω0, m0s, R1, R2f, T2s, Rx, N),
    ),
    alg,
)

for i = 1:length(t)
    dxf_fd[i] = (gBloch_sol_dB1(t[i])[1] - gBloch_sol(t[i])[1]) / dB1
    dyf_fd[i] = (gBloch_sol_dB1(t[i])[2] - gBloch_sol(t[i])[2]) / dB1
    dzf_fd[i] = (gBloch_sol_dB1(t[i])[3] - gBloch_sol(t[i])[3]) / dB1
    dzs_fd[i] = (gBloch_sol_dB1(t[i])[4] - gBloch_sol(t[i])[4]) / dB1

    dxf[i] = gBloch_sol_grad(t[i])[36]
    dyf[i] = gBloch_sol_grad(t[i])[37]
    dzf[i] = gBloch_sol_grad(t[i])[38]
    dzs[i] = gBloch_sol_grad(t[i])[39]
end

@test dxf ≈ dxf_fd rtol = max_error
@test dyf ≈ dyf_fd rtol = max_error
@test dzf ≈ dzf_fd rtol = max_error
@test dzs ≈ dzs_fd rtol = max_error

## plot the functions for debug purposes
# using Plots
# plotlyjs()
# theme(:lime);
# plot(t, dxf, ticks=:native, label="xf", legend=:bottomleft)
# plot!(t, dxf_fd, label="xf_fd")
# plot!(t, dyf, ticks=:native, label="yf")
# plot!(t, dyf_fd, label="yf_fd")
# plot!(t, dzf, ticks=:native, label="zf")
# plot!(t, dzf_fd, label="zf_fd")
# plot!(t, dzs, label="zs")
# plot!(t, dzs_fd, label="zs_fd")
