import Cubature
using QuadGK
using ApproxFun
using DifferentialEquations
using BenchmarkTools
using Plots
plotlyjs(ticks=:native)
theme(:lime);

include("../src/MT_Hamiltonians.jl")
using Main.MT_Hamiltonians

## set parameters
ω1 = π / 500e-6
ω0 = 200.0
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

# ApproxFun
g = (τ) -> quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8), 0.0, 1.0)[1]
dg_oT2 = (τ) -> quadgk(ct -> exp(- τ^2 * (3.0 * ct^2 - 1)^2 / 8.0) * (τ^2 * (3.0 * ct^2 - 1)^2 / 4.0), 0.0, 1.0)[1]
x = Fun(identity, 0..100)
ga = g(x)
dg_oT2_a = dg_oT2(x)


## baseline IDE solution
u0 = [0.5 * (1-m0s), 0.0, 0.5 * (1-m0s), m0s, 1.0]
gBloch_sol = solve(DDEProblem(gBloch_Hamiltonian!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, T2s, Rx, N)), alg)
plot(gBloch_sol)

## free precession
u1 = gBloch_sol[end]
FP_sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u1, (TRF, TE), (ω0, m0s, R1, R2f, Rx)), Tsit5())
plot(FP_sol)

## Analytical gradients (using ApproxFun)
grad_list = [grad_m0s(), grad_R1(), grad_R2f(), grad_Rx(), grad_T2s(), grad_ω0(), grad_ω1()]
u0 = zeros(5 * (length(grad_list)+1),1)
u0[1] = 0.5 * (1-m0s)
u0[3] = 0.5 * (1-m0s)
u0[4] = m0s
u0[5] = 1.0

gBloch_sol_grad = solve(DDEProblem(gBloch_Hamiltonian_Gradient_ApproxFun!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, T2s, Rx, ga, dg_oT2_a, grad_list)), alg)

u1 = gBloch_sol_grad[end]
FP_sol_grad = solve(ODEProblem(FreePrecession_Hamiltonian_Gradient!, u1, (TRF, TE), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5())

## FD derivative wrt. m0s
dm0s = 1e-9

u0 = u0[1:5]
gBloch_sol_dm0s = solve(DDEProblem(gBloch_Hamiltonian!, u0, h, (0.0, TRF), (ω1, ω0, (m0s + dm0s), R1, R2f, T2s, Rx, N)), alg)
u1 = gBloch_sol_dm0s[end]
FP_sol_dm0s = solve(ODEProblem(FreePrecession_Hamiltonian!, u1, (TRF, TE), (ω0, (m0s + dm0s), R1, R2f, Rx)), Tsit5())

t = 0 : 1e-5 : TE
dxf = similar(t)
dyf = similar(t)
dzf = similar(t)
dzs = similar(t)
dxf_fd = similar(t)
dyf_fd = similar(t)
dzf_fd = similar(t)
dzs_fd = similar(t)

for i = 1 : length(t)
    if t[i] <= TRF
        dxf_fd[i] = (gBloch_sol_dm0s(t[i])[1] - gBloch_sol(t[i])[1]) /dm0s
        dyf_fd[i] = (gBloch_sol_dm0s(t[i])[2] - gBloch_sol(t[i])[2]) /dm0s
        dzf_fd[i] = (gBloch_sol_dm0s(t[i])[3] - gBloch_sol(t[i])[3]) /dm0s
        dzs_fd[i] = (gBloch_sol_dm0s(t[i])[4] - gBloch_sol(t[i])[4]) /dm0s

        dxf[i] = gBloch_sol_grad(t[i])[6]
        dyf[i] = gBloch_sol_grad(t[i])[7]
        dzf[i] = gBloch_sol_grad(t[i])[8]
        dzs[i] = gBloch_sol_grad(t[i])[9]
    else
        dxf_fd[i] = (FP_sol_dm0s(t[i])[1] - FP_sol(t[i])[1]) /dm0s
        dyf_fd[i] = (FP_sol_dm0s(t[i])[2] - FP_sol(t[i])[2]) /dm0s
        dzf_fd[i] = (FP_sol_dm0s(t[i])[3] - FP_sol(t[i])[3]) /dm0s
        dzs_fd[i] = (FP_sol_dm0s(t[i])[4] - FP_sol(t[i])[4]) /dm0s

        dxf[i] = FP_sol_grad(t[i])[6]
        dyf[i] = FP_sol_grad(t[i])[7]
        dzf[i] = FP_sol_grad(t[i])[8]
        dzs[i] = FP_sol_grad(t[i])[9]
    end
end

plot(t, dxf, ticks=:native, label="xf", legend=:topleft)
plot!(t, dxf_fd, label="xf_fd")
plot!(t, dyf, ticks=:native, label="yf")
plot!(t, dyf_fd, label="yf_fd")
plot!(t, dzf, ticks=:native, label="zf")
plot!(t, dzf_fd, label="zf_fd")
plot!(t, dzs, label="zs")
plot!(t, dzs_fd, label="zs_fd")


## test derivative wrt. R1 
dR1 = 1e-9

gBloch_sol_dR1 = solve(DDEProblem(gBloch_Hamiltonian!, u0, h, (0.0, TRF), (ω1, ω0, m0s, (R1+ dR1), R2f, T2s, Rx, N)), alg)
u1 = gBloch_sol_dR1[end]
FP_sol_dR1 = solve(ODEProblem(FreePrecession_Hamiltonian!, u1, (TRF, TE), (ω0, m0s, (R1+ dR1), R2f, Rx)), Tsit5())

for i = 1: length(t)
    if t[i] <= TRF
        dxf_fd[i] = (gBloch_sol_dR1(t[i])[1] - gBloch_sol(t[i])[1]) /dR1
        dyf_fd[i] = (gBloch_sol_dR1(t[i])[2] - gBloch_sol(t[i])[2]) /dR1
        dzf_fd[i] = (gBloch_sol_dR1(t[i])[3] - gBloch_sol(t[i])[3]) /dR1
        dzs_fd[i] = (gBloch_sol_dR1(t[i])[4] - gBloch_sol(t[i])[4]) /dR1

        dxf[i] = gBloch_sol_grad(t[i])[11]
        dyf[i] = gBloch_sol_grad(t[i])[12]
        dzf[i] = gBloch_sol_grad(t[i])[13]
        dzs[i] = gBloch_sol_grad(t[i])[14]
    else
        dxf_fd[i] = (FP_sol_dR1(t[i])[1] - FP_sol(t[i])[1]) /dR1
        dyf_fd[i] = (FP_sol_dR1(t[i])[2] - FP_sol(t[i])[2]) /dR1
        dzf_fd[i] = (FP_sol_dR1(t[i])[3] - FP_sol(t[i])[3]) /dR1
        dzs_fd[i] = (FP_sol_dR1(t[i])[4] - FP_sol(t[i])[4]) /dR1

        dxf[i] = FP_sol_grad(t[i])[11]
        dyf[i] = FP_sol_grad(t[i])[12]
        dzf[i] = FP_sol_grad(t[i])[13]
        dzs[i] = FP_sol_grad(t[i])[14]
    end
end

plot(t, dxf, ticks=:native, label="xf", legend=:topleft)
plot!(t, dxf_fd, label="xf_fd")
plot!(t, dyf, ticks=:native, label="yf")
plot!(t, dyf_fd, label="yf_fd")
plot!(t, dzf, ticks=:native, label="zf")
plot!(t, dzf_fd, label="zf_fd")
plot!(t, dzs, label="zs")
plot!(t, dzs_fd, label="zs_fd")

## test derivative wrt. R2f
dR2f = 1e-9

gBloch_sol_dR2f = solve(DDEProblem(gBloch_Hamiltonian!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, (R2f+dR2f), T2s, Rx, N)), alg)
u1 = gBloch_sol_dR2f[end]
FP_sol_dR2f = solve(ODEProblem(FreePrecession_Hamiltonian!, u1, (TRF, TE), (ω0, m0s, R1, (R2f+dR2f), Rx)), Tsit5())

for i = 1: length(t)
    if t[i] <= TRF
        dxf_fd[i] = (gBloch_sol_dR2f(t[i])[1] - gBloch_sol(t[i])[1]) /dR2f
        dyf_fd[i] = (gBloch_sol_dR2f(t[i])[2] - gBloch_sol(t[i])[2]) /dR2f
        dzf_fd[i] = (gBloch_sol_dR2f(t[i])[3] - gBloch_sol(t[i])[3]) /dR2f
        dzs_fd[i] = (gBloch_sol_dR2f(t[i])[4] - gBloch_sol(t[i])[4]) /dR2f

        dxf[i] = gBloch_sol_grad(t[i])[16]
        dyf[i] = gBloch_sol_grad(t[i])[17]
        dzf[i] = gBloch_sol_grad(t[i])[18]
        dzs[i] = gBloch_sol_grad(t[i])[19]
    else
        dxf_fd[i] = (FP_sol_dR2f(t[i])[1] - FP_sol(t[i])[1]) /dR2f
        dyf_fd[i] = (FP_sol_dR2f(t[i])[2] - FP_sol(t[i])[2]) /dR2f
        dzf_fd[i] = (FP_sol_dR2f(t[i])[3] - FP_sol(t[i])[3]) /dR2f
        dzs_fd[i] = (FP_sol_dR2f(t[i])[4] - FP_sol(t[i])[4]) /dR2f

        dxf[i] = FP_sol_grad(t[i])[16]
        dyf[i] = FP_sol_grad(t[i])[17]
        dzf[i] = FP_sol_grad(t[i])[18]
        dzs[i] = FP_sol_grad(t[i])[19]
    end
end

plot(t, dxf, ticks=:native, label="xf", legend=:topleft)
plot!(t, dxf_fd, label="xf_fd")
plot!(t, dyf, ticks=:native, label="yf")
plot!(t, dyf_fd, label="yf_fd")
plot!(t, dzf, ticks=:native, label="zf")
plot!(t, dzf_fd, label="zf_fd")
plot!(t, dzs, label="zs")
plot!(t, dzs_fd, label="zs_fd")

## test derivative wrt. Rx
dRx = 1e-9

gBloch_sol_dRx = solve(DDEProblem(gBloch_Hamiltonian!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, T2s, (Rx + dRx), N)), alg)
u1 = gBloch_sol_dRx[end]
FP_sol_dRx = solve(ODEProblem(FreePrecession_Hamiltonian!, u1, (TRF, TE), (ω0, m0s, R1, R2f, (Rx+dRx))), Tsit5())

for i = 1: length(t)
    if t[i] <= TRF
        dxf_fd[i] = (gBloch_sol_dRx(t[i])[1] - gBloch_sol(t[i])[1]) /dRx
        dyf_fd[i] = (gBloch_sol_dRx(t[i])[2] - gBloch_sol(t[i])[2]) /dRx
        dzf_fd[i] = (gBloch_sol_dRx(t[i])[3] - gBloch_sol(t[i])[3]) /dRx
        dzs_fd[i] = (gBloch_sol_dRx(t[i])[4] - gBloch_sol(t[i])[4]) /dRx

        dxf[i] = gBloch_sol_grad(t[i])[21]
        dyf[i] = gBloch_sol_grad(t[i])[22]
        dzf[i] = gBloch_sol_grad(t[i])[23]
        dzs[i] = gBloch_sol_grad(t[i])[24]
    else
        dxf_fd[i] = (FP_sol_dRx(t[i])[1] - FP_sol(t[i])[1]) /dRx
        dyf_fd[i] = (FP_sol_dRx(t[i])[2] - FP_sol(t[i])[2]) /dRx
        dzf_fd[i] = (FP_sol_dRx(t[i])[3] - FP_sol(t[i])[3]) /dRx
        dzs_fd[i] = (FP_sol_dRx(t[i])[4] - FP_sol(t[i])[4]) /dRx

        dxf[i] = FP_sol_grad(t[i])[21]
        dyf[i] = FP_sol_grad(t[i])[22]
        dzf[i] = FP_sol_grad(t[i])[23]
        dzs[i] = FP_sol_grad(t[i])[24]
    end
end

plot(t, dxf, ticks=:native, label="xf", legend=:topleft)
plot!(t, dxf_fd, label="xf_fd")
plot!(t, dyf, ticks=:native, label="yf")
plot!(t, dyf_fd, label="yf_fd")
plot!(t, dzf, ticks=:native, label="zf")
plot!(t, dzf_fd, label="zf_fd")
plot!(t, dzs, label="zs")
plot!(t, dzs_fd, label="zs_fd")

## test derivative wrt. T2s
dT2s = 1e-15

gBloch_sol_dT2s = solve(DDEProblem(gBloch_Hamiltonian!, u0, h, (0.0, TRF), (ω1, ω0, m0s, R1, R2f, (T2s + dT2s), Rx, N)), alg)
u1 = gBloch_sol_dT2s[end]
FP_sol_dT2s = solve(ODEProblem(FreePrecession_Hamiltonian!, u1, (TRF, TE), (ω0, m0s, R1, R2f, Rx)), Tsit5())

for i = 1: length(t)
    if t[i] <= TRF
        dxf_fd[i] = (gBloch_sol_dT2s(t[i])[1] - gBloch_sol(t[i])[1]) /dT2s
        dyf_fd[i] = (gBloch_sol_dT2s(t[i])[2] - gBloch_sol(t[i])[2]) /dT2s
        dzf_fd[i] = (gBloch_sol_dT2s(t[i])[3] - gBloch_sol(t[i])[3]) /dT2s
        dzs_fd[i] = (gBloch_sol_dT2s(t[i])[4] - gBloch_sol(t[i])[4]) /dT2s

        dxf[i] = gBloch_sol_grad(t[i])[26]
        dyf[i] = gBloch_sol_grad(t[i])[27]
        dzf[i] = gBloch_sol_grad(t[i])[28]
        dzs[i] = gBloch_sol_grad(t[i])[29]
    else
        dxf_fd[i] = (FP_sol_dT2s(t[i])[1] - FP_sol(t[i])[1]) /dT2s
        dyf_fd[i] = (FP_sol_dT2s(t[i])[2] - FP_sol(t[i])[2]) /dT2s
        dzf_fd[i] = (FP_sol_dT2s(t[i])[3] - FP_sol(t[i])[3]) /dT2s
        dzs_fd[i] = (FP_sol_dT2s(t[i])[4] - FP_sol(t[i])[4]) /dT2s

        dxf[i] = FP_sol_grad(t[i])[26]
        dyf[i] = FP_sol_grad(t[i])[27]
        dzf[i] = FP_sol_grad(t[i])[28]
        dzs[i] = FP_sol_grad(t[i])[29]
    end
end

plot(t, dxf, ticks=:native, label="xf", legend=:bottomleft)
plot!(t, dxf_fd, label="xf_fd")
plot!(t, dyf, ticks=:native, label="yf")
plot!(t, dyf_fd, label="yf_fd")
plot!(t, dzf, ticks=:native, label="zf")
plot!(t, dzf_fd, label="zf_fd")
plot!(t, dzs, label="zs")
plot!(t, dzs_fd, label="zs_fd")


## test derivative wrt. ω0
dω0 = 1e-9

gBloch_sol_dω0 = solve(DDEProblem(gBloch_Hamiltonian!, u0, h, (0.0, TRF), (ω1, (ω0 + dω0), m0s, R1, R2f, T2s, Rx, N)), alg)
u1 = gBloch_sol_dω0[end]
FP_sol_dω0 = solve(ODEProblem(FreePrecession_Hamiltonian!, u1, (TRF, TE), ((ω0 + dω0), m0s, R1, R2f, Rx)), Tsit5())

for i = 1: length(t)
    if t[i] <= TRF
        dxf_fd[i] = (gBloch_sol_dω0(t[i])[1] - gBloch_sol(t[i])[1]) /dω0
        dyf_fd[i] = (gBloch_sol_dω0(t[i])[2] - gBloch_sol(t[i])[2]) /dω0
        dzf_fd[i] = (gBloch_sol_dω0(t[i])[3] - gBloch_sol(t[i])[3]) /dω0
        dzs_fd[i] = (gBloch_sol_dω0(t[i])[4] - gBloch_sol(t[i])[4]) /dω0

        dxf[i] = gBloch_sol_grad(t[i])[31]
        dyf[i] = gBloch_sol_grad(t[i])[32]
        dzf[i] = gBloch_sol_grad(t[i])[33]
        dzs[i] = gBloch_sol_grad(t[i])[34]
    else
        dxf_fd[i] = (FP_sol_dω0(t[i])[1] - FP_sol(t[i])[1]) /dω0
        dyf_fd[i] = (FP_sol_dω0(t[i])[2] - FP_sol(t[i])[2]) /dω0
        dzf_fd[i] = (FP_sol_dω0(t[i])[3] - FP_sol(t[i])[3]) /dω0
        dzs_fd[i] = (FP_sol_dω0(t[i])[4] - FP_sol(t[i])[4]) /dω0

        dxf[i] = FP_sol_grad(t[i])[31]
        dyf[i] = FP_sol_grad(t[i])[32]
        dzf[i] = FP_sol_grad(t[i])[33]
        dzs[i] = FP_sol_grad(t[i])[34]
    end
end

plot(t, dxf, ticks=:native, label="xf", legend=:bottomleft)
plot!(t, dxf_fd, label="xf_fd")
plot!(t, dyf, ticks=:native, label="yf")
plot!(t, dyf_fd, label="yf_fd")
plot!(t, dzf, ticks=:native, label="zf")
plot!(t, dzf_fd, label="zf_fd")
plot!(t, dzs, label="zs")
plot!(t, dzs_fd, label="zs_fd")


## test derivative wrt. ω1
dω1 = 1e-9

gBloch_sol_dω1 = solve(DDEProblem(gBloch_Hamiltonian!, u0, h, (0.0, TRF), ((ω1+dω1), ω0, m0s, R1, R2f, T2s, Rx, N)), alg)
u1 = gBloch_sol_dω1[end]
FP_sol_dω1 = solve(ODEProblem(FreePrecession_Hamiltonian!, u1, (TRF, TE), (ω0, m0s, R1, R2f, Rx)), Tsit5())

for i = 1: length(t)
    if t[i] <= TRF
        dxf_fd[i] = (gBloch_sol_dω1(t[i])[1] - gBloch_sol(t[i])[1]) /dω1
        dyf_fd[i] = (gBloch_sol_dω1(t[i])[2] - gBloch_sol(t[i])[2]) /dω1
        dzf_fd[i] = (gBloch_sol_dω1(t[i])[3] - gBloch_sol(t[i])[3]) /dω1
        dzs_fd[i] = (gBloch_sol_dω1(t[i])[4] - gBloch_sol(t[i])[4]) /dω1

        dxf[i] = gBloch_sol_grad(t[i])[36]
        dyf[i] = gBloch_sol_grad(t[i])[37]
        dzf[i] = gBloch_sol_grad(t[i])[38]
        dzs[i] = gBloch_sol_grad(t[i])[39]
    else
        dxf_fd[i] = (FP_sol_dω1(t[i])[1] - FP_sol(t[i])[1]) /dω1
        dyf_fd[i] = (FP_sol_dω1(t[i])[2] - FP_sol(t[i])[2]) /dω1
        dzf_fd[i] = (FP_sol_dω1(t[i])[3] - FP_sol(t[i])[3]) /dω1
        dzs_fd[i] = (FP_sol_dω1(t[i])[4] - FP_sol(t[i])[4]) /dω1

        dxf[i] = FP_sol_grad(t[i])[36]
        dyf[i] = FP_sol_grad(t[i])[37]
        dzf[i] = FP_sol_grad(t[i])[38]
        dzs[i] = FP_sol_grad(t[i])[39]
    end
end

plot(t, dxf, ticks=:native, label="xf", legend=:bottomleft)
plot!(t, dxf_fd, label="xf_fd")
plot!(t, dyf, ticks=:native, label="yf")
plot!(t, dyf_fd, label="yf_fd")
plot!(t, dzf, ticks=:native, label="zf")
plot!(t, dzf_fd, label="zf_fd")
plot!(t, dzs, label="zs")
plot!(t, dzs_fd, label="zs_fd")
