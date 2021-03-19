using Revise
using QuadGK
using DifferentialEquations
using SpecialFunctions
using ApproxFun
using BenchmarkTools
using Printf
using Plots
plotlyjs(ticks=:native)
theme(:lime);

include("../src/MT_Hamiltonians.jl")
using Main.MT_Hamiltonians
import Main.MT_Hamiltonians:gBloch_Hamiltonian_ApproxFun!

## define lineshapes
g_SL = (τ) -> quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8), 0.0, sqrt(1 / 3), 1.0)[1]

g_L = (τ) -> exp(- τ)
g_G = (τ) -> exp(- τ^2 / 2.0)

f_PSD = (τ) -> quadgk(ct -> (4 / τ / abs(1 - 3 * ct^2) * (exp(- τ^2 / 8 * (1 - 3 * ct^2)^2) - 1) 
    + sqrt(2π) * erf(τ / 2 / sqrt(2) * abs(1 - 3 * ct^2))) / abs(1 - 3 * ct^2), 0.0, 1.0)[1]
R_SL_PSD = (TRF, T2, ω1) -> f_PSD(TRF / T2) * ω1^2 * T2

f_Sinc = (τ) -> quadgk(ct -> erf(τ / 4 / sqrt(2) * abs(1 - 3 * ct^2)) / abs(1 - 3 * ct^2), 0.0, 1.0)[1]
R_SL_Sinc = (TRF, T2, ω1) -> f_Sinc(TRF / T2) * ω1^2 * T2 * sqrt(2π)

function gBloch_Hamiltonian_ApproxFun!(du, u, h, p::NTuple{3,Any}, t)
    ωy, T2s, g = p

    du[1] = -ωy^2 * quadgk(x -> g((t - x) / T2s) * h(p, x)[1], 0.0, t)[1]
end

## pulsed MT: Compare generalized Bloch to Literature approaches
c = 1.0
ω1 = π / 500e-6 / c
R1 = 0.0 / c
T2 = 10e-6 * c
u0 = [1.0]
TRF = 500e-6 * c
# h(p, t) = [1.0]
h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5)

x = Fun(identity, 0..100)
g_SLa = g_SL(x)

sol_approx = solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h, (0.0, TRF), (ω1, T2, g_SLa)), MethodOfSteps(DP8()))

t = 0:(1e-5 * c):(0.0, TRF)[2]
plot(t, (hcat(sol_approx(t).u...)'), ticks=:native, label="generalized Bloch")

Rrf = R_SL_PSD((0.0, TRF)[2], T2, ω1)
zss = R1 / (R1 + Rrf)
plot!(t, (1 - zss) .* exp.(- (R1 + Rrf) .* t) .+ zss, label="Graham's 1st approx.")

Rrf = R_SL_Sinc((0.0, TRF)[2], T2, ω1)
zss = R1 / (R1 + Rrf)
plot!(t, (1 - zss) .* exp.(- (R1 + Rrf) .* t) .+ zss, label="Sled's approx.")

##
S = Chebyshev(100e-6..(TRF)) * Chebyshev((100.0)..(π / 500e-6))

p = points(S, 10^3) # 105 padua points
# f̃ = (xy) -> R_SL_PSD(xy[1] * T2, T2, xy[2])
fDDE = (xy) -> solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h, (0.0, xy[1]), (xy[2], T2, g_SLa)), MethodOfSteps(DP8()))[end][1]
fapprox = Fun(S, transform(S, fDDE.(p)))

## test ApproxFun for varying ω1
ω1 = 650.0 : 10 : π/500e-6
t = TRF
fn = similar(ω1)
fDDEn = similar(ω1)
for i = 1:length(ω1)
    fn[i] = fapprox([t, ω1[i]])
    fDDEn[i] = fDDE([t, ω1[i]])
end
plot(ω1, fDDEn)
plot!(ω1, fn)

## test ApproxFun for varying TRF
ω1 = π / 500e-6
TRF = 500e-6
t = 0.0 : 1e-5 : TRF
fDDEn = similar(t)
for i = 1:length(t)
    fDDEn[i] = fDDE([t[i], ω1])
end

Rrf = - log(fapprox([TRF, ω1])) / TRF
fn_exp = similar(t)
for i = 1:length(t)
    fn_exp[i] = exp(- t[i] * Rrf)
end

plot(t, fDDEn)
plot!(t, fn_exp)


##
ω0 = 0.0
B1 = 1.0
m0s = .03526
R1 = 1.0
R2f = 1.0 / 65e-3
Rx = 30.0
# u0x = [0.0 0.0 1-m0s m0s 1.0]
u0x = [-0.0717803415882632, 0.0, 0.6673881489930942, 0.03523934569182852, 1.0]

sol_X = solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0x, h, (0.0, TRF), (ω1, B1, ω0, m0s, R1, R2f, T2, Rx, g_SLa, [], [])), MethodOfSteps(DP8()))

Rrf = - log(fapprox([TRF, ω1])) / TRF
sol_XL = solve(ODEProblem(Linear_Hamiltonian!, u0x, (0.0, TRF), (ω1, B1, ω0, m0s, R1, R2f, Rx, Rrf)), Vern6())

sol_Graham = solve(ODEProblem(Graham_Hamiltonian!, u0x, (0.0, TRF), (ω1, B1, ω0, TRF, m0s, R1, R2f, T2, Rx)), Vern6())

zs_X   = similar(t)
zs_XL  = similar(t)
zs_Gr  = similar(t)
for i = 1:length(t)
    zs_X[i]   = sol_X(t[i])[4]
    zs_XL[i]  = sol_XL(t[i])[4]
    zs_Gr[i]  = sol_Graham(t[i])[4]
end
plot(t, zs_X, label="zs exchange")
plot!(t, zs_XL, label="zs x-linearized")
plot!(t, zs_Gr, label="zs Graham")



##
using MAT
include("../src/MT_Diff_Equation_Sovlers.jl")
using Main.MT_Diff_Equation_Sovlers
using Revise
Revise.track("src/MT_Diff_Equation_Sovlers.jl")
Revise.track("src/MT_Hamiltonians.jl")

##
file = matopen(expanduser("~/mygs/20200806_MT_inVivo/control_MT_v3p2_TR3p5ms_discretized.mat"))
# file = matopen(expanduser("~/mygs/20200928_Phantom_NewSweeping_MT_v6p2/control_MT_v6p2_TR3p5ms_discretized.mat"))
# file = matopen(expanduser("~/mygs/20201016_InVivo_MT_v7_v0/control_v7p1.mat"))
# file = matopen(expanduser("~/mygs/20201016_InVivo_MT_v7_v0/control_v7p2.mat"))
control = read(file, "control")
TRF = [500e-6; control[1:end - 1,2]]
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
ω1 = α ./ TRF
TR = 3.5e-3

m0s = .1
T1 = 1.0
T2f = 65e-3
Rx = 30.0
T2s = 10e-6
ω0 = 0.0
B1 = 1.0

##
gra_list = [
    MT_Diff_Equation_Sovlers.grad_m0s(), 
    MT_Diff_Equation_Sovlers.grad_R1(), 
    MT_Diff_Equation_Sovlers.grad_R2f(), 
    MT_Diff_Equation_Sovlers.grad_Rx(), 
    MT_Diff_Equation_Sovlers.grad_T2s(), 
    MT_Diff_Equation_Sovlers.grad_ω0(), 
    MT_Diff_Equation_Sovlers.grad_B1()]
s = gBloch_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, 1/T1, 1/T2f, Rx, T2s, gra_list, 2);

Rrf_T = PreCompute_Saturation(minimum(TRF), maximum(TRF), T2s, T2s, minimum(ω1), maximum(ω1), 1.0, 1.0)
sl = LinearApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, 1/T1, 1/T2f, Rx, T2s, gra_list, 2, Rrf_T);
plot(s[4,:])
plot!(sl[4,:])

##
ii = 39
plot(s[ii,:])
plot!(sl[ii,:])
