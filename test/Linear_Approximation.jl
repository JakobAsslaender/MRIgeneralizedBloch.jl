using DifferentialEquations
using BenchmarkTools
using MRIgeneralizedBloch
using MRIgeneralizedBloch: apply_hamiltonian_gbloch!, hamiltonian_linear
using Test

## set variables
α = (.01:.01:1) * π
TRF = 100e-6 # s
ω1 = α ./ TRF # rad/s
m0s = 0.1
m0f = 1-m0s
R1 = 1.0 # 1/s
R2f = 1 / 50e-3 # 1/s
T2s = 10e-6 # s
Rx = 70 # 1/s

## pre-calcualtions
G = interpolate_greens_function(greens_superlorentzian, 0, maximum(TRF) / T2s)
h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5)

print("Time to pre-compute saturation: ")
(R2s, _, _) = @time precompute_R2sl(100e-6, 1e-3, 5e-6, 15e-6, minimum(α), maximum(α), 1-eps(), 1+eps())

## init
u0_5D = [0,0,m0f,  m0s,1]
u0_6D = [0,0,m0f,0,m0s,1]

## simulate and plot magnetization at the end of RF-pulses with different flip angles
M_full = zeros(length(ω1), 4)
M_appx = similar(M_full)
Threads.@threads for i in eachindex(ω1)
    M_full[i,:] = solve(DDEProblem(apply_hamiltonian_gbloch!, u0_5D, h, (0.0, TRF), (ω1[i], 1, 0, m0s, R1, R2f, T2s, Rx, G)), MethodOfSteps(DP8()))[end][1:4]
    u = exp(hamiltonian_linear(ω1[i], 1, 0, TRF, m0s, R1, R2f, Rx, R2s(TRF, ω1[i], 1, T2s))) * u0_6D
    M_appx[i,:] = u[[1:3;5]]
end

## Test
@test M_appx[:,1] ≈ M_full[:,1] rtol = 1e-3
@test M_appx[:,2] ≈ M_full[:,2] rtol = 1e-3
@test M_appx[:,3] ≈ M_full[:,3] rtol = 1e-3
@test M_appx[:,4] ≈ M_full[:,4] rtol = 1e-3

## benchmark the different solvers (excecute one line at a time to provoke individual results to be printed)
print("Time to solve the full gene. Bloch IDE for 100us π-pulse:")
@btime solve(DDEProblem(MRIgeneralizedBloch.apply_hamiltonian_gbloch!, u0_5D, h, (0.0, TRF), (ω1[end], 1, 0, m0s, R1, R2f, T2s, Rx, G)), MethodOfSteps(DP8()))

print("Time to solve the linear approximation for 100us π-pulse:")
@btime exp(hamiltonian_linear(ω1[end-1], 1, 0, TRF, m0s, R1, R2f, Rx, R2s(TRF, ω1[end-1], 1, T2s))) * u0_6D