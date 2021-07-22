using DifferentialEquations
using BenchmarkTools
using LinearAlgebra
using MRIgeneralizedBloch
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

m₀ˢ = 0.1
m₀ᶠ = 1-m₀ˢ
R₁ = 1 # 1/s
R₂ᶠ = 1 / 50e-3 # 1/s
Rₓ = 70; # 1/s

G = interpolate_greens_function(greens_superlorentzian, 0, 1e-3 / 5e-6);

(R₂ˢˡ, ∂R₂ˢˡ∂T₂ˢ, ∂R₂ˢˡ∂B₁) = precompute_R2sl(100e-6, 1e-3, 5e-6, 15e-6, 0.01π, π, 1-eps(), 1+eps(); greens=G);

α = π
T₂ˢ = 5e-6 : 1e-7 : 15e-6 # s
Tʳᶠ = 100e-6 : 100e-6 : 1e-3 # s
p = plot(xlabel = "T₂ˢ [μs]", ylabel = "T₂ˢˡ(Tʳᶠ, α, B₁, T₂ˢ) / T₂ˢ")
for Tʳᶠᵢ in Tʳᶠ
    plot!(p, T₂ˢ*1e6, 1 ./ (T₂ˢ .* R₂ˢˡ.(Tʳᶠᵢ, α, 1, T₂ˢ)), label=string(Tʳᶠᵢ*1e6, "μs"))
end
display(p)

Tʳᶠ = 100e-6 # s
α = (0.1 : .1 : 1) * π
p = plot(xlabel = "T₂ˢ [μs]", ylabel = "T₂ˢˡ(Tʳᶠ, α, B₁, T₂ˢ) / T₂ˢ")
for αᵢ in α
    plot!(p, 1e6*T₂ˢ, 1 ./ (T₂ˢ .* R₂ˢˡ.(Tʳᶠ, αᵢ, 1, T₂ˢ)), label=string(αᵢ/π, "π"))
end
display(p)

T₂ˢ = 10e-6 # μs
m0_5D = [0,0,m₀ᶠ,m₀ˢ,1]
mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0 : m0_5D; # intialize history function, here with the ability to just call a single index

param = (π/Tʳᶠ, 1, 0, m₀ˢ, R₁, R₂ᶠ, T₂ˢ, Rₓ, G)
prob = DDEProblem(apply_hamiltonian_gbloch!, m0_5D, mfun, (0.0, Tʳᶠ), param)
sol_pi_full = solve(prob);

t = (0 : .01 : 1) * Tʳᶠ # s
Mpi_full = zeros(length(t),4)
for i in eachindex(t)
    Mpi_full[i,:] = sol_pi_full(t[i])[1:4]
end

m0_6D = [0,0,m₀ᶠ,0,m₀ˢ,1]

Mpi_appx = similar(Mpi_full)
for i in eachindex(t)
    H = exp(hamiltonian_linear(π/Tʳᶠ, 1, 0, t[i], m₀ˢ, R₁, R₂ᶠ, Rₓ, R₂ˢˡ(Tʳᶠ, π, 1, T₂ˢ)))
    Mpi_appx[i,:] = (H * m0_6D)[[1:3;5]]
end

p = plot(xlabel="t [s]", ylabel="m/m₀")
plot!(p, t, Mpi_full[:,1] / m₀ᶠ, label="xᶠ/m₀ᶠ original model")
plot!(p, t, Mpi_appx[:,1] / m₀ᶠ, label="xᶠ/m₀ᶠ linear approximation")
plot!(p, t, Mpi_full[:,3] / m₀ᶠ, label="zᶠ/m₀ᶠ original model")
plot!(p, t, Mpi_appx[:,3] / m₀ᶠ, label="zᶠ/m₀ᶠ linear approximation")
plot!(p, t, Mpi_full[:,4] / m₀ˢ, label="zˢ/m₀ˢ original model")
plot!(p, t, Mpi_appx[:,4] / m₀ˢ, label="zˢ/m₀ˢ linear approximation")

α = (.01:.01:1) * π

M_full = zeros(length(α), 4)
M_appx = similar(M_full)
for i in eachindex(α)
    param = (α[i]/Tʳᶠ, 1, 0, m₀ˢ, R₁, R₂ᶠ, T₂ˢ, Rₓ, G)
    prob = DDEProblem(apply_hamiltonian_gbloch!, m0_5D, mfun, (0.0, Tʳᶠ), param)
    M_full[i,:] = solve(prob)[end][1:4]

    u = exp(hamiltonian_linear(α[i]/Tʳᶠ, 1, 0, Tʳᶠ, m₀ˢ, R₁, R₂ᶠ, Rₓ, R₂ˢˡ(Tʳᶠ, α[i], 1, T₂ˢ))) * m0_6D
    M_appx[i,:] = u[[1:3;5]]
end

p = plot(xlabel="α/π", ylabel="m/m₀")
plot!(p, α/π, M_appx[:,1] / m₀ᶠ, label="xᶠ/m₀ˢ original model")
plot!(p, α/π, M_full[:,1] / m₀ᶠ, label="xᶠ/m₀ˢ linear approximation")
plot!(p, α/π, M_appx[:,3] / m₀ᶠ, label="zᶠ/m₀ˢ original model")
plot!(p, α/π, M_full[:,3] / m₀ᶠ, label="zᶠ/m₀ˢ linear approximation")
plot!(p, α/π, M_appx[:,4] / m₀ˢ, label="zˢ/m₀ˢ original model")
plot!(p, α/π, M_full[:,4] / m₀ˢ, label="zˢ/m₀ˢ linear approximation")

norm(M_appx[:,1] .- M_full[:,1]) / norm(M_full[:,1])

norm(M_appx[:,3] .- M_full[:,3]) / norm(M_full[:,3])

norm(M_appx[:,4] .- M_full[:,4]) / norm(M_full[:,4])

param = (α[end]/Tʳᶠ, 1, 0, m₀ˢ, R₁, R₂ᶠ, T₂ˢ, Rₓ, G)
prob = DDEProblem(apply_hamiltonian_gbloch!, m0_5D, mfun, (0.0, Tʳᶠ), param)
@benchmark solve($prob)

@benchmark exp(hamiltonian_linear($(α[end]/Tʳᶠ), 1, 0, $Tʳᶠ, $m₀ˢ, $R₁, $R₂ᶠ, $Rₓ, R₂ˢˡ($Tʳᶠ, $α[end], 1, $T₂ˢ))) * $m0_6D

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

