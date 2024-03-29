#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/Linear_Approximation.ipynb)

# # Linear Approximation
# The following code demonstrates the linear approximation of the generalized Bloch model and replicates Figs. 7 and 8 in the paper.

# For this analysis we need the following packages:
using DifferentialEquations
using BenchmarkTools
using LinearAlgebra
using MRIgeneralizedBloch
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb

# and we simulate a coupled spin system with the following parameters:
m₀ˢ = 0.1
m₀ᶠ = 1-m₀ˢ
R₁ = 1 # 1/s
R₂ᶠ = 1 / 50e-3 # 1/s
Rₓ = 70; # 1/s

# ## Linearized ``T_2^{s,l}``
# We demonstrate the linear approximation at the example of the Green's function corresponding to the [super-Lorentzian lineshape](http://dx.doi.org/10.1002/mrm.1910330404), which we interpolate to improve the performance:
G = interpolate_greens_function(greens_superlorentzian, 0, 1e-3 / 5e-6);

# The function [`precompute_R2sl`](@ref) returns another function, `R₂ˢˡ(Tʳᶠ, α, B1, T₂ˢ)`, that interpolates the linearized relaxation rate, as well as functions that describe its derivatives wrt. ``T_2^s`` and ``B_1``, respectively:
R₂ˢˡ, ∂R₂ˢˡ∂T₂ˢ, ∂R₂ˢˡ∂B₁ = precompute_R2sl(TRF_min=5, TRF_max=100, T2s_min=1, T2s_max=1, ω1_max=π/5, B1_max=1, greens=G);
# The derivatives are not used here and are just assigned for demonstration purposes.

# In order to replicate Fig. 7, we plot `R₂ˢˡ(Tʳᶠ, α, B₁, T₂ˢ)` for a varying ``α`` and ``T_\text{RF}/T_2^s``:
α = (0.01:.01:1) * π
TʳᶠoT₂ˢ = 5:100

TʳᶠoT₂ˢ_m = repeat(reshape(TʳᶠoT₂ˢ, 1, :), length(α), 1)
α_m = repeat(α, 1, size(TʳᶠoT₂ˢ_m, 2))

p = plot(xlabel="Tʳᶠ/T₂ˢ", ylabel="α/π", colorbar_title="T₂ˢˡ/T₂ˢ")
contour!(p, TʳᶠoT₂ˢ, α ./ π, 1 ./ R₂ˢˡ.(TʳᶠoT₂ˢ_m, α_m, 1, 1), fill = true)
#md Main.HTMLPlot(p) #hide

#src export
using Printf #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/Linearized_T2s.txt")), "w") #src
write(io, "alphaopi TRFoT2s T2sloT2s \n") #src

for αi in α #src
    for TʳᶠoT₂ˢi in TʳᶠoT₂ˢ #src
        write(io, @sprintf("%1.3f %1.3f %1.3f \n", αi/π, TʳᶠoT₂ˢi, 1 / R₂ˢˡ(TʳᶠoT₂ˢi, αi, 1, 1))) #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src


# ## Spin Dynamics during a single RF Pulse
# To replicate Fig. 8a, we simulate and plot the dynamics of a coupled spin system during a single π-pulse, starting from thermal equilibrium.
Tʳᶠ = 100e-6 # s
T₂ˢ = 10e-6 # μs
m0_5D = [0,0,m₀ᶠ,m₀ˢ,1]
mfun(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0 : m0_5D; # initialize history function, here with the ability to just call a single index

# The full generalized Bloch model is solved by
param = (π/Tʳᶠ, 1, 0, m₀ˢ, R₁, R₂ᶠ, Rₓ, R₁, T₂ˢ, G)
prob = DDEProblem(apply_hamiltonian_gbloch!, m0_5D, mfun, (0.0, Tʳᶠ), param)
sol_pi_full = solve(prob);

# and we evaluate the interpolated solution at the following time points:
t = (0:.01:1) * Tʳᶠ # s
Mpi_full = zeros(length(t),4)
for i in eachindex(t)
    Mpi_full[i,:] = sol_pi_full(t[i])[1:4]
end

# Further, we calculate the linear approximation, which is simulated in a 6D-space as it explicitly models ``x^s``:
m0_6D = [0,0,m₀ᶠ,0,m₀ˢ,1]

Mpi_appx = similar(Mpi_full)
for i in eachindex(t)
    H = exp(hamiltonian_linear(π/Tʳᶠ, 1, 0, t[i], m₀ˢ, R₁, Rₓ, R₁, R₂ᶠ, R₂ˢˡ(Tʳᶠ, π, 1, T₂ˢ)))
    Mpi_appx[i,:] = (H * m0_6D)[[1:3;5]]
end

# and plot the original generalized Bloch model and its linear approximation for comparison:
p = plot(xlabel="t [s]", ylabel="m/m₀")
plot!(p, t, Mpi_full[:,1] / m₀ᶠ, label="xᶠ/m₀ᶠ original model")
plot!(p, t, Mpi_appx[:,1] / m₀ᶠ, label="xᶠ/m₀ᶠ linear approximation")
plot!(p, t, Mpi_full[:,3] / m₀ᶠ, label="zᶠ/m₀ᶠ original model")
plot!(p, t, Mpi_appx[:,3] / m₀ᶠ, label="zᶠ/m₀ᶠ linear approximation")
plot!(p, t, Mpi_full[:,4] / m₀ˢ, label="zˢ/m₀ˢ original model")
plot!(p, t, Mpi_appx[:,4] / m₀ˢ, label="zˢ/m₀ˢ linear approximation")
#md Main.HTMLPlot(p) #hide

# We observe slight deviations of zˢ during the pulse, but a virtually perfect match at the end of the RF pulse.

# ## RF Pulses with Different Flip Angles
# To replicate Fig. 8b, we simulate the spin dynamics during multiple RF pulses with different flip angles α, each simulation starting from thermal equilibrium, and analyze the magnetization at the end of each pulse:
α = (.01:.01:1) * π

M_full = zeros(length(α), 4)
M_appx = similar(M_full)
for i in eachindex(α)
    param = (α[i]/Tʳᶠ, 1, 0, m₀ˢ, R₁, R₂ᶠ, Rₓ, R₁, T₂ˢ, G)
    prob = DDEProblem(apply_hamiltonian_gbloch!, m0_5D, mfun, (0.0, Tʳᶠ), param)
    M_full[i,:] = solve(prob).u[end][1:4]

    u = exp(hamiltonian_linear(α[i]/Tʳᶠ, 1, 0, Tʳᶠ, m₀ˢ, R₁, R₂ᶠ, Rₓ, R₁, R₂ˢˡ(Tʳᶠ, α[i], 1, T₂ˢ))) * m0_6D
    M_appx[i,:] = u[[1:3;5]]
end

p = plot(xlabel="α/π", ylabel="m/m₀")
plot!(p, α/π, M_appx[:,1] / m₀ᶠ, label="xᶠ/m₀ˢ original model")
plot!(p, α/π, M_full[:,1] / m₀ᶠ, label="xᶠ/m₀ˢ linear approximation")
plot!(p, α/π, M_appx[:,3] / m₀ᶠ, label="zᶠ/m₀ˢ original model")
plot!(p, α/π, M_full[:,3] / m₀ᶠ, label="zᶠ/m₀ˢ linear approximation")
plot!(p, α/π, M_appx[:,4] / m₀ˢ, label="zˢ/m₀ˢ original model")
plot!(p, α/π, M_full[:,4] / m₀ˢ, label="zˢ/m₀ˢ linear approximation")
#md Main.HTMLPlot(p) #hide

# Visually, the linear approximation matches the full simulation well. The normalized root-mean-squared error of the linear approximation for ``x^f`` is
norm(M_appx[:,1] .- M_full[:,1]) / norm(M_full[:,1])
# for ``z^f`` it is
norm(M_appx[:,3] .- M_full[:,3]) / norm(M_full[:,3])
# and for ``z^s`` it is
norm(M_appx[:,4] .- M_full[:,4]) / norm(M_full[:,4])
# which confirms the good concordance.

# ## Benchmark
# We analyze the execution time for solving the full integro-differential equation:
param = (α[end]/Tʳᶠ, 1, 0, m₀ˢ, R₁, R₂ᶠ, Rₓ, R₁, T₂ˢ, G)
prob = DDEProblem(apply_hamiltonian_gbloch!, m0_5D, mfun, (0.0, Tʳᶠ), param)
@benchmark solve($prob)

# The `$` symbol *interpolates* the variable, which improves the accuracy of the timing measurement. We can compare this time to the time it takes to calculate the linear approximation, including the time it takes to evaluate the interpolated `R₂ˢˡ`:
@benchmark exp(hamiltonian_linear($(α[end]/Tʳᶠ), 1, 0, $Tʳᶠ, $m₀ˢ, $R₁, $R₂ᶠ, $Rₓ, $R₁, R₂ˢˡ($Tʳᶠ, $α[end], 1, $T₂ˢ))) * $m0_6D

# We can see that linear approximation is about 4 orders of magnitude faster compared to the full model.

#src export data for t vs. M figure
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/Linearized_gBloch_M_during_Pulse.txt")), "w") #src
write(io, "t_us xf_full xf_appx zf_full zf_appx zs_full zs_appx \n") #src
for i = 1:3:length(t) #src
    write(io, @sprintf("%1.3f ", t[i]*1e6)) #src
    write(io, @sprintf("%1.3f ", Mpi_full[i,1] / m₀ᶠ)) #src
    write(io, @sprintf("%1.3f ", Mpi_appx[i,1] / m₀ᶠ)) #src
    write(io, @sprintf("%1.3f ", Mpi_full[i,3] / m₀ᶠ)) #src
    write(io, @sprintf("%1.3f ", Mpi_appx[i,3] / m₀ᶠ)) #src
    write(io, @sprintf("%1.3f ", Mpi_full[i,4] / m₀ˢ)) #src
    write(io, @sprintf("%1.3f ", Mpi_appx[i,4] / m₀ˢ)) #src
    write(io, "\n") #src
end #src
close(io) #src

#src export data for α vs. M figure
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/Linearized_gBloch_vary_alpha.txt")), "w") #src
write(io, "alpha/pi xf_full xf_appx zf_full zf_appx zs_full zs_appx \n") #src
for i = 1:3:length(α) #src
    write(io, @sprintf("%1.3f ", α[i]/π)) #src
    write(io, @sprintf("%1.3f ", M_full[i,1] / m₀ᶠ)) #src
    write(io, @sprintf("%1.3f ", M_appx[i,1] / m₀ᶠ)) #src
    write(io, @sprintf("%1.3f ", M_full[i,3] / m₀ᶠ)) #src
    write(io, @sprintf("%1.3f ", M_appx[i,3] / m₀ᶠ)) #src
    write(io, @sprintf("%1.3f ", M_full[i,4] / m₀ˢ)) #src
    write(io, @sprintf("%1.3f ", M_appx[i,4] / m₀ˢ)) #src
    write(io, "\n") #src
end #src
close(io) #src