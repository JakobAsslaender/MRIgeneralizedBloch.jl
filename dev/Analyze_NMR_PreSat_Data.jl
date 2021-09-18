#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/Analyze_NMR_PreSat_Data.ipynb)

# # Continuous Wave Saturation Experiments
# The following code analyzes data from a steady-state experiment similar to the original work of [Henkelman et al](https://doi.org/10.1002/mrm.1910290607). In this experiment, the magnetization of the coupled spin system is saturated with off-resonant continuous waves of the exponentially spaced frequencies:
Δ = exp.(range(log(0.01e3), log(100e3), length=20)) * 2π # rad/s
# and the amplitudes:
ω1_dB = -60:5:-5 # dB
ω1 = @. 10^(ω1_dB / 20) * π / 2 / 11.4e-6 # rad/s
# The waves were applied for 7 seconds to ensure a steady state. Thereafter, the magnetization was excited with a π/2-pulse and an FID was acquired. The repetition times was 30s to ensure full recovery to thermal equilibrium. 

# We fit the data with Henkelman's closed form solution to this steady-state problem while assuming a Lorentzian lineshape for the free spin pool, and different lineshapes for the semi-solid spin pool:
g_Lorentzian(Δ, T2) = T2 / π / (1 + (T2 * Δ)^2)
g_Gaussian(Δ, T2) = T2 / sqrt(2π) * exp(-(T2 * Δ)^2 / 2)
g_superLorentzian(Δ, T2) = T2 * sqrt(2 / π) * quadgk(ct -> exp(- 2 *  (T2 * Δ / (3 * ct^2 - 1))^2) / abs(3 * ct^2 - 1), 0, sqrt(1 / 3), 1)[1];


# For this data analysis we need the following packages:
using MRIgeneralizedBloch
using QuadGK
using LsqFit
using LinearAlgebra
using Statistics
using Printf
using Plots
plotlyjs(bg=RGBA(31 / 255, 36 / 255, 36 / 255, 1.0), ticks=:native); # hide #!nb
#nb plotlyjs(ticks=:native);

# The raw data is stored in a separate [github repository](https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData) and the following functions return the URL to the individual files:
MnCl2_data(ω1_dB) = string("https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210419_1mM_MnCl2/ja_PreSat_v2%20(", ω1_dB, ")/1/data.2d?raw=true")
BSA_data(ω1_dB)   = string("https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210416_15%25BSA_2ndBatch/ja_PreSat_v2%20(", ω1_dB, ")/1/data.2d?raw=true");

# which can be loaded with functions implemented in this file:
include(string(pathof(MRIgeneralizedBloch), "/../../docs/src/load_NMR_data.jl"));

# We store the off-resonance frequencies and wave amplitudes in a single matrix for convenience:
x = zeros(Float64, length(ω1) * length(Δ), 2)
x[:,1] = repeat(Δ, length(ω1))
x[:,2] = vec(repeat(ω1, 1, length(Δ))');

# ## MnCl``_2`` Sample
# We load the first data point of each FID:
M = zeros(Float64, length(Δ), length(ω1))
for i = 1:length(ω1_dB)
    M[:,i] = load_first_datapoint(MnCl2_data(ω1_dB[i]); set_phase=:abs)
end
M ./= maximum(M);
# In contrast to the inversion-recovery experiment, the phase of the signal was not stable. Therefore, we took the absolute value of the signal by setting the flag `set_phase=:abs`.

# The MnCl``_2``-data can be described with a single compartment model:
function single_compartment_model(x, p) 
    (m0, R1, T2) = p
    
    Δ  = @view x[:,1]
    ω1 = @view x[:,2]

    Rrf = @. π * ω1^2 * g_Lorentzian(Δ, T2)
    m = @. m0 * R1 / (R1 + Rrf)
    return m
end;
# (cf. Eqs. (14) and (15) in the [paper](https://arxiv.org/pdf/2107.11000.pdf)).

# As this model is merely a function of the relaxation times T₁ and T₂, we forgo a fitting routine and use the estimates from the [Inversion Recovery Experiments](@ref) instead:
R1 = 1.479 # 1/s
T2 = 0.075 # s
nothing # hide #md

# Visually, this model describes the data well:
p = plot(xlabel="Δ [rad/s]", ylabel="M / max(M)", xaxis=:log, legend=:none)
[scatter!(p, Δ, M[:,i], color=i) for i=1:length(ω1)]
[plot!(p, Δ, reshape(single_compartment_model(x, [1,R1,T2]), length(Δ), length(ω1))[:,i], color=i) for i=1:length(ω1)]
display(p) #!md
#md Main.HTMLPlot(p) #hide

# ## Bovine Serum Albumin Sample
# We acquired the same data for the BSA sample, which we load:
for i = 1:length(ω1_dB)
    M[:,i] = load_first_datapoint(BSA_data(ω1_dB[i]); set_phase=:abs)
end
M ./= maximum(M);

# We model the steady-state magnetization as described by [Henkelman et al.](https://doi.org/10.1002/mrm.1910290607):
function Henkelman_model(x, p; lineshape=:superLorentzian) 
    (m0, m0s, R1f, R1s, T2f, T2s, Rx) = p

    m0s /= 1 - m0s # switch from m0s + m0f = 1 to m0f = 1 normalization

    Δ  = @view x[:,1]
    ω1 = @view x[:,2]

    Rrf_f = @. π * ω1^2 * g_Lorentzian(Δ, T2f)

    if lineshape == :Lorentzian
        Rrf_s = @. π * ω1^2 * g_Lorentzian(Δ, T2s)
    elseif lineshape == :Gaussian
        Rrf_s = @. π * ω1^2 * g_Gaussian(Δ, T2s)
    elseif lineshape == :superLorentzian
        Rrf_s = @. π * ω1^2 * g_superLorentzian(Δ, T2s)
    end

    m = @. m0 * (R1s * Rx * m0s + Rrf_s * R1f + R1f * R1s + R1f * Rx) / ((R1f + Rrf_f + Rx * m0s) * (R1s + Rrf_s + Rx) - Rx^2 * m0s)
    return m
end;

# Here, we use a fitting routine to demonstrate the best possible fit with each of the three lineshapes. We define an initialization for the fitting routine `p0 = [m0, m0s, R1f, R1s, T2f, T2s, Rx]` and set some reasonable bounds:
p0   = [  1,0.01,   1,   5,0.052,  1e-5, 40]
pmin = [  0,   0,   0,   0,0.052,  1e-6,  1]
pmax = [Inf,   1, Inf, Inf,0.052, 10e-3,100];
# Note that we fixed T₂ᶠ = 52ms to the value estimated with the [Inversion Recovery Experiments](@ref) as T₂ᶠ is poorly defined by this saturation experiment. 

# ### Super-Lorentzian Lineshape
# Fitting the model with a super-Lorentzian lineshape to the data achieves good concordance:
fit = curve_fit((x, p) -> Henkelman_model(x, p; lineshape=:superLorentzian), x, vec(M), p0, lower=pmin, upper=pmax)
fit_std = stderror(fit)

p = plot(xlabel="Δ [rad/s]", ylabel="M / max(M)", xaxis=:log, legend=:none)
[scatter!(p, Δ, M[:,i], color=i) for i=1:length(ω1)]
[plot!(p, Δ, reshape(Henkelman_model(x, fit.param), length(Δ), length(ω1))[:,i], color=i) for i=1:length(ω1)]
display(p) #!md
#md Main.HTMLPlot(p) #hide


# ### Lorentzian Lineshape
# The Lorentzian lineshape, on the other hand, does not fit the data well:
fit = curve_fit((x, p) -> Henkelman_model(x, p; lineshape=:Lorentzian), x, vec(M), p0, lower=pmin, upper=pmax)
fit_std = stderror(fit)

p = plot(xlabel="Δ [rad/s]", ylabel="M / max(M)", xaxis=:log, legend=:none)
[scatter!(p, Δ, M[:,i], color=i) for i=1:length(ω1)]
[plot!(p, Δ, reshape(Henkelman_model(x, fit.param), length(Δ), length(ω1))[:,i], color=i) for i=1:length(ω1)]
display(p) #!md
#md Main.HTMLPlot(p) #hide


# ### Gaussian Lineshape
# And the Gaussian lineshape does not not fit the data well either:
fit = curve_fit((x, p) -> Henkelman_model(x, p; lineshape=:Lorentzian), x, vec(M), p0, lower=pmin, upper=pmax)
fit_std = stderror(fit)

p = plot(xlabel="Δ [rad/s]", ylabel="M / max(M)", xaxis=:log, legend=:none)
[scatter!(p, Δ, M[:,i], color=i) for i=1:length(ω1)]
[plot!(p, Δ, reshape(Henkelman_model(x, fit.param), length(Δ), length(ω1))[:,i], color=i) for i=1:length(ω1)]
display(p) #!md
#md Main.HTMLPlot(p) #hide

