Δ = exp.(range(log(0.01e3), log(100e3), length=20)) * 2π # rad/s

ω1_dB = -60:5:-5 # dB
ω1 = @. 10^(ω1_dB / 20) * π / 2 / 11.4e-6 # rad/s

g_Lorentzian(Δ, T2) = T2 / π / (1 + (T2 * Δ)^2)
g_Gaussian(Δ, T2) = T2 / sqrt(2π) * exp(-(T2 * Δ)^2 / 2)
g_superLorentzian(Δ, T2) = T2 * sqrt(2 / π) * quadgk(ct -> exp(- 2 *  (T2 * Δ / (3 * ct^2 - 1))^2) / abs(3 * ct^2 - 1), 0, sqrt(1 / 3), 1)[1];

using MRIgeneralizedBloch
using QuadGK
using LsqFit
using LinearAlgebra
using Statistics
using Printf
using Plots
plotlyjs(bg=RGBA(31 / 255, 36 / 255, 36 / 255, 1.0), ticks=:native); # hide

MnCl2_data(ω1_dB) = string("https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210419_1mM_MnCl2/ja_PreSat_v2%20(", ω1_dB, ")/1/data.2d?raw=true")
BSA_data(ω1_dB)   = string("https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210416_15%25BSA_2ndBatch/ja_PreSat_v2%20(", ω1_dB, ")/1/data.2d?raw=true");

include(string(pathof(MRIgeneralizedBloch), "/../../docs/src/load_NMR_data.jl"));

x = zeros(Float64, length(ω1) * length(Δ), 2)
x[:,1] = repeat(Δ, length(ω1))
x[:,2] = vec(repeat(ω1, 1, length(Δ))');

M = zeros(Float64, length(Δ), length(ω1))
for i = 1:length(ω1_dB)
    M[:,i] = load_first_datapoint(MnCl2_data(ω1_dB[i]); set_phase=:abs)
end
M ./= maximum(M);

function single_compartment_model(x, p)
    (m0, R1, T2) = p

    Δ  = @view x[:,1]
    ω1 = @view x[:,2]

    Rrf = @. π * ω1^2 * g_Lorentzian(Δ, T2)
    m = @. m0 * R1 / (R1 + Rrf)
    return m
end;

R1 = 1.479 # 1/s
T2 = 0.075 # s

p = plot(xlabel="Δ [rad/s]", ylabel="M / max(M)", xaxis=:log, legend=:none)
[scatter!(p, Δ, M[:,i], color=i) for i=1:length(ω1)]
[plot!(p, Δ, reshape(single_compartment_model(x, [1,R1,T2]), length(Δ), length(ω1))[:,i], color=i) for i=1:length(ω1)]
display(p)

for i = 1:length(ω1_dB)
    M[:,i] = load_first_datapoint(BSA_data(ω1_dB[i]); set_phase=:abs)
end
M ./= maximum(M);

function Henkelman_model(x, p; lineshape=:superLorentzian)
    (m0, m0s, R1f, R1s, T2f, T2s, Rex) = p

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

    m = @. m0 * (R1s * Rex * m0s + Rrf_s * R1f + R1f * R1s + R1f * Rex) / ((R1f + Rrf_f + Rex * m0s) * (R1s + Rrf_s + Rex) - Rex^2 * m0s)
    return m
end;

p0   = [  1,0.01,   1,   5,0.052,  1e-5, 40]
pmin = [  0,   0,   0,   0,0.052,  1e-6,  1]
pmax = [Inf,   1, Inf, Inf,0.052, 10e-3,100];

fit = curve_fit((x, p) -> Henkelman_model(x, p; lineshape=:superLorentzian), x, vec(M), p0, lower=pmin, upper=pmax)
fit_std = stderror(fit)

p = plot(xlabel="Δ [rad/s]", ylabel="M / max(M)", xaxis=:log, legend=:none)
[scatter!(p, Δ, M[:,i], color=i) for i=1:length(ω1)]
[plot!(p, Δ, reshape(Henkelman_model(x, fit.param), length(Δ), length(ω1))[:,i], color=i) for i=1:length(ω1)]
display(p)

fit = curve_fit((x, p) -> Henkelman_model(x, p; lineshape=:Lorentzian), x, vec(M), p0, lower=pmin, upper=pmax)
fit_std = stderror(fit)

p = plot(xlabel="Δ [rad/s]", ylabel="M / max(M)", xaxis=:log, legend=:none)
[scatter!(p, Δ, M[:,i], color=i) for i=1:length(ω1)]
[plot!(p, Δ, reshape(Henkelman_model(x, fit.param), length(Δ), length(ω1))[:,i], color=i) for i=1:length(ω1)]
display(p)

fit = curve_fit((x, p) -> Henkelman_model(x, p; lineshape=:Lorentzian), x, vec(M), p0, lower=pmin, upper=pmax)
fit_std = stderror(fit)

p = plot(xlabel="Δ [rad/s]", ylabel="M / max(M)", xaxis=:log, legend=:none)
[scatter!(p, Δ, M[:,i], color=i) for i=1:length(ω1)]
[plot!(p, Δ, reshape(Henkelman_model(x, fit.param), length(Δ), length(ω1))[:,i], color=i) for i=1:length(ω1)]
display(p)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
