using MRIgeneralizedBloch
using DifferentialEquations
using LinearAlgebra
using LsqFit
using Statistics
using HypothesisTests
using Printf
using Plots
plotlyjs(bg=RGBA(31 / 255, 36 / 255, 36 / 255, 1.0), ticks=:native); #hide

MnCl2_data(TRF_scale) = string("https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210419_1mM_MnCl2/ja_IR_v2%20(", TRF_scale, ")/1/data.2d?raw=true")
BSA_data(TRF_scale)   = string("https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210416_15%25BSA_2ndBatch/ja_IR_v2%20(", TRF_scale, ")/1/data.2d?raw=true");

include(string(pathof(MRIgeneralizedBloch), "/../../docs/src/load_NMR_data.jl"));

M = load_Data(MnCl2_data(1))
M = M[:, 1]; # select Tᵢ = 5s

T_dwell = 100e-6 # s
TE = T_dwell * ((1:length(M)) .+ 7) # s

TEreal = [TE; TE]
Mreal = [real(M); imag(M)];

FID_model(t, p) = @. [p[1] * exp(-t[1:end÷2] / p[3]) * cos(p[4] * t[1:end÷2]); p[2] * exp(-t[end÷2+1:end] / p[3]) * sin(p[4] * t[end÷2+1:end])];

fit = curve_fit(FID_model, TEreal, Mreal, [1, 1, 0.1, 0])
T₂star_MnCl2 = fit.param[3] # s

stderror(fit)[3] # s

Mfitted = FID_model(TEreal, fit.param)
Mfitted = Mfitted[1:end÷2] + 1im * Mfitted[end÷2+1:end]
p = plot(xlabel="TE [s]", ylabel="|FID(TE)| [a.u.]")
plot!(p, TE, abs.(M), label="data")
plot!(p, TE, abs.(Mfitted), label=@sprintf("fit with T₂* = %2.3f ms", 1e3 * T₂star_MnCl2))

norm(fit.resid) / norm(M)

ShapiroWilkTest(fit.resid)

Tʳᶠmin = 22.8e-6 # s - shortest Tʳᶠ possible on the NMR
TRF_scale = [1; 2; 5:5:40] # scaling factor
Tʳᶠ = TRF_scale * Tʳᶠmin # s

Tᵢ = exp.(range(log(3e-3), log(5), length=20)) # s
Tᵢ .+= 12 * Tʳᶠmin + (13 * 15.065 - 5) * 1e-6 # s - correction factors

ω₁ = π ./ Tʳᶠ # rad/s
Tᵢplot = exp.(range(log(Tᵢ[1]), log(Tᵢ[end]), length=500)); # s

M = zeros(Float64, length(Tᵢ), length(TRF_scale))
for i ∈ eachindex(TRF_scale)
    M[:, i] = load_first_datapoint(MnCl2_data(TRF_scale[i]))
end
M ./= maximum(M);

standard_IR_model(t, p) = @. p[1] - p[3] * exp(-t * p[2]);

p0 = [1.0, 1.0, 2.0];

R₁ = similar(M[1, :])
Minv = similar(R₁)
residual = similar(R₁)
p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i ∈ eachindex(TRF_scale)
    Mi = @view M[:, i]

    fit = curve_fit(standard_IR_model, Tᵢ, Mi, p0)

    R₁[i] = fit.param[2]
    Minv[i] = fit.param[3] / fit.param[1] - 1

    residual[i] = norm(fit.resid) / norm(Mi)

    scatter!(p, Tᵢ, Mi, label=@sprintf("Tʳᶠ = %1.2es - data", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, standard_IR_model(Tᵢplot, fit.param), label=@sprintf("fit with R₁ = %.3f/s; MInv = %.3f", R₁[i], Minv[i]), color=i)
end
display(p)

Minv[1]

R₁[1] # 1/s

Minv[end]

R₁[end] # 1/s

mean(R₁) # 1/s

std(R₁) # 1/s

mean(residual)

function Bloch_IR_model(p, Tʳᶠ, Tᵢ, T2)
    (m0, m0_inv, R₁) = p
    R2 = 1 / T2

    M = zeros(Float64, length(Tᵢ), length(Tʳᶠ))
    for i ∈ eachindex(Tʳᶠ)
        # simulate inversion pulse
        ω₁ = π / Tʳᶠ[i]
        H = [-R2 -ω₁ 0 ;
              ω₁ -R₁ R₁;
               0   0 0 ]

        m_inv = m0_inv * (exp(H * Tʳᶠ[i])*[0, 1, 1])[2]

        # simulate T1 recovery
        H = [-R₁ R₁*m0;
               0     0]

        for j ∈ eachindex(Tᵢ)
            M[j, i] = m0 * (exp(H .* (Tᵢ[j] - Tʳᶠ[i] / 2)) * [m_inv, 1])[1]
        end
    end
    return vec(M)
end;

fit = curve_fit((x, p) -> Bloch_IR_model(p, Tʳᶠ, Tᵢ, T₂star_MnCl2), 1:length(M), vec(M), [1, 0.8, 1])

p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i ∈ eachindex(Tʳᶠ)
    scatter!(p, Tᵢ, M[:, i], label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, Bloch_IR_model(fit.param, Tʳᶠ[i], Tᵢplot, T₂star_MnCl2), label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
end
display(p)

R₁_MnCl2 = fit.param[3] # 1/s

stderror(fit)[3] # 1/s

norm(fit.resid) / norm(M)

M = load_Data(BSA_data(1));
M = M[:, 1] # select Tᵢ = 5s
Mreal = [real(M); imag(M)]

fit = curve_fit(FID_model, TEreal, Mreal, [1.0, 1.0, 0.1, 0.0]);

T₂star_BSA = fit.param[3] # s

stderror(fit)[3] # s

Mfitted = FID_model(TEreal, fit.param)
Mfitted = Mfitted[1:end÷2] + 1im * Mfitted[end÷2+1:end]
p = plot(xlabel="TE [s]", ylabel="|FID(TE)| [a.u.]")
plot!(p, TE, abs.(M), label="data")
plot!(p, TE, abs.(Mfitted), label=@sprintf("fit with T₂* = %2.3f ms", 1e3 * T₂star_BSA))

norm(fit.resid) / norm(M)

ShapiroWilkTest(fit.resid)

M = zeros(Float64, length(Tᵢ), length(TRF_scale))
for i ∈ eachindex(TRF_scale)
    M[:, i] = load_first_datapoint(BSA_data(TRF_scale[i]))
end
M ./= maximum(M)


p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i ∈ eachindex(TRF_scale)
    Mi = @view M[:, i]

    fit = curve_fit(standard_IR_model, Tᵢ, Mi, p0)

    R₁[i] = fit.param[2]
    Minv[i] = fit.param[3] / fit.param[1] - 1
    residual[i] = norm(fit.resid) / norm(Mi)

    scatter!(p, Tᵢ, Mi, label=@sprintf("Tʳᶠ = %1.2es - data", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, standard_IR_model(Tᵢplot, fit.param), label=@sprintf("fit with R₁ = %.3f/s; MInv = %.3f", R₁[i], Minv[i]), color=i)
end
display(p)

mean(residual)

Minv[1]

R₁[1] # 1/s

Minv[end]

R₁[end] # 1/s

mean(R₁) # 1/s

std(R₁) # 1/s

function gBloch_IR_model(p, G, Tʳᶠ, TI, R2f)
    (m0, m0f_inv, m0s, R₁, T₂ˢ, Rx) = p
    m0f = 1 - m0s
    ω₁ = π ./ Tʳᶠ

    m0vec = [0, 0, m0f, m0s, 1]
    m_fun(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5)


    H = [-R₁-m0s*Rx     m0f*Rx R₁*m0f;
             m0s*Rx -R₁-m0f*Rx R₁*m0s;
              0          0         0 ]

    M = zeros(Float64, length(TI), length(Tʳᶠ))
    for i ∈ eachindex(Tʳᶠ)
        param = (ω₁[i], 1, 0, m0s, R₁, R2f, Rx, R₁, T₂ˢ, G)
        prob = DDEProblem(apply_hamiltonian_gbloch!, m0vec, m_fun, (0.0, Tʳᶠ[i]), param)
        m = solve(prob).u[end]

        for j ∈ eachindex(TI)
            M[j, i] = m0 * (exp(H .* (TI[j] - Tʳᶠ[i] / 2))*[m0f_inv * m[3], m[4], 1])[1]
        end
    end
    return vec(M)
end;

T₂ˢ_min = 5e-6 # s
G_superLorentzian = interpolate_greens_function(greens_superlorentzian, 0, maximum(Tʳᶠ) / T₂ˢ_min);

p0   = [  1, 0.932,  0.1,   1, 10e-6, 50]
pmin = [  0, 0.100,   .0, 0.3,  1e-9, 10]
pmax = [Inf,   Inf,  1.0, Inf, 20e-6,1e3]

fit_gBloch = curve_fit((x, p) -> gBloch_IR_model(p, G_superLorentzian, Tʳᶠ, Tᵢ, 1 / T₂star_BSA), [], vec(M), p0, lower=pmin, upper=pmax);

p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i ∈ eachindex(Tʳᶠ)
    scatter!(p, Tᵢ, M[:, i], label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, gBloch_IR_model(fit_gBloch.param, G_superLorentzian, Tʳᶠ[i], Tᵢplot, 1 / T₂star_BSA), label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
end
display(p)

norm(fit_gBloch.resid) / norm(M)

m0 = fit_gBloch.param[1]

Minv = fit_gBloch.param[2]

m0s = fit_gBloch.param[3]

R₁ = fit_gBloch.param[4] # 1/s

T₂ˢ = 1e6fit_gBloch.param[5] # μs

Rx = fit_gBloch.param[6] # 1/s

stderror(fit_gBloch)

function Graham_IR_model(p, Tʳᶠ, TI, R2f)
    (m0, m0f_inv, m0s, R₁, T₂ˢ, Rx) = p
    m0f = 1 - m0s
    ω₁ = π ./ Tʳᶠ

    m0vec = [0, 0, m0f, m0s, 1]

    H = [-R₁-m0s*Rx     m0f*Rx R₁*m0f;
             m0s*Rx -R₁-m0f*Rx R₁*m0s;
              0          0         0 ]

    M = zeros(Float64, length(TI), length(Tʳᶠ))
    for i ∈ eachindex(Tʳᶠ)
        param = (ω₁[i], 1, 0, Tʳᶠ[i], m0s, R₁, R2f, Rx, R₁, T₂ˢ)
        prob = ODEProblem(apply_hamiltonian_graham_superlorentzian!, m0vec, (0.0, Tʳᶠ[i]), param)
        m = solve(prob).u[end]

        for j ∈ eachindex(TI)
            M[j, i] = m0 * (exp(H .* (TI[j] - Tʳᶠ[i] / 2)) * [m0f_inv * m[3], m[4], 1])[1]
        end
    end
    return vec(M)
end

fit_Graham = curve_fit((x, p) -> Graham_IR_model(p, Tʳᶠ, Tᵢ, 1 / T₂star_BSA), [], vec(M), p0, lower=pmin, upper=pmax);

p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i ∈ eachindex(Tʳᶠ)
    scatter!(p, Tᵢ, M[:, i], label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, Graham_IR_model(fit_Graham.param, Tʳᶠ[i], Tᵢplot, 1 / T₂star_BSA), label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
end
display(p)

norm(fit_Graham.resid) / norm(M)

m0 = fit_Graham.param[1]

Minv = fit_Graham.param[2]

m0s = fit_Graham.param[3]

R₁ = fit_Graham.param[4] # 1/s

T₂ˢ = 1e6fit_Graham.param[5] # μs

Rx = fit_Graham.param[6] # 1/s

stderror(fit_Graham)

function Sled_IR_model(p, G, Tʳᶠ, TI, R2f)
    (m0, m0f_inv, m0s, R₁, T₂ˢ, Rx) = p
    m0f = 1 - m0s
    ω₁ = π ./ Tʳᶠ

    m0vec = [0, 0, m0f, m0s, 1]

    H = [-R₁-m0s*Rx     m0f*Rx R₁*m0f;
             m0s*Rx -R₁-m0f*Rx R₁*m0s;
              0          0         0 ]

    M = zeros(Float64, length(TI), length(Tʳᶠ))
    for i ∈ eachindex(Tʳᶠ)
        param = (ω₁[i], 1, 0, m0s, R₁, R2f, Rx, R₁, T₂ˢ, G)
        prob = ODEProblem(apply_hamiltonian_sled!, m0vec, (0.0, Tʳᶠ[i]), param)
        m = solve(prob).u[end]

        for j ∈ eachindex(TI)
            M[j, i] = m0 * (exp(H .* (TI[j] - Tʳᶠ[i] / 2))*[m0f_inv * m[3], m[4], 1])[1]
        end
    end
    return vec(M)
end

fit_Sled = curve_fit((x, p) -> Sled_IR_model(p, G_superLorentzian, Tʳᶠ, Tᵢ, 1 / T₂star_BSA), [], vec(M), p0, lower=pmin, upper=pmax);

p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i ∈ eachindex(Tʳᶠ)
    scatter!(p, Tᵢ, M[:, i], label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, Sled_IR_model(fit_Sled.param, G_superLorentzian, Tʳᶠ[i], Tᵢplot, 1 / T₂star_BSA), label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
end
display(p)

norm(fit_Sled.resid) / norm(M)

m0 = fit_Sled.param[1]

Minv = fit_Sled.param[2]

m0s = fit_Sled.param[3]

R₁ = fit_Sled.param[4] # 1/s

T₂ˢ = 1e6fit_Sled.param[5] # μs

Rx = fit_Sled.param[6] # 1/s

stderror(fit_Sled)

resid_gBlo = similar(Tʳᶠ)
resid_Sled = similar(Tʳᶠ)
resid_Grah = similar(Tʳᶠ)
for i ∈ eachindex(Tʳᶠ)
    resid_gBlo[i] = norm(gBloch_IR_model(fit_gBloch.param, G_superLorentzian, Tʳᶠ[i], Tᵢ, 1 / T₂star_BSA) .- M[:, i]) / norm(M[:, i])
    resid_Grah[i] = norm(Graham_IR_model(fit_Graham.param, Tʳᶠ[i], Tᵢ, 1 / T₂star_BSA) .- M[:, i]) / norm(M[:, i])
    resid_Sled[i] = norm(Sled_IR_model(fit_Sled.param, G_superLorentzian, Tʳᶠ[i], Tᵢ, 1 / T₂star_BSA) .- M[:, i]) / norm(M[:, i])
end

p = plot(xlabel="Tʳᶠ [s]", ylabel="relative residual")
scatter!(p, Tʳᶠ, resid_gBlo, label="generalized Bloch model")
scatter!(p, Tʳᶠ, resid_Grah, label="Graham's spectral model")
scatter!(p, Tʳᶠ, resid_Sled, label="Sled's model")

for i ∈ eachindex(Tʳᶠ)
    resid_gBlo[i] = norm(gBloch_IR_model(fit_gBloch.param, G_superLorentzian, Tʳᶠ[i], Tᵢ, 1 / T₂star_BSA) .- M[:, i]) / norm(M[:, i])
    resid_Grah[i] = norm(Graham_IR_model(fit_gBloch.param, Tʳᶠ[i], Tᵢ, 1 / T₂star_BSA) .- M[:, i]) / norm(M[:, i])
    resid_Sled[i] = norm(Sled_IR_model(fit_gBloch.param, G_superLorentzian, Tʳᶠ[i], Tᵢ, 1 / T₂star_BSA) .- M[:, i]) / norm(M[:, i])
end

p = plot(xlabel="Tʳᶠ [s]", ylabel="relative residual")
scatter!(p, Tʳᶠ, resid_gBlo, label="generalized Bloch model")
scatter!(p, Tʳᶠ, resid_Grah, label="Graham's spectral model")
scatter!(p, Tʳᶠ, resid_Sled, label="Sled's model")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
