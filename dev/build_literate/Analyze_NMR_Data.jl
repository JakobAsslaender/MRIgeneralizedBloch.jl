using MRIgeneralizedBloch
using DifferentialEquations
using LinearAlgebra
using LsqFit
using Statistics
import Pingouin
using Printf
using Formatting
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native)

MnCl2_data(TRF_scale) = string("https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210419_1mM_MnCl2/ja_IR_v2%20(", TRF_scale, ")/1/data.2d?raw=true")
BSA_data(TRF_scale)   = string("https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210416_15%25BSA_2ndBatch/ja_IR_v2%20(", TRF_scale, ")/1/data.2d?raw=true")
nothing #hide

include(string(pathof(MRIgeneralizedBloch), "/../../docs/src/load_NMR_data.jl"))
nothing #hide

M = load_Data(MnCl2_data(1))
M = M[:,1] # select Ti = 5s
nothing #hide

T_dwell = 100e-6 # s
TE = T_dwell * ((1:length(M)) .+ 7) # s

TEreal = [TE;TE]
Mreal = [real(M);imag(M)]
nothing #hide

FID_model(t, p) = @. [p[1] * exp(- t[1:end ÷ 2] / p[3]) * cos(p[4] * t[1:end ÷ 2]); p[2] * exp(- t[end ÷ 2 + 1:end] / p[3]) * sin(p[4] * t[end ÷ 2 + 1:end])]
nothing #hide

fit = curve_fit(FID_model, TEreal, Mreal, [1, 1, 0.1, 0])
T2star_MnCl2 = fit.param[3] # s

stderror(fit)[3] # s

Mfitted = FID_model(TEreal, fit.param)
Mfitted = Mfitted[1:end÷2] + 1im * Mfitted[end÷2+1:end]
p = plot(xlabel="TE [s]", ylabel="|FID(TE)| [a.u.]")
plot!(p, TE, abs.(M), label="data")
plot!(p, TE, abs.(Mfitted), label=@sprintf("fit with T2* = %2.3f ms", 1e3 * T2star_MnCl2))

norm(fit.resid) / norm(M)

Pingouin.normality(fit.resid, α=0.05)

TRFmin = 22.8e-6 # s - shortest TRF possible on the NMR
TRF_scale = [1;2;5:5:40] # scaling factor
TRF = TRF_scale * TRFmin # s

Ti = exp.(range(log(3e-3), log(5), length=20)) # s
Ti .+= 12 * TRFmin + (13 * 15.065 - 5) * 1e-6 # s - correction factors

ω1 = π ./ TRF # rad/s
TIplot = exp.(range(log(Ti[1]), log(Ti[end]), length=500)) # s
nothing #hide

M = zeros(Float64, length(Ti), length(TRF_scale))
for i = 1:length(TRF_scale)
    M[:,i] = load_spectral_integral(MnCl2_data(TRF_scale[i]))
end
M ./= maximum(M)
nothing #hide

standard_IR_model(t, p) = @. p[1] - p[3] * exp(- t * p[2])
nothing #hide

p0 = [1.0, 1.0, 2.0]
nothing #hide

R1 = similar(M[1,:])
residual = similar(R1)
p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti) [a.u.]")
for i = 1:length(TRF_scale)
    Mi = @view M[:,i]

    fit = curve_fit(standard_IR_model, Ti, Mi, p0)

    R1[i] = fit.param[2]
    Minv = fit.param[3] / fit.param[1] - 1

    residual[i] = norm(fit.resid) / norm(Mi)

    scatter!(p, Ti, Mi, label=@sprintf("TRF = %1.2es - data", TRF[i]), color=i)
    plot!(p, TIplot, standard_IR_model(TIplot, fit.param), label=@sprintf("fit with R1 = %.3f/s; MInv = %.3f", R1[i], Minv), color=i)
end
gui() #hide

mean(R1) # 1/s

std(R1) # 1/s

mean(residual)

Pingouin.normality(R1, α=0.05)

function Bloch_IR_model(p, TRF, Ti, T2)
    (m0, m0_inv, R1) = p
    R2 = 1 / T2

    M = zeros(Float64, length(Ti), length(TRF))
    for i = 1:length(TRF)
        # simulate inversion pulse
        ω1 = π / TRF[i]
        H = [-R2 -ω1  0;
              ω1 -R1 R1;
               0   0  0]

        m_inv = m0_inv * (exp(H * TRF[i]) * [0,1,1])[2]

        # simulate T1 recovery
        H = [-R1 R1 * m0;
               0       0]

        for j = 1:length(Ti)
            M[j,i] = m0 * (exp(H .* (Ti[j] - TRF[i] / 2)) * [m_inv,1])[1]
        end
    end
    return vec(M)
end
nothing #hide

fit = curve_fit((x, p) -> Bloch_IR_model(p, TRF, Ti, T2star_MnCl2), 1:length(M), vec(M), [ 1, .8, 1])

p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti) [a.u.]")
for i=1:length(TRF)
    scatter!(p, Ti, M[:,i], label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
    plot!(p, TIplot, Bloch_IR_model(fit.param, TRF[i], TIplot, T2star_MnCl2), label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
end
gui() #hide

R1_MnCl2 = fit.param[3] # 1/s

stderror(fit)[3] # 1/s

norm(fit.resid) / norm(M)

M = load_Data(BSA_data(1));
M = M[:,1] # select Ti = 5s
Mreal = [real(M);imag(M)]

fit = curve_fit(FID_model, TEreal, Mreal, [1.0, 1.0, .1, 0.0])
nothing #hide

T2star_BSA = fit.param[3] # s

stderror(fit)[3] # s

Mfitted = FID_model(TEreal, fit.param)
Mfitted = Mfitted[1:end÷2] + 1im * Mfitted[end÷2+1:end]
p = plot(xlabel="TE [s]", ylabel="|FID(TE)| [a.u.]")
plot!(p, TE, abs.(M), label="data")
plot!(p, TE, abs.(Mfitted), label=@sprintf("fit with T2* = %2.3f ms", 1e3 * T2star_BSA))

norm(fit.resid) / norm(M)

Pingouin.normality(fit.resid, α=0.05)

M = zeros(Float64, length(Ti), length(TRF_scale))
for i = 1:length(TRF_scale)
    M[:,i] = load_spectral_integral(BSA_data(TRF_scale[i]))
end
M ./= maximum(M)

R1 = similar(M[1,:])
residual = similar(R1)
p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti) [a.u.]")
for i = 1:length(TRF_scale)
    Mi = @view M[:,i]

    fit = curve_fit(standard_IR_model, Ti, Mi, p0)

    R1[i] = fit.param[2]
    Minv = fit.param[3] / fit.param[1] - 1

    residual[i] = norm(fit.resid) / norm(Mi)

    scatter!(p, Ti, Mi, label=@sprintf("TRF = %1.2es - data", TRF[i]), color=i)
    plot!(p, TIplot, standard_IR_model(TIplot, fit.param), label=@sprintf("fit with R1 = %.3f/s; MInv = %.3f", R1[i], Minv), color=i)
end
gui() #hide

mean(residual)

mean(R1) # 1/s

std(R1) # 1/s

Pingouin.normality(R1, α=0.05)

function gBloch_IR_model(p, G, TRF, TI, R2f)
    (m0, m0f_inv, m0s, R1, T2s, Rx) = p
    m0f = 1 - m0s
    ω1 = π ./ TRF

    m0vec = [0, 0, m0f, m0s, 1]
    m_fun(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5)


    H = [-R1 - m0s * Rx       m0f * Rx R1 * m0f;
               m0s * Rx -R1 - m0f * Rx R1 * m0s;
         0              0              0       ]

    M = zeros(Float64, length(TI), length(TRF))
    for i = 1:length(TRF)
        m = solve(DDEProblem(apply_hamiltonian_gbloch!, m0vec, m_fun, (0.0, TRF[i]), (ω1[i], 1, 0, m0s, R1, R2f, T2s, Rx, G)))[end]

        for j = 1:length(TI)
            M[j,i] = m0 * (exp(H .* (TI[j] - TRF[i] / 2)) * [m0f_inv * m[3],m[4],1])[1]
        end
    end
    return vec(M)
end
nothing #hide

T2s_min = 5e-6 # s
G_superLorentzian = interpolate_greens_function(greens_superlorentzian, 0, maximum(TRF)/T2s_min)
nothing #hide

p0   = [  1, 0.932,  0.1,   1, 10e-6, 50]
pmin = [  0, 0.100,   .0, 0.3,  1e-9, 10]
pmax = [Inf,   Inf,  1.0, Inf, 20e-6,1e3]
nothing #hide

fit = curve_fit((x, p) -> gBloch_IR_model(p, G_superLorentzian, TRF, Ti, 1/T2star_BSA), [], vec(M), p0, lower=pmin, upper=pmax)
nothing #hide

p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti) [a.u.]")
for i=1:length(TRF)
    scatter!(p, Ti, M[:,i], label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
    plot!(p, TIplot, gBloch_IR_model(fit.param, G_superLorentzian, TRF[i], TIplot, 1/T2star_BSA), label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
end
gui() #hide

norm(fit.resid) / norm(M)

m0 = fit.param[1]

Minv = fit.param[2]

m0s = fit.param[3]

R1 = fit.param[4] # 1/s

T2s = 1e6fit.param[5] # μs

Rx = fit.param[6] # 1/s

function Graham_IR_model(p, TRF, TI, R2f)
    (m0, m0f_inv, m0s, R1, T2s, Rx) = p
    m0f = 1 - m0s
    ω1 = π ./ TRF

    m0vec = [0, 0, m0f, m0s, 1]

    H = [-R1 - m0s * Rx       m0f * Rx R1 * m0f;
               m0s * Rx -R1 - m0f * Rx R1 * m0s;
         0              0              0       ]

    M = zeros(Float64, length(TI), length(TRF))
    for i = 1:length(TRF)
        m = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, m0vec, (0.0, TRF[i]), (ω1[i], 1, 0, TRF[i], m0s, R1, R2f, T2s, Rx)))[end]

        for j = 1:length(TI)
            M[j,i] = m0 * (exp(H .* (TI[j] - TRF[i] / 2)) * [m0f_inv * m[3],m[4],1])[1]
        end
    end
    return vec(M)
end

fit = curve_fit((x, p) -> Graham_IR_model(p, TRF, Ti, 1/T2star_BSA), [], vec(M), p0, lower=pmin, upper=pmax)
nothing #hide

p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti) [a.u.]")
for i=1:length(TRF)
    scatter!(p, Ti, M[:,i], label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
    plot!(p, TIplot, Graham_IR_model(fit.param, TRF[i], TIplot, 1/T2star_BSA), label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
end
gui() #hide

norm(fit.resid) / norm(M)

m0 = fit.param[1]

Minv = fit.param[2]

m0s = fit.param[3]

R1 = fit.param[4] # 1/s

T2s = 1e6fit.param[5] # μs

Rx = fit.param[6] # 1/s

function Sled_IR_model(p, G, TRF, TI, R2f)
    (m0, m0f_inv, m0s, R1, T2s, Rx) = p
    m0f = 1 - m0s
    ω1 = π ./ TRF

    m0vec = [0, 0, m0f, m0s, 1]

    H = [-R1 - m0s * Rx       m0f * Rx R1 * m0f;
               m0s * Rx -R1 - m0f * Rx R1 * m0s;
         0              0              0       ]

    M = zeros(Float64, length(TI), length(TRF))
    for i = 1:length(TRF)
        m = solve(ODEProblem(apply_hamiltonian_sled!, m0vec, (0.0, TRF[i]), (ω1[i], 1, 0, m0s, R1, R2f, T2s, Rx, G)))[end]

        for j = 1:length(TI)
            M[j,i] = m0 * (exp(H .* (TI[j] - TRF[i] / 2)) * [m0f_inv * m[3],m[4],1])[1]
        end
    end
    return vec(M)
end

fit = curve_fit((x, p) -> Sled_IR_model(p, G_superLorentzian, TRF, Ti, 1/T2star_BSA), [], vec(M), p0, lower=pmin, upper=pmax)
nothing #hide

p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti) [a.u.]")
for i=1:length(TRF)
    scatter!(p, Ti, M[:,i], label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
    plot!(p, TIplot, Sled_IR_model(fit.param, G_superLorentzian, TRF[i], TIplot, 1/T2star_BSA), label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
end
gui() #hide

norm(fit.resid) / norm(M)

m0 = fit.param[1]

Minv = fit.param[2]

m0s = fit.param[3]

R1 = fit.param[4] # 1/s

T2s = 1e6fit.param[5] # μs

Rx = fit.param[6] # 1/s

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

