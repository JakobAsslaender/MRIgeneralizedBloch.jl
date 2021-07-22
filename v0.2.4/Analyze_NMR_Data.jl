#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/Analyze_NMR_Data.ipynb)

# # NMR Data Analysis
# The following code replicates the NMR data analysis in Fig. 4, including the full MnCl``_2`` analysis that is not shown in the paper in the interest of brevity.

# For this analysis we need the following packages:
using MRIgeneralizedBloch
using DifferentialEquations
using LinearAlgebra
using LsqFit
using Statistics
import Pingouin
using Printf
using Formatting
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb
#nb plotlyjs(ticks=:native);

# The raw data is stored in a separate [github repository](https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData) and the following functions return the URL to the individual files:
MnCl2_data(TRF_scale) = string("https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210419_1mM_MnCl2/ja_IR_v2%20(", TRF_scale, ")/1/data.2d?raw=true")
BSA_data(TRF_scale)   = string("https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210416_15%25BSA_2ndBatch/ja_IR_v2%20(", TRF_scale, ")/1/data.2d?raw=true");

# which can be loaded with functions implemented in this file:
include(string(pathof(MRIgeneralizedBloch), "/../../docs/src/load_NMR_data.jl"));

# ## MnCl``_2`` Sample
# ### ``T_2^{*,f}`` Estimation
# We estimate ``T_2^{*,f}`` by fitting a mono-exponential decay curve to the FID of the acquisition with ``T_\text{RF} = 22.8``μs and ``T_\text{i} = 5``s.
M = load_Data(MnCl2_data(1))
M = M[:,1]; # select Tᵢ = 5s

# The data was measured at the following time points in units of seconds:
T_dwell = 100e-6 # s
TE = T_dwell * ((1:length(M)) .+ 7) # s

# Note that the signal is an FID, so the phrase *echo* time is a bit misleading. 

# The function [curve_fit](https://julianlsolvers.github.io/LsqFit.jl/latest/api/#LsqFit.curve_fit) from the [LsqFit.jl](https://julianlsolvers.github.io/LsqFit.jl/latest/) package is only implemented for real-valued models. To accommodate this, we need to split the data into its real and imaginary part:
TEreal = [TE;TE]
Mreal = [real(M);imag(M)];

# Here, we are using a simple mono-exponential model with a complex-valued scaling factor `p[1] + 1im p[2]`, the decay time ``T_2^{*,f} =`` `p[3]`, and the Larmor frequency `p[4]`:
FID_model(t, p) = @. [p[1] * exp(- t[1:end ÷ 2] / p[3]) * cos(p[4] * t[1:end ÷ 2]); p[2] * exp(- t[end ÷ 2 + 1:end] / p[3]) * sin(p[4] * t[end ÷ 2 + 1:end])];

# Fitting this model to the NMR data estimates ``T_2^{*,f}``:
fit = curve_fit(FID_model, TEreal, Mreal, [1, 1, 0.1, 0])
T₂star_MnCl2 = fit.param[3] # s
# seconds and its uncertainty (also in units of seconds)
stderror(fit)[3] # s

# Visually, the plot and the data show good agreement:
Mfitted = FID_model(TEreal, fit.param)
Mfitted = Mfitted[1:end÷2] + 1im * Mfitted[end÷2+1:end]
p = plot(xlabel="TE [s]", ylabel="|FID(TE)| [a.u.]")
plot!(p, TE, abs.(M), label="data")
plot!(p, TE, abs.(Mfitted), label=@sprintf("fit with T₂* = %2.3f ms", 1e3 * T₂star_MnCl2))
#md Main.HTMLPlot(p) #hide

# The relative residual norm of the fit, i.e. ``||residual||_2/||M||_2`` is
norm(fit.resid) / norm(M)

# Despite its small ``\ell_2``-norm, the Shapiro-Wilk test indicates that the residual is not Gaussian or normal distributed at a significance level of `α=0.05`
Pingouin.normality(fit.resid, α=0.05)
# We note that mono-exponential ``T_2^*`` decays assume a Lorentzian distributed magnetic field, which is in general an assumption rather than a well-founded theory. 

# ### Mono-Exponential IR Model
# We performed several experiments in which we inverted the thermal equilibrium magnetization with rectangular π-pulses with the following pulse durations (in seconds):
Tʳᶠmin = 22.8e-6 # s - shortest Tʳᶠ possible on the NMR
TRF_scale = [1;2;5:5:40] # scaling factor 
Tʳᶠ = TRF_scale * Tʳᶠmin # s

# and acquired inversion recovery data at exponentially spaced inversion times (in seconds):
Tᵢ = exp.(range(log(3e-3), log(5), length=20)) # s
Tᵢ .+= 12 * Tʳᶠmin + (13 * 15.065 - 5) * 1e-6 # s - correction factors


# We calculate the Rabi frequencies of the RF pulses and a finer grid of ``T_\text{i}`` to plot the IR model:
ω₁ = π ./ Tʳᶠ # rad/s
Tᵢplot = exp.(range(log(Tᵢ[1]), log(Tᵢ[end]), length=500)); # s

# After loading and normalizing the data
M = zeros(Float64, length(Tᵢ), length(TRF_scale))
for i = 1:length(TRF_scale)
    M[:,i] = load_spectral_integral(MnCl2_data(TRF_scale[i]))
end
M ./= maximum(M);

# we analyze each inversion recovery curve that corresponds to a different ``T_\text{RF}`` separately. This allows us to fit a simple mono-exponential model
standard_IR_model(t, p) = @. p[1] - p[3] * exp(- t * p[2]);
# where `p[1]` is the thermal equilibrium magnetization, `p[2]` ``= T_1``, and `p[1] - p[3]` is the magnetization right after the inversion pulse or, equivalently, `Minv = p[1] / p[3] - 1` is the inversion efficiency, which is 1 for an ideal π-pulse and smaller otherwise. The parameters are initialized with
p0 = [1.0, 1.0, 2.0];

# and we can loop over ``T_\text{RF}`` to perform the fits:
R₁ = similar(M[1,:])
residual = similar(R₁)
p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i = 1:length(TRF_scale)
    Mi = @view M[:,i]

    fit = curve_fit(standard_IR_model, Tᵢ, Mi, p0)

    R₁[i] = fit.param[2]
    Minv = fit.param[3] / fit.param[1] - 1

    residual[i] = norm(fit.resid) / norm(Mi)

    scatter!(p, Tᵢ, Mi, label=@sprintf("Tʳᶠ = %1.2es - data", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, standard_IR_model(Tᵢplot, fit.param), label=@sprintf("fit with R₁ = %.3f/s; MInv = %.3f", R₁[i], Minv), color=i)
end
display(p) #!md
#md Main.HTMLPlot(p) #hide

# Here, the data measured with different ``T_\text{RF}`` are indicated by markers in different colors, and the corresponding fits are the line plots in the same color. The fitted parameters are denoted in the legend and the mean value of all R₁ estimates is 
mean(R₁) # 1/s

# 1/s and its standard deviation in units of 1/s is 
std(R₁) # 1/s

# The relative residual norm of the fits is on average 
mean(residual)

# Further, we cannot reject the null hypothesis that the estimated R₁ values are Gaussian distributed:
Pingouin.normality(R₁, α=0.05)


# ### Global IR Fit 
# As an alternative to individual fits to the inversion recovery curves with different ``T_\text{RF}``, we can also perform a global fit that accounts for the ``T_2^{*,f}`` decay during the inversion pulse. The model first simulates the ``T_2^{*,f}`` decay during the inversion pulse, followed by ``T_1`` recovery:
function Bloch_IR_model(p, Tʳᶠ, Tᵢ, T2)
    (m0, m0_inv, R₁) = p
    R2 = 1 / T2
    
    M = zeros(Float64, length(Tᵢ), length(Tʳᶠ))
    for i = 1:length(Tʳᶠ)
        ## simulate inversion pulse
        ω₁ = π / Tʳᶠ[i]
        H = [-R2 -ω₁ 0 ;
              ω₁ -R₁ R₁;
               0   0 0 ]
        
        m_inv = m0_inv * (exp(H * Tʳᶠ[i]) * [0,1,1])[2]

        ## simulate T1 recovery
        H = [-R₁ R₁*m0;
               0     0]

        for j = 1:length(Tᵢ)
            M[j,i] = m0 * (exp(H .* (Tᵢ[j] - Tʳᶠ[i] / 2)) * [m_inv,1])[1]
        end
    end
    return vec(M)
end;

# We use the previously estimated ``T_2^{*,f}`` value for the fit:
fit = curve_fit((x, p) -> Bloch_IR_model(p, Tʳᶠ, Tᵢ, T₂star_MnCl2), 1:length(M), vec(M), [ 1, .8, 1])

p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i=1:length(Tʳᶠ)
    scatter!(p, Tᵢ, M[:,i], label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, Bloch_IR_model(fit.param, Tʳᶠ[i], Tᵢplot, T₂star_MnCl2), label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
end
display(p) #!md
#md Main.HTMLPlot(p) #hide

# With this global fit, we get a very similar relaxation rate in units of 1/s
R₁_MnCl2 = fit.param[3] # 1/s

# with an uncertainty (also in units of 1/s) of 
stderror(fit)[3] # 1/s

# Note that the relative residual norm is somewhat increased compared to individual fits to each inversion recovery curve:
norm(fit.resid) / norm(M)



# ## Bovine Serum Albumin Sample
# ### ``T_2^{*,f}`` Estimation
# We repeat the ``T_2^{*,f}`` estimation for the bovine serum albumin (BSA) sample by fitting a mono-exponential decay curve to the FID of the acquisition with ``T_\text{RF} = 22.8``μs and ``T_\text{i} = 5``s.

M = load_Data(BSA_data(1));
M = M[:,1] # select Tᵢ = 5s
Mreal = [real(M);imag(M)]

fit = curve_fit(FID_model, TEreal, Mreal, [1.0, 1.0, .1, 0.0]);

# The estimated ``T_2^{*,f}`` of the BSA sample is 
T₂star_BSA = fit.param[3] # s
# seconds with an uncertainty of 
stderror(fit)[3] # s
# seconds. 

# Visually, the plot and the data align well for the BSA sample, too:
Mfitted = FID_model(TEreal, fit.param)
Mfitted = Mfitted[1:end÷2] + 1im * Mfitted[end÷2+1:end]
p = plot(xlabel="TE [s]", ylabel="|FID(TE)| [a.u.]")
plot!(p, TE, abs.(M), label="data")
plot!(p, TE, abs.(Mfitted), label=@sprintf("fit with T₂* = %2.3f ms", 1e3 * T₂star_BSA))
#md Main.HTMLPlot(p) #hide

# The relative residual norm (``||residual||_2/||M||_2``) is
norm(fit.resid) / norm(M)

# Despite the small residual, the Shapiro-Wilk test indicates that the residual is not normal distributed for this sample either:
Pingouin.normality(fit.resid, α=0.05)


# ### Mono-Exponential IR Model
# We also fit a mono-exponential model to each inversion recovery curve of the BSA data: 
M = zeros(Float64, length(Tᵢ), length(TRF_scale))
for i = 1:length(TRF_scale)
    M[:,i] = load_spectral_integral(BSA_data(TRF_scale[i]))
end
M ./= maximum(M)

R₁ = similar(M[1,:])
residual = similar(R₁)
p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i = 1:length(TRF_scale)
    Mi = @view M[:,i]
    
    fit = curve_fit(standard_IR_model, Tᵢ, Mi, p0)

    R₁[i] = fit.param[2]
    Minv = fit.param[3] / fit.param[1] - 1

    residual[i] = norm(fit.resid) / norm(Mi)

    scatter!(p, Tᵢ, Mi, label=@sprintf("Tʳᶠ = %1.2es - data", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, standard_IR_model(Tᵢplot, fit.param), label=@sprintf("fit with R₁ = %.3f/s; MInv = %.3f", R₁[i], Minv), color=i)
end
display(p) #!md
#md Main.HTMLPlot(p) #hide

# Zooming into early phase of the recovery curve reveals the poor fit quality, in particular for long ``T_\text{RF}``. This is also reflected by a substantially larger relative residual norm compared to the MnCl``_2`` sample:
mean(residual)

# The mean of all R₁ fits in units of 1/s is 
mean(R₁) # 1/s

# and its standard deviation is substantially larger compared to the same fit of the MnCl``_2`` sample:
std(R₁) # 1/s

# In contrast to the MnCl``_2`` sample, we can reject the null hypothesis that the R₁ rates, estimated with a mono-exponential model from the BSA sample with different ``T_\text{RF}``, are Gaussian distributed:
Pingouin.normality(R₁, α=0.05)




# ### Global IR Fit - Generalized Bloch Model
# In order to repeat the global fit that includes all ``T_\text{RF}`` values, we have to account for the spin dynamics in the semi-solid pool during the RF-pulse. First, we do this with the proposed generalized Bloch model:
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
    for i = 1:length(Tʳᶠ)
        param = (ω₁[i], 1, 0, m0s, R₁, R2f, T₂ˢ, Rx, G)
        prob = DDEProblem(apply_hamiltonian_gbloch!, m0vec, m_fun, (0.0, Tʳᶠ[i]), param)
        m = solve(prob)[end]

        for j = 1:length(TI)
            M[j,i] = m0 * (exp(H .* (TI[j] - Tʳᶠ[i] / 2)) * [m0f_inv * m[3],m[4],1])[1]
        end
    end
    return vec(M)
end;

# Here, we use assume a super-Lorentzian lineshape, whose Green's function is interpolated to speed up the fitting routine:
T₂ˢ_min = 5e-6 # s
G_superLorentzian = interpolate_greens_function(greens_superlorentzian, 0, maximum(Tʳᶠ)/T₂ˢ_min);

# The fit is initialized with `p0 = [m0, m0f_inv, m0_s, R₁, T2_s, Rx]` and we set some reasonable bounds to the fitted parameters:
p0   = [  1, 0.932,  0.1,   1, 10e-6, 50]
pmin = [  0, 0.100,   .0, 0.3,  1e-9, 10]
pmax = [Inf,   Inf,  1.0, Inf, 20e-6,1e3]

fit = curve_fit((x, p) -> gBloch_IR_model(p, G_superLorentzian, Tʳᶠ, Tᵢ, 1/T₂star_BSA), [], vec(M), p0, lower=pmin, upper=pmax);

# Visually, the plot and the data align well:
p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i=1:length(Tʳᶠ)
    scatter!(p, Tᵢ, M[:,i], label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, gBloch_IR_model(fit.param, G_superLorentzian, Tʳᶠ[i], Tᵢplot, 1/T₂star_BSA), label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
end
display(p) #!md
#md Main.HTMLPlot(p) #hide

# which becomes particularly apparent when zooming into the beginning of the inversion recovery curves. Further, the relative residual norm is much smaller compared to the mono-exponential fit:
norm(fit.resid) / norm(M)

# The estimated parameters are
m0 = fit.param[1]
#-
Minv = fit.param[2]
#-
m0s = fit.param[3]
#-
R₁ = fit.param[4] # 1/s
#-
T₂ˢ = 1e6fit.param[5] # μs
#-
Rx = fit.param[6] # 1/s
# with the uncertainties (in the same order)
stderror(fit)

#src #############################################################################
#src # export data
#src #############################################################################
#src measured data
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_data.txt")), "w") #src
write(io, "TI_s") #src
for i = 1:length(Tʳᶠ) #src
    write(io, " z_$(@sprintf("%.2e", Tʳᶠ[i]))") #src
end #src
write(io, " \n")  #src

for j = 1:length(Tᵢ) #src
    write(io, "$(@sprintf("%.2e", Tᵢ[j])) ") #src
    for i = 1:length(Tʳᶠ) #src
        write(io, "$(@sprintf("%.2e", M[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src

#src export fitted curves
Mp = reshape(gBloch_IR_model(fit.param, G_superLorentzian, Tʳᶠ, Tᵢplot, 1/T₂star_BSA), length(Tᵢplot), length(Tʳᶠ)) #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_gBloch_fit.txt")), "w") #src
write(io, "TI_s") #src
for i = 1:length(Tʳᶠ) #src
    write(io, " z_$(@sprintf("%.2e", Tʳᶠ[i]))") #src
end #src
write(io, " \n")  #src

for j = 1:length(Tᵢplot) #src
    write(io, "$(@sprintf("%.2e", Tᵢplot[j])) ") #src
    for i = 1:length(Tʳᶠ) #src
        write(io, "$(@sprintf("%.2e", Mp[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src






# ### Global IR Fit - Graham's Single Frequency Approximation
# For comparison, we repeat the same fit with [Graham's single frequency approximation](http://dx.doi.org/10.1002/jmri.1880070520):
function Graham_IR_model(p, Tʳᶠ, TI, R2f)
    (m0, m0f_inv, m0s, R₁, T₂ˢ, Rx) = p
    m0f = 1 - m0s
    ω₁ = π ./ Tʳᶠ

    m0vec = [0, 0, m0f, m0s, 1]
    
    H = [-R₁-m0s*Rx     m0f*Rx R₁*m0f;
             m0s*Rx -R₁-m0f*Rx R₁*m0s;
              0          0         0 ]

    M = zeros(Float64, length(TI), length(Tʳᶠ))
    for i = 1:length(Tʳᶠ)
        param = (ω₁[i], 1, 0, Tʳᶠ[i], m0s, R₁, R2f, T₂ˢ, Rx)
        prob = ODEProblem(apply_hamiltonian_graham_superlorentzian!, m0vec, (0.0, Tʳᶠ[i]), param)
        m = solve(prob)[end]

        for j = 1:length(TI)
            M[j,i] = m0 * (exp(H .* (TI[j] - Tʳᶠ[i] / 2)) * [m0f_inv * m[3],m[4],1])[1]
        end
    end
    return vec(M)
end

fit = curve_fit((x, p) -> Graham_IR_model(p, Tʳᶠ, Tᵢ, 1/T₂star_BSA), [], vec(M), p0, lower=pmin, upper=pmax);

# Visually, the plot and the data align substantially worse:
p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i=1:length(Tʳᶠ)
    scatter!(p, Tᵢ, M[:,i], label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, Graham_IR_model(fit.param, Tʳᶠ[i], Tᵢplot, 1/T₂star_BSA), label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
end
display(p) #!md
#md Main.HTMLPlot(p) #hide

# which becomes particularly apparent when zooming into the beginning of the inversion recovery curves. Further, the relative residual norm is much larger compared to the generalized Bloch fit:
norm(fit.resid) / norm(M)

# The estimated parameters are
m0 = fit.param[1]
#-
Minv = fit.param[2]
#-
m0s = fit.param[3]
#-
R₁ = fit.param[4] # 1/s
#-
T₂ˢ = 1e6fit.param[5] # μs
#-
Rx = fit.param[6] # 1/s
# with the uncertainties (in the same order)
stderror(fit)


#src #############################################################################
#src # export data
#src #############################################################################
#src export fitted curves
Mp = reshape(Graham_IR_model(fit.param, Tʳᶠ, Tᵢplot, 1/T₂star_BSA), length(Tᵢplot), length(Tʳᶠ)) #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_Graham_fit.txt")), "w") #src
write(io, "TI_s") #src
for i = 1:length(Tʳᶠ) #src
    write(io, " z_$(@sprintf("%.2e", Tʳᶠ[i]))") #src
end #src
write(io, " \n")  #src

for j = 1:length(Tᵢplot) #src
    write(io, "$(@sprintf("%.2e", Tᵢplot[j])) ") #src
    for i = 1:length(Tʳᶠ) #src
        write(io, "$(@sprintf("%.2e", Mp[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src




# ### Global IR Fit - Sled's Model
# We also performed the fit with [Sled's model](http://dx.doi.org/10.1006/jmre.2000.2059):
function Sled_IR_model(p, G, Tʳᶠ, TI, R2f)
    (m0, m0f_inv, m0s, R₁, T₂ˢ, Rx) = p
    m0f = 1 - m0s
    ω₁ = π ./ Tʳᶠ

    m0vec = [0, 0, m0f, m0s, 1]
    
    H = [-R₁-m0s*Rx     m0f*Rx R₁*m0f;
             m0s*Rx -R₁-m0f*Rx R₁*m0s;
              0          0         0 ]

    M = zeros(Float64, length(TI), length(Tʳᶠ))
    for i = 1:length(Tʳᶠ)
        param = (ω₁[i], 1, 0, m0s, R₁, R2f, T₂ˢ, Rx, G)
        prob = ODEProblem(apply_hamiltonian_sled!, m0vec, (0.0, Tʳᶠ[i]), param)
        m = solve(prob)[end]

        for j = 1:length(TI)
            M[j,i] = m0 * (exp(H .* (TI[j] - Tʳᶠ[i] / 2)) * [m0f_inv * m[3],m[4],1])[1]
        end
    end
    return vec(M)
end

fit = curve_fit((x, p) -> Sled_IR_model(p, G_superLorentzian, Tʳᶠ, Tᵢ, 1/T₂star_BSA), [], vec(M), p0, lower=pmin, upper=pmax);

# Visually, the plot and the data do not align well either:
p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i=1:length(Tʳᶠ)
    scatter!(p, Tᵢ, M[:,i], label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, Sled_IR_model(fit.param, G_superLorentzian, Tʳᶠ[i], Tᵢplot, 1/T₂star_BSA), label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
end
display(p) #!md
#md Main.HTMLPlot(p) #hide

# which becomes particularly apparent when zooming into the beginning of the inversion recovery curves. Further, the relative residual norm is also large compared to the generalized Bloch fit:
norm(fit.resid) / norm(M)

# The estimated parameters are
m0 = fit.param[1]
#-
Minv = fit.param[2]
#-
m0s = fit.param[3]
#-
R₁ = fit.param[4] # 1/s
#-
T₂ˢ = 1e6 * fit.param[5] # μs
#-
Rx = fit.param[6] # 1/s
# with the uncertainties (in the same order)
stderror(fit)


#src #############################################################################
#src # export data
#src #############################################################################
#src export fitted curves
Mp = reshape(Sled_IR_model(fit.param, G_superLorentzian, Tʳᶠ, Tᵢplot, 1/T₂star_BSA), length(Tᵢplot), length(Tʳᶠ)) #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_Sled_fit.txt")), "w") #src
write(io, "TI_s") #src
for i = 1:length(Tʳᶠ) #src
    write(io, " z_$(@sprintf("%.2e", Tʳᶠ[i]))") #src
end #src
write(io, " \n")  #src
    
for j = 1:length(Tᵢplot) #src
    write(io, "$(@sprintf("%.2e", Tᵢplot[j])) ") #src
    for i = 1:length(Tʳᶠ) #src
        write(io, "$(@sprintf("%.2e", Mp[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src
