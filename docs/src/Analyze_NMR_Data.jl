#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/Analyze_NMR_Data.ipynb) [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/build_literate/Analyze_NMR_Data.ipynb)
#nb # For interactive plots, uncomment the line `plotlyjs(ticks=:native);` and run the notebook. 

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
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #!nb
#nb ## plotlyjs(ticks=:native);

# The raw data is stored in a separate [github repository](https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData) and the following functions return the URL to the individual files:
MnCl2_data(TRF_scale) = string("https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210419_1mM_MnCl2/ja_IR_v2%20(", TRF_scale, ")/1/data.2d?raw=true")
BSA_data(TRF_scale)   = string("https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210416_15%25BSA_2ndBatch/ja_IR_v2%20(", TRF_scale, ")/1/data.2d?raw=true");

# which can be loaded with functions implemented in this file:
include(string(pathof(MRIgeneralizedBloch), "/../../docs/src/load_NMR_data.jl"));

# ## MnCl``_2`` Probe
# ### ``T_2^{*,f}`` Estimation
# We estimate ``T_2^{*,f}`` by fitting a mono-exponential decay curve to the FID of the acquisition with ``T_\text{RF} = 22.8``μs and ``T_\text{i} = 5``s.
M = load_Data(MnCl2_data(1))
M = M[:,1]; # select Ti = 5s

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
T2star_MnCl2 = fit.param[3] # s
# seconds and its uncertainty (also in units of seconds)
stderror(fit)[3] # s

# Visually, the plot and the data show good agreement:
Mfitted = FID_model(TEreal, fit.param)
Mfitted = Mfitted[1:end÷2] + 1im * Mfitted[end÷2+1:end]
p = plot(xlabel="TE [s]", ylabel="|FID(TE)| [a.u.]")
plot!(p, TE, abs.(M), label="data")
plot!(p, TE, abs.(Mfitted), label=@sprintf("fit with T2* = %2.3f ms", 1e3 * T2star_MnCl2))
#md Main.HTMLPlot(p) #hide

# The relative residual norm of the fit, i.e. ``||residual||_2/||M||_2`` is
norm(fit.resid) / norm(M)

# Despite its small ``\ell_2``-norm, the Shapiro-Wilk test indicates that the residual is not Gaussian or normal distributed at a significance level of `α=0.05`
Pingouin.normality(fit.resid, α=0.05)
# We note that mono-exponential ``T_2^*`` decays assume a Lorentzian distributed magnetic field, which is in general an assumption rather than a well-founded theory. 

# ### Mono-Exponential IR Model
# We performed several experiments in which we inverted the thermal equilibrium magnetization with rectangular π-pulses with the following pulse durations (in seconds):
TRFmin = 22.8e-6 # s - shortest TRF possible on the NMR
TRF_scale = [1;2;5:5:40] # scaling factor 
TRF = TRF_scale * TRFmin # s

# and acquired inversion recovery data at exponentially spaced inversion times (in seconds):
Ti = exp.(range(log(3e-3), log(5), length=20)) # s
Ti .+= 12 * TRFmin + (13 * 15.065 - 5) * 1e-6 # s - correction factors


# We calculate the Rabi frequencies of the RF pulses and a finer grid of ``T_\text{i}`` to plot the IR model:
ω1 = π ./ TRF # rad/s
TIplot = exp.(range(log(Ti[1]), log(Ti[end]), length=500)); # s

# After loading and normalizing the data
M = zeros(Float64, length(Ti), length(TRF_scale))
for i = 1:length(TRF_scale)
    M[:,i] = load_spectral_integral(MnCl2_data(TRF_scale[i]))
end
M ./= maximum(M);

# we analyze each inversion recovery curve that corresponds to a different ``T_\text{RF}`` separately. This allows us to fit a simple mono-exponential model
standard_IR_model(t, p) = @. p[1] - p[3] * exp(- t * p[2]);
# where `p[1]` is the thermal equilibrium magnetization, `p[2]` ``= T_1``, and `p[1] - p[3]` is the magnetization right after the inversion pulse or, equivalently, `Minv = p[1] / p[3] - 1` is the inversion efficiency, which is 1 for an ideal π-pulse and smaller otherwise. The parameters are initialized with
p0 = [1.0, 1.0, 2.0];

# and we can loop over ``T_\text{RF}`` to perform the fits:
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
display(p) #!md
#md Main.HTMLPlot(p) #hide

# Here, the data measured with different ``T_\text{RF}`` are indicated by markers in different colors, and the corresponding fits are the line plots in the same color. The fitted parameters are denoted in the legend and the mean value of all R1 estimates is 
mean(R1) # 1/s

# 1/s and its standard deviation in units of 1/s is 
std(R1) # 1/s

# The relative residual norm of the fits is on average 
mean(residual)

# Further, we cannot reject the null hypothesis that the estimated R1 values are Gaussian distributed:
Pingouin.normality(R1, α=0.05)


# ### Global IR fit 
# As an alternative to individual fits to the inversion recovery curves with different ``T_\text{RF}``, we can also perform a global fit that accounts for the ``T_2^{*,f}`` decay during the inversion pulse. The model first simulates the ``T_2^{*,f}`` decay during the inversion pulse, followed by ``T_1`` recovery:
function Bloch_IR_model(p, TRF, Ti, T2)
    (m0, m0_inv, R1) = p
    R2 = 1 / T2
    
    M = zeros(Float64, length(Ti), length(TRF))
    for i = 1:length(TRF)
        ## simulate inversion pulse
        ω1 = π / TRF[i]
        H = [-R2 -ω1  0;
              ω1 -R1 R1;
               0   0  0]
        
        m_inv = m0_inv * (exp(H * TRF[i]) * [0,1,1])[2]

        ## simulate T1 recovery
        H = [-R1 R1 * m0;
               0       0]

        for j = 1:length(Ti)
            M[j,i] = m0 * (exp(H .* (Ti[j] - TRF[i] / 2)) * [m_inv,1])[1]
        end
    end
    return vec(M)
end

# We use the previously estimated ``T_2^{*,f}`` value for the fit:
fit = curve_fit((x, p) -> Bloch_IR_model(p, TRF, Ti, T2star_MnCl2), 1:length(M), vec(M), [ 1, .8, 1])

p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti) [a.u.]")
for i=1:length(TRF)
    scatter!(p, Ti, M[:,i], label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
    plot!(p, TIplot, Bloch_IR_model(fit.param, TRF[i], TIplot, T2star_MnCl2), label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
end
display(p) #!md #hide
#md Main.HTMLPlot(p) #hide

# With this global fit, we get a very similar relaxation rate in units of 1/s
R1_MnCl2 = fit.param[3] # 1/s

# with an uncertainty (also in units of 1/s) of 
stderror(fit)[3] # 1/s

# Note that the relative residual norm is somewhat increased compared to individual fits to each inversion recovery curve:
norm(fit.resid) / norm(M)



# ## Bovine Serum Albumin Probe
# ### ``T_2^{*,f}`` Estimation
# We repeat the ``T_2^{*,f}`` estimation for the bovine serum albumin (BSA) probe by fitting a mono-exponential decay curve to the FID of the acquisition with ``T_\text{RF} = 22.8``μs and ``T_\text{i} = 5``s.

M = load_Data(BSA_data(1));
M = M[:,1] # select Ti = 5s
Mreal = [real(M);imag(M)]

fit = curve_fit(FID_model, TEreal, Mreal, [1.0, 1.0, .1, 0.0]);

# The estimated ``T_2^{*,f}`` of the BSA probe is 
T2star_BSA = fit.param[3] # s
# seconds with an uncertainty of 
stderror(fit)[3] # s
# seconds. 

# Visually, the plot and the data align well for the BSA probe, too:
Mfitted = FID_model(TEreal, fit.param)
Mfitted = Mfitted[1:end÷2] + 1im * Mfitted[end÷2+1:end]
p = plot(xlabel="TE [s]", ylabel="|FID(TE)| [a.u.]")
plot!(p, TE, abs.(M), label="data")
plot!(p, TE, abs.(Mfitted), label=@sprintf("fit with T2* = %2.3f ms", 1e3 * T2star_BSA))
#md Main.HTMLPlot(p) #hide

# The relative residual norm (``||residual||_2/||M||_2``) is
norm(fit.resid) / norm(M)

# Despite the small residual, the Shapiro-Wilk test indicates that the residual is not normal distributed for this probe either:
Pingouin.normality(fit.resid, α=0.05)


# ### Mono-Exponential IR Model
# We also fit a mono-exponential model to each inversion recovery curve of the BSA data: 
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
display(p) #!md #hide
#md Main.HTMLPlot(p) #hide

# Zooming into early phase of the recovery curve reveals the poor fit quality, in particular for long ``T_\text{RF}``. This is also reflected by a substantially larger relative residual norm compared to the MnCl``_2`` probe:
mean(residual)

# The mean of all R1 fits in units of 1/s is 
mean(R1) # 1/s

# and its standard deviation is substantially larger compared to the same fit of the MnCl``_2`` probe:
std(R1) # 1/s

# In contrast to the MnCl``_2`` probe, we can reject the null hypothesis that the R1 rates, estimated with a mono-exponential model from the BSA probe with different ``T_\text{RF}``, are Gaussian distributed:
Pingouin.normality(R1, α=0.05)




# ### Global IR Fit - Generalized Bloch Model
# In order to repeat the global fit that includes all ``T_\text{RF}`` values, we have to account for the spin dynamics in the semi-solid pool during the RF-pulse. First, we do this with the proposed generalized Bloch model:
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
end;

# Here, we use assume a super-Lorentzian lineshape, whose Green's function is interpolated to speed up the fitting routine:
T2s_min = 5e-6 # s
G_superLorentzian = interpolate_greens_function(greens_superlorentzian, 0, maximum(TRF)/T2s_min);

# The fit is initialized with `p0 = [m0, m0f_inv, m0_s, R1, T2_s, Rx]` and set some reasonable bounds to the fitted parameters:
p0   = [  1, 0.932,  0.1,   1, 10e-6, 50];
pmin = [  0, 0.100,   .0, 0.3,  1e-9, 10];
pmax = [Inf,   Inf,  1.0, Inf, 20e-6,1e3];

fit = curve_fit((x, p) -> gBloch_IR_model(p, G_superLorentzian, TRF, Ti, 1/T2star_BSA), [], vec(M), p0, lower=pmin, upper=pmax);

# Visually, the plot and the data align well:
p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti) [a.u.]")
for i=1:length(TRF)
    scatter!(p, Ti, M[:,i], label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
    plot!(p, TIplot, gBloch_IR_model(fit.param, G_superLorentzian, TRF[i], TIplot, 1/T2star_BSA), label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
end
display(p) #!md #hide
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
R1 = fit.param[4] # 1/s
#-
T2s = 1e6fit.param[5] # μs
#-
Rx = fit.param[6] # 1/s



#src #############################################################################
#src # export data
#src #############################################################################
#src measured data
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_data.txt")), "w") #src
write(io, "TI_s") #src
for i = 1:length(TRF) #src
    write(io, " z_$(@sprintf("%.2e", TRF[i]))") #src
end #src
write(io, " \n")  #src

for j = 1:length(Ti) #src
    write(io, "$(@sprintf("%.2e", Ti[j])) ") #src
    for i = 1:length(TRF) #src
        write(io, "$(@sprintf("%.2e", M[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src

#src export fitted curves
Mp = reshape(gBloch_IR_model(fit.param, G_superLorentzian, TRF, TIplot, 1/T2star_BSA), length(TIplot), length(TRF)) #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_gBloch_fit.txt")), "w") #src
write(io, "TI_s") #src
for i = 1:length(TRF) #src
    write(io, " z_$(@sprintf("%.2e", TRF[i]))") #src
end #src
write(io, " \n")  #src

for j = 1:length(TIplot) #src
    write(io, "$(@sprintf("%.2e", TIplot[j])) ") #src
    for i = 1:length(TRF) #src
        write(io, "$(@sprintf("%.2e", Mp[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src






# ### Global IR Fit - Graham's Single Frequency Approximation
# For comparison, we repeat the same fit with [Graham's single frequency approximation](http://dx.doi.org/10.1002/jmri.1880070520):
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

fit = curve_fit((x, p) -> Graham_IR_model(p, TRF, Ti, 1/T2star_BSA), [], vec(M), p0, lower=pmin, upper=pmax);

# Visually, the plot and the data align substantially worse:
p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti) [a.u.]")
for i=1:length(TRF)
    scatter!(p, Ti, M[:,i], label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
    plot!(p, TIplot, Graham_IR_model(fit.param, TRF[i], TIplot, 1/T2star_BSA), label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
end
display(p) #!md #hide
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
R1 = fit.param[4] # 1/s
#-
T2s = 1e6fit.param[5] # μs
#-
Rx = fit.param[6] # 1/s


#src #############################################################################
#src # export data
#src #############################################################################
#src export fitted curves
Mp = reshape(Graham_IR_model(fit.param, TRF, TIplot, 1/T2star_BSA), length(TIplot), length(TRF)) #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_Graham_fit.txt")), "w") #src
write(io, "TI_s") #src
for i = 1:length(TRF) #src
    write(io, " z_$(@sprintf("%.2e", TRF[i]))") #src
end #src
write(io, " \n")  #src

for j = 1:length(TIplot) #src
    write(io, "$(@sprintf("%.2e", TIplot[j])) ") #src
    for i = 1:length(TRF) #src
        write(io, "$(@sprintf("%.2e", Mp[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src




# ### Global IR Fit - Sled's Model
# We also performed the fit with [Sled's model](http://dx.doi.org/10.1006/jmre.2000.2059):
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

fit = curve_fit((x, p) -> Sled_IR_model(p, G_superLorentzian, TRF, Ti, 1/T2star_BSA), [], vec(M), p0, lower=pmin, upper=pmax);

# Visually, the plot and the data do align well either:
p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti) [a.u.]")
for i=1:length(TRF)
    scatter!(p, Ti, M[:,i], label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
    plot!(p, TIplot, Sled_IR_model(fit.param, G_superLorentzian, TRF[i], TIplot, 1/T2star_BSA), label=@sprintf("TRF = %1.2es", TRF[i]), color=i)
end
display(p) #!md #hide
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
R1 = fit.param[4] # 1/s
#-
T2s = 1e6fit.param[5] # μs
#-
Rx = fit.param[6] # 1/s

#src #############################################################################
#src # export data
#src #############################################################################
#src export fitted curves
Mp = reshape(Sled_IR_model(fit.param, G_superLorentzian, TRF, TIplot, 1/T2star_BSA), length(TIplot), length(TRF)) #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_Sled_fit.txt")), "w") #src
write(io, "TI_s") #src
for i = 1:length(TRF) #src
    write(io, " z_$(@sprintf("%.2e", TRF[i]))") #src
end #src
write(io, " \n")  #src
    
for j = 1:length(TIplot) #src
    write(io, "$(@sprintf("%.2e", TIplot[j])) ") #src
    for i = 1:length(TRF) #src
        write(io, "$(@sprintf("%.2e", Mp[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src
