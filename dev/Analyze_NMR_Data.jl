# # NMR Data Analysis
# The following code replicates the NMR data anlysis in Fig. 4, including the full MnCl``_2`` analysis that is not shown in the paper in the interest of brevity.

# For these simulations we need the following packages:
using MRIgeneralizedBloch
using DifferentialEquations
using ApproxFun
using SpecialFunctions
using QuadGK
using LinearAlgebra
using FFTW
using LsqFit
using Statistics
import Pingouin
using Printf
using Formatting
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); nothing #hide

# and some helper functions implemented in this file:
include(string(pathof(MRIgeneralizedBloch), "/../../docs/src/load_NMR_data.jl"))
nothing #hide

# ## MnCl``_2`` Probe
# ### ``T_2^{*,f}`` Estimation
# We estimate ``T_2^{*,f}`` by fitting a mono-exponential decay curve to the FID of the acquisition with ``T_\text{RF} = 22.8``μs and ``T_\text{i} = 5``s.

M = load_Data(string("~/mygs/asslaj01/NMR_MT_gBloch/20210419_1mM_MnCl2/ja_IR_v2 (1)/1/data.2d"))
M = M[:,1] # select Ti = 5s
nothing #hide

# The data was measured at the following timepoints in units of seconds:
T_dwell = 100e-6 # s
TE = T_dwell * ((1:length(M)) .+ 7) # s

# Note that the signal is an FID, so the phrase *echo* time is a bit missleading. 

# The function [curve_fit](https://julianlsolvers.github.io/LsqFit.jl/latest/api/#LsqFit.curve_fit) from the [LsqFit.jl](https://julianlsolvers.github.io/LsqFit.jl/latest/) package allows only real-valued models. To accomodate this, we need to split the data into its real and imaginary part:
TEreal = [TE;TE]
Mreal = [real(M);imag(M)]
nothing #hide

# The model we are using here is a simple mono-exponential model with the decay time ``T_2^{*,f} =`` `p[3]`, a complex-valued scaling factor `p[1] + 1im p[2]` and a Larmor frequency `p[4]`:
FID_model(t, p) = @. [p[1] * exp(- t[1:end ÷ 2] / p[3]) * cos(p[4] * t[1:end ÷ 2]); p[2] * exp(- t[end ÷ 2 + 1:end] / p[3]) * sin(p[4] * t[end ÷ 2 + 1:end])]
nothing #hide

# Fitting this model to the NMR data reveals the ``T_2^{*,f}`` in the unit seconds
fit = curve_fit(FID_model, TEreal, Mreal, [1, 1, 0.1, 0])
T2star_MnCl2 = fit.param[3] # s

# and its uncertainty (also in units of seconds)
stderror(fit)[3] # s

# Visually, the plot and the data show good agreement:
Mfit = FID_model(TEreal, fit.param)
Mfit = Mfit[1:end÷2] + 1im * Mfit[end÷2+1:end]
p = plot(xlabel="TE [s]", ylabel="FID(TE)")
plot!(p, TE, abs.(M), label="data")
plot!(p, TE, abs.(Mfit), label=@sprintf("fit with T2* = %2.3f ms", 1e3 * T2star_MnCl2))
#md Main.HTMLPlot(p) #hide

# The relative residual of the fit, i.e. ``||residual||_2/||M||_2`` is
norm(fit.resid) / norm(M)

# Despite the small residual, the Shapiro-Wilk test indicates that the residual is not Gaussian or normal distributed at a significance level of `α=0.05`
Pingouin.normality(fit.resid, α=0.05)
# We note that mono-exponential ``T_2^*`` decays assume a Lorentzian distributed magnetic field, which is in general an assumption rather than a well-founded theory. 

# ### Mono-Exponential IR Model
# We inverted the thermal equilibrium magnetization with rectangluar π-pulses with the following pulse durations:
TRFmin = 22.8e-6 # s - shortest TRF possible on the NMR
TRF_scale = [1;2;5:5:40] # scaling factor 
TRF = TRF_scale * TRFmin # s

# and acquired inversion recovery data at the following inversion times
Ti = exp.(range(log(3e-3), log(5), length=20)) # s
Ti .+= 12 * TRFmin + (13 * 15.065 - 5) * 1e-6 # s - correction factors


# We calcualte the Rabi frequencies of the RF pulses and a finer grid of ``T_\text{i}`` to plot the IR model:
ω1 = π ./ TRF # rad/s
TIplot = exp.(range(log(Ti[1]), log(Ti[end]), length=500)) # s
nothing #hide

# After loading and normalizing the data
M = zeros(Float64, length(Ti), length(TRF_scale))
for i = 1:length(TRF_scale)
    M[:,i] = load_spectral_integral(string("~/mygs/asslaj01/NMR_MT_gBloch/20210419_1mM_MnCl2/ja_IR_v2 (", TRF_scale[i], ")/1/data.2d"))
end
M ./= maximum(M)
nothing #hide

# we treat each inversion recovery with different ``T_\text{RF}`` separately. This allows us to fit a simple mono-exponential model
standard_IR_model(t, p) = @. p[1] - p[3] * exp(- t * p[2])
nothing #hide
# where `p[1]` is the thermal equilibrium magnetization, `p[2]` ``= T_1``, and `p[1] - p[3]` is the magnetization right after the inversion pulse or, eqivalently, `Minv = p[1] / p[3] - 1` is the inverion efficiency, which is 1 for an ideal π-pulse. 

# The parameters are initialized with
p0 = [1.0, 1.0, 2.0]
nothing #hide

# and we can loop over ``T_\text{RF}`` to perform the fits:
R1 = similar(M[1,:])
residual = similar(R1)
p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti)")
for i = 1:length(TRF_scale)
    Mi = @view M[:,i]

    fit = curve_fit(standard_IR_model, Ti, Mi, p0)

    R1[i] = fit.param[2]
    Minv = fit.param[3] / fit.param[1] - 1

    residual[i] = norm(fit.resid) / norm(Mi)

    scatter!(p, Ti, Mi, label=@sprintf("TRF = %1.2es", TRF[i]))
    plot!(p, TIplot, standard_IR_model(TIplot, fit.param), label=@sprintf("R1 = %.3f/s; MInv = %.3f", R1[i], Minv))
end
gui() #hide
#md Main.HTMLPlot(p) #hide

# The mean of all R1 fits in units of ``1/s`` is 
mean(R1)

# and its standard deviation in units of ``1/s`` is 
std(R1)

# The relative residual of the fits is on average 
mean(residual)

# with a standard deviation of
std(residual)

# The Shapiro-Wilk test is not able to reject the null hypothesis that the residual is Gaussian or normal distributed at a significance level of `α=0.05`
Pingouin.normality(R1, α=0.05)

# ### Global IR fit 
# As an alterantive to individual fits to the inversion recovery curves with different ``T_\text{RF}``, we can also perform a global fit that accounts for the ``T_2^{*,f}`` decay during the inversion pulse. The model first simulates the ``T_2^{*,f}`` decay during the inversion pulse, followed by the ``T_1`` recovery:
function Bloch_IR_model(_, p, TRF, Ti, T2)
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

        ## simulate recovery
        H = [-R1 R1 * m0;
               0       0]

        for j = 1:length(Ti)
            M[j,i] = m0 * (exp(H .* (Ti[j] - TRF[i] / 2)) * [m_inv,1])[1]
        end
    end
    return vec(M)
end
nothing #hide

# We use the previously estimated ``T_2^{*,f}`` value for the fit:
fit = curve_fit((x, p) -> Bloch_IR_model(x, p, TRF, Ti, T2star_MnCl2), 1:length(M), vec(M), [ 1, .8, 1])

p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti)", legend=false)
scatter!(p, Ti, M)
plot!(p, TIplot, reshape(Bloch_IR_model(1:length(M), fit.param, TRF, TIplot, T2star_MnCl2), length(TIplot), length(TRF)))
#md Main.HTMLPlot(p) #hide

# With this global fit, we get a very similar relaxation rate in units of ``1/s``
R1 = fit.param[3]

# With an uncertainty (also in units of 1/s) of 
stderror(fit)[3]

# Note that the relative residual is somehwat increased when compared to individual fits to each inversion recovery curve:
norm(fit.resid) / norm(M)









# ## Bovine Serum Albumin Probe
# ### ``T_2^{*,f}`` Estimation
# We repeat the ``T_2^{*,f}`` estimation for the bovine serum albumin (BSA) probe by fitting a mono-exponential decay curve to the FID of the acquisition with ``T_\text{RF} = 22.8``μs and ``T_\text{i} = 5``s.

M = load_Data(string("~/mygs/asslaj01/NMR_MT_gBloch/20210416_15%BSA_2ndBatch/ja_IR_v2 (1)/1/data.2d"));
M = M[:,1] # select Ti = 5s
Mreal = [real(M);imag(M)]

fit = curve_fit(FID_model, TEreal, Mreal, [1.0, 1.0, .1, 0.0])
nothing #hide

# The fitted ``T_2^{*,f}`` of the BSA probe is 
T2star_BSA = fit.param[3] # s
# seconds with an uncertainty of 
stderror(fit)[3]
# seconds. 

# Visually, the plot and the data show good agreement:
Mfit = FID_model(TEreal, fit.param)
Mfit = Mfit[1:end÷2] + 1im * Mfit[end÷2+1:end]
p = plot(xlabel="TE [s]", ylabel="FID(TE)")
plot!(p, TE, abs.(M), label="data")
plot!(p, TE, abs.(Mfit), label=@sprintf("fit with T2* = %2.3f ms", 1e3 * T2star_BSA))
#md Main.HTMLPlot(p) #hide


# The relative residual (||residual||_2/||M||_2) is
norm(fit.resid) / norm(M)

# Despite the small residual, the Shapiro-Wilk test indicates that the residual is not normal distributed for this probe either:
Pingouin.normality(fit.resid, α=0.05)


# ### Mono-Exponential IR Model
# We also fit a mono-exponential model to the BSA data, separately for each ``T_\text{RF}``. 
M = zeros(Float64, length(Ti), length(TRF_scale))
for i = 1:length(TRF_scale)
    M[:,i] = load_spectral_integral(string("~/mygs/asslaj01/NMR_MT_gBloch/20210416_15%BSA_2ndBatch/ja_IR_v2 (", TRF_scale[i], ")/1/data.2d"))
end
M ./= maximum(M)

R1 = similar(M[1,:])
residual = similar(R1)
p = plot(xlabel="Ti [s]", ylabel="zf(TRF, Ti)")
for i = 1:length(TRF_scale)
    Mi = @view M[:,i]
    
    fit = curve_fit(standard_IR_model, Ti, Mi, p0)

    R1[i] = fit.param[2]
    Minv = fit.param[3] / fit.param[1] - 1

    residual[i] = norm(fit.resid) / norm(Mi)

    scatter!(p, Ti, Mi, label=@sprintf("TRF = %1.2es", TRF[i]))
    plot!(p, TIplot, standard_IR_model(TIplot, fit.param), label=@sprintf("R1 = %.3f/s; MInv = %.3f", R1[i], Minv))
end
gui()

#md Main.HTMLPlot(p) #hide

# The mean of all R1 fits in units of ``1/s`` is 
mean(R1)

# and its standard deviation in units of ``1/s`` is 
std(R1)

# The relative residual of the fits is on average 
mean(residual)

# with a standard deviation of
std(residual)

# The Shapiro-Wilk test is not able to reject the null hypothesis that the residual is Gaussian or normal distributed at a significance level of `α=0.05`
Pingouin.normality(R1, α=0.05)


# ### Global IR Fit - Generalized Bloch Model
# In order to repeat the global fit that includes all ``T_\text{RF}`` values, we have to account for the spin dynamics in the the semi-solid spin pool during the RF-pulse. First, we do this with the proposed generalized Bloch model:
function gBloch_IR_model(_, p, G, TRF, TI, R2f)
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




## gBloch fit setup
G_superLorentzian = interpolate_greens_function(greens_superlorentzian, 0, maximum(TRF)/5e-6)

## global gBloch fit
#       m0, m0f_inv, m0_s, R1,  T2_s, Rx
p0   = [ 1,    .932,   .1,  1, 10e-6, 50];
pmin = [ 0,    .100,   .0, .3,  1e-9, 10];
pmax = [Inf,    Inf,  1.0,Inf, 20e-6,1e3];

fit = curve_fit((x, p) -> gBloch_IR_model(undef, p, G_superLorentzian, TRF, Ti, 1/T2star_BSA), [], vec(M), p0, lower=pmin, upper=pmax, show_trace=false, lambda=1e-3)

fit_std = stderror(fit)

println(@sprintf("Global BSA fit with model %s: ||residual||_2/||M||_2 = %1.2e", model, norm(fit.resid) / norm(M)))
print_result("m0", fit.param[1], fit_std[1], "")
print_result("m0f_inv", fit.param[2], fit_std[2], "")
print_result("m0_s", fit.param[3], fit_std[3], "")
print_result("R1", fit.param[4], fit_std[4], "1/s")
print_result("T2_s", fit.param[end - 1], fit_std[end - 1], "s")
print_result("Rx", fit.param[end], fit_std[end], "1/s")
println(" ")
    

Mp = reshape(gBloch_IR_model(undef, fit.param, G_superLorentzian, TRF, TIplot, 1/T2star_BSA), length(TIplot), length(TRF))


# fitold = curve_fit((x, p) -> gBloch_IR_model(x, p, G_superLorentzian, ω1, TRF, Ti, 1/T2star_BSA, :gBloch), [], vec(M), p0, lower=pmin, upper=pmax, show_trace=false, lambda=1e-3)
# Mp = reshape(gBloch_IR_model(undef, fitold.param, G_superLorentzian, ω1, TRF, TIplot, 1/T2star_BSA, :gBloch), length(TIplot), length(TRF))

scatter(Ti, M)
plot!(TIplot, Mp, legend=false, title=model)




@benchmark curve_fit((x, p) -> gBloch_IR_model(undef, p, G_superLorentzian, TRF, Ti, 1/T2star_BSA), [], vec(M), p0, lower=pmin, upper=pmax, show_trace=false, lambda=1e-3)

@benchmark curve_fit((x, p) -> gBloch_IR_model(x, p, G_superLorentzian, ω1, TRF, Ti, 1/T2star_BSA, :gBloch), [], vec(M), p0, lower=pmin, upper=pmax, show_trace=false, lambda=1e-3)







## #############################################################################
# export data
################################################################################
# measured data
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_data.txt")), "w")
write(io, "TI_s")
for i = 1:length(TRF)
    write(io, " z_$(@sprintf("%.2e", TRF[i]))")
end
write(io, " \n") 

for j = 1:length(Ti)
    write(io, "$(@sprintf("%.2e", Ti[j])) ")
    for i = 1:length(TRF)
        write(io, "$(@sprintf("%.2e", M[j,i])) ")
    end
    write(io, " \n")
end
close(io)

## export fitted curves
for imodel = 1:length(models)
    Mp = reshape(gBloch_IR_model([], fits[imodel].param, G_superLorentzian, ω1, TRF, TIplot, 1 / T2star_BSA, models[imodel]), length(TIplot), length(TRF))
    io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_", models[imodel], "_fit.txt")), "w")
    write(io, "TI_s")
    for i = 1:length(TRF)
        write(io, " z_$(@sprintf("%.2e", TRF[i]))")
    end
    write(io, " \n") 
    
    for j = 1:length(TIplot)
        write(io, "$(@sprintf("%.2e", TIplot[j])) ")
        for i = 1:length(TRF)
            write(io, "$(@sprintf("%.2e", Mp[j,i])) ")
        end
        write(io, " \n")
    end
    close(io)
end