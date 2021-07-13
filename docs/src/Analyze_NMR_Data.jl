# # NMR Data Analysis
# The following code replicates the NMR data anlysis in Fig. 4, including the full MnCl2 analysis that did we refrained from showing in the paper in the interest of brevity.

# For these simulations we need the following packages:
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
include(string(pathof(MRIgeneralizedBloch), "docs/load_NMR_data.jl"))
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); nothing #hide

# We measure the inversion recovery data with the following inversion times:
TI = exp.(range(log(3e-3), log(5), length=20)) # s
TI .+= 12 * TRFmin + (13 * 15.065 - 5) * 1e-6 # s - correction factors

# The inversion pulses were rectangular π-pulses with the pulse durations
TRFmin = 2 * 11.4e-6 # s - shortest TRF possible on the NMR
TRF_scale = [1;2;5:5:40] # scaling factor 
TRF = TRF_scale * TRFmin # s

# and Rabi frequencies
ω1 = π ./ TRF # rad/s

# Readout was performed with a dwell time of
T_dwell = 100e-6 # s
TIplot = exp.(range(log(TI[1]), log(TI[end]), length=500)) #hide
nothing #hide

# ## MnCl``_2`` Probe
# ### ``T_2^{*,f}`` Estimation
# We estimated ``T_2^{*,f}`` by fitting a mono-exponential decay curve to the FID of the acquisition with ``T_\text{RF} = 22.8``μs and ``T_\text{i} = 5``s.

M = load_Data(string("~/mygs/asslaj01/NMR_MT_gBloch/20210419_1mM_MnCl2/ja_IR_v2 (", TRF_scale[1], ")/1/data.2d"));
M = M[:,1]
TE = T_dwell * ((1:length(M)) .+ 7)

# For the LsqFit.jl package, we need to split the data into its real and imaginary part:
TEs = [TE;TE]
Ms = [real(M);imag(M)]

# The model we are using here is a simple mono-exponential model with a complex-valued scaling factor and a constant Larmor frequency:
m(t, p) = @. [p[1] * exp(- t[1:end ÷ 2] / p[3]) * cos(p[4] * t[1:end ÷ 2]); p[2] * exp(- t[end ÷ 2 + 1:end] / p[3]) * sin(p[4] * t[end ÷ 2 + 1:end])]

# Performing the actual fit reveals the ``T_2^{*,f}``
fit = curve_fit(m, TEs, Ms, [1.0, 1.0, .1, 0.0])
T2_MnCl2 = fit.param[3]

# and its standard deviation
fit_std = stderror(fit)[3]

# Visually, the plot and the data show good agreement:
plot(TE, abs.(M), label="data")
Mfit = m(TEs, fit.param)
Mfit = Mfit[1:end÷2] + 1im * Mfit[end÷2+1:end]
plot!(TE, abs.(Mfit), label=@sprintf("fit with T2 = %.3f", T2_MnCl2))

# The relative residuum of the fit, i.e. ``||residuum||_2/||M||_2`` is
norm(fit.resid) / norm(M)

# Despite the small residuum, the Shapiro-Wilk test indicates that the residuum is not Gaussian or normal distributed at a significance level of `α=0.05`
Pingouin.normality(fit.resid, α=0.05)

# ### Mono-Exponential IR Model
# The MnCl2 data is fitted with a mono-exponential model, for each ``T_\text{RF}`` separately. 

# Load and normalize IR data:
M = zeros(Float64, length(TI), length(TRF_scale))
for i = 1:length(TRF_scale)
    M[:,i] = load_spectral_integral(string("~/mygs/asslaj01/NMR_MT_gBloch/20210419_1mM_MnCl2/ja_IR_v2 (", TRF_scale[i], ")/1/data.2d"))
end
M ./= maximum(M)

# Here, we use a simple inversion recovery model

m(t, p) = @. p[1] - p[3] * exp(- t * p[2])

# where `p[1]` is the thermal equilibrium magnetization, `p[2]` ``= T_1``, and `p[1] - p[3]` is the magnetization right after the inversion pulse, or, in other words `Minv = p[1] / p[3] - 1` is the inverion efficiency, which is 1 for an ideal π-pulse. 

# The parameters are initialized with
p0 = [1.0, 1.0, 2.0]

# Fit mono-exponential model for each ``T_\text{RF}`` individually:
R1 = similar(M[1,:])
residuum = similar(R1)
for i = 1:length(TRF_scale)
    Mi = @view M[:,i]

    fit = curve_fit(m, TI, Mi, p0)

    R1[i] = fit.param[2]
    R1_std = stderror(fit)[2]
    Minv = fit.param[3] / fit.param[1] - 1

    residuum[i] = norm(fit.resid) / norm(Mi)

    println(@sprintf("TRF = %1.2e; ||residuum||_2/||M||_2 = %1.2e", TRF[i], norm(fit.resid) / norm(Mi)))
    print_result("R1", R1[i], R1_std, "1/s")

    if i == 1
        p = scatter(TI, Mi, label=@sprintf("TRF = %1.2es", TRF[i]), ticks=:native)
    else
        p = scatter!(TI, Mi, label=@sprintf("TRF = %1.2es", TRF[i]), ticks=:native)
    end
    p = plot!(TIplot, m(TIplot, p), label=@sprintf("R1 = %.3f/s; MInv = %.3f", R1[i], Minv))
end
#md Main.HTMLPlot(p) #hide
gui() #src

# The mean of all R1 fits in units of ``1/s`` is 
mean(R1)

# and its standard deviation in units of ``1/s`` is 
std(R1)

# The relative residuum of the fits is on average 
mean(residuum)

# with a standard deviation of
std(residuum)

# The Shapiro-Wilk test is not able to reject the null hypothesis that the residuum is Gaussian or normal distributed at a significance level of `α=0.05`
Pingouin.normality(R1, α=0.05)
# Note that this fit is based on significantly less data points compared to the ``T_2^{*,f}`` fit, which hampers a direct comparison of the statistical analysis. 


# ## Global IR fit 
# As an alterantive to individual fits to the inversion recovery curves with different ``T_\text{RF}``, we can also perform a global pulse that accounts for the ``T_2^{*,f}`` decay during the inversion pulse.

# For this purpose we can write a function that simulates the ``T_2^{*,f}`` decay during the inversion pulse, followed by ``T_1`` recovery:
function Bloch_IR_model(x, p, TRF, TI, T2)
    (m0, m0_inv, R1) = p
    
    M = zeros(Float64, length(TI), length(TRF))
    for i = 1:length(TRF)
        # simulate inversion pulse
        R2 = 1 / T2
        ω1 = π / TRF[i]
        H = [-R2 -ω1  0;
              ω1 -R1 R1;
               0   0  0]
        
        m_inv = m0_inv * (exp(H * TRF[i]) * [0,1,1])[2]

        # simulate recovery
        H = [-R1 R1 * m0;
               0       0]

        for j = 1:length(TI)
            M[j,i] = m0 * (exp(H .* (TI[j] - TRF[i] / 2)) * [m_inv,1])[1]
        end
    end
    return vec(M)
end

# We provide the previously estimated ``T_2^{*,f}`` value to the fit:
fit = curve_fit((x, p) -> Bloch_IR_model(x, p, TRF, TI, T2_MnCl2), 1:length(M), vec(M), [ 1, .8, 1])

p = scatter(TI, M, ticks=:native)
p = plot!(TIplot, reshape(Bloch_IR_model(x, fit.param, TRF, TIplot, T2_MnCl2), length(TIplot), length(TRF)), legend=false)
#md Main.HTMLPlot(p) #hide

# With this global fit, we get a very similar relaxation rate in units of ``1/s``
fit.param[3]

# With an uncertainty of 
fit_std = stderror(fit)
fit_std[3]

# Note that the relative residuum is somehwat increased when compared to individual fits to each inversion recovery curve:
norm(fit.resid) / norm(M)



## #############################################################################
# BSA
################################################################################
println("------------- BSA -------------")
# estimate T2*
M = load_Data(string("~/mygs/asslaj01/NMR_MT_gBloch/20210416_15%BSA_2ndBatch/ja_IR_v2 (", TRF_scale[1], ")/1/data.2d"));
M = M[:,1]
TE = T_dwell * ((1:length(M)) .+ 7)

# split real and imaginary part
TE = [TE;TE]
M = [real(M);imag(M)]

m(t, p) = @. [p[1] * exp(- t[1:end ÷ 2] / p[3]) * cos(p[4] * t[1:end ÷ 2]); p[2] * exp(- t[end ÷ 2 + 1:end] / p[3]) * sin(p[4] * t[end ÷ 2 + 1:end])]
fit = curve_fit(m, TE, M, [1.0, 1.0, .1, 0.0])
fit_std = stderror(fit)

T2_BSA = fit.param[3]

print_result("T2", T2_BSA, fit_std[3], "s")
println(@sprintf("||residuum||_2/||M||_2 = %1.2e", norm(fit.resid) / norm(M)))
println(Pingouin.normality(fit.resid, α=0.05))

plot(TE, M)
plot!(TE, m(TE, fit.param), label=@sprintf("T2 = %.3fs", T2_BSA))

## load IR data
M = zeros(Float64, length(TI), length(TRF_scale))
for i = 1:length(TRF_scale)
    M[:,i] = load_spectral_integral(string("~/mygs/asslaj01/NMR_MT_gBloch/20210416_15%BSA_2ndBatch/ja_IR_v2 (", TRF_scale[i], ")/1/data.2d"))
end
M ./= maximum(M)

## Fit mono-exponential model for each individual TRF
println("individual IR fits for each TRF: ")
for i = 1:length(TRF_scale)
    Mi = @view M[:,i]
    m(t, p) = @. p[1] - p[3] * exp(- t * p[2])
    p0 = [1.0, 1.0, 2.0]

    fit = curve_fit(m, TI, Mi, p0)
    fit_std = stderror(fit)

    R1[i] = fit.param[2]
    residuum[i] = norm(fit.resid) / norm(Mi)
    p = fit.param

    println(@sprintf("TRF = %1.2e; ||residuum||_2/||M||_2 = %1.2e", TRF[i], norm(fit.resid) / norm(Mi)))
    print_result("R1", p[2], fit_std[2], "/s")

    if i == 1
        scatter(TI, Mi, label=@sprintf("TRF = %1.2es", TRF[i]), ticks=:native)
    else
        scatter!(TI, Mi, label=@sprintf("TRF = %1.2es", TRF[i]), ticks=:native)
    end
    plot!(TIplot, m(TIplot, p), label=@sprintf("R1 = %.3f/s; MInv = %.3f", p[2], p[3] / p[1] - 1))
end
println(" ")
println("Statistics over all fits:")
print_result("R1", mean(R1), std(R1), "/s")
print_result("resiuum", mean(residuum), std(residuum), "")
println(Pingouin.normality(R1, α=0.05))
gui()

## gBloch fit setup
T2s_min = 5e-6
T2s_max = 15e-6

g_SL = (τ) -> quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8), 0.0, sqrt(1 / 3), 1.0)[1]
x = Fun(identity, 0..(maximum(TRF) / T2s_min))
g_SLa = g_SL(x)


## global gBloch fit
#       m0, m0f_inv, m0_s, R1,  T2_s, Rx
p0   = [ 1,    .932,   .1,  1, 10e-6, 50];
pmin = [ 0,    .100,   .0, .3,  1e-9, 10];
pmax = [Inf,    Inf,  1.0,Inf, 20e-6,1e3];

models = [:Graham, :Sled, :gBloch]
fits = similar(models, Any)
p = similar(models, Plots.Plot)
for imodel = 1:length(models)
    model = models[imodel]
    fit = curve_fit((x, p) -> gBloch_IR_model([], p, g_SLa, ω1, TRF, TI, 1 / T2_BSA, model), [], vec(M), p0, lower=pmin, upper=pmax, show_trace=false, lambda=1e-3)
    fits[imodel] = fit
    fit_std = stderror(fit)

    println(@sprintf("Global BSA fit with model %s: ||residuum||_2/||M||_2 = %1.2e", model, norm(fit.resid) / norm(M)))
    print_result("m0", fit.param[1], fit_std[1], "")
    print_result("m0f_inv", fit.param[2], fit_std[2], "")
    print_result("m0_s", fit.param[3], fit_std[3], "")
    print_result("R1", fit.param[4], fit_std[4], "1/s")
    print_result("T2_s", fit.param[end - 1], fit_std[end - 1], "s")
    print_result("Rx", fit.param[end], fit_std[end], "1/s")
    println(" ")
    
    scatter(TI, M, ticks=:native)
    Mp = reshape(gBloch_IR_model([], fit.param, g_SLa, ω1, TRF, TIplot, 1 / T2_BSA, model), length(TIplot), length(TRF))
    p[imodel] = plot!(TIplot, Mp, legend=false, title=model)
    display(p[imodel])
    sleep(.3)
end

plot(p[1], p[2], p[3])

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

for j = 1:length(TI)
    write(io, "$(@sprintf("%.2e", TI[j])) ")
    for i = 1:length(TRF)
        write(io, "$(@sprintf("%.2e", M[j,i])) ")
    end
    write(io, " \n")
end
close(io)

## export fitted curves
for imodel = 1:length(models)
    Mp = reshape(gBloch_IR_model([], fits[imodel].param, g_SLa, ω1, TRF, TIplot, 1 / T2_BSA, models[imodel]), length(TIplot), length(TRF))
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