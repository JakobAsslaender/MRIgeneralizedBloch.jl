#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/Analyze_NMR_IR_Data.ipynb)

# # Inversion Recovery Experiments
# The following code replicates the NMR data analysis in Figs. 4-6 and complements the paper with additional analyses that are not shown in the paper in the interest of brevity.

# For this analysis we need the following packages:
using MRIgeneralizedBloch
using DifferentialEquations
using LinearAlgebra
using LsqFit
using Statistics
using HypothesisTests
using Printf
using Plots
plotlyjs(bg=RGBA(31 / 255, 36 / 255, 36 / 255, 1.0), ticks=:native); #hide #!nb

# The raw data is stored in a separate [github repository](https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData) and the following functions return the URL to the individual files:
MnCl2_data(TRF_scale) = "https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210419_1mM_MnCl2/ja_IR_v2%20($TRF_scale)/1/data.2d?raw=true"
BSA_data(TRF_scale)   = "https://github.com/JakobAsslaender/MRIgeneralizedBloch_NMRData/blob/main/20210416_15%25BSA_2ndBatch/ja_IR_v2%20($TRF_scale)/1/data.2d?raw=true";

# which can be loaded with functions implemented in this file:
include("$(pathof(MRIgeneralizedBloch))/../../docs/src/load_NMR_data.jl");

# ## MnCl``_2`` Sample
# ### ``T_2^{*,f}`` Estimation
# We estimate ``T_2^{*,f}`` by fitting a mono-exponential decay curve to the FID of the acquisition with ``T_\text{RF} = 22.8``μs and ``T_\text{i} = 5``s.
M = load_Data(MnCl2_data(1))
M = M[:, 1]; # select Tᵢ = 5s

# The data was measured at the following time points in units of seconds:
T_dwell = 100e-6 # s
TE = T_dwell * ((1:length(M)) .+ 7) # s

# Note that the signal is an FID, so the phrase *echo* time is a bit misleading.

# The function [curve_fit](https://julianlsolvers.github.io/LsqFit.jl/latest/api/#LsqFit.curve_fit) from the [LsqFit.jl](https://julianlsolvers.github.io/LsqFit.jl/latest/) package is only implemented for real-valued models. To accommodate this, we need to split the data into its real and imaginary part:
TEreal = [TE; TE]
Mreal = [real(M); imag(M)];

# Here, we are using a simple mono-exponential model with a complex-valued scaling factor `p[1] + 1im p[2]`, the decay time ``T_2^{*,f} =`` `p[3]`, and the Larmor frequency `p[4]`:
FID_model(t, p) = @. [p[1] * exp(-t[1:end÷2] / p[3]) * cos(p[4] * t[1:end÷2]); p[2] * exp(-t[end÷2+1:end] / p[3]) * sin(p[4] * t[end÷2+1:end])];

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

# The relative residual norm of the fit, i.e. ``||\text{residual}||_2/||M||_2`` is
norm(fit.resid) / norm(M)

# Despite its small ``\ell_2``-norm, the Shapiro-Wilk test rejects the null hypothesis `h_0` that the residual is Gaussian or normal distributed with a very small `p` value:
ShapiroWilkTest(fit.resid)
# We note that mono-exponential ``T_2^*`` decays assume a Lorentzian distributed magnetic field, which is in general an assumption rather than a well-founded theory.

# ### Mono-Exponential IR Model
# We performed several experiments in which we inverted the thermal equilibrium magnetization with rectangular π-pulses with the following pulse durations (in seconds):
Tʳᶠmin = 22.8e-6 # s - shortest Tʳᶠ possible on the NMR
TRF_scale = [1; 2; 5:5:40] # scaling factor
Tʳᶠ = TRF_scale * Tʳᶠmin # s

# and acquired inversion recovery data at exponentially spaced inversion times (in seconds):
Tᵢ = exp.(range(log(3e-3), log(5), length=20)) # s
Tᵢ .+= 12 * Tʳᶠmin + (13 * 15.065 - 5) * 1e-6 # s - correction factors


# We calculate the Rabi frequencies of the RF pulses and a finer grid of ``T_\text{i}`` to plot the IR model:
ω₁ = π ./ Tʳᶠ # rad/s
Tᵢplot = exp.(range(log(Tᵢ[1]), log(Tᵢ[end]), length=500)); # s

# After loading and normalizing the data
M = zeros(Float64, length(Tᵢ), length(TRF_scale))
for i ∈ eachindex(TRF_scale)
    M[:, i] = load_first_datapoint(MnCl2_data(TRF_scale[i]))
end
M ./= maximum(M);

#src #############################################################################
#src # export MnCl2 data
#src #############################################################################
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_data_MnCl2.txt")), "w") #src
write(io, "TI_s") #src
for i ∈ eachindex(Tʳᶠ) #src
    write(io, " z_$(@sprintf("%.2e", Tʳᶠ[i]))") #src
end #src
write(io, " \n")  #src

for j ∈ eachindex(Tᵢ) #src
    write(io, "$(@sprintf("%.2e", Tᵢ[j])) ") #src
    for i ∈ eachindex(Tʳᶠ) #src
        write(io, "$(@sprintf("%.2e", M[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src
#src #############################################################################

# we analyze each inversion recovery curve that corresponds to a different ``T_\text{RF}`` separately. This allows us to fit a simple mono-exponential model
standard_IR_model(t, p) = @. p[1] - p[3] * exp(-t * p[2]);
# where `p[1]` is the thermal equilibrium magnetization, `p[2]` ``= T_1``, and `p[1] - p[3]` is the magnetization right after the inversion pulse or, equivalently, `Minv = p[1] / p[3] - 1` is the inversion efficiency, which is 1 for an ideal π-pulse and smaller otherwise. The parameters are initialized with
p0 = [1.0, 1.0, 2.0];

# and we can loop over ``T_\text{RF}`` to perform the fits:
param = similar(M[1, :], Vector{Float64}) #src
R₁ = similar(M[1, :])
Minv = similar(R₁)
residual = similar(R₁)
p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i ∈ eachindex(TRF_scale)
    Mi = @view M[:, i]

    fit = curve_fit(standard_IR_model, Tᵢ, Mi, p0)
    param[i] = fit.param #src

    R₁[i] = fit.param[2]
    Minv[i] = fit.param[3] / fit.param[1] - 1

    residual[i] = norm(fit.resid) / norm(Mi)

    scatter!(p, Tᵢ, Mi, label=@sprintf("Tʳᶠ = %1.2es - data", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, standard_IR_model(Tᵢplot, fit.param), label=@sprintf("fit with R₁ = %.3f/s; MInv = %.3f", R₁[i], Minv[i]), color=i)
end
display(p) #!md
#md Main.HTMLPlot(p) #hide

#src #############################################################################
#src # export fitted curves
#src #############################################################################
Mp = [standard_IR_model(Tᵢplot, param[i]) for i ∈ eachindex(Tʳᶠ)] #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_monoExp_fit_MnCl2.txt")), "w") #src
write(io, "TI_s") #src
for i ∈ eachindex(Tʳᶠ) #src
    write(io, " z_$(@sprintf("%.2e", Tʳᶠ[i]))") #src
end #src
write(io, " \n")  #src

for j ∈ eachindex(Tᵢplot) #src
    write(io, "$(@sprintf("%.2e", Tᵢplot[j])) ") #src
    for i ∈ eachindex(Tʳᶠ) #src
        write(io, "$(@sprintf("%.2e", Mp[i][j])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src
#src #############################################################################

# Here, the data measured with different ``T_\text{RF}`` are indicated by markers in different colors, and the corresponding fits are the line plots in the same color. The fitted parameters are denoted in the legend. In the paper, we highlight the estimated inversion efficiency and the relaxation rate of the dataset acquired with ``T_\text{RF}=22.8``μs:
Minv[1]
#-
R₁[1] # 1/s
# and of the dataset acquired with ``T_\text{RF}=912``μs:
Minv[end]
#-
R₁[end] # 1/s
# The mean value of all R₁ estimates is
mean(R₁) # 1/s

# 1/s and its standard deviation in units of 1/s is
std(R₁) # 1/s

# The relative residual norm of the fits is on average
mean(residual)

# ### Global IR Fit
# As an alternative to individual fits to the inversion recovery curves with different ``T_\text{RF}``, we can also perform a global fit that accounts for the ``T_2^{*,f}`` decay during the inversion pulse. The model first simulates the ``T_2^{*,f}`` decay during the inversion pulse, followed by ``T_1`` recovery:
function Bloch_IR_model(p, Tʳᶠ, Tᵢ, T2)
    (m0, m0_inv, R₁) = p
    R2 = 1 / T2

    M = zeros(Float64, length(Tᵢ), length(Tʳᶠ))
    for i ∈ eachindex(Tʳᶠ)
        ## simulate inversion pulse
        ω₁ = π / Tʳᶠ[i]
        H = [-R2 -ω₁ 0 ;
              ω₁ -R₁ R₁;
               0   0 0 ]

        m_inv = m0_inv * (exp(H * Tʳᶠ[i])*[0, 1, 1])[2]

        ## simulate T1 recovery
        H = [-R₁ R₁*m0;
               0     0]

        for j ∈ eachindex(Tᵢ)
            M[j, i] = m0 * (exp(H .* (Tᵢ[j] - Tʳᶠ[i] / 2)) * [m_inv, 1])[1]
        end
    end
    return vec(M)
end;

# We use the previously estimated ``T_2^{*,f}`` value for the fit:
fit = curve_fit((x, p) -> Bloch_IR_model(p, Tʳᶠ, Tᵢ, T₂star_MnCl2), 1:length(M), vec(M), [1, 0.8, 1])

p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i ∈ eachindex(Tʳᶠ)
    scatter!(p, Tᵢ, M[:, i], label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
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
M = M[:, 1] # select Tᵢ = 5s
Mreal = [real(M); imag(M)]

fit = curve_fit(FID_model, TEreal, Mreal, [1.0, 1.0, 0.1, 0.0]);

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

# The relative residual norm (``||\text{residual}||_2/||M||_2``) is
norm(fit.resid) / norm(M)

# Despite the small residual, the Shapiro-Wilk test indicates that the residual is not normal distributed for this sample either:
ShapiroWilkTest(fit.resid)


# ### Mono-Exponential IR Model
# We also fit a mono-exponential model to each inversion recovery curve of the BSA data:
M = zeros(Float64, length(Tᵢ), length(TRF_scale))
for i ∈ eachindex(TRF_scale)
    M[:, i] = load_first_datapoint(BSA_data(TRF_scale[i]))
end
M ./= maximum(M)
#src #############################################################################
#src # export BSA data
#src #############################################################################
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_data_BSA.txt")), "w") #src
write(io, "TI_s") #src
for i ∈ eachindex(Tʳᶠ) #src
    write(io, " z_$(@sprintf("%.2e", Tʳᶠ[i]))") #src
end #src
write(io, " \n")  #src

for j ∈ eachindex(Tᵢ) #src
    write(io, "$(@sprintf("%.2e", Tᵢ[j])) ") #src
    for i ∈ eachindex(Tʳᶠ) #src
        write(io, "$(@sprintf("%.2e", M[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src
#src #############################################################################

p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i ∈ eachindex(TRF_scale)
    Mi = @view M[:, i]

    fit = curve_fit(standard_IR_model, Tᵢ, Mi, p0)
    param[i] = fit.param #src

    R₁[i] = fit.param[2]
    Minv[i] = fit.param[3] / fit.param[1] - 1
    residual[i] = norm(fit.resid) / norm(Mi)

    scatter!(p, Tᵢ, Mi, label=@sprintf("Tʳᶠ = %1.2es - data", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, standard_IR_model(Tᵢplot, fit.param), label=@sprintf("fit with R₁ = %.3f/s; MInv = %.3f", R₁[i], Minv[i]), color=i)
end
display(p) #!md
#md Main.HTMLPlot(p) #hide

#src #############################################################################
#src # export fitted curves
#src #############################################################################
Mp = [standard_IR_model(Tᵢplot, param[i]) for i ∈ eachindex(Tʳᶠ)] #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_monoExp_fit_BSA.txt")), "w") #src
write(io, "TI_s") #src
for i ∈ eachindex(Tʳᶠ) #src
    write(io, " z_$(@sprintf("%.2e", Tʳᶠ[i]))") #src
end #src
write(io, " \n")  #src

for j ∈ eachindex(Tᵢplot) #src
    write(io, "$(@sprintf("%.2e", Tᵢplot[j])) ") #src
    for i ∈ eachindex(Tʳᶠ) #src
        write(io, "$(@sprintf("%.2e", Mp[i][j])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src
#src #############################################################################


# Zooming into the early phase of the recovery curve reveals the poor fit quality, in particular for long ``T_\text{RF}``. This is also reflected by a substantially larger relative residual norm compared to the MnCl``_2`` sample:
mean(residual)

# In the paper, we highlight the estimated inversion efficiency and the relaxation rate of the dataset acquired with ``T_\text{RF}=22.8``μs
Minv[1]
#-
R₁[1] # 1/s
# and of the dataset acquired with ``T_\text{RF}=912``μs
Minv[end]
#-
R₁[end] # 1/s
# The mean value of all R₁ estimates is
mean(R₁) # 1/s

# and its standard deviation is substantially larger compared to the same fit of the MnCl``_2`` sample:
std(R₁) # 1/s




# ### Global IR Fit - Generalized Bloch Model
# In order to repeat the global fit that includes all ``T_\text{RF}`` values, we have to account for the spin dynamics in the semi-solid pool during the RF-pulse. First, we do this with the proposed generalized Bloch model:
function gBloch_IR_model(p, G, Tʳᶠ, TI, R2f)
    (m0, m0f_inv, m0s, R₁, T₂ˢ, Rex) = p
    m0f = 1 - m0s
    ω₁ = π ./ Tʳᶠ

    m0vec = [0, 0, m0f, m0s, 1]
    m_fun(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5)


    H = [-R₁-m0s*Rex     m0f*Rex R₁*m0f;
             m0s*Rex -R₁-m0f*Rex R₁*m0s;
              0           0          0 ]

    M = zeros(Float64, length(TI), length(Tʳᶠ))
    for i ∈ eachindex(Tʳᶠ)
        param = (ω₁[i], 1, 0, m0s, R₁, R2f, Rex, R₁, T₂ˢ, G)
        prob = DDEProblem(apply_hamiltonian_gbloch!, m0vec, m_fun, (0.0, Tʳᶠ[i]), param)
        m = solve(prob, MethodOfSteps(Tsit5())).u[end]

        for j ∈ eachindex(TI)
            M[j, i] = m0 * (exp(H .* (TI[j] - Tʳᶠ[i] / 2))*[m0f_inv * m[3], m[4], 1])[1]
        end
    end
    return vec(M)
end;

# Here, we use assume a super-Lorentzian lineshape, whose Green's function is interpolated to speed up the fitting routine:
T₂ˢ_min = 5e-6 # s
G_superLorentzian = interpolate_greens_function(greens_superlorentzian, 0, maximum(Tʳᶠ) / T₂ˢ_min);

# The fit is initialized with `p0 = [m0, m0f_inv, m0_s, R₁, T2_s, Rex]` and we set some reasonable bounds to the fitted parameters:
p0   = [  1, 0.932,  0.1,   1, 10e-6, 50]
pmin = [  0, 0.100,   .0, 0.3,  1e-9, 10]
pmax = [Inf,   Inf,  1.0, Inf, 20e-6,1e3]

fit_gBloch = curve_fit((x, p) -> gBloch_IR_model(p, G_superLorentzian, Tʳᶠ, Tᵢ, 1 / T₂star_BSA), [], vec(M), p0, lower=pmin, upper=pmax);

# Visually, the plot and the data align well:
p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i ∈ eachindex(Tʳᶠ)
    scatter!(p, Tᵢ, M[:, i], label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, gBloch_IR_model(fit_gBloch.param, G_superLorentzian, Tʳᶠ[i], Tᵢplot, 1 / T₂star_BSA), label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
end
display(p) #!md
#md Main.HTMLPlot(p) #hide

# which becomes particularly apparent when zooming into the beginning of the inversion recovery curves. Further, the relative residual norm is much smaller compared to the mono-exponential fit:
norm(fit_gBloch.resid) / norm(M)

# The estimated parameters are
m0 = fit_gBloch.param[1]
#-
Minv = fit_gBloch.param[2]
#-
m0s = fit_gBloch.param[3]
#-
R₁ = fit_gBloch.param[4] # 1/s
#-
T₂ˢ = 1e6fit_gBloch.param[5] # μs
#-
Rex = fit_gBloch.param[6] # 1/s
# with the uncertainties (in the same order)
stderror(fit_gBloch)


#src #############################################################################
#src # export fitted curves
#src #############################################################################
Mp = reshape(gBloch_IR_model(fit_gBloch.param, G_superLorentzian, Tʳᶠ, Tᵢplot, 1 / T₂star_BSA), length(Tᵢplot), length(Tʳᶠ)) #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_gBloch_fit.txt")), "w") #src
write(io, "TI_s") #src
for i ∈ eachindex(Tʳᶠ) #src
    write(io, " z_$(@sprintf("%.2e", Tʳᶠ[i]))") #src
end #src
write(io, " \n")  #src

for j ∈ eachindex(Tᵢplot) #src
    write(io, "$(@sprintf("%.2e", Tᵢplot[j])) ") #src
    for i ∈ eachindex(Tʳᶠ) #src
        write(io, "$(@sprintf("%.2e", Mp[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src
#src #############################################################################





# ### Global IR Fit - Graham's Spectral Model
# For comparison, we repeat the same fit with [Graham's spectral model](http://dx.doi.org/10.1002/jmri.1880070520):
function Graham_IR_model(p, Tʳᶠ, TI, R2f)
    (m0, m0f_inv, m0s, R₁, T₂ˢ, Rex) = p
    m0f = 1 - m0s
    ω₁ = π ./ Tʳᶠ

    m0vec = [0, 0, m0f, m0s, 1]

    H = [-R₁-m0s*Rex     m0f*Rex R₁*m0f;
             m0s*Rex -R₁-m0f*Rex R₁*m0s;
              0           0          0 ]

    M = zeros(Float64, length(TI), length(Tʳᶠ))
    for i ∈ eachindex(Tʳᶠ)
        param = (ω₁[i], 1, 0, Tʳᶠ[i], m0s, R₁, R2f, Rex, R₁, T₂ˢ)
        prob = ODEProblem(apply_hamiltonian_graham_superlorentzian!, m0vec, (0.0, Tʳᶠ[i]), param)
        m = solve(prob).u[end]

        for j ∈ eachindex(TI)
            M[j, i] = m0 * (exp(H .* (TI[j] - Tʳᶠ[i] / 2)) * [m0f_inv * m[3], m[4], 1])[1]
        end
    end
    return vec(M)
end

fit_Graham = curve_fit((x, p) -> Graham_IR_model(p, Tʳᶠ, Tᵢ, 1 / T₂star_BSA), [], vec(M), p0, lower=pmin, upper=pmax);

# Visually, the plot and the data align substantially worse:
p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i ∈ eachindex(Tʳᶠ)
    scatter!(p, Tᵢ, M[:, i], label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, Graham_IR_model(fit_Graham.param, Tʳᶠ[i], Tᵢplot, 1 / T₂star_BSA), label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
end
display(p) #!md
#md Main.HTMLPlot(p) #hide

# which becomes particularly apparent when zooming into the beginning of the inversion recovery curves. Further, the relative residual norm is much larger compared to the generalized Bloch fit:
norm(fit_Graham.resid) / norm(M)

# The estimated parameters are
m0 = fit_Graham.param[1]
#-
Minv = fit_Graham.param[2]
#-
m0s = fit_Graham.param[3]
#-
R₁ = fit_Graham.param[4] # 1/s
#-
T₂ˢ = 1e6fit_Graham.param[5] # μs
#-
Rex = fit_Graham.param[6] # 1/s
# with the uncertainties (in the same order)
stderror(fit_Graham)


#src #############################################################################
#src # export data
#src #############################################################################
#src export fitted curves
Mp = reshape(Graham_IR_model(fit_Graham.param, Tʳᶠ, Tᵢplot, 1 / T₂star_BSA), length(Tᵢplot), length(Tʳᶠ)) #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_Graham_fit.txt")), "w") #src
write(io, "TI_s") #src
for i ∈ eachindex(Tʳᶠ) #src
    write(io, " z_$(@sprintf("%.2e", Tʳᶠ[i]))") #src
end #src
write(io, " \n")  #src

for j ∈ eachindex(Tᵢplot) #src
    write(io, "$(@sprintf("%.2e", Tᵢplot[j])) ") #src
    for i ∈ eachindex(Tʳᶠ) #src
        write(io, "$(@sprintf("%.2e", Mp[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src




# ### Global IR Fit - Sled's Model
# We also performed the fit with [Sled's model](http://dx.doi.org/10.1006/jmre.2000.2059):
function Sled_IR_model(p, G, Tʳᶠ, TI, R2f)
    (m0, m0f_inv, m0s, R₁, T₂ˢ, Rex) = p
    m0f = 1 - m0s
    ω₁ = π ./ Tʳᶠ

    m0vec = [0, 0, m0f, m0s, 1]

    H = [-R₁-m0s*Rex     m0f*Rex R₁*m0f;
             m0s*Rex -R₁-m0f*Rex R₁*m0s;
              0           0          0 ]

    M = zeros(Float64, length(TI), length(Tʳᶠ))
    for i ∈ eachindex(Tʳᶠ)
        param = (ω₁[i], 1, 0, m0s, R₁, R2f, Rex, R₁, T₂ˢ, G)
        prob = ODEProblem(apply_hamiltonian_sled!, m0vec, (0.0, Tʳᶠ[i]), param)
        m = solve(prob).u[end]

        for j ∈ eachindex(TI)
            M[j, i] = m0 * (exp(H .* (TI[j] - Tʳᶠ[i] / 2))*[m0f_inv * m[3], m[4], 1])[1]
        end
    end
    return vec(M)
end

fit_Sled = curve_fit((x, p) -> Sled_IR_model(p, G_superLorentzian, Tʳᶠ, Tᵢ, 1 / T₂star_BSA), [], vec(M), p0, lower=pmin, upper=pmax);

# Visually, the plot and the data do not align well either:
p = plot(xlabel="Tᵢ [s]", ylabel="zᶠ(Tʳᶠ, Tᵢ) [a.u.]")
for i ∈ eachindex(Tʳᶠ)
    scatter!(p, Tᵢ, M[:, i], label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
    plot!(p, Tᵢplot, Sled_IR_model(fit_Sled.param, G_superLorentzian, Tʳᶠ[i], Tᵢplot, 1 / T₂star_BSA), label=@sprintf("Tʳᶠ = %1.2es", Tʳᶠ[i]), color=i)
end
display(p) #!md
#md Main.HTMLPlot(p) #hide

# which becomes particularly apparent when zooming into the beginning of the inversion recovery curves. Further, the relative residual norm is also large compared to the generalized Bloch fit:
norm(fit_Sled.resid) / norm(M)

# The estimated parameters are
m0 = fit_Sled.param[1]
#-
Minv = fit_Sled.param[2]
#-
m0s = fit_Sled.param[3]
#-
R₁ = fit_Sled.param[4] # 1/s
#-
T₂ˢ = 1e6fit_Sled.param[5] # μs
#-
Rex = fit_Sled.param[6] # 1/s
# with the uncertainties (in the same order)
stderror(fit_Sled)


#src #############################################################################
#src # export data
#src #############################################################################
#src export fitted curves
Mp = reshape(Sled_IR_model(fit_Sled.param, G_superLorentzian, Tʳᶠ, Tᵢplot, 1 / T₂star_BSA), length(Tᵢplot), length(Tʳᶠ)) #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_Sled_fit.txt")), "w") #src
write(io, "TI_s") #src
for i ∈ eachindex(Tʳᶠ) #src
    write(io, " z_$(@sprintf("%.2e", Tʳᶠ[i]))") #src
end #src
write(io, " \n")  #src

for j ∈ eachindex(Tᵢplot) #src
    write(io, "$(@sprintf("%.2e", Tᵢplot[j])) ") #src
    for i ∈ eachindex(Tʳᶠ) #src
        write(io, "$(@sprintf("%.2e", Mp[j,i])) ") #src
    end #src
    write(io, " \n") #src
end #src
close(io) #src

# ### Analysis of the Residuals
# In order to visualize how well the three models align with the data at different ``T_\text{RF}``, we calculate the ``\ell_2``-norm of the residuals after subtracting the modeled from the measured signal and normalize it by the ``\ell_2``-norm of the signal:

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
#md Main.HTMLPlot(p) #hide

#src #############################################################################
#src # export data
#src #############################################################################
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_residuum_each_fit.txt")), "w") #src
write(io, "TRF_ms Graham_percent Sled_percent gBloch_percent \n")  #src

for i ∈ eachindex(Tʳᶠ) #src
    write(io, "$(@sprintf("%.2e", 1e3 * Tʳᶠ[i])) ") #src
    write(io, "$(@sprintf("%.2e", 1e2 * resid_Grah[i])) ") #src
    write(io, "$(@sprintf("%.2e", 1e2 * resid_Sled[i])) ") #src
    write(io, "$(@sprintf("%.2e", 1e2 * resid_gBlo[i])) ") #src
    write(io, " \n") #src
end #src
close(io) #src


# This analysis examines the residuals from the actual fits, i.e. it uses the biophysical parameters of respective fit to model the signal. The disadvantage of this approach is that residuals at long ``T_\text{RF}`` are negatively affected by the poor fits of Graham's and Sled's models at short ``T_\text{RF}``. This problem is overcome by subtracting the measured signal from signal that is simulated with the biophysical parameters that were estimated by fitting the generalized Bloch model:

for i ∈ eachindex(Tʳᶠ)
    resid_gBlo[i] = norm(gBloch_IR_model(fit_gBloch.param, G_superLorentzian, Tʳᶠ[i], Tᵢ, 1 / T₂star_BSA) .- M[:, i]) / norm(M[:, i])
    resid_Grah[i] = norm(Graham_IR_model(fit_gBloch.param, Tʳᶠ[i], Tᵢ, 1 / T₂star_BSA) .- M[:, i]) / norm(M[:, i])
    resid_Sled[i] = norm(Sled_IR_model(fit_gBloch.param, G_superLorentzian, Tʳᶠ[i], Tᵢ, 1 / T₂star_BSA) .- M[:, i]) / norm(M[:, i])
end

p = plot(xlabel="Tʳᶠ [s]", ylabel="relative residual")
scatter!(p, Tʳᶠ, resid_gBlo, label="generalized Bloch model")
scatter!(p, Tʳᶠ, resid_Grah, label="Graham's spectral model")
scatter!(p, Tʳᶠ, resid_Sled, label="Sled's model")
#md Main.HTMLPlot(p) #hide

# One can observe reduced residuals for Graham's and Sled's models for long ``T_\text{RF}`` as a trade off for larger residuals at short ``T_\text{RF}``. Yet, the residuals at long ``T_\text{RF}`` are still substantially larger compared to ones of the generalized Bloch model.

#src #############################################################################
#src # export data
#src #############################################################################
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/IR_residuum_gBloch_est.txt")), "w") #src
write(io, "TRF_ms Graham_percent Sled_percent gBloch_percent \n")  #src

for i ∈ eachindex(Tʳᶠ) #src
    write(io, "$(@sprintf("%.2e", 1e3 * Tʳᶠ[i])) ") #src
    write(io, "$(@sprintf("%.2e", 1e2 * resid_Grah[i])) ") #src
    write(io, "$(@sprintf("%.2e", 1e2 * resid_Sled[i])) ") #src
    write(io, "$(@sprintf("%.2e", 1e2 * resid_gBlo[i])) ") #src
    write(io, " \n") #src
end #src
close(io) #src