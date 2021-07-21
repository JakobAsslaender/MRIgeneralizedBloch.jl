#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/Greens_functions.ipynb)

# # Green's Functions
# The Green's functions are given by the Fourier transform of lineshapes. For a Lorentzian lineshape, the Green's function is
# ```math
# 	G(t,\tau) = \exp (-R_2^s (t-\tau)) \;\; \forall \;\; t \geq \tau,
# ```
# for a Gaussian lineshape it is
# ```math
# 	G(t,\tau) = \exp(- {R_2^s}^2 (t-\tau)^2 / 2)),
# ```
# and for [super-Lorentzian lineshape](http://dx.doi.org/10.1002/mrm.1910330404) it is 
# ```math
# 	G(t,\tau) = \int_0^{1} \exp \left(- {R_2^s}^2 (t - \tau)^2 \cdot  \frac{(3 \zeta^2 - 1)^2}{8} \right) d\zeta.
# ```
# As evident from these equations, the Green's functions are merely a function of ``\kappa = R_2^s \cdot (t - \tau) = (t - \tau) / T_2^s``, and in this package we implemented the functions as such: [`greens_lorentzian(κ)`](@ref), [`greens_gaussian(κ)`](@ref), and [`greens_superlorentzian(κ)`](@ref). These functions can be used to reproduce Fig. 1 in the generalized Bloch paper:

using MRIgeneralizedBloch
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb
#nb plotlyjs(ticks=:native);
#-
T₂ˢ = 10e-6 # s
t = 0 : 1e-6 : 1e-3
p = plot(yaxis=:log, ylim=(1e-6,1), xlabel="(t-τ) [ms]", ylabel="G((t-τ)/T₂ˢ)")
plot!(p, 1e3t, greens_lorentzian.(t ./ T₂ˢ), label="Lorentzian lineshape")
plot!(p, 1e3t, greens_gaussian.(t ./ T₂ˢ), label="Gaussian lineshape")
plot!(p, 1e3t, greens_superlorentzian.(t ./ T₂ˢ), label="super-Lorentzian l.")
#md Main.HTMLPlot(p) #hide

#src ###############################################################
#src export data for plotting
#src ###############################################################
using Printf #src
io = open(expanduser(string("~/Documents/Paper/2021_MT_IDE/Figures/Greens_Functions.txt")), "w") #src
write(io, "t_ms tau G_Lorentzian G_Gaussian G_superLorentzian \n") #src
for i = 1:length(t) #src
    write(io, @sprintf("%1.3e %1.3e %1.3e %1.3e %1.3e \n",  #src
    t[i] * 1e3, #src
    t[i] / T₂ˢ, #src
    greens_lorentzian(t[i] / T₂ˢ), #src
    greens_gaussian(t[i] / T₂ˢ), #src
    greens_superlorentzian(t[i] / T₂ˢ) #src
    )) #src
end #src
close(io) #src
