#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/Greens_functions.ipynb) [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/build_literate/Greens_functions.ipynb)

# # Green's Functions
# The Green's functions are given by the Fourier transform of lineshapes. For a Lorentzian lineshape, the Green's function is
# ```math
# 	G(t,\tau) = \exp (-R_2^s (t-\tau)) \;\; \forall \;\; t \geq \tau,
# ```
# for a Gaussian lineshape it is
# ```math
# 	G(t,\tau) = \exp(- {R_2^s}^2 (t-\tau)^2 / 2)),
# ```
# and for super-Lorentzian it is 
# ```math
# 	G(t,\tau) = \int_0^{1} \exp \left(- {R_2^s}^2 (t - \tau)^2 \cdot  \frac{(3 \zeta^2 - 1)^2}{8} \right) d\zeta.
# ```
# As evident from these equations, the Green's functions are merely a function of ``\kappa = R_2^s \cdot (t - \tau) = (t - \tau) / T_2^s``, and in this package we implemented the functions as such: `greens_lorentzian(κ)`, `greens_gaussian(κ)`, and `greens_superlorentzian(κ)`. These functions can be used to reproduce Fig. 1 in the generalized Bloch paper:

using MRIgeneralizedBloch
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #!nb
#nb plotlyjs(ticks=:native);
#-
T2s = 10e-6 # s
t = 0 : 1e-6 : 1e-3
p = plot(1e3t, greens_lorentzian.(t ./ T2s), yaxis=:log, ylim=(1e-6,1), label="Lorentzian lineshape", xlabel="(t-τ) [ms]", ylabel="G((t-τ)/T2s)")
p = plot!(1e3t, greens_gaussian.(t ./ T2s), label="Gaussian lineshape")
p = plot!(1e3t, greens_superlorentzian.(t ./ T2s), label="super-Lorentzian l.")
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
    t[i] / T2s, #src
    greens_lorentzian(t[i] / T2s), #src
    greens_gaussian(t[i] / T2s), #src
    greens_superlorentzian(t[i] / T2s) #src
    )) #src
end #src
close(io) #src
