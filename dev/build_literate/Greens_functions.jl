using MRIgeneralizedBloch
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

T₂ˢ = 10e-6 # s
t = 0 : 1e-6 : 1e-3
p = plot(yaxis=:log, ylim=(1e-6,1), xlabel="(t-τ) [ms]", ylabel="G((t-τ)/T₂ˢ)")
plot!(p, 1e3t, greens_lorentzian.(t ./ T₂ˢ), label="Lorentzian lineshape")
plot!(p, 1e3t, greens_gaussian.(t ./ T₂ˢ), label="Gaussian lineshape")
plot!(p, 1e3t, greens_superlorentzian.(t ./ T₂ˢ), label="super-Lorentzian l.")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

