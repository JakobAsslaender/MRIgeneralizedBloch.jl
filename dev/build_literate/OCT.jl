using MRIgeneralizedBloch
using MAT
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide

# set parameters
ω0 = 0
B1 = 1
m0s = 0.15
R1f = 0.5
R2f = 1 / 65e-3
R1s = 3
T2s = 10e-6
Rx = 30
TR = 3.5e-3

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

