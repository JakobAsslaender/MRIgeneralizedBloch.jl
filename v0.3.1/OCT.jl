#md # [![](https://mybinder.org/badge_logo.svg)](@__BINDER_ROOT_URL__/build_literate/OCT.ipynb)

# # Optimal Control


using MRIgeneralizedBloch
using MAT
using Plots
plotlyjs(bg = RGBA(31/255,36/255,36/255,1.0), ticks=:native); #hide #!nb

## set parameters
ω0 = 0
B1 = 1
m0s = 0.15
R1f = 0.5
R2f = 1 / 65e-3
R1s = 3
T2s = 10e-6
Rx = 30
TR = 3.5e-3
