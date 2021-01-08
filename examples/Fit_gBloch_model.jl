using LinearAlgebra
using BenchmarkTools
using MAT
using LsqFit
using Revise
using Plots
plotlyjs(ticks=:native)
theme(:lime);

include("src/MT_Diff_Equation_Sovlers.jl")
using Main.MT_Diff_Equation_Sovlers
Revise.track("src/MT_Diff_Equation_Sovlers.jl")
Revise.track("src/MT_Hamiltonians.jl")

## load data
vars = matread(expanduser("~/mygs/20200917_InVivo_MT_1mm_MWI_1p7mm/20201021_nonSweeping_Recos_n_Fits_Symmetric_Basis/NLLS_fits/x_mid1648_reg7e-06_R_13_basis_v3.2_sweep_0_std_B0_pio2_symmetric_B1_0.9pm0.2_fit.mat"))

TR = vars["TR"]
control = vars["control"]
TRF = [vars["TRFmax"]; control[1:end - 1,2]]
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
ω1 = α ./ TRF
u = vars["u"]
xm = vars["xm"]
mask = vars["mask"]

x = zeros(ComplexF64, size(mask,1), size(mask,2), 13);
for i=1:13
    xi = @view x[:,:,i]
    xi[mask] = xm[:,i];
end
# heatmap(abs.(x[:,:,2]))

B0 = matread(expanduser("~/mygs/20200917_InVivo_MT_1mm_MWI_1p7mm/B0map_MID1643_reg_to_MID1648.mat"))["B0"]
B1 = matread(expanduser("~/mygs/20200917_InVivo_MT_1mm_MWI_1p7mm/B1map_MID1644_reg_to_MID1648.mat"))["B1"]
B0 = B0[50,:,:]
B1 = B1[50,:,:]

## set up Fit
#      re(M0) i(M0) m0s    R1  R2f   Rx    T2s
p0   = [-1.0,  0.1, 0.1,    1,  15,  15, 10e-6];
pmin = [-1e3, -1e3,   0,  0.1, 1.0,   5, 10e-6];
pmax = [ 1e3,  1e3, 0.5,  100, 100,  40, 10e-6];

ix = 96
iy = 87
yfit = x[ix,iy,:]
yfit ./= yfit[1]
# plot(real(yfit))
# plot!(imag(yfit))
yfit = [real(yfit); imag(yfit)]

ω0 = B0[ix,iy]
ω1s = B1[ix,iy] .* ω1

function model(t,p) 
    println("paramters = ", p)
    M0 = p[1] + 1im * p[2] 
    m = M0 .* gBloch_calculate_signal(ω1s, ω0, TR, TRF, p[3], p[4], p[5], p[6], p[7], 2)
    m = u' * m;
    m = [real(m); imag(m)]
    return m
end

function jacobian_model(t,p)
    M0 = p[1] + 1im * p[2] 
    J = transpose(gBloch_calculate_signal_gradients(ω1s, ω0, TR, TRF, p[3], p[4], p[5], p[6], p[7], 2))
    J[:,2:end] .*= M0
    J = u' * J;

    Jv = zeros(2*size(u,2), length(p))
    Jv[:,1] = [real(J[:,1]); imag(J[:,1])];
    Jv[:,2] = [-imag(J[:,1]); real(J[:,1])];
    Jv[:,3:end] = [real(J[:,2:end]); imag(J[:,2:end])];
    return Jv
end

## fit
fit = curve_fit(model, jacobian_model, 1:length(yfit), yfit, p0, lower=pmin, upper=pmax, show_trace=true)
param = fit.param
println("m0s = ", param[3])
println("T1  = ", 1/param[4], "s")
println("T2f = ", 1/param[5]*1e3, "ms")
println("Rx  = ", param[6], "/s")
println("T2s = ", param[7]*1e6, "μs")

## ####################################################################################
## old stuff
## ####################################################################################

## simulate fake data
model(t,p) = p[1] .* real(gBloch_calculate_signal(ω1, ω0, TR, TRF, p[2], p[3], p[4], p[5], p[6], 2))

m0s = 0.15
R1 = 1.0
R2f = 1 / 65e-3
Rx = 30.0
T2s = 10e-6

t = 1:1142 # not used
p0   = [1.0, m0s,   R1, R2f, Rx,   T2s]
pmin = [0.0,   0,  0.1, 1.0,  1,  5e-6];
pmax = [0.0,   1,  100, 100,100, 15e-5];

ydata = model(t, p0)
# ydata .+= 0.1 * randn(length(ydata))

## fit 
function jacobian_model(x,p)
    J = real(gBloch_calculate_signal_gradients(ω1, ω0, TR, TRF, p[2], p[3], p[4], p[5], p[6], 2))'
    J[:,2:end] .*= p[1]
    return J
end

# fit = curve_fit(model, t, real(ydata), p0, show_trace=true)
fit = curve_fit(model, jacobian_model, t, real(ydata), 2 .* p0, show_trace=true)
param = fit.param




##
# @. model(x, p) = p[1]*exp(-x*p[2])
# xdata = range(0, stop=10, length=20)
# ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
# p0 = [0.5, 0.5]
# fit = curve_fit(model, xdata, ydata, p0, show_trace=true)
