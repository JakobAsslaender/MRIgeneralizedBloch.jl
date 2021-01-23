using LinearAlgebra
using BenchmarkTools
using MAT
using LsqFit
using Revise
using Plots
# plotlyjs(ticks=:native)
theme(:lime);

include("../src/readcfl.jl")
using Main.Readcfl
include("../src/MT_Diff_Equation_Sovlers.jl")
using Main.MT_Diff_Equation_Sovlers
Revise.track("src/MT_Diff_Equation_Sovlers.jl")
# Revise.track("../src/MT_Hamiltonians.jl")

## read old v3.2 data
# vars = matread(expanduser("~/mygs/20200917_InVivo_MT_1mm_MWI_1p7mm/20201021_nonSweeping_Recos_n_Fits_Symmetric_Basis/NLLS_fits_Julia/x_mid1648_reg7e-06_R_13_basis_v3.2_sweep_0_std_B0_pio2_symmetric_B1_0.9pm0.2_fit.mat"))

# TR = vars["TR"]
# control = vars["control"]
# TRF = [vars["TRFmax"]; control[1:end - 1,2]]
# α = [π; control[1:end - 1,1] .+ control[2:end,1]]
# ω1 = α ./ TRF

# u = vars["u"]
# xm = vars["xm"]
# mask = vars["mask"]

# x = zeros(ComplexF64, size(mask,1), size(mask,2), 13);
# for i=1:13
#     xi = @view x[:,:,i]
#     xi[mask] = xm[:,i];
# end
# # heatmap(abs.(x[:,:,2]))

# B0 = matread(expanduser("~/mygs/20200917_InVivo_MT_1mm_MWI_1p7mm/B0map_MID1643_reg_to_MID1648.mat"))["B0"]
# B1 = matread(expanduser("~/mygs/20200917_InVivo_MT_1mm_MWI_1p7mm/B1map_MID1644_reg_to_MID1648.mat"))["B1"]
# B0 = B0[50,:,:]
# B1 = B1[50,:,:]

# ## choose voxel
# ix = 96
# iy = 87
# yfit = x[ix,iy,:]
# yfit ./= yfit[1]
# # plot(real(yfit))
# # plot!(imag(yfit))
# yfit = [real(yfit); imag(yfit)]

## read new v3.2 data
x = readcfl(expanduser("~/mygs/20210108_InVivo_MT_0p3sweeping/x_mid1555_reg7e-06_R_13_svd_basis_v3.2_sweep_0_std_B0_pio2_symmetric_B1_0.9pm0.2"))
x = x[end:-1:1,end:-1:1,:,1,1,1,:]

##
ix = 150
iy = 80
iz = 92
heatmap(abs.(x[:,:,iz,2]), c=:grays, clims=(0, .001))

##
yfit = x[ix,iy,iz,:]
yfit ./= yfit[1]
# plot(real(yfit))
# plot!(imag(yfit))
yfit = [real(yfit); imag(yfit)]

control = matread(expanduser("~/mygs/20200806_MT_inVivo/control_MT_v3p2_TR3p5ms_discretized.mat"))["control"]
TR = 3.5e-3
TRF = [500e-6; control[1:end - 1,2]]
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
ω1 = α ./ TRF

u = matread(expanduser("~/mygs/20200917_InVivo_MT_1mm_MWI_1p7mm/20201021_nonSweeping_Recos_n_Fits_Symmetric_Basis/basis_v3.2_sweep_0_std_B0_pio2_symmetric_B1_0.9pm0.2.mat"))["u"]
u = u[:,1:13]

B1 = matread(expanduser("~/mygs/20210108_InVivo_MT_0p3sweeping/B1_Maps_registered.mat"))["B1_AFI"]


## fit m0s    R1  R2f   Rx    T2s
grad_list = [grad_m0s(), grad_R1(), grad_R2f(), grad_Rx(), grad_T2s(), grad_ω0(), grad_B1()]
#      re(M0) i(M0) m0s    R1  R2f   Rx    T2s
p0   = [-1.0,  0.1, 0.1,    1,  15,  15, 10e-6,  0.0, 1.0];
pmin = [-1e3, -1e3,   0,  0.1, 1.0,   5, 10e-7, -1e4, 0.5];
pmax = [ 1e3,  1e3, 0.5,  100, 100,  40, 10e-3,  1e4, 1.5];

# ω0 = B0[ix,iy]
B1vx = B1[ix,iy,iz]
ω0 = 0.0
# ω1s = ω1

function model(t,p) 
    println("paramters = ", p)
    M0 = p[1] + 1im * p[2] 
    m = M0 .* gBloch_calculate_signal(ω1, TRF, TR, p[8], p[9], p[3], p[4], p[5], p[6], p[7], 2)
    m = u' * m;
    m = [real(m); imag(m)]
    return m
end

function jacobian_model(t,p)
    M0 = p[1] + 1im * p[2] 
    J = transpose(gBloch_calculate_signal(ω1, TRF, TR, p[8], p[9], p[3], p[4], p[5], p[6], p[7], grad_list, 2))
    J[:,2:end] .*= M0
    J = u' * J;

    Jv = zeros(2*size(J,1), length(p))
    Jv[:,1] = [real(J[:,1]); imag(J[:,1])];
    Jv[:,2] = [-imag(J[:,1]); real(J[:,1])];
    Jv[:,3:end] = [real(J[:,2:end]); imag(J[:,2:end])];
    return Jv
end

fit = curve_fit(model, jacobian_model, 1:length(yfit), yfit, p0, lower=pmin, upper=pmax, show_trace=true)
param = fit.param
println("m0s = ", param[3])
println("T1  = ", 1/param[4], "s")
println("T2f = ", 1/param[5]*1e3, "ms")
println("Rx  = ", param[6], "/s")
println("T2s = ", param[7]*1e6, "μs")
