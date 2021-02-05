using LinearAlgebra
# using BenchmarkTools
using MAT
using LsqFit
# using Revise
# using Plots
# plotlyjs(ticks=:native)
# theme(:lime);

include("../src/MT_Diff_Equation_Sovlers.jl")
using Main.MT_Diff_Equation_Sovlers
# Revise.track("../src/MT_Diff_Equation_Sovlers.jl")
# Revise.track("../src/MT_Hamiltonians.jl")

## load data
x_path = expanduser("~/mygs/20200917_InVivo_MT_1mm_MWI_1p7mm/20201021_nonSweeping_Recos_n_Fits_Symmetric_Basis/NLLS_fits/");
qM_str = "x_mid1648_reg7e-06_R_13_basis_v3.2_sweep_0_std_B0_pio2_symmetric_B1_0.9pm0.2_fit"

vars = matread(string(x_path, qM_str, ".mat"))

TR = vars["TR"]
control = vars["control"]
TRF = [vars["TRFmax"]; control[1:end - 1,2]]
α = [π; control[1:end - 1,1] .+ control[2:end,1]]
ω1 = α ./ TRF
u = vars["u"]
xm = vars["xm"]
mask = vars["mask"]

ijob = parse(Int32, ENV["SLURM_ARRAY_TASK_ID"])
njobs = 1000;

B0 = matread(string(x_path, "../../B0map_MID1643_reg_to_MID1648.mat"))["B0"]
B1 = matread(string(x_path, "../../B1map_MID1644_reg_to_MID1648.mat"))["B1"]
B0 = B0[50,:,:]
B1 = B1[50,:,:]

## fit m0s    R1  R2f   Rx
grad_list = [grad_m0s(), grad_R1(), grad_R2f(), grad_Rx()]
#      re(M0) i(M0) m0s    R1  R2f   Rx
p0   = [-1.0,  0.1, 0.1,    1,  15,  15];
pmin = [-1e3, -1e3,   0,  0.1, 1.0,   5];
pmax = [ 1e3,  1e3, 0.5,  100, 100,  40];

T2s = 10e-6
ω0 = 0.0
ω1s = ω1

function model(t,p) 
    println("paramters = ", p)
    M0 = p[1] + 1im * p[2] 
    @time m = M0 .* Graham_calculate_signal(ω1s, TRF, TR, ω0, B1, p[3], p[4], p[5], p[6], T2s, 2)
    m = u' * m;
    m = [real(m); imag(m)]
    return m
end

function jacobian_model(t,p)
    M0 = p[1] + 1im * p[2] 
    @time J = transpose(Graham_calculate_signal_gradients(ω1s, TRF, TR, ω0, B1, p[3], p[4], p[5], p[6], T2s, grad_list, 2))
    J[:,2:end] .*= M0
    J = u' * J;

    Jv = zeros(2*size(u,2), length(p))
    Jv[:,1] = [real(J[:,1]); imag(J[:,1])];
    Jv[:,2] = [-imag(J[:,1]); real(J[:,1])];
    Jv[:,3:end] = [real(J[:,2:end]); imag(J[:,2:end])];
    return Jv
end


## set up voxels for job 
nx = convert(Int, ceil(size(xm,1)/njobs));
ixs = ((ijob-1)*nx+1):min(ijob*nx, size(xm,1));
println("Today, I'll fit ", length(ixs), " voxel");

## loop through those voxels
qM  = zeros(size(xm,1),length(p0));
res = zeros(size(xm,1));
for i in ixs
    println(i)

    yfit = xm[i,:]
    yfit ./= yfit[1]
    yfit = [real(yfit); imag(yfit)]

    fit = curve_fit(model, jacobian_model, 1:length(yfit), yfit, p0, lower=pmin, upper=pmax, show_trace=true, maxIter=30)
    param = fit.param
    println("m0s = ", param[3])
    println("T1  = ", 1/param[4], "s")
    println("T2f = ", 1/param[5]*1e3, "ms")
    println("Rx  = ", param[6], "/s")
    flush(stdout)

    qM[i,:] = param
    res[i] = norm(fit.resid)

    matwrite(string(x_path, qM_str, "_Graham_ijob_", ijob, ".mat"), Dict(
    "qM" => qM,
    "res" => res,
    "mask" => mask,
    "xm" => xm
    ));
end

exit()