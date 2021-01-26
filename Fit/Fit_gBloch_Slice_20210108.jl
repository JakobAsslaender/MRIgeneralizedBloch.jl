using LinearAlgebra
using MAT
using LsqFit
include("../src/MT_Diff_Equation_Sovlers.jl")
using Main.MT_Diff_Equation_Sovlers

##
# todo = "save_x"
todo = "fit"
ijob = parse(Int32, ENV["SLURM_ARRAY_TASK_ID"])
njobs = 1000;

iz = 92

x_path = expanduser("~/mygs/20210108_InVivo_MT_0p3sweeping/")
save_path = string(x_path, "NLLS_fits_Julia/")
x_filename = "x_mid1555_reg7e-06_R_13_svd_basis_v3.2_sweep_0_std_B0_pio2_symmetric_B1_0.9pm0.2"
qM_filename = "qM_mid1555_reg7e-06_R_13_svd_basis_v3.2_sweep_0_std_B0_pio2_symmetric_B1_0.9pm0.2_gBloch_B1map"

## ########################################################################################
if todo == "save_x"
    include("../src/readcfl.jl")
    using Main.Readcfl
    using Statistics
    using Plots
# plotlyjs(ticks=:native)
# theme(:lime);

##
    control = matread(expanduser("~/mygs/20200806_MT_inVivo/control_MT_v3p2_TR3p5ms_discretized.mat"))["control"]
    u = matread(expanduser("~/mygs/20200917_InVivo_MT_1mm_MWI_1p7mm/20201021_nonSweeping_Recos_n_Fits_Symmetric_Basis/basis_v3.2_sweep_0_std_B0_pio2_symmetric_B1_0.9pm0.2.mat"))["u"]

    TR = 3.5e-3
    TRF = [500e-6; control[1:end - 1,2]]
    α = [π; control[1:end - 1,1] .+ control[2:end,1]]
    ω1 = α ./ TRF

    u = u[:,1:13]

    ##
    x = readcfl(string(x_path, x_filename))
    x = x[end:-1:1,end:-1:1,:,1,1,1,:]
    x = x[:,:,iz,:]
    heatmap(abs.(x[:,:,2]), c=:grays, clims=(0, .001))

##
    mask = mapslices(norm, x, dims=3)
    mask = mask[:,:,1]
    mask = round.(mask ./ mean(mask) ./ 3.0)
    mask = convert.(Bool, min.(mask, 1.0))
    mask[1:60,:] .= false
    mask[201:end,:] .= false
    mask[:,1:20] .= false
    mask[:,171:end] .= false
    heatmap(mask)

##
    xm = zeros(ComplexF64, sum(mask), size(x, 3))
    for i = 1:size(xm, 2)
        xtmp = @view x[:,:,i]
        xm[:,i] = xtmp[mask]
    end

##
    B1 = matread(string(x_path, "B1_Maps_registered.mat"))["B1_AFI"]
    B1 = B1[end:-1:1,end:-1:1,iz]
    B1 = B1[mask]

##
    matwrite(string(save_path, x_filename, "_iz", iz, "_fit.mat"), Dict(
    "xm" => xm,
    "mask" => mask,
    "u" => u,
    "B1" => B1,
    "w1" => ω1,
    "TRF" => TRF,
    "TR" => TR
    ));

## ########################################################################################
elseif todo == "fit"
    xm = matread(string(save_path, x_filename, "_iz", iz, "_fit.mat"))["xm"]
    mask = matread(string(save_path, x_filename, "_iz", iz, "_fit.mat"))["mask"]
    u = matread(string(save_path, x_filename, "_iz", iz, "_fit.mat"))["u"]
    B1 = matread(string(save_path, x_filename, "_iz", iz, "_fit.mat"))["B1"]
    ω1 = matread(string(save_path, x_filename, "_iz", iz, "_fit.mat"))["w1"]
    TRF = matread(string(save_path, x_filename, "_iz", iz, "_fit.mat"))["TRF"]
    TR = matread(string(save_path, x_filename, "_iz", iz, "_fit.mat"))["TR"]

## fit m0s    R1  R2f   Rx
    grad_list = [grad_m0s(), grad_R1(), grad_R2f(), grad_Rx()]
#      re(M0) i(M0) m0s    R1  R2f   Rx
    p0   = [-0.5,  0.1, 0.1,    1,  15,  15];
    pmin = [-1e3, -1e3,   0,  0.1, 1.0,   5];
    pmax = [ 1e3,  1e3, 0.5,  100, 100,  40];

    T2s = 10e-6
    ω0 = 0.0

    function model(t, p, B1) 
        println("paramters = ", p)
        M0 = p[1] + 1im * p[2] 
        m = M0 .* gBloch_calculate_signal(ω1, TRF, TR, ω0, B1, p[3], p[4], p[5], p[6], T2s, 2)
        m = u' * m;
        m = [real(m); imag(m)]
        return m
    end

    function jacobian_model(t, p, B1)
        M0 = p[1] + 1im * p[2] 
        J = transpose(gBloch_calculate_signal(ω1, TRF, TR, ω0, B1, p[3], p[4], p[5], p[6], T2s, grad_list, 2))
        J[:,2:end] .*= M0
        J = u' * J;

        Jv = zeros(2 * size(u, 2), length(p))
        Jv[:,1] = [real(J[:,1]); imag(J[:,1])];
        Jv[:,2] = [-imag(J[:,1]); real(J[:,1])];
        Jv[:,3:end] = [real(J[:,2:end]); imag(J[:,2:end])];
        return Jv
    end


## set up voxels for job 
    nx = convert(Int, ceil(size(xm, 1) / njobs));
    ixs = ((ijob - 1) * nx + 1):min(ijob * nx, size(xm, 1));
    println("Today, I'll fit ", length(ixs), " voxel");

## loop through those voxels
    qM  = zeros(size(xm, 1), length(p0));
    res = zeros(size(xm, 1));
    for i in ixs
        println(i)
    
        xfit = xm[i,:]
        xfit ./= xfit[1]
        xfit = [real(xfit); imag(xfit)]

        fit = curve_fit((t, p) -> model(t, p, B1[i]), (t, p) -> jacobian_model(t, p, B1[i]), 1:length(xfit), xfit, p0, lower=pmin, upper=pmax, show_trace=true, maxIter=30)
        param = fit.param
        println("m0s = ", param[3])
        println("T1  = ", 1 / param[4], "s")
        println("T2f = ", 1 / param[5] * 1e3, "ms")
        println("Rx  = ", param[6], "/s")
        flush(stdout)

        qM[i,:] = param
        res[i] = norm(fit.resid)

        matwrite(string(save_path, qM_filename, "_ijob_", ijob, ".mat"), Dict(
            "qM" => qM,
            "res" => res,
        ));
    end
end

##
exit()