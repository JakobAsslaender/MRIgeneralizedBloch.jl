module MT_Diff_Equation_Sovlers
include("MT_Hamiltonians.jl")

using .MT_Hamiltonians
using QuadGK
using DifferentialEquations
using ApproxFun

export gBloch_calculate_magnetization
export gBloch_calculate_signal
export LinearApprox_calculate_magnetization
export LinearApprox_calculate_signal
export Graham_calculate_magnetization
export Graham_calculate_signal

export grad_m0s
export grad_R1
export grad_R2f
export grad_Rx
export grad_T2s
export grad_ω0
export grad_B1

function gBloch_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Niter)
    # define g(τ) using ApproxFun
    g = (τ) -> quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8), 0.0, 1.0)[1]
    x = Fun(identity, 0..100)
    ga = g(x)

    # init and memory allocation 
    s = zeros(5, length(TRF))
    u0 = [0.0, 0.0, (1 - m0s), m0s, 1.0]
    h5(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5)
    alg = MethodOfSteps(DP8())

    # prep pulse 
    sol = solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h5, (0.0, TRF[2]), (-ω1[2] / 2, B1, ω0, m0s, R1, R2f, T2s, Rx, ga)), alg, save_everystep=false)
    u0 = sol[end]

    T_FP = TR / 2 - TRF[2] / 2 - TRF[1] / 2
    sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx)), Vern6(), save_everystep=false)
    u0 = sol[end]

    for ic = 0:(Niter - 1)
        # free precession for TRF/2
        sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, TRF[1]/2), (ω0, m0s, R1, R2f, Rx)), Vern6(), save_everystep=false)
        u0 = sol[end]
        # calculate saturation of RF pulse
        sol = solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h5, (0.0, TRF[1]), ((-1)^(1+ic) * ω1[1], B1, ω0, m0s, R1, R2f, T2s, Rx, ga)), alg, save_everystep=false)

        # inversion pulse with crusher gradients (assumed to be instantanious)
        u0[1] *= -sin(B1 * ω1[1] * TRF[1]/2)^2
        u0[2] *= sin(B1 * ω1[1] * TRF[1]/2)^2
        u0[3] *= cos(B1 * ω1[1] * TRF[1])
        u0[4] = sol[end][4]

        # free precession
        T_FP = TR - TRF[2] / 2
        TE = TR / 2
        sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx)), Vern6(), save_everystep=false, saveat=TE)
        s[:,1] = sol[2]
        s[1,1] *= (-1)^(1 + ic)
        u0 = sol[end]

        for ip = 2:length(TRF)
            sol = solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h5, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, m0s, R1, R2f, T2s, Rx, ga)), alg, save_everystep=false)
            u0 = sol[end]

            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx)), Vern6(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(DimensionMismatch("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol[2]
            s[1,ip] *= (-1)^(ip + ic)
            u0 = sol[end]
        end
    end
    return s
end

function gBloch_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Niter)
    s = gBloch_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Niter)
    s = s[1,:] + 1im * s[2,:]
    return s
end

function gBloch_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)

    # Define g(τ) and its derivative as ApproxFuns
    g = (τ) -> quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8), 0.0, 1.0)[1]
    x = Fun(identity, 0..100)
    ga = g(x)
    if any(isa.(grad_list, grad_T2s))
        dg_oT2 = (τ) -> quadgk(ct -> exp(- τ^2 * (3.0 * ct^2 - 1)^2 / 8.0) * (τ^2 * (3.0 * ct^2 - 1)^2 / 4.0), 0.0, 1.0)[1]
        dg_oT2_a = dg_oT2(x)
    else
        dg_oT2_a = []
    end

    # initialization and memory allocation
    N_s = 5 * (1 + length(grad_list))
    h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(N_s)
    alg = MethodOfSteps(DP8())
    s = zeros(N_s, length(TRF))
    u0 = zeros(N_s)
    u0[3] = (1 - m0s)
    u0[4] = m0s
    u0[5] = 1.0

    # prep pulse 
    sol = solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h, (0.0, TRF[2]), (-ω1[2] / 2, B1, ω0, m0s, R1, R2f, T2s, Rx, ga, dg_oT2_a, grad_list)), alg, save_everystep=false)  
    u0 = sol[end]
    
    T_FP = TR / 2 - TRF[2] / 2 - TRF[1] / 2
    sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Vern6(), save_everystep=false)
    u0 = sol[end]
    
    for ic = 0:(Niter - 1)
        # free precession for TRF/2
        sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, TRF[1]/2), (ω0, m0s, R1, R2f, Rx, grad_list)), Vern6(), save_everystep=false)
        u0 = sol[end]
        # calculate saturation of RF pulse
        sol = solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h, (0.0, TRF[1]), ((-1)^(1+ic) * ω1[1], B1, ω0, m0s, R1, R2f, T2s, Rx, ga, dg_oT2_a, grad_list)), alg, save_everystep=false)

        # inversion pulse with crusher gradients (assumed to be instantanious)
        u0[1:5:end] .*= -sin(B1 * ω1[1] * TRF[1]/2)^2
        u0[2:5:end] .*= sin(B1 * ω1[1] * TRF[1]/2)^2
        u0[3:5:end] .*= cos(B1 * ω1[1] * TRF[1])
        u0[4:5:end] = sol[end][4:5:end]

        for i in eachindex(grad_list)
            if isa(grad_list[i], grad_B1)
                u0[(i-1)*5+1] -= u0[1] * 2 * sin(B1 * ω1[1] * TRF[1]/2) * cos(B1 * ω1[1] * TRF[1]/2) * ω1[1] * TRF[1]/2
                u0[(i-1)*5+2] += u0[2] * 2 * sin(B1 * ω1[1] * TRF[1]/2) * cos(B1 * ω1[1] * TRF[1]/2) * ω1[1] * TRF[1]/2
                u0[(i-1)*5+3] -= u0[3] * sin(B1 * ω1[1] * TRF[1]) * ω1[1] * TRF[1]
            end
        end

        # free precession
        T_FP = TR - TRF[2] / 2
        TE = TR / 2
        sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Vern6(), save_everystep=false, saveat=TE)
        s[:,1] = sol[2]
        s[1:5:end,1] .*= (-1)^(1 + ic)
        u0 = sol[end]

        for ip = 2:length(TRF)
            sol = solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, m0s, R1, R2f, T2s, Rx, ga, dg_oT2_a, grad_list)), alg, save_everystep=false)
            u0 = sol[end]
    
            local T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Vern6(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(DimensionMismatch("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol[2]
            s[1:5:end,ip] .*= (-1)^(ip + ic)
            u0 = sol[end]
        end
    end
    return s
end

function gBloch_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    s = gBloch_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    s = s[1:5:end,:] + 1im * s[2:5:end,:]
    return s
end

function Graham_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Niter)
    s = zeros(5, length(TRF))
    u0 = [0.0, 0.0, (1 - m0s), m0s, 1.0]

    # prep pulse 
    sol = solve(ODEProblem(Graham_Hamiltonian!, u0, (0.0, TRF[2]), (-ω1[2] / 2, B1, ω0, TRF[2], m0s, R1, R2f, T2s, Rx)), Vern6(), save_everystep=false)
    u0 = sol[end]
    
    T_FP = TR / 2 - TRF[2] / 2 - TRF[1] / 2
    sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx)), Vern6(), save_everystep=false)
    u0 = sol[end]
    
    for ic = 0:(Niter - 1)
        for ip = 1:length(TRF)
            sol = solve(ODEProblem(Graham_Hamiltonian!, u0, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, TRF[ip], m0s, R1, R2f, T2s, Rx)), Vern6(), save_everystep=false)
            u0 = sol[end]
    
            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx)), Vern6(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(DimensionMismatch("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol[2]
            s[1,ip] *= (-1)^(ip + ic)
            u0 = sol[end]
        end
    end
    return s
end

function Graham_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Niter)
    s = Graham_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Niter)
    s = s[1,:] + 1im * s[2,:]
    return s
end

function Graham_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)

    # initialization and memory allocation 
    N_s = 5 * (1 + length(grad_list))
    s = zeros(N_s, length(TRF))
    u0 = zeros(N_s)
    u0[3] = (1 - m0s)
    u0[4] = m0s
    u0[5] = 1.0

    # prep pulse 
    sol = solve(ODEProblem(Graham_Hamiltonian!, u0, (0.0, TRF[2]), (-ω1[2] / 2, B1, ω0, TRF[2], m0s, R1, R2f, T2s, Rx, grad_list)), Vern6(), save_everystep=false)    
    u0 = sol[end]
    
    T_FP = TR / 2 - TRF[2] / 2 - TRF[1] / 2
    sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Vern6(), save_everystep=false)
    u0 = sol[end]
    
    for ic = 0:(Niter - 1)
        for ip = 1:length(TRF)
            sol = solve(ODEProblem(Graham_Hamiltonian!, u0, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, TRF[ip], m0s, R1, R2f, T2s, Rx, grad_list)), Vern6(), save_everystep=false)
            u0 = sol[end]
    
            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Vern6(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(DimensionMismatch("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol[2]
            s[1:5:end,ip] .*= (-1)^(ip + ic)
            u0 = sol[end]
        end
    end
    return s
end

function Graham_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    s = Graham_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    s = s[1:5:end,:] + 1im * s[2:5:end,:]
    return s
end

function LinearApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Niter)

    s = zeros(5, length(TRF))
    u0 = [0.0, 0.0, (1 - m0s), m0s, 1.0]

    # approximate saturation
    g_SL = (τ) -> quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8), 0.0, sqrt(1 / 3), 1.0)[1]
    x = Fun(identity, 0..maximum(TRF/T2s))
    g_SLa = g_SL(x)
    h(p, t) = [1.0]

    function gBloch_Hamiltonian_ApproxFun!(du, u, h, p::NTuple{3,Any}, t)
        ωy, T2s, g = p
        du[1] = -ωy^2 * quadgk(x -> g((t - x) / T2s) * h(p, x)[1], 0.0, t)[1]
    end
    
    S = Chebyshev(minimum(TRF/T2s)..maximum(TRF/T2s)) * Chebyshev(minimum(B1*ω1*T2s)..maximum(B1*ω1*T2s))
    Npoints = 10^3; _points = points(S, Npoints) 
    fDDE = (xy) -> solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, [1.0], h, (0.0, xy[1]*T2s), (xy[2]/T2s, T2s, g_SLa)), MethodOfSteps(DP8()))[end][1]
    fapprox = Fun(S, transform(S, fDDE.(_points)))

    # prep pulse 
    Rrf = - log(fapprox([TRF[2]/T2s, B1*ω1[2]*T2s/2])) / TRF[2]
    sol = solve(ODEProblem(Linear_Hamiltonian!, u0, (0.0, TRF[2]), (-ω1[2] / 2, B1, ω0, m0s, R1, R2f, Rx, Rrf)), Vern6(), save_everystep=false)
    u0 = sol[end]

    T_FP = TR / 2 - TRF[2] / 2 - TRF[1] / 2
    sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx)), Vern6(), save_everystep=false)
    u0 = sol[end]
    
    for ic = 0:(Niter - 1)
        for ip = 1:length(TRF)
            Rrf = - log(fapprox([TRF[ip]/T2s, B1*ω1[ip]*T2s])) / TRF[ip]
            sol = solve(ODEProblem(Linear_Hamiltonian!, u0, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, m0s, R1, R2f, Rx, Rrf)), Vern6(), save_everystep=false)
            u0 = sol[end]
    
            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx)), Vern6(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(DimensionMismatch("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol[2]
            s[1,ip] *= (-1)^(ip + ic)
            u0 = sol[end]
        end
    end
    return s
end

function LinearApprox_calculate_signal(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Niter)
    s = LinearApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Niter)
    s = s[1,:] + 1im * s[2,:]
    return s
end

# function LinearApprox_calculate_magnetization(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)

#     # initialization and memory allocation 
#     N_s = 5 * (1 + length(grad_list))
#     s = zeros(N_s, length(TRF))
#     u0 = zeros(N_s)
#     u0[3] = (1 - m0s)
#     u0[4] = m0s
#     u0[5] = 1.0

#     # prep pulse 
#     Rrf = - log(fapprox([TRF[2] / T2s, B1*ω1[2]])) / TRF[2]
#     sol = solve(ODEProblem(LinearApprox_Hamiltonian!, u0, (0.0, TRF[2]), (-ω1[2] / 2, ω0, TRF[2], m0s, R1, R2f, T2s, Rx, grad_list)), Vern6(), save_everystep=false)    
#     u0 = sol[end]
    
#     T_FP = TR / 2 - TRF[2] / 2 - TRF[1] / 2
#     sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Vern6(), save_everystep=false)
#     u0 = sol[end]
    
#     for ic = 0:(Niter - 1)
#         for ip = 1:length(TRF)
#             Rrf = - log(fapprox([TRF[ip] / T2s, B1*ω1[ip]])) / TRF[ip]
#             sol = solve(ODEProblem(LinearApprox_Hamiltonian!, u0, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, m0s, R1, R2f, Rx, Rrf, grad_list)), Vern6(), save_everystep=false)
#             u0 = sol[end]
    
#             T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
#             TE = TR / 2 - TRF[ip] / 2
#             sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Vern6(), save_everystep=false, saveat=TE)
#             if sol.t[2] / TE - 1 > 1e-10
#                 throw(DimensionMismatch("sol.t[2] is not equal to TE"))
#             end
#             s[:,ip] = sol[2]
#             s[1:5:end,ip] .*= (-1)^(ip + ic)
#             u0 = sol[end]
#         end
#     end
#     return s
# end

end