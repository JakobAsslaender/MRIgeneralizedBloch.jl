module MT_Diff_Equation_Sovlers
include("MT_Hamiltonians.jl")

using .MT_Hamiltonians
using QuadGK
using DifferentialEquations
using ApproxFun

export gBloch_calculate_magnetization
export gBloch_calculate_magnetization_gradients
export gBloch_calculate_signal
export gBloch_calculate_signal_gradients
export Graham_calculate_magnetization
export Graham_calculate_magnetization_gradients
export Graham_calculate_signal
export Graham_calculate_signal_gradients

export grad_m0s
export grad_R1
export grad_R2f
export grad_Rx
export grad_T2s
export grad_ω0
export grad_ω1

function gBloch_calculate_magnetization(ω1, ω0::Float64, TR, TRF, m0s, R1, R2f, Rx, T2s, Niter)
    # define g(τ) using ApproxFun
    g = (τ) -> quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8), 0.0, 1.0)[1]
    x = Fun(identity, 0..100)
    ga = g(x)

    # init and memory allocation 
    s = zeros(5, length(TRF))
    u0 = [0.0, 0.0, (1 - m0s), m0s, 1.0]
    h5(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(5)
    alg = MethodOfSteps(Tsit5())

    # prep pulse 
    sol = solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h5, (0.0, TRF[2]), (-ω1[2]/2, ω0, m0s, R1, R2f, T2s, Rx, ga)), alg, save_everystep=false)
    u0 = sol[end]

    T_FP = TR/2 - TRF[2]/2 - TRF[1]/2
    sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx)), Tsit5(), save_everystep=false)
    u0 = sol[end]

    for ic = 0:(Niter - 1)
        for ip = 1:length(TRF)
            sol = solve(DDEProblem(gBloch_Hamiltonian_ApproxFun!, u0, h5, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], ω0, m0s, R1, R2f, T2s, Rx, ga)), alg, save_everystep=false)
            u0 = sol[end]

            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx)), Tsit5(), save_everystep=false, saveat=TE)
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

function gBloch_calculate_signal(ω1, ω0::Float64, TR, TRF, m0s, R1, R2f, Rx, T2s, Niter)
    s = gBloch_calculate_magnetization(ω1, ω0, TR, TRF, m0s, R1, R2f, Rx, T2s, Niter)
    s = s[1,:] + 1im * s[2,:]
    return s
end

function gBloch_calculate_magnetization_gradients(ω1, ω0::Float64, TR, TRF, m0s, R1, R2f, Rx, T2s, grad_list, Niter)

    # Define g(τ) and its derivative as ApproxFuns
    g = (τ) -> quadgk(ct -> exp(- τ^2 * (3 * ct^2 - 1)^2 / 8), 0.0, 1.0)[1]
    dg_oT2 = (τ) -> quadgk(ct -> exp(- τ^2 * (3.0 * ct^2 - 1)^2 / 8.0) * (τ^2 * (3.0 * ct^2 - 1)^2 / 4.0), 0.0, 1.0)[1]
    x = Fun(identity, 0..100)
    ga = g(x)
    dg_oT2_a = dg_oT2(x)

    # initialization and memory allocation
    N_s = 5 * (1 + length(grad_list))
    h(p, t; idxs=nothing) = typeof(idxs) <: Number ? 0.0 : zeros(N_s)
    alg = MethodOfSteps(Tsit5())
    s = zeros(N_s, length(TRF))
    u0 = zeros(N_s)
    u0[3] = (1 - m0s)
    u0[4] = m0s
    u0[5] = 1.0

    # prep pulse 
    sol = solve(DDEProblem(gBloch_Hamiltonian_Gradient_ApproxFun!, u0, h, (0.0, TRF[2]), (-ω1[2]/2, ω0, m0s, R1, R2f, T2s, Rx, ga, dg_oT2_a, grad_list)), alg, save_everystep=false)  
    u0 = sol[end]
    
    T_FP = TR/2 - TRF[2]/2 - TRF[1]/2
    sol = solve(ODEProblem(FreePrecession_Hamiltonian_Gradient!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false)
    u0 = sol[end]
    
    for ic = 0:(Niter - 1)
        for ip = 1:length(TRF)
            sol = solve(DDEProblem(gBloch_Hamiltonian_Gradient_ApproxFun!, u0, h, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], ω0, m0s, R1, R2f, T2s, Rx, ga, dg_oT2_a, grad_list)), alg, save_everystep=false)
            u0 = sol[end]
    
            local T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(FreePrecession_Hamiltonian_Gradient!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
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

function gBloch_calculate_signal_gradients(ω1, ω0::Float64, TR,TRF, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    s = gBloch_calculate_magnetization_gradients(ω1, ω0, TR, TRF, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    s = s[1:5:end,:] + 1im * s[2:5:end,:]
    return s
end

function Graham_calculate_magnetization(ω1, ω0::Float64, TR, TRF, m0s, R1, R2f, Rx, T2s, Niter)

    s = zeros(5, length(TRF))
    u0 = [0.0, 0.0, (1 - m0s), m0s, 1.0]

    # prep pulse 
    sol = solve(ODEProblem(Graham_Hamiltonian!, u0, (0.0, TRF[2]), (-ω1[2]/2, ω0, TRF[2], m0s, R1, R2f, T2s, Rx)), Tsit5(), save_everystep=false)
    u0 = sol[end]
    
    T_FP = TR/2 - TRF[2]/2 - TRF[1]/2
    sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx)), Tsit5(), save_everystep=false)
    u0 = sol[end]
    
    for ic = 0:(Niter - 1)
        for ip = 1:length(TRF)
            sol = solve(ODEProblem(Graham_Hamiltonian!, u0, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], ω0, TRF[ip], m0s, R1, R2f, T2s, Rx)), Tsit5(), save_everystep=false)
            u0 = sol[end]
    
            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(FreePrecession_Hamiltonian!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx)), Tsit5(), save_everystep=false, saveat=TE)
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

function Graham_calculate_signal(ω1, ω0::Float64, TR, TRF, m0s, R1, R2f, Rx, T2s, Niter)
    s = Graham_calculate_magnetization(ω1, ω0, TR, TRF, m0s, R1, R2f, Rx, T2s, Niter)
    s = s[1,:] + 1im * s[2,:]
    return s
end

function Graham_calculate_magnetization_gradients(ω1, ω0::Float64, TR, TRF, m0s, R1, R2f, Rx, T2s, grad_list, Niter)

    # initialization and memory allocation 
    N_s = 5 * (1 + length(grad_list))
    s = zeros(N_s, length(TRF))
    u0 = zeros(N_s)
    u0[3] = (1 - m0s)
    u0[4] = m0s
    u0[5] = 1.0

    # prep pulse 
    sol = solve(ODEProblem(Graham_Hamiltonian_Gradient!, u0, (0.0, TRF[2]), (-ω1[2]/2, ω0, TRF[2], m0s, R1, R2f, T2s, Rx, grad_list)), Tsit5(), save_everystep=false)    
    u0 = sol[end]
    
    T_FP = TR/2 - TRF[2]/2 - TRF[1]/2
    sol = solve(ODEProblem(FreePrecession_Hamiltonian_Gradient!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false)
    u0 = sol[end]
    
    for ic = 0:(Niter - 1)
        for ip = 1:length(TRF)
            sol = solve(ODEProblem(Graham_Hamiltonian_Gradient!, u0, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], ω0, TRF[ip], m0s, R1, R2f, T2s, Rx, grad_list)), Tsit5(), save_everystep=false)
            u0 = sol[end]
    
            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(FreePrecession_Hamiltonian_Gradient!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
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

function Graham_calculate_signal_gradients(ω1, ω0::Float64, TR, TRF, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    s = Graham_calculate_magnetization_gradients(ω1, ω0, TR, TRF, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    s = s[1:5:end,:] + 1im * s[2:5:end,:]
    return s
end

end