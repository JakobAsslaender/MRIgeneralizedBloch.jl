##
function calculatemagnetization_gbloch_ide(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)

    # Define G(τ) and its derivative as ApproxFuns
    G = interpolate_greens_function(greens_superlorentzian, 0, maximum(TRF) / T2s)
    if any(isa.(grad_list, grad_T2s))
        dG_oT2 = interpolate_greens_function(dG_o_dT2s_x_T2s_superlorentzian, 0, maximum(TRF) / T2s)
    else
        dG_oT2 = []
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
    sol = solve(DDEProblem(apply_hamiltonian_gbloch!, u0, h, (0.0, TRF[2]), (-ω1[2] / 2, B1, ω0, m0s, R1, R2f, T2s, Rx, G, dG_oT2, grad_list)), alg)  
    u0 = sol[end]
    
    T_FP = (TR - TRF[2]) / 2 - TRF[1] / 2
    sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5())
    u0 = sol[end]
    
    for ic = 0:(Niter - 1)
        # free precession for TRF/2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, TRF[1] / 2), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5())
        u0 = sol[end]
        
        # inversion pulse with crusher gradients (assumed to be instantanious)
        u00 = u0[1:3]
        u0[1:5:end] .*= -sin(B1 * ω1[1] * TRF[1] / 2)^2
        u0[2:5:end] .*= sin(B1 * ω1[1] * TRF[1] / 2)^2
        u0[3:5:end] .*= cos(B1 * ω1[1] * TRF[1])

        # calculate saturation of RF pulse
        sol = solve(DDEProblem(apply_hamiltonian_gbloch_inversion!, u0, h, (0.0, TRF[1]), ((-1)^(1 + ic) * ω1[1], B1, ω0, m0s, 0.0, 0.0, T2s, 0.0, G, dG_oT2, grad_list)), alg)
        u0[4:5:end] = sol[end][4:5:end]

        for i in eachindex(grad_list)
            if isa(grad_list[i], grad_B1)
                u0[5i + 1] -= u00[1] * sin(B1 * ω1[1] * TRF[1] / 2) * cos(B1 * ω1[1] * TRF[1] / 2) * ω1[1] * TRF[1]
                u0[5i + 2] += u00[2] * sin(B1 * ω1[1] * TRF[1] / 2) * cos(B1 * ω1[1] * TRF[1] / 2) * ω1[1] * TRF[1]
                u0[5i + 3] -= u00[3] * sin(B1 * ω1[1] * TRF[1]) * ω1[1] * TRF[1]
            end
        end

        # free precession
        T_FP = TR - TRF[2] / 2
        TE = TR / 2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
        s[:,1] = sol[2]
        s[1:5:end,1] .*= (-1)^(1 + ic)
        s[2:5:end,1] .*= (-1)^(1 + ic)
        u0 = sol[end]

        for ip = 2:length(TRF)
            sol = solve(DDEProblem(apply_hamiltonian_gbloch!, u0, h, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, m0s, R1, R2f, T2s, Rx, G, dG_oT2, grad_list)), alg)
            u0 = sol[end]
    
            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(DimensionMismatch("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol[2]
            s[1:5:end,ip] .*= (-1)^(ip + ic)
            s[2:5:end,ip] .*= (-1)^(ip + ic)
            u0 = sol[end]
        end
    end
    return transpose(s)
end

function calculatesignal_gbloch_ide(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Niter)
    s = calculatemagnetization_gbloch_ide(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, [], Niter)
    return s[:,1] + 1im * s[:,2]
end

function calculatesignal_gbloch_ide(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    s = calculatemagnetization_gbloch_ide(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    return s[:, 1:5:end] + 1im * s[:, 2:5:end]
end

function calculatemagnetization_graham_ode(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    # initialization and memory allocation 
    N_s = 5 * (1 + length(grad_list))
    s = zeros(N_s, length(TRF))
    u0 = zeros(N_s)
    u0[3] = (1 - m0s)
    u0[4] = m0s
    u0[5] = 1.0

    # prep pulse 
    sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, u0, (0.0, TRF[2]), (-ω1[2] / 2, B1, ω0, TRF[2], m0s, R1, R2f, T2s, Rx, grad_list)), Tsit5(), save_everystep=false)    
    u0 = sol[end]
    
    T_FP = TR / 2 - TRF[2] / 2 - TRF[1] / 2
    sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false)
    u0 = sol[end]
    
    for ic = 0:(Niter - 1)
        # free precession for TRF/2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, TRF[1] / 2), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false)
        u0 = sol[end]

        # inversion pulse with crusher gradients (assumed to be instantanious)
        u00 = u0[1:3]
        u0[1:5:end] .*= -sin(B1 * ω1[1] * TRF[1] / 2)^2
        u0[2:5:end] .*= sin(B1 * ω1[1] * TRF[1] / 2)^2
        u0[3:5:end] .*= cos(B1 * ω1[1] * TRF[1])

        # calculate saturation of RF pulse
        sol = solve(ODEProblem(Graham_Hamiltonian_superLorentzian_InversionPulse!, u0, (0.0, TRF[1]), ((-1)^(1 + ic) * ω1[1], B1, ω0, TRF[1], m0s, 0.0, 0.0, T2s, 0.0, grad_list)), Tsit5(), save_everystep=false)
        u0[4:5:end] = sol[end][4:5:end]

        for i in eachindex(grad_list)
            if isa(grad_list[i], grad_B1)
                u0[5i + 1] -= u00[1] * sin(B1 * ω1[1] * TRF[1] / 2) * cos(B1 * ω1[1] * TRF[1] / 2) * ω1[1] * TRF[1]
                u0[5i + 2] += u00[2] * sin(B1 * ω1[1] * TRF[1] / 2) * cos(B1 * ω1[1] * TRF[1] / 2) * ω1[1] * TRF[1]
                u0[5i + 3] -= u00[3] * sin(B1 * ω1[1] * TRF[1]) * ω1[1] * TRF[1]
            end
        end

        # free precession
        T_FP = TR - TRF[2] / 2
        TE = TR / 2
        sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
        s[:,1] = sol[2]
        s[1:5:end,1] .*= (-1)^(1 + ic)
        s[2:5:end,1] .*= (-1)^(1 + ic)
        u0 = sol[end]

        for ip = 2:length(TRF)
            sol = solve(ODEProblem(apply_hamiltonian_graham_superlorentzian!, u0, (0.0, TRF[ip]), ((-1)^(ip + ic) * ω1[ip], B1, ω0, TRF[ip], m0s, R1, R2f, T2s, Rx, grad_list)), Tsit5(), save_everystep=false)
            u0 = sol[end]
    
            T_FP = TR - TRF[ip] / 2 - TRF[mod(ip, length(TRF)) + 1] / 2
            TE = TR / 2 - TRF[ip] / 2
            sol = solve(ODEProblem(apply_hamiltonian_freeprecession!, u0, (0.0, T_FP), (ω0, m0s, R1, R2f, Rx, grad_list)), Tsit5(), save_everystep=false, saveat=TE)
            if sol.t[2] / TE - 1 > 1e-10
                throw(DimensionMismatch("sol.t[2] is not equal to TE"))
            end
            s[:,ip] = sol[2]
            s[1:5:end,ip] .*= (-1)^(ip + ic)
            s[2:5:end,ip] .*= (-1)^(ip + ic)
            u0 = sol[end]
        end
    end
    return transpose(s)
end

function calculatesignal_graham_ode(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, Niter)
    s = calculatemagnetization_graham_ode(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, [], Niter)
    return s[:,1] + 1im * s[:,2]
end

function calculatesignal_graham_ode(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    s = calculatemagnetization_graham_ode(ω1, TRF, TR, ω0, B1, m0s, R1, R2f, Rx, T2s, grad_list, Niter)
    return s[:,1:5:end] + 1im * s[:,2:5:end]
end