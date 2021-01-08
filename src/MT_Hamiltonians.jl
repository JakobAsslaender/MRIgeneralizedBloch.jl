## Define functions
module MT_Hamiltonians
import Cubature
using QuadGK
# using ApproxFun
using SpecialFunctions

export gBloch_Hamiltonian!
export gBloch_Hamiltonian_Gradient!
export gBloch_Hamiltonian_ApproxFun!
export gBloch_Hamiltonian_Gradient_ApproxFun!
export FreePrecession_Hamiltonian!
export FreePrecession_Hamiltonian_Gradient!
export Graham_Hamiltonian!
export Graham_Hamiltonian_Gradient!
export gBloch_Hamiltonian_SemiSolid!

function gBloch_Hamiltonian!(du, u, h, p::NTuple{9,Any}, t)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, zs_idx, N = p

    gt = (t, T2s, ct) -> exp(- (t / T2s)^2 * (3 * ct^2 - 1)^2 / 8)

    function fy!(x,y,gt,h,p,T2s,zs_idx,t)
        for i = 1:size(x,2)
            y[i] = gt(t - x[2,i],T2s,x[1,i]) * h(p, x[2,i]; idxs=zs_idx)
        end
    end

    dy1 = Cubature.pcubature_v((x,y) -> fy!(x,y,gt,h,p,T2s,zs_idx,t), [0.0, max(0.0, t - N * T2s)], [1.0, t])[1]

    if t > (N * T2s)
        dy2 = T2s * sqrt(2π / 3) * Cubature.pcubature(x -> h(p, x[1]; idxs=zs_idx) / (t - x[1]), [0.0], [t - N * T2s])[1]
        
        du[4] = -ωy^2 * ((dy1) + (dy2))
    else
        du[4] = -ωy^2 * (dy1)
    end

    du[1] = - R2f * u[1] - ωz  * u[2] + ωy * u[3]
    du[2] =   ωz  * u[1] - R2f * u[2]
    du[3] = - ωy  * u[1] - (R1 + Rx * m0s) * u[3] +       Rx * (1-m0s)  * u[4] + (1 - m0s) * R1 * u[5]
    du[4] +=             +       Rx * m0s  * u[3] - (R1 + Rx * (1-m0s)) * u[4] +      m0s  * R1 * u[5]
    du[5] = 0.0
end

function gBloch_Hamiltonian!(du, u, h, p::NTuple{8,Any}, t)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, N = p
    gBloch_Hamiltonian!(du, u, h, (ωy, ωz, m0s, R1, R2f, T2s, Rx, 4, N), t)
end

function gBloch_Hamiltonian_Gradient!(du, u, h, p, t)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, N = p

    # start integral for T2s gradient 
    dgt = (t, T2s, ct) -> exp(- (t / T2s)^2 * (3.0 * ct^2 - 1)^2 / 8.0) * (t^2/T2s^3 * (3.0 * ct^2 - 1)^2 / 4.0)

    function dfy!(x,y,dgt,h,p,T2s,zs_idx,t)
        for i = 1:size(x,2)
            y[i] = dgt(t - x[2,i],T2s,x[1,i]) * h(p, x[2,i]; idxs=zs_idx)
        end
    end

    dy1 = Cubature.pcubature_v((x,y) -> dfy!(x,y,dgt,h,p,T2s,4,t), [0.0, max(0.0, t - N * T2s)], [1.0, t])[1]

    if t > (N * T2s)
        dy2 = sqrt(2π / 3) * Cubature.pcubature(x -> h(p, x[1]; idxs=4) / (t - x[1]), [0.0], [t - N * T2s])[1]
    end
    
    # Apply Hamiltonian to M and all its derivatives
    for i = 1 : 5 : 26
        @inbounds du_v = @view du[i:(i+4)]
        @inbounds u_v = @view u[i:(i+4)]
        gBloch_Hamiltonian!(du_v, u_v, h, (ωy, ωz, m0s, R1, R2f, T2s, Rx, (i+3), N), t)
    end

    # dM / dm0s
    du[8] -= Rx * u[3] + Rx * u[4] + R1
    du[9] += Rx * u[3] + Rx * u[4] + R1
    du[10] = 0.0

    # dM / dR1
    du[13] += - u[3] + (1 - m0s)
    du[14] += - u[4] + m0s
    du[15] = 0.0

    # dM / dR2f
    du[16] -= u[1]
    du[17] -= u[2]
    du[20] = 0.0

    # dM / dRx
    du[23] += - m0s * u[3] + (1-m0s) * u[4]
    du[24] +=   m0s * u[3] - (1-m0s) * u[4]
    du[25] = 0.0

    # dM / dT2s
    if t > (N * T2s)
        du[29] += -ωy^2 * (fetch(dy1) + fetch(dy2))
    else
        du[29] += -ωy^2 * fetch(dy1)
    end
    du[30] = 0.0
end

function gBloch_Hamiltonian_ApproxFun!(du, u, h, p::NTuple{9,Any}, t)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, zs_idx, g = p

    du[1] = - R2f * u[1] - ωz  * u[2] + ωy * u[3]
    du[2] =   ωz  * u[1] - R2f * u[2]
    du[3] = - ωy  * u[1] - (R1 + Rx * m0s) * u[3] +       Rx * (1-m0s)  * u[4] + (1 - m0s) * R1 * u[5]
    du[4] = -ωy^2 * quadgk(x -> g((t - x)/T2s) * h(p, x; idxs=zs_idx), eps(), t)[1] + Rx * m0s  * u[3] - (R1 + Rx * (1-m0s)) * u[4] + m0s * R1 * u[5]
    du[5] = 0.0
end

function gBloch_Hamiltonian_ApproxFun!(du, u, h, p::NTuple{8,Any}, t)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, g = p
    gBloch_Hamiltonian_ApproxFun!(du, u, h, (ωy, ωz, m0s, R1, R2f, T2s, Rx, 4, g), t)
end

function gBloch_Hamiltonian_Gradient_ApproxFun!(du, u, h, p, t)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, g, dg_oT2 = p
    
    # Apply Hamiltonian to M and all its derivatives
    for i = 1 : 5 : 26
        @inbounds du_v = @view du[i:(i+4)]
        @inbounds u_v = @view u[i:(i+4)]
        gBloch_Hamiltonian_ApproxFun!(du_v, u_v, h, (ωy, ωz, m0s, R1, R2f, T2s, Rx, (i+3), g), t)
    end

    # dM / dm0s
    du[8] -= Rx * u[3] + Rx * u[4] + R1
    du[9] += Rx * u[3] + Rx * u[4] + R1
    du[10] = 0.0

    # dM / dR1
    du[13] += - u[3] + (1 - m0s)
    du[14] += - u[4] + m0s
    du[15] = 0.0

    # dM / dR2f
    du[16] -= u[1]
    du[17] -= u[2]
    du[20] = 0.0

    # dM / dRx
    du[23] += - m0s * u[3] + (1-m0s) * u[4]
    du[24] +=   m0s * u[3] - (1-m0s) * u[4]
    du[25] = 0.0

    # dM / dT2s
    du[29] -= ωy^2/T2s * quadgk(x -> dg_oT2((t - x)/T2s) * h(p, x; idxs=4), 0.0, t)[1]
    du[30] = 0.0
end

function FreePrecession_Hamiltonian!(du, u, p, t)
    ωz, m0s, R1, R2f, Rx = p

    du[1] = - R2f * u[1] - ωz  * u[2]
    du[2] =   ωz  * u[1] - R2f * u[2]
    du[3] = - (R1 + Rx * m0s) * u[3] + Rx * (1-m0s)  * u[4] + (1 - m0s) * R1 * u[5]
    du[4] =   Rx * m0s  * u[3] - (R1 + Rx * (1-m0s)) * u[4] + m0s  * R1 * u[5]
    du[5] = 0.0
end

function FreePrecession_Hamiltonian_Gradient!(du, u, p, t)
    ωz, m0s, R1, R2f, Rx = p

    # Apply Hamiltonian to M and all its derivatives
    for i = 1 : 5 : 26
        @inbounds du_v = @view du[i:(i+4)]
        @inbounds u_v = @view u[i:(i+4)]
        FreePrecession_Hamiltonian!(du_v, u_v, (ωz, m0s, R1, R2f, Rx), t)
    end

    # dM / dm0s
    du[8] -= Rx * u[3] + Rx * u[4] + R1
    du[9] += Rx * u[3] + Rx * u[4] + R1
    du[10] = 0.0

    # dM / dR1
    du[13] += - u[3] + (1 - m0s)
    du[14] += - u[4] + m0s
    du[15] = 0.0

    # dM / dR2f
    du[16] -= u[1]
    du[17] -= u[2]
    du[20] = 0.0

    # dM / dRx
    du[23] += - m0s * u[3] + (1-m0s) * u[4]
    du[24] +=   m0s * u[3] - (1-m0s) * u[4]
    du[25] = 0.0

    # dM / dT2s
    du[30] = 0.0
end

function Graham_Hamiltonian!(du, u, p, t)
    ωy, ωz, TRF, m0s, R1, R2f, T2s, Rx = p

    f_PSD = (τ) -> quadgk(ct -> 1.0 / abs(1 - 3*ct^2) * (4/τ / abs(1 - 3*ct^2) * (exp(- τ^2/8 * (1 - 3*ct^2)^2) - 1) + sqrt(2π) * erf(τ/2/sqrt(2) * abs(1 - 3*ct^2))), 0.0, 1.0)[1]
    
    du[1] = - R2f * u[1] - ωz  * u[2] + ωy * u[3]
    du[2] =   ωz  * u[1] - R2f * u[2]
    du[3] = - ωy  * u[1] - (R1 + Rx * m0s) * u[3] + Rx * (1-m0s) * u[4] + (1 - m0s) * R1 * u[5]
    du[4] = - f_PSD(TRF/T2s) * ωy^2 * T2s/π * u[4] + R1 * (m0s * u[5] - u[4]) + Rx * (m0s * u[3] - (1-m0s) * u[4])
    du[5] = 0.0
end

function Graham_Hamiltonian_Gradient!(du, u, p, t)
    ωy, ωz, TRF, m0s, R1, R2f, T2s, Rx = p
    
    # Apply Hamiltonian to M and all its derivatives
    for i = 1 : 5 : 26
        @inbounds du_v = @view du[i:(i+4)]
        @inbounds u_v = @view u[i:(i+4)]
        Graham_Hamiltonian!(du_v, u_v, (ωy, ωz, TRF, m0s, R1, R2f, T2s, Rx), t)
    end

    # dM / dm0s
    du[8] -= Rx * u[3] + Rx * u[4] + R1
    du[9] += Rx * u[3] + Rx * u[4] + R1
    du[10] = 0.0

    # dM / dR1
    du[13] += - u[3] + (1 - m0s)
    du[14] += - u[4] + m0s
    du[15] = 0.0

    # dM / dR2f
    du[16] -= u[1]
    du[17] -= u[2]
    du[20] = 0.0

    # dM / dRx
    du[23] += - m0s * u[3] + (1-m0s) * u[4]
    du[24] +=   m0s * u[3] - (1-m0s) * u[4]
    du[25] = 0.0

    # dM / dT2s
    df_PSD = (τ) -> quadgk(ct -> 8/τ * (exp(-τ^2/8 * (3*ct^2-1)^2) - 1) / (3*ct^2-1)^2 + sqrt(2π) * erf(τ/sqrt(8) * abs(3*ct^2-1)) / abs(3*ct^2-1), 0.0, 1.0)[1]
    
    du[29] -= df_PSD(TRF/T2s) * ωy^2/π * u[4]
    du[30] = 0.0
end

function gBloch_Hamiltonian_SemiSolid!(du, u, h, p, t)
    ωy, T2s, N = p

    gt = (t, T2s, ct) -> exp(- (t / T2s)^2 * (3 * ct^2 - 1)^2 / 8)

    function fy!(x,y,gt,h,p,T2s,t)
        for i = 1:size(x,2)
            y[i] = gt(t - x[2,i],T2s,x[1,i]) * h(p, x[2,i]; idxs=1)
        end
    end

    dy1 = Cubature.pcubature_v((x,y) -> fy!(x,y,gt,h,p,T2s,t), [0.0, max(0.0, t - N * T2s)], [1.0, t])[1]

    if t > (N * T2s)
        dy2 = T2s * sqrt(2π / 3) * Cubature.pcubature(x -> h(p, x[1]; idxs=1) / (t - x[1]), [0.0], [t - N * T2s])[1]
        
        du[1] = -ωy^2 * ((dy1) + (dy2))
    else
        du[1] = -ωy^2 * (dy1)
    end
end
 
end