## Define functions
module MT_Hamiltonians
import Cubature
using QuadGK
using SpecialFunctions
using ApproxFun

export gBloch_Hamiltonian!
export gBloch_Hamiltonian_Gradient!
export gBloch_Hamiltonian_ApproxFun!
export gBloch_Hamiltonian_Gradient_ApproxFun!
export FreePrecession_Hamiltonian!
export FreePrecession_Hamiltonian_Gradient!
export Graham_Hamiltonian!
export Graham_Hamiltonian_Gradient!
export gBloch_Hamiltonian_SemiSolid!
export grad_m0s
export grad_R1
export grad_R2f
export grad_Rx
export grad_T2s
export grad_ω0
export grad_ω1


struct grad_m0s end
struct grad_R1 end
struct grad_R2f end
struct grad_Rx end
struct grad_T2s end
struct grad_ω0 end
struct grad_ω1 end

function add_partial_derivative!(du, u, h, p, t, grad_type::grad_m0s)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, g, dg_oT2 = p

    du[3] -= Rx * u[3] + Rx * u[4] + R1
    du[4] += Rx * u[3] + Rx * u[4] + R1
    return nothing
end

function add_partial_derivative!(du, u, h, p, t, grad_type::grad_R1)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, g, dg_oT2 = p

    du[3] += - u[3] + (1 - m0s)
    du[4] += - u[4] + m0s
    return nothing
end

function add_partial_derivative!(du, u, h, p, t, grad_type::grad_R2f)
    du[1] -= u[1]
    du[2] -= u[2]
    return nothing
end

function add_partial_derivative!(du, u, h, p, t, grad_type::grad_Rx)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, g, dg_oT2 = p

    du[3] += - m0s * u[3] + (1 - m0s) * u[4]
    du[4] +=   m0s * u[3] - (1 - m0s) * u[4]
    return nothing
end

# version for gBloch with using ApproxFun
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Fun,Fun}, t, grad_type::grad_T2s)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, g, dg_oT2 = p
    
    du[4] -= ωy^2 / T2s * quadgk(x -> dg_oT2((t - x) / T2s) * h(x), 0.0, t)[1]
    return nothing
end

# version for free precession (does nothing)
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Array{Any,1},Array{Any,1}}, t, grad_type::grad_T2s)
    return nothing
end

# version for Graham's model
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Number,Any}, t, grad_type::grad_T2s)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, TRF = p
    
    df_PSD = (τ) -> quadgk(ct -> 8 / τ * (exp(-τ^2 / 8 * (3 * ct^2 - 1)^2) - 1) / (3 * ct^2 - 1)^2 + sqrt(2π) * erf(τ / sqrt(8) * abs(3 * ct^2 - 1)) / abs(3 * ct^2 - 1), 0.0, 1.0)[1]
        
    du[4] -= df_PSD(TRF / T2s) * ωy^2 * u[4]
    return nothing
end

function add_partial_derivative!(du, u, h, p, t, grad_type::grad_ω0)
    du[1] -= u[2]
    du[2] += u[1]
    return nothing
end

# version for gBloch (using ApproxFun)
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Fun,Fun}, t, grad_type::grad_ω1)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, g, dg_oT2 = p
    
    du[1] += u[3]
    du[3] -= u[1]
    du[4] -= 2 * ωy * quadgk(x -> g((t - x) / T2s) * h(x), eps(), t)[1]
    return nothing
end

# version for free precession (does nothing)
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Array{Any,1},Array{Any,1}}, t, grad_type::grad_ω1)
    return nothing
end

# version for Graham
function add_partial_derivative!(du, u, h, p::Tuple{Any,Any,Any,Any,Any,Any,Any,Number,Any}, t, grad_type::grad_ω1)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, TRF = p

    f_PSD = (τ) -> quadgk(ct -> 1.0 / abs(1 - 3 * ct^2) * (4 / τ / abs(1 - 3 * ct^2) * (exp(- τ^2 / 8 * (1 - 3 * ct^2)^2) - 1) + sqrt(2π) * erf(τ / 2 / sqrt(2) * abs(1 - 3 * ct^2))), 0.0, 1.0)[1]

    du[1] += u[3]
    du[3] -= u[1]
    du[4] -= f_PSD(TRF / T2s) * 2 * ωy * T2s / π * u[4]
    return nothing
end


function gBloch_Hamiltonian!(du, u, h, p::NTuple{9,Any}, t)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, zs_idx, N = p

    gt = (t, T2s, ct) -> exp(- (t / T2s)^2 * (3 * ct^2 - 1)^2 / 8)

    function fy!(x, y, gt, h, p, T2s, zs_idx, t)
        for i = 1:size(x, 2)
            y[i] = gt(t - x[2,i], T2s, x[1,i]) * h(p, x[2,i]; idxs=zs_idx)
        end
    end

    dy1 = Cubature.pcubature_v((x, y) -> fy!(x, y, gt, h, p, T2s, zs_idx, t), [0.0, max(0.0, t - N * T2s)], [1.0, t])[1]

    if t > (N * T2s)
        dy2 = T2s * sqrt(2π / 3) * Cubature.pcubature(x -> h(p, x[1]; idxs=zs_idx) / (t - x[1]), [0.0], [t - N * T2s])[1]
        
        du[4] = -ωy^2 * ((dy1) + (dy2))
    else
        du[4] = -ωy^2 * (dy1)
    end

    du[1] = - R2f * u[1] - ωz  * u[2] + ωy * u[3]
    du[2] =   ωz  * u[1] - R2f * u[2]
    du[3] = - ωy  * u[1] - (R1 + Rx * m0s) * u[3] +       Rx * (1 - m0s)  * u[4] + (1 - m0s) * R1 * u[5]
    du[4] +=             +       Rx * m0s  * u[3] - (R1 + Rx * (1 - m0s)) * u[4] +      m0s  * R1 * u[5]
    du[5] = 0.0
end

function gBloch_Hamiltonian!(du, u, h, p::NTuple{8,Any}, t)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, N = p
    gBloch_Hamiltonian!(du, u, h, (ωy, ωz, m0s, R1, R2f, T2s, Rx, 4, N), t)
end

function gBloch_Hamiltonian_Gradient!(du, u, h, p, t)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, N = p

    # start integral for T2s gradient 
    dgt = (t, T2s, ct) -> exp(- (t / T2s)^2 * (3.0 * ct^2 - 1)^2 / 8.0) * (t^2 / T2s^3 * (3.0 * ct^2 - 1)^2 / 4.0)

    function dfy!(x, y, dgt, h, p, T2s, zs_idx, t)
        for i = 1:size(x, 2)
            y[i] = dgt(t - x[2,i], T2s, x[1,i]) * h(p, x[2,i]; idxs=zs_idx)
        end
    end

    dy1 = Cubature.pcubature_v((x, y) -> dfy!(x, y, dgt, h, p, T2s, 4, t), [0.0, max(0.0, t - N * T2s)], [1.0, t])[1]

    if t > (N * T2s)
        dy2 = sqrt(2π / 3) * Cubature.pcubature(x -> h(p, x[1]; idxs=4) / (t - x[1]), [0.0], [t - N * T2s])[1]
    end
    
    # Apply Hamiltonian to M and all its derivatives
    for i = 1:5:26
        @inbounds du_v = @view du[i:(i + 4)]
        @inbounds u_v = @view u[i:(i + 4)]
        gBloch_Hamiltonian!(du_v, u_v, h, (ωy, ωz, m0s, R1, R2f, T2s, Rx, (i + 3), N), t)
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
    du[23] += - m0s * u[3] + (1 - m0s) * u[4]
    du[24] +=   m0s * u[3] - (1 - m0s) * u[4]
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
    du[3] = - ωy  * u[1] - (R1 + Rx * m0s) * u[3] +       Rx * (1 - m0s)  * u[4] + (1 - m0s) * R1 * u[5]
    du[4] = -ωy^2 * quadgk(x -> g((t - x) / T2s) * h(p, x; idxs=zs_idx), eps(), t)[1] + Rx * m0s  * u[3] - (R1 + Rx * (1 - m0s)) * u[4] + m0s * R1 * u[5]
end

function gBloch_Hamiltonian_ApproxFun!(du, u, h, p::NTuple{8,Any}, t)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, g = p
    gBloch_Hamiltonian_ApproxFun!(du, u, h, (ωy, ωz, m0s, R1, R2f, T2s, Rx, 4, g), t)
end

function gBloch_Hamiltonian_Gradient_ApproxFun!(du, u, h, p, t)
    ωy, ωz, m0s, R1, R2f, T2s, Rx, g, dg_oT2, grad_list = p
    
    # Apply Hamiltonian to M
    u_v1 = @view u[1:5]
    du_v1 = @view du[1:5]
    gBloch_Hamiltonian_ApproxFun!(du_v1, u_v1, h, (ωy, ωz, m0s, R1, R2f, T2s, Rx, 4, g), t)

    # Apply Hamiltonian to all derivatives and add partial derivatives
    for i = 1:length(grad_list)
        du_v = @view du[5 * i + 1:5 * (i + 1)]
        u_v  = @view u[5 * i + 1:5 * (i + 1)]
        gBloch_Hamiltonian_ApproxFun!(du_v, u_v, h, (ωy, ωz, m0s, R1, R2f, T2s, Rx, (5 * i + 4), g), t)

        add_partial_derivative!(du_v, u_v1, x -> h(p, x; idxs=4), (ωy, ωz, m0s, R1, R2f, T2s, Rx, g, dg_oT2), t, grad_list[i])
    end
end

function FreePrecession_Hamiltonian!(du, u, p, t)
    ωz, m0s, R1, R2f, Rx = p

    du[1] = - R2f * u[1] - ωz  * u[2]
    du[2] =   ωz  * u[1] - R2f * u[2]
    du[3] = - (R1 + Rx * m0s) * u[3] + Rx * (1 - m0s)  * u[4] + (1 - m0s) * R1 * u[5]
    du[4] =   Rx * m0s  * u[3] - (R1 + Rx * (1 - m0s)) * u[4] + m0s  * R1 * u[5]
end

function FreePrecession_Hamiltonian_Gradient!(du, u, p, t)
    ωz, m0s, R1, R2f, Rx, grad_list = p
    # ωy, ωz, m0s, R1, R2f, T2s, Rx, g, dg_oT2, grad_list = p

    # Apply Hamiltonian to M
    u_v1 = @view u[1:5]
    du_v1 = @view du[1:5]
    FreePrecession_Hamiltonian!(du_v1, u_v1, (ωz, m0s, R1, R2f, Rx), t)

    # Apply Hamiltonian to M and all its derivatives
    for i = 1:length(grad_list)
        du_v = @view du[5 * i + 1:5 * (i + 1)]
        u_v  = @view u[5 * i + 1:5 * (i + 1)]
        FreePrecession_Hamiltonian!(du_v, u_v, (ωz, m0s, R1, R2f, Rx), t)

        add_partial_derivative!(du_v, u_v1, [], (0.0, ωz, m0s, R1, R2f, [], Rx, [], []), t, grad_list[i])
    end
end

function Graham_Hamiltonian!(du, u, p, t)
    ωy, ωz, TRF, m0s, R1, R2f, T2s, Rx = p

    f_PSD = (τ) -> quadgk(ct -> 1.0 / abs(1 - 3 * ct^2) * (4 / τ / abs(1 - 3 * ct^2) * (exp(- τ^2 / 8 * (1 - 3 * ct^2)^2) - 1) + sqrt(2π) * erf(τ / 2 / sqrt(2) * abs(1 - 3 * ct^2))), 0.0, 1.0)[1]
    
    du[1] = - R2f * u[1] - ωz  * u[2] + ωy * u[3]
    du[2] =   ωz  * u[1] - R2f * u[2]
    du[3] = - ωy  * u[1] - (R1 + Rx * m0s) * u[3] + Rx * (1 - m0s) * u[4] + (1 - m0s) * R1 * u[5]
    du[4] = - f_PSD(TRF / T2s) * ωy^2 * T2s * u[4] + R1 * (m0s * u[5] - u[4]) + Rx * (m0s * u[3] - (1 - m0s) * u[4])
end

function Graham_Hamiltonian_Gradient!(du, u, p, t)
    ωy, ωz, TRF, m0s, R1, R2f, T2s, Rx, grad_list = p
    
     # Apply Hamiltonian to M
    u_v1 = @view u[1:5]
    du_v1 = @view du[1:5]
    Graham_Hamiltonian!(du_v1, u_v1, (ωy, ωz, TRF, m0s, R1, R2f, T2s, Rx), t)
 
     # Apply Hamiltonian to M and all its derivatives
    for i = 1:length(grad_list)
        du_v = @view du[5 * i + 1:5 * (i + 1)]
        u_v  = @view u[5 * i + 1:5 * (i + 1)]
        Graham_Hamiltonian!(du_v, u_v, (ωy, ωz, TRF, m0s, R1, R2f, T2s, Rx), t)
 
        add_partial_derivative!(du_v, u_v1, [], (ωy, ωz, m0s, R1, R2f, T2s, Rx, TRF, []), t, grad_list[i])
    end
end
 
end