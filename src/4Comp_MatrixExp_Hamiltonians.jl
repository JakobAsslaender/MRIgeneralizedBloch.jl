##
function hamiltonian_linear(ω1, B1, ω0, T, m0_M, m0_NM, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM)
    m0_IEW = 1 - m0_M - m0_NM - m0_MW
    
    k_MW_M = Rx_M_MW * m0_M  * (m0_MW + m0_M)
    k_M_MW = Rx_M_MW * m0_MW * (m0_M + m0_MW)

    k_IEW_NM = Rx_IEW_NM * m0_NM  * (m0_IEW + m0_NM)
    k_NM_IEW = Rx_IEW_NM * m0_IEW * (m0_NM + m0_IEW)

    k_MW_IEW = Rx_MW_IEW * m0_IEW * (m0_MW + m0_IEW)
    k_IEW_MW = Rx_MW_IEW * m0_MW  * (m0_IEW + m0_MW)

    H = @SMatrix [
-R2_MW - k_MW_IEW                -ω0                     B1 * ω1         0              0              k_IEW_MW                   0                                          0         0                 0               0;
               ω0  -R2_MW - k_MW_IEW                           0         0              0                     0            k_IEW_MW                                          0         0                 0               0;
         -B1 * ω1                  0  -R1_MW - k_MW_M - k_MW_IEW         0         k_M_MW                     0                   0                                   k_IEW_MW         0                 0   R1_MW * m0_MW;
                0                  0                           0     -R2_M        B1 * ω1                     0                   0                                          0         0                 0               0;
                0                  0                      k_MW_M  -B1 * ω1 -R1_M - k_M_MW                     0                   0                                          0         0                 0     R1_M * m0_M;
         k_MW_IEW                  0                           0         0              0    -R2_IEW - k_IEW_MW                 -ω0                                    B1 * ω1         0                 0               0;
                0           k_MW_IEW                           0         0              0                    ω0  -R2_IEW - k_IEW_MW                               0         0                 0               0;
                0                  0                    k_MW_IEW         0              0              -B1 * ω1                   0              -R1_IEW - k_IEW_NM - k_IEW_MW         0          k_NM_IEW R1_IEW * m0_IEW;
                0                  0                           0         0              0                     0                   0                                          0    -R2_NM           B1 * ω1               0;
                0                  0                           0         0              0                     0                   0                                   k_IEW_NM  -B1 * ω1 -R1_NM - k_NM_IEW   R1_NM * m0_NM;
                0                  0                           0         0              0                     0                   0                                          0         0                 0               0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0_M, m0_NM, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM, dR2dT2_M, dR2dB1_M, dR2dT2_NM, dR2dB1_NM, _)
    return hamiltonian_linear(ω1, B1, ω0, T, m0_M, m0_NM, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM)
end

function propagator_linear_inversion_pulse(ω1, T, B1, R2_M, R2_NM, _, _, _, _, _)
    H_M = @SMatrix [   -R2_M         B1 * ω1;
                    -B1 * ω1               0]
    U_M = exp(H_M * T)

    H_NM = @SMatrix [   -R2_NM         B1 * ω1;
                      -B1 * ω1               0]
    U_NM = exp(H_NM * T)

    U = @SMatrix [
        sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0;
        0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0;
        0 0 0 U_M[1,1]   U_M[1,2]       0 0 0 0 0 0;
        0 0 0 U_M[2,1]   U_M[2,2]       0 0 0 0 0 0;
        0 0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
        0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
        0 0 0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0;
        0 0 0 0 0 0 0 0 U_NM[1,1]  U_NM[1,2]      0;
        0 0 0 0 0 0 0 0 U_NM[2,1]  U_NM[2,2]      0;
        0 0 0 0 0 0 0 0 0          0              1]
    return U
end


function propagator_linear_inversion_pulse(ω1, T, B1, R2_M, R2_NM, _, _, _, _, grad_type::grad_param)
    H_M = @SMatrix [   -R2_M         B1 * ω1;
                    -B1 * ω1               0]
    U_M = exp(H_M * T)

    H_NM = @SMatrix [   -R2_NM         B1 * ω1;
                      -B1 * ω1               0]
    U_NM = exp(H_NM * T)

    U = @SMatrix [
        sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 U_M[1,1]   U_M[1,2]       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 U_M[2,1]   U_M[2,2]       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 U_NM[1,1]  U_NM[1,2]      0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 U_NM[2,1]  U_NM[2,2]      0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 U_M[1,1]   U_M[1,2]       0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 U_M[2,1]   U_M[2,2]       0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 U_NM[1,1]  U_NM[1,2]      0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 U_NM[2,1]  U_NM[2,2]      0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0          0              1]
    return U
end

function z_rotation_propagator(rfphase_increment, x::Matrix)
    @assert size(x) == (21, 21)

    sϕ, cϕ = sincos(rfphase_increment)
    u_rot = @SMatrix [cϕ -sϕ 0 0 0  0   0 0 0 0  0   0 0 0 0  0   0 0 0 0 0;
                      sϕ  cϕ 0 0 0  0   0 0 0 0  0   0 0 0 0  0   0 0 0 0 0;
                      0    0 1 0 0  0   0 0 0 0  0   0 0 0 0  0   0 0 0 0 0;
                      0    0 0 1 0  0   0 0 0 0  0   0 0 0 0  0   0 0 0 0 0;
                      0    0 0 0 1  0   0 0 0 0  0   0 0 0 0  0   0 0 0 0 0;
                      0    0 0 0 0 cϕ -sϕ 0 0 0  0   0 0 0 0  0   0 0 0 0 0;
                      0    0 0 0 0 sϕ  cϕ 0 0 0  0   0 0 0 0  0   0 0 0 0 0;
                      0    0 0 0 0  0   0 1 0 0  0   0 0 0 0  0   0 0 0 0 0;
                      0    0 0 0 0  0   0 0 1 0  0   0 0 0 0  0   0 0 0 0 0;
                      0    0 0 0 0  0   0 0 0 1  0   0 0 0 0  0   0 0 0 0 0;
                      0    0 0 0 0  0   0 0 0 0 cϕ -sϕ 0 0 0  0   0 0 0 0 0;
                      0    0 0 0 0  0   0 0 0 0 sϕ  cϕ 0 0 0  0   0 0 0 0 0;
                      0    0 0 0 0  0   0 0 0 0 0    0 1 0 0  0   0 0 0 0 0;
                      0    0 0 0 0  0   0 0 0 0 0    0 0 1 0  0   0 0 0 0 0;
                      0    0 0 0 0  0   0 0 0 0 0    0 0 0 1  0   0 0 0 0 0;
                      0    0 0 0 0  0   0 0 0 0 0    0 0 0 0 cϕ -sϕ 0 0 0 0;
                      0    0 0 0 0  0   0 0 0 0 0    0 0 0 0 sϕ  cϕ 0 0 0 0;
                      0    0 0 0 0  0   0 0 0 0 0    0 0 0 0  0   0 1 0 0 0;
                      0    0 0 0 0  0   0 0 0 0 0    0 0 0 0  0   0 0 1 0 0;
                      0    0 0 0 0  0   0 0 0 0 0    0 0 0 0  0   0 0 0 1 0;
                      0    0 0 0 0  0   0 0 0 0 0    0 0 0 0  0   0 0 0 0 1]
    return u_rot
end



function xs_destructor(x::Matrix)
    if size(x) == (6, 6)
        return Diagonal([1,1,1,0,1,1])
    elseif size(x) == (11, 11)
        return Diagonal([1,1,1,0,1,1,1,1,0,1,1])
    elseif size(x) == (21, 21)
        return Diagonal([1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1])
    else
        error("`xs_destructor` is only implemented for sizes `(6, 6)`, `(11, 11)`, `(21, 21)`, ")
    end
end