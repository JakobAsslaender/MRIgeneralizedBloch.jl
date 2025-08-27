##
function hamiltonian_linear(ω1, B1, ω0, T, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG)
    k_CFW_C = Rx_C_CFW * (1 + m0_C/m0_CFW)
    k_C_CFW = Rx_C_CFW * (1 + m0_CFW/m0_C)

    k_BW_PG = Rx_BW_PG * (1 + m0_PG/m0_BW)
    k_PG_BW = Rx_BW_PG * (1 + m0_BW/m0_PG)

    k_CFW_BW = Rx_CFW_BW * (1 + m0_BW/m0_CFW)
    k_BW_CFW = Rx_CFW_BW * (1 + m0_CFW/m0_BW)

    k_BW_C = Rx_C_BW * (1 + m0_C/m0_BW)
    k_C_BW = Rx_C_BW * (1 + m0_BW/m0_C)

    k_C_PG = Rx_PG_C * (1 + m0_PG/m0_C)
    k_PG_C = Rx_PG_C * (1 + m0_C/m0_PG)

    H = @SMatrix [#----CFW----------------------------|---------------C----------------------------|-------------------------BW-----------------------------|----------------------------PG-----------------------|
           -R2_CFW     -ω0                      B1 * ω1         0                                  0        0        0                                      0         0                         0                 0;
                ω0 -R2_CFW                            0         0                                  0        0        0                                      0         0                         0                 0;
          -B1 * ω1       0 -R1_CFW - k_CFW_C - k_CFW_BW         0                            k_C_CFW        0        0                               k_BW_CFW         0                         0   R1_CFW * m0_CFW;
                 0       0                            0     -R2_C                            B1 * ω1        0        0                                      0         0                         0                 0;
                 0       0                      k_CFW_C  -B1 * ω1  -R1_C - k_C_CFW - k_C_BW - k_C_PG        0        0                                 k_BW_C         0                    k_PG_C       R1_C * m0_C;
                 0       0                            0         0                                  0   -R2_BW      -ω0                                B1 * ω1         0                         0                 0;
                 0       0                            0         0                                  0       ω0   -R2_BW                                      0         0                         0                 0;
                 0       0                     k_CFW_BW         0                             k_C_BW -B1 * ω1        0   -R1_BW - k_BW_PG - k_BW_CFW - k_BW_C         0                   k_PG_BW     R1_BW * m0_BW;
                 0       0                            0         0                                  0        0        0                                      0    -R2_PG                   B1 * ω1                 0;
                 0       0                            0         0                             k_C_PG        0        0                                k_BW_PG  -B1 * ω1 -R1_PG - k_PG_BW - k_PG_C     R1_PG * m0_PG;
                 0       0                            0         0                                  0        0        0                                      0         0                         0                 0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG, dR2dT2_C, dR2dB1_C, dR2dT2_PG, dR2dB1_PG, _)
    return hamiltonian_linear(ω1, B1, ω0, T, m0_C, m0_PG, m0_BW, m0_CFW, Rx_C_CFW, Rx_CFW_BW, Rx_BW_PG, Rx_C_BW, Rx_PG_C, R1_C, R1_PG, R1_BW, R1_CFW, R2_CFW, R2_BW, R2_C, R2_PG)
end

function propagator_linear_inversion_pulse(ω1, T, B1, R2_C, R2_PG, _, _, _, _, _)
    H_C = @SMatrix [   -R2_C         B1 * ω1;
                    -B1 * ω1               0]
    U_C = exp(H_C * T)

    H_PG = @SMatrix [   -R2_PG         B1 * ω1;
                      -B1 * ω1               0]
    U_PG = exp(H_PG * T)

    U = @SMatrix [
        sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0;
        0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0;
        0 0 0 U_C[1,1]   U_C[1,2]       0 0 0 0 0 0;
        0 0 0 U_C[2,1]   U_C[2,2]       0 0 0 0 0 0;
        0 0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
        0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
        0 0 0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0;
        0 0 0 0 0 0 0 0 U_PG[1,1]  U_PG[1,2]      0;
        0 0 0 0 0 0 0 0 U_PG[2,1]  U_PG[2,2]      0;
        0 0 0 0 0 0 0 0 0          0              1]
    return U
end


function propagator_linear_inversion_pulse(ω1, T, B1, R2_C, R2_PG, _, _, _, _, grad_type::grad_param)
    H_C = @SMatrix [   -R2_C         B1 * ω1;
                    -B1 * ω1               0]
    U_C = exp(H_C * T)

    H_PG = @SMatrix [   -R2_PG         B1 * ω1;
                      -B1 * ω1               0]
    U_PG = exp(H_PG * T)

    U = @SMatrix [
        sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 U_C[1,1]   U_C[1,2]       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 U_C[2,1]   U_C[2,2]       0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 U_PG[1,1]  U_PG[1,2]      0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 U_PG[2,1]  U_PG[2,2]      0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 U_C[1,1]   U_C[1,2]       0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 U_C[2,1]   U_C[2,2]       0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 U_PG[1,1]  U_PG[1,2]      0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 U_PG[2,1]  U_PG[2,2]      0;
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