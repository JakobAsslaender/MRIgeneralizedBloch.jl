##
function hamiltonian_linear(ω1, B1, ω0, T, m0_M, m0_NM, m0_IEW, m0_MW, Rx_M_MW, Rx_MW_IEW, Rx_IEW_NM, R1_M, R1_NM, R1_IEW, R1_MW, R2_MW, R2_IEW, R2_M, R2_NM)
    k_MW_M = Rx_M_MW * (1 + m0_M/m0_MW)
    k_M_MW = Rx_M_MW * (1 + m0_MW/m0_M)

    k_IEW_NM = Rx_IEW_NM * (1 + m0_NM/m0_IEW)
    k_NM_IEW = Rx_IEW_NM * (1 + m0_IEW/m0_NM)

    k_MW_IEW = Rx_MW_IEW * (1 + m0_IEW/m0_MW)
    k_IEW_MW = Rx_MW_IEW * (1 + m0_MW/m0_IEW)

    H = @SMatrix [
           -R2_MW     -ω0                    B1 * ω1         0              0        0        0                             0         0                 0               0;
               ω0  -R2_MW                          0         0              0        0        0                             0         0                 0               0;
         -B1 * ω1       0 -R1_MW - k_MW_M - k_MW_IEW         0         k_M_MW        0        0                      k_IEW_MW         0                 0   R1_MW * m0_MW;
                0       0                          0     -R2_M        B1 * ω1        0        0                             0         0                 0               0;
                0       0                     k_MW_M  -B1 * ω1 -R1_M - k_M_MW        0        0                             0         0                 0     R1_M * m0_M;
                0       0                          0         0              0  -R2_IEW      -ω0                       B1 * ω1         0                 0               0;
                0       0                          0         0              0       ω0  -R2_IEW                             0         0                 0               0;
                0       0                   k_MW_IEW         0              0 -B1 * ω1        0 -R1_IEW - k_IEW_NM - k_IEW_MW         0          k_NM_IEW R1_IEW * m0_IEW;
                0       0                          0         0              0        0        0                             0    -R2_NM           B1 * ω1               0;
                0       0                          0         0              0        0        0                      k_IEW_NM  -B1 * ω1 -R1_NM - k_NM_IEW   R1_NM * m0_NM;
                0       0                          0         0              0        0        0                             0         0                 0               0]
    return H * T
end

function propagator_linear_inversion_pulse(ω1, T, B1, R2_M, R2_NM)
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

function z_rotation_propagator(ϕ)
    u_rot = @SMatrix [cos(ϕ) -sin(ϕ) 0 0 0     0       0  0 0 0 0;
                      sin(ϕ)  cos(ϕ) 0 0 0     0       0  0 0 0 0;
                          0       0  1 0 0     0       0  0 0 0 0;
                          0       0  0 1 0     0       0  0 0 0 0;
                          0       0  0 0 1     0       0  0 0 0 0;
                          0       0  0 0 0 cos(ϕ) -sin(ϕ) 0 0 0 0;
                          0       0  0 0 0 sin(ϕ)  cos(ϕ) 0 0 0 0;
                          0       0  0 0 0     0       0  1 0 0 0;
                          0       0  0 0 0     0       0  0 1 0 0;
                          0       0  0 0 0     0       0  0 0 1 0;
                          0       0  0 0 0     0       0  0 0 0 1]
    return u_rot
end

function xs_destructor()
    @SMatrix [
        1 0 0 0 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0 0 0;
        0 0 0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 1 0;
        0 0 0 0 0 0 0 0 0 0 1]
end

function A0()
    @SMatrix [
        1 0 0 0 0 0 0 0 0 0 0;    
        0 1 0 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0 0 0;
        0 0 0 1 0 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0 0 0;
        0 0 0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 1 0 0 0 0;        
        0 0 0 0 0 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 0 0 0 1 0;
        0 0 0 0 0 0 0 0 0 0 0]
end
    
function C()
    @SVector [0,0,0,0,0,0,0,0,0,0,1]
end