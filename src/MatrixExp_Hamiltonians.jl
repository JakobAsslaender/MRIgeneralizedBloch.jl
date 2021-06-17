##
function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1, R2f, Rx, R2s)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0         0;
               ω0  -R2f               0         0               0         0;
         -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f  R1 * m0f;
                0     0               0      -R2s         B1 * ω1         0;
                0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f  R1 * m0s;
                0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1, R2f, Rx, R2s, _, _, _)
    return hamiltonian_linear(ω1, B1, ω0, T, m0s, R1, R2f, Rx, R2s)
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1, R2f, Rx, R2s, _, _, grad_type::grad_m0s)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f        0     0               0         0               0  R1 * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f        0     0               0         0               0  R1 * m0s;
                0     0               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0     0               0         0               0       ω0  -R2f               0         0               0         0;
                0     0             -Rx         0             -Rx -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f       -R1;
                0     0               0         0               0        0     0               0      -R2s         B1 * ω1         0;
                0     0              Rx         0              Rx        0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f        R1;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1, R2f, Rx, R2s, _, _, grad_type::grad_R1)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f        0     0               0         0               0  R1 * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f        0     0               0         0               0  R1 * m0s;
                0     0               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0     0               0         0               0       ω0  -R2f               0         0               0         0;
                0     0              -1         0               0 -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f       m0f;
                0     0               0         0               0        0     0               0      -R2s         B1 * ω1         0;
                0     0               0         0              -1        0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f       m0s;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1, R2f, Rx, R2s, _, _, grad_type::grad_R2f)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f        0     0               0         0               0  R1 * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f        0     0               0         0               0  R1 * m0s;
               -1     0               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0    -1               0         0               0       ω0  -R2f               0         0               0         0;
                0     0               0         0               0 -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f         0;
                0     0               0         0               0        0     0               0      -R2s         B1 * ω1         0;
                0     0               0         0               0        0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f         0;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1, R2f, Rx, R2s, _, _, grad_type::grad_Rx)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f        0     0               0         0               0  R1 * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f        0     0               0         0               0  R1 * m0s;
                0     0               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0     0               0         0               0       ω0  -R2f               0         0               0         0;
                0     0            -m0s         0             m0f -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f         0;
                0     0               0         0               0        0     0               0      -R2s         B1 * ω1         0;
                0     0             m0s         0            -m0f        0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f         0;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1, R2f, Rx, R2s, dR2sdT2s, _, grad_type::grad_T2s)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f        0     0               0         0               0  R1 * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f        0     0               0         0               0  R1 * m0s;
                0     0               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0     0               0         0               0       ω0  -R2f               0         0               0         0;
                0     0               0         0               0 -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f         0;
                0     0               0 -dR2sdT2s               0        0     0               0      -R2s         B1 * ω1         0;
                0     0               0         0               0        0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f         0;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1, R2f, Rx, R2s, _, _, grad_type::grad_ω0)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f        0     0               0         0               0  R1 * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f        0     0               0         0               0  R1 * m0s;
                0    -1               0         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                1     0               0         0               0       ω0  -R2f               0         0               0         0;
                0     0               0         0               0 -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f         0;
                0     0               0         0               0        0     0               0      -R2s         B1 * ω1         0;
                0     0               0         0               0        0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f         0;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1, R2f, Rx, R2s, _, dR2sdB1, grad_type::grad_B1)
    m0f = 1 - m0s
    H = @SMatrix [
             -R2f   -ω0         B1 * ω1         0               0        0     0               0         0               0         0;
               ω0  -R2f               0         0               0        0     0               0         0               0         0;
         -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f        0     0               0         0               0  R1 * m0f;
                0     0               0      -R2s         B1 * ω1        0     0               0         0               0         0;
                0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f        0     0               0         0               0  R1 * m0s;
                0     0              ω1         0               0     -R2f   -ω0         B1 * ω1         0               0         0;
                0     0               0         0               0       ω0  -R2f               0         0               0         0;
              -ω1     0               0         0               0 -B1 * ω1     0  -R1 - Rx * m0s         0        Rx * m0f         0;
                0     0               0  -dR2sdB1              ω1        0     0               0      -R2s         B1 * ω1         0;
                0     0               0       -ω1               0        0     0        Rx * m0s  -B1 * ω1  -R1 - Rx * m0f         0;
                0     0               0         0               0        0     0               0         0               0         0]
    return H * T
end

function propagator_linear_inversion_pulse(ω1, T, B1, R2s, _, _, _)
    Hs = @SMatrix [    -R2s         B1 * ω1;
                   -B1 * ω1               0]
    Us = exp(Hs * T)

    U = @SMatrix [
        sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
        0 0 cos(B1 * ω1 * T)        0 0 0;
        0 0 0 Us[1,1]    Us[1,2]        0;
        0 0 0 Us[2,1]    Us[2,2]        0;
        0 0 0 0          0              1]
    return U
end

function propagator_linear_inversion_pulse(ω1, T, B1, R2s, _, _, grad_type::grad_param)
    Hs = @SMatrix [    -R2s         B1 * ω1;
                   -B1 * ω1               0]
    Us = exp(Hs * T)

    U = @SMatrix [
        sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0;
        0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0;
        0 0 0 Us[1,1]    Us[1,2]        0 0 0 0 0 0;
        0 0 0 Us[2,1]    Us[2,2]        0 0 0 0 0 0;
        0 0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
        0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
        0 0 0 0 0 0 0 cos(B1 * ω1 * T)        0 0 0;
        0 0 0 0 0 0 0 0 Us[1,1]    Us[1,2]        0;
        0 0 0 0 0 0 0 0 Us[2,1]    Us[2,2]        0;
        0 0 0 0 0 0 0 0 0          0              1]
    return U
end

function propagator_linear_inversion_pulse(ω1, T, B1, R2s, dR2sdT2s, _, grad_type::grad_T2s)
    Hs = @SMatrix [    -R2s  B1 * ω1;
                   -B1 * ω1        0]
    Us = exp(Hs * T)

    dHsdT2s = @SMatrix [-dR2sdT2s   0;
                                0   0]
    
    # Higham's Complex Step Approximation:
    h = 1im * eps()
    dU = real.(exp((Hs + h * dHsdT2s) * T) ./ h)

    U = @SMatrix [
        sin(B1 * ω1 * T / 2)^2    0 0 0 0 0 0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2)^2   0 0 0 0 0 0 0 0 0;
        0 0 cos(B1 * ω1 * T)          0 0 0 0 0 0 0 0;
        0 0 0 Us[1,1]    Us[1,2]          0 0 0 0 0 0;
        0 0 0 Us[2,1]    Us[2,2]          0 0 0 0 0 0;
        0 0 0 0 0 sin(B1 * ω1 * T / 2)^2    0 0 0 0 0;
        0 0 0 0 0 0 -sin(B1 * ω1 * T / 2)^2   0 0 0 0;
        0 0 0 0 0 0 0 cos(B1 * ω1 * T)          0 0 0;
        0 0 0 dU[1,1] dU[1,2] 0 0 0 Us[1,1] Us[1,2] 0;
        0 0 0 dU[2,1] dU[2,2] 0 0 0 Us[2,1] Us[2,2] 0;
        0 0 0 0       0       0 0 0 0       0       1]
    return U
end

function propagator_linear_inversion_pulse(ω1, T, B1, R2s, _, dR2sdB1, grad_type::grad_B1)
    Hs = @SMatrix [    -R2s  B1 * ω1;
                   -B1 * ω1        0]
    Us = exp(Hs * T)

    dHsdB1 = @SMatrix [-dR2sdB1  ω1;
                            -ω1   0]
                            
    # Higham's Complex Step Approximation:
    h = 1im * eps()
    dU = real.(exp((Hs + h * dHsdB1) * T) ./ h)

    U = @SMatrix [
        sin(B1 * ω1 * T / 2)^2  0 0 0 0 0 0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0 0 0 0 0 0;
        0 0 cos(B1 * ω1 * T)        0 0 0 0 0 0 0 0;
        0 0 0 Us[1,1]    Us[1,2]        0 0 0 0 0 0;
        0 0 0 Us[2,1]    Us[2,2]        0 0 0 0 0 0;
        sin(B1 * ω1 * T / 2) * cos(B1 * ω1 * T / 2) * ω1 * T  0 0 0 0 sin(B1 * ω1 * T / 2)^2  0 0 0 0 0;
        0 -sin(B1 * ω1 * T / 2) * cos(B1 * ω1 * T / 2) * ω1 * T 0 0 0 0 -sin(B1 * ω1 * T / 2)^2 0 0 0 0;
        0 0 -sin(B1 * ω1 * T) * ω1 * T                            0 0 0 0 cos(B1 * ω1 * T)        0 0 0;
        0 0 0 dU[1,1] dU[1,2] 0 0 0 Us[1,1] Us[1,2] 0;
        0 0 0 dU[2,1] dU[2,2] 0 0 0 Us[2,1] Us[2,2] 0;
        0 0 0 0       0       0 0 0 0       0       1]
    return U
end

function z_rotation_propagator(sweep_phase, _)
    sweep_phase += π
    u_rot = @SMatrix [cos(sweep_phase) -sin(sweep_phase) 0 0 0 0;
                      sin(sweep_phase)  cos(sweep_phase) 0 0 0 0;
                                     0                0  1 0 0 0;
                                     0                0  0 1 0 0;
                                     0                0  0 0 1 0;
                                     0                0  0 0 0 1]
    return u_rot
end

function z_rotation_propagator(sweep_phase, grad::grad_param)
    sweep_phase += π
    u_rot = @SMatrix [cos(sweep_phase) -sin(sweep_phase) 0 0 0 0                 0                0 0 0 0;
                      sin(sweep_phase)  cos(sweep_phase) 0 0 0 0                 0                0 0 0 0;
                      0                 0                1 0 0 0                 0                0 0 0 0;
                      0                 0                0 1 0 0                 0                0 0 0 0;
                      0                 0                0 0 1 0                 0                0 0 0 0;
                      0                 0                0 0 0 cos(sweep_phase) -sin(sweep_phase) 0 0 0 0;
                      0                 0                0 0 0 sin(sweep_phase)  cos(sweep_phase) 0 0 0 0;
                      0                 0                0 0 0 0                 0                1 0 0 0;
                      0                 0                0 0 0 0                 0                0 1 0 0;
                      0                 0                0 0 0 0                 0                0 0 1 0;
                      0                 0                0 0 0 0                 0                0 0 0 1]
    return u_rot
end