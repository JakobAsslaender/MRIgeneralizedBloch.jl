function d_hamiltonian_linear_dω1(B1, T, dR2sdω1)
    H = @SMatrix [
           0  0  B1        0   0  0;
           0  0   0        0   0  0;
         -B1  0   0        0   0  0;
           0  0   0 -dR2sdω1  B1  0;
           0  0   0      -B1   0  0;
           0  0   0        0   0  0]
    return H * T
end

function d_hamiltonian_linear_dω1(B1, T, dR2sdω1, _)
  return d_hamiltonian_linear_dω1(B1, T, dR2sdω1)
end
function d_hamiltonian_linear_dω1(B1, T, dR2sdω1, _, _)
    return d_hamiltonian_linear_dω1(B1, T, dR2sdω1)
end
function d_hamiltonian_linear_dω1(B1, T, dR2sdω1, _, _, _)
    return d_hamiltonian_linear_dω1(B1, T, dR2sdω1)
end

function d_hamiltonian_linear_dω1(B1, T, dR2sdω1, _, _, grad_type::grad_param)
    H = @SMatrix [
           0  0  B1            0   0   0  0   0         0   0  0;
           0  0   0            0   0   0  0   0         0   0  0;
         -B1  0   0            0   0   0  0   0         0   0  0;
           0  0   0     -dR2sdω1  B1   0  0   0         0   0  0;
           0  0   0          -B1   0   0  0   0         0   0  0;
           0  0   0            0   0   0  0  B1         0   0  0;
           0  0   0            0   0   0  0   0         0   0  0;
           0  0   0            0   0 -B1  0   0         0   0  0;
           0  0   0            0   0   0  0   0  -dR2sdω1  B1  0;
           0  0   0            0   0   0  0   0       -B1   0  0;
           0  0   0            0   0   0  0   0         0   0  0]
    return H * T
end

function d_hamiltonian_linear_dω1(B1, T, dR2sdω1, _, dR2sdB1dω1, grad_type::grad_B1)
    H = @SMatrix [
           0  0  B1           0   0   0  0   0        0   0  0;
           0  0   0           0   0   0  0   0        0   0  0;
         -B1  0   0           0   0   0  0   0        0   0  0;
           0  0   0    -dR2sdω1  B1   0  0   0        0   0  0;
           0  0   0         -B1   0   0  0   0        0   0  0;
           0  0   1           0   0   0  0  B1        0   0  0;
           0  0   0           0   0   0  0   0        0   0  0;
          -1  0   0           0   0 -B1  0   0        0   0  0;
           0  0   0 -dR2sdB1dω1   1   0  0   0 -dR2sdω1  B1  0;
           0  0   0          -1   0   0  0   0      -B1   0  0;
           0  0   0           0   0   0  0   0        0   0  0]
    return H * T
end


function d_hamiltonian_linear_dω1(B1, T, dR2sdω1, dR2sdT2sdω1, _, grad_type::grad_T2s)
  H = @SMatrix [
              0  0  B1            0   0   0  0   0         0   0  0;
              0  0   0            0   0   0  0   0         0   0  0;
            -B1  0   0            0   0   0  0   0         0   0  0;
              0  0   0     -dR2sdω1  B1   0  0   0         0   0  0;
              0  0   0          -B1   0   0  0   0         0   0  0;
              0  0   0            0   0   0  0  B1         0   0  0;
              0  0   0            0   0   0  0   0         0   0  0;
              0  0   0            0   0 -B1  0   0         0   0  0;
              0  0   0 -dR2sdT2sdω1   0   0  0   0  -dR2sdω1  B1  0;
              0  0   0            0   0   0  0   0       -B1   0  0;
              0  0   0            0   0   0  0   0         0   0  0]
  return H * T
end


function d_hamiltonian_linear_dTRF_add(T, dR2sdTRF)
  H = @SMatrix [
         0  0   0         0   0  0;
         0  0   0         0   0  0;
         0  0   0         0   0  0;
         0  0   0 -dR2sdTRF   0  0;
         0  0   0         0   0  0;
         0  0   0         0   0  0]
  return H * T
end

function d_hamiltonian_linear_dTRF_add(T, dR2sdTRF, _, _, grad_type::grad_param)
  H = @SMatrix [
         0  0   0          0   0   0  0   0          0   0  0;
         0  0   0          0   0   0  0   0          0   0  0;
         0  0   0          0   0   0  0   0          0   0  0;
         0  0   0  -dR2sdTRF   0   0  0   0          0   0  0;
         0  0   0          0   0   0  0   0          0   0  0;
         0  0   0          0   0   0  0   0          0   0  0;
         0  0   0          0   0   0  0   0          0   0  0;
         0  0   0          0   0   0  0   0          0   0  0;
         0  0   0          0   0   0  0   0  -dR2sdTRF   0  0;
         0  0   0          0   0   0  0   0          0   0  0;
         0  0   0          0   0   0  0   0          0   0  0]
  return H * T
end

function d_hamiltonian_linear_dTRF_add(T, dR2sdTRF, dR2sdT2sdTRF, _, grad_type::grad_T2s)
  H = @SMatrix [
         0  0   0              0   0   0  0   0          0   0  0;
         0  0   0              0   0   0  0   0          0   0  0;
         0  0   0              0   0   0  0   0          0   0  0;
         0  0   0      -dR2sdTRF   0   0  0   0          0   0  0;
         0  0   0              0   0   0  0   0          0   0  0;
         0  0   0              0   0   0  0   0          0   0  0;
         0  0   0              0   0   0  0   0          0   0  0;
         0  0   0              0   0   0  0   0          0   0  0;
         0  0   0  -dR2sdT2sdTRF   0   0  0   0  -dR2sdTRF   0  0;
         0  0   0              0   0   0  0   0          0   0  0;
         0  0   0              0   0   0  0   0          0   0  0]
  return H * T
end


function d_hamiltonian_linear_dTRF_add(T, dR2sdTRF, _, dR2sdB1dTRF, grad_type::grad_B1)
  H = @SMatrix [
         0  0   0             0   0   0  0   0          0   0  0;
         0  0   0             0   0   0  0   0          0   0  0;
         0  0   0             0   0   0  0   0          0   0  0;
         0  0   0     -dR2sdTRF   0   0  0   0          0   0  0;
         0  0   0             0   0   0  0   0          0   0  0;
         0  0   0             0   0   0  0   0          0   0  0;
         0  0   0             0   0   0  0   0          0   0  0;
         0  0   0             0   0   0  0   0          0   0  0;
         0  0   0  -dR2sdB1dTRF   0   0  0   0  -dR2sdTRF   0  0;
         0  0   0             0   0   0  0   0          0   0  0;
         0  0   0             0   0   0  0   0          0   0  0]
  return H * T
end