using MRIgeneralizedBloch
using Test

@testset "Greens_Interpolation" begin
    include("Greens_Interpolation.jl")
end

@testset "apply_hamiltonian_gbloch!" begin
    for i = 1:10 # test different random initializations
        include("gBloch_Hamiltonian.jl")
    end
end

@testset "apply_hamiltonian_freeprecession!" begin
    for i = 1:10 # test different random initializations
        include("FreePrecession_Hamiltonian.jl")
    end
end

@testset "apply_hamiltonian_graham_superlorentzian!" begin
    for i = 1:10 # test different random initializations
        include("Graham_Hamiltonian_superLorentzian.jl")
    end
end

@testset "apply_hamiltonian_sled!" begin
    for i = 1:10 # test different random initializations
        include("Sled_Hamiltonian_superLorentzian.jl")
    end
end

@testset "Linear_Approximation" begin
    include("Linear_Approximation.jl")
end

@testset "apply_hamiltonian_gbloch! gradients" begin
    include("gBloch_Hamiltonian_Gradients.jl")
end

@testset "apply_hamiltonian_freeprecession! gradients" begin
    include("FreePrecession_Hamiltonian_Gradients.jl")
end

@testset "Linear_Approx gradients" begin
    include("Linear_Approx_Gradients.jl")
end

@testset "apply_hamiltonian_graham_superlorentzian! gradients" begin
    include("Graham_Hamiltonian_Gradients.jl")
end

@testset "inversionpulse" begin
    include("inversionpulse.jl")
end

@testset "cf. Solvers" begin
    include("Solvers_cf.jl")
end

@testset "Solvers_Gradients" begin
    include("Solvers_Gradients.jl")
end

@testset "OCT Gradients" begin
    for i = 1:10 # test different random initializations
        include("OCT.jl")
    end
end