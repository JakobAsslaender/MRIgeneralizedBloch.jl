function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, dR2sdT2s, dR2sdB1, grad_type::grad_m0s)
    #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =# @inbounds begin
            #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =#
            begin
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:409 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:410 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:411 =#
                begin
                    #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:510 =#
                    SMatrix{11, 11}((*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, (*)((*)(-1, Rx), T), 0, (*)(Rx, T), 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, (*)((*)(-1, Rx), T), 0, (*)(Rx, T), 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, (*)((*)(R1f, T), (+)(1, (*)(-1, m0s))), 0, (*)((*)(R1s, T), m0s), 0, 0, (*)((*)(-1, R1f), T), 0, (*)(R1s, T), 0)
                end
            end
        end
end
 
function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, dR2sdT2s, dR2sdB1, grad_type::grad_R1f)
    #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =# @inbounds begin
            #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =#
            begin
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:409 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:410 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:411 =#
                begin
                    #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:510 =#
                    SMatrix{11, 11}((*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, (*)(-1, T), 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, (*)((*)(R1f, T), (+)(1, (*)(-1, m0s))), 0, (*)((*)(R1s, T), m0s), 0, 0, (*)(T, (+)(1, (*)(-1, m0s))), 0, 0, 0)
                end
            end
        end
end
 
function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, dR2sdT2s, dR2sdB1, grad_type::grad_R2f)
    #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =# @inbounds begin
            #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =#
            begin
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:409 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:410 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:411 =#
                begin
                    #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:510 =#
                    SMatrix{11, 11}((*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, (*)(-1, T), 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, (*)(-1, T), 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, (*)((*)(R1f, T), (+)(1, (*)(-1, m0s))), 0, (*)((*)(R1s, T), m0s), 0, 0, 0, 0, 0, 0)
                end
            end
        end
end
 
function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, dR2sdT2s, dR2sdB1, grad_type::grad_Rx)
    #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =# @inbounds begin
            #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =#
            begin
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:409 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:410 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:411 =#
                begin
                    #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:510 =#
                    SMatrix{11, 11}((*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, (*)((*)(-1, T), m0s), 0, (*)(T, m0s), 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, (*)(T, (+)(1, (*)(-1, m0s))), 0, (*)(T, (+)(-1, m0s)), 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, (*)((*)(R1f, T), (+)(1, (*)(-1, m0s))), 0, (*)((*)(R1s, T), m0s), 0, 0, 0, 0, 0, 0)
                end
            end
        end
end
 
function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, dR2sdT2s, dR2sdB1, grad_type::grad_R1s)
    #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =# @inbounds begin
            #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =#
            begin
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:409 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:410 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:411 =#
                begin
                    #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:510 =#
                    SMatrix{11, 11}((*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, 0, (*)(-1, T), 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, (*)((*)(R1f, T), (+)(1, (*)(-1, m0s))), 0, (*)((*)(R1s, T), m0s), 0, 0, 0, 0, (*)(T, m0s), 0)
                end
            end
        end
end
 
function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, dR2sdT2s, dR2sdB1, grad_type::grad_T2s)
    #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =# @inbounds begin
            #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =#
            begin
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:409 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:410 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:411 =#
                begin
                    #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:510 =#
                    SMatrix{11, 11}((*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, (*)((*)(-1, T), dR2sdT2s), 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, (*)((*)(R1f, T), (+)(1, (*)(-1, m0s))), 0, (*)((*)(R1s, T), m0s), 0, 0, 0, 0, 0, 0)
                end
            end
        end
end
 
function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, dR2sdT2s, dR2sdB1, grad_type::grad_B1)
    #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =# @inbounds begin
            #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =#
            begin
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:409 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:410 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:411 =#
                begin
                    #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:510 =#
                    SMatrix{11, 11}((*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, (*)((*)(-1, T), ω1), 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), (*)(T, ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, (*)((*)(-1, T), dR2sdB1), (*)((*)(-1, T), ω1), 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, (*)(T, ω1), 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, (*)((*)(R1f, T), (+)(1, (*)(-1, m0s))), 0, (*)((*)(R1s, T), m0s), 0, 0, 0, 0, 0, 0)
                end
            end
        end
end
 
function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, dR2sdT2s, dR2sdB1, grad_type::grad_ω0)
    #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =# @inbounds begin
            #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =#
            begin
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:409 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:410 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:411 =#
                begin
                    #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:510 =#
                    SMatrix{11, 11}((*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, T, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, (*)(-1, T), 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, (*)((*)(R1f, T), (+)(1, (*)(-1, m0s))), 0, (*)((*)(R1s, T), m0s), 0, 0, 0, 0, 0, 0)
                end
            end
        end
end
 
function hamiltonian_linear(ω1, B1, ω0, T, m0s, R1f, R2f, Rx, R1s, R2s, dR2sdT2s, dR2sdB1, grad_type::grad_R1a)
    #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =# @inbounds begin
            #= /home/runner/.julia/packages/Symbolics/UPr9b/src/build_function.jl:366 =#
            begin
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:409 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:410 =#
                #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:411 =#
                begin
                    #= /home/runner/.julia/packages/SymbolicUtils/aooYZ/src/code.jl:510 =#
                    SMatrix{11, 11}((*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, (*)(-1, T), 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, 0, (*)(-1, T), 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2f), T), (*)(T, ω0), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, T), ω0), (*)((*)(-1, R2f), T), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(B1, T), ω1), 0, (*)((+)((*)(-1, R1f), (*)((*)(-1, Rx), m0s)), T), 0, (*)((*)(Rx, T), m0s), 0, 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(-1, R2s), T), (*)((*)((*)(-1, B1), T), ω1), 0, 0, 0, 0, 0, 0, 0, 0, (*)((*)(Rx, T), (+)(1, (*)(-1, m0s))), (*)((*)(B1, T), ω1), (*)((+)((*)(-1, R1s), (*)((*)(-1, Rx), (+)(1, (*)(-1, m0s)))), T), 0, 0, 0, (*)((*)(R1f, T), (+)(1, (*)(-1, m0s))), 0, (*)((*)(R1s, T), m0s), 0, 0, (*)(T, (+)(1, (*)(-1, m0s))), 0, (*)(T, m0s), 0)
                end
            end
        end
end
 
