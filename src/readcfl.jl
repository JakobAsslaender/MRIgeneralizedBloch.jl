module Readcfl

export readcfl

# READCFL Read complex data from file.
#    READCFL(filenameBase) read in reconstruction data stored in filenameBase.cfl 
#    (complex float) based on dimensions stored in filenameBase.hdr.
#    Parameters:
#        filenameBase:   path and filename of cfl file (without extension)
#    Written to edit data with the Berkeley Advanced Reconstruction Toolbox (BART).
function readcfl(filenameBase)
    dims = readReconHeader(filenameBase);
    data = Array{ComplexF32}(undef, Tuple(dims))

    filename = string(filenameBase, ".cfl");
    fid = open(filename);

    for i in eachindex(data)
        data[i] = read(fid, Float32) + 1im * read(fid, Float32)
    end

    close(fid);
    return data
end

function readReconHeader(filenameBase)
    filename = string(filenameBase, ".hdr");
    fid = open(filename);
    
    line = ["#"]
    while line[1] == "#"
        line = split(readline(fid))
    end

    dims = parse.(Int, line)
    close(fid);
    return dims
end

end