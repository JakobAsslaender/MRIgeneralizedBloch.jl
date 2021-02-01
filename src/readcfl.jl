module Readcfl

export readcfl

"""
    readcfl(filename(no extension)) -> Array{ComplexF32,N} where N is defined the filename.hdr file
    readcfl(filename.cfl) -> Array{ComplexF32,N} where N is defined the filename.hdr file
    readcfl(filename.hdr) -> Array{ComplexF32,N} where N is defined the filename.hdr file

Reads complex data from files created by the Berkeley Advanced Reconstruction Toolbox (BART).
The output is an Array of ComplexF32 with the dimensions stored in a .hdr file.

Parameters:
    filenameBase:   path and filename of the cfl and hdr files, which can either be without extension, end on .cfl, or end on .hdr

Copyright: Jakob Asslaender, NYU School of Medicine, 2021 (jakob.aslaender@nyumc.org)
"""
function readcfl(filename)

    if filename[end-3:end] == ".cfl"
        filenameBase = filename[1:end-4]
    elseif filename[end-3:end] == ".hdr"
        filenameBase = filename[1:end-4]
        filename = string(filenameBase, ".cfl");
    else
        filenameBase = filename
        filename = string(filenameBase, ".cfl");
    end

    dims = readreconheader(filenameBase);
    data = Array{ComplexF32}(undef, Tuple(dims))

    fid = open(filename);

    for i in eachindex(data)
        data[i] = read(fid, Float32) + 1im * read(fid, Float32)
    end

    close(fid);
    return data
end

function readreconheader(filenameBase)
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