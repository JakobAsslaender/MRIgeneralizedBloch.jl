using HTTP
using BufferedStreams

function load_first_datapoint(filename; set_phase=:real)
    data = load_Data(filename)
    M = data[1,:]

    if set_phase == :real
        phase = angle(M[end])
        M = M .* exp(-1im * phase)
        M = real.(M)
    elseif set_phase == :abs
        M = abs.(M)
    end
    return M
end

function load_Data(filename)
    if filename[1:4] == "http"
        io = Base.BufferStream()
        mytask = @async readdata(io)
        HTTP.request("GET", filename, response_stream=io)
        data = fetch(mytask)
    else
        filename = expanduser(filename)
        io = open(filename)
        data = readdata(io)
    end

    close(io)
    return data
end

function readdata(io)
    for _=1:12
        Char.(read(io, Char))
    end
    if read(io, Int32) != 501; error; end # 500 real; 501 complex; 502 double real; 503 xy_real; 504 xy_complex

    size1 = read(io, Int32)
    size2 = read(io, Int32)
    if read(io, Int32) != 1; error; end
    if read(io, Int32) != 1; error; end

    data = zeros(ComplexF32, size1, size2)
    for i = 1:length(data)
        data[i] = read(io, Float32) + 1im * read(io, Float32)
    end
    
    return data[7:end - 7,:]
end