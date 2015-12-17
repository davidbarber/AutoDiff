function settrue!(x::BitArray,b)
    fill!(x,false)
    x[b]=true
end

function settrue(N::Integer,b)
    x=BitArray(N)
    x[b]=true
    return x
end
