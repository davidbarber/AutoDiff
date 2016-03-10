# Array subindexing:
function Fgetcol(x::Array,incol)
    col=Int(incol[1])
    if col>0
        tmp=zeros(size(x,1),1)
        tmp[:]=x[:,col]
        return (tmp, zeros(size(x)))
    else
        return (zeros(size(x,1)), zeros(size(x)))
    end
end

function Fgetcol_inplace(value,auxvalue,x,incol)
    col=Int(incol[1])
    if col>0
        tmp=zeros(size(x,1),1)
        tmp[:]=x[:,col]
        copy!(value,tmp)
        copy!(auxvalue,zeros(size(x)))
    else
        return
        copy!(value,zeros(size(x,1)))
        copy!(auxvalue,zeros(size(x)))
    end
end

function Dgetcol(derivativeIDX,f_c,faux_c,grad_c,grad_n,x,incol)
    col=Int(incol[1]) # this is ugly since currently we force all values stored on the graph to be real arrays.
    if derivativeIDX==1
        if col>0
            faux_c[:,col]=grad_c
            axpy!(1.0,faux_c,grad_n)
        end
    end
end

#TODO: gpu version

export Fgetcol

Derivative[Fgetcol]=Dgetcol
Inplace[Fgetcol]=Fgetcol_inplace

import Base.*
getindex(A::ADnode,:,i::Int)=ADnode(Fgetcol,[A ADint(i)])
getcol(A::ADnode,i)=ADnode(Fgetcol,[A ADint(i)])

export getindex
export getcol
