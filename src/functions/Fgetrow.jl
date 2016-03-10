# Array subindexing:
function Fgetrow(x::Array,inrow)
    row=Int(inrow[1])
    if row>0
       tmp=zeros(1,size(x,2))
        tmp[:]=x[row,:]
        return (tmp, zeros(size(x)))
    else
        return (zeros(size(x,2)), zeros(size(x)))
    end
end

function Fgetrow_inplace(value,auxvalue,x,inrow)
    row=Int(inrow[1])
    if row>0
        tmp=zeros(1,size(x,2))
        tmp[:]=x[row,:]
        copy!(value,tmp)
        copy!(auxvalue,zeros(size(x)))
    else
        return
        copy!(value,zeros(size(x,2)))
        copy!(auxvalue,zeros(size(x)))
    end
end

function Dgetrow(derivativeIDX,f_c,faux_c,grad_c,grad_n,x,inrow)
    row=Int(inrow[1]) # this is ugly since currently we force all values stored on the graph to be real arrays.
    if derivativeIDX==1
        if row>0
            faux_c[row,:]=grad_c
            axpy!(1.0,faux_c,grad_n)
        end
    end
end

#TODO: gpu version

export Fgetrow

Derivative[Fgetrow]=Dgetrow
Inplace[Fgetrow]=Fgetrow_inplace

import Base.*
#getindex(A::ADnode,:,i::Int)=ADnode(Fgetcol,[A ADint(i)])
getindex(A::ADnode,i::Int,::Colon)=ADnode(Fgetrow,[A ADint(i)])
getrow(A::ADnode,i)=ADnode(Fgetrow,[A ADint(i)])

export getindex
export getrow
