# A.*B

FAhadamardprodB(A,B)=(A.*B,nothing)

function FAhadamardprodB_inplace(value,aux,A,B)
    copy!(value,A.*B)
end

function DAhadamardprodB(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,B)
    if derivativeIDX==1
        axpy!(1.0,grad_c.*B,grad_n) 
    elseif derivativeIDX==2
        axpy!(1.0,grad_c.*A,grad_n)
    end
end

if (PROC=="GPU") || (PROC=="GPU32")
    
    function FAhadamardprodB(A::CudaArray,B::CudaArray)
        tmp=CudaArray(Float64,size(A))
        vmult!(1.0,A,B,tmp)
        return (tmp,nothing)
    end    
    
    function FAhadamardprodB_inplace(value::CudaArray,aux,A::CudaArray,B::CudaArray)
        vmult!(1.0,A,B,value)
    end


    function DAhadamardprodB(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,B::CudaArray)
    if derivativeIDX==1
        vmultupdate!(1.0,grad_c,B,grad_n)
    elseif derivativeIDX==2
        vmultupdate!(1.0,grad_c,A,grad_n)
    end
end
    
end


Derivative[FAhadamardprodB]=DAhadamardprodB
Inplace[FAhadamardprodB]=FAhadamardprodB_inplace

import Base. .*
.*(A::ADnode,B::ADnode)=ADnode(FAhadamardprodB,[A B])

export .*

export FAhadamardprodB
