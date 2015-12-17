# A./*B

FAhadamarddivB(A,B)=(A./B,nothing)

function FAhadamarddivB_inplace(value,aux,A,B)
    copy!(value,A./B)
end

function DAhadamarddivB(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,B)
    if derivativeIDX==1
        axpy!(1.0,grad_c./B,grad_n) # should really use BLAS here
    elseif derivativeIDX==2
        axpy!(-1.0,grad_c.*A./(B.*B),grad_n)
    end
end

if GPU    
    function FAhadamarddivB(A::CudaArray,B::CudaArray)
        tmp=CudaArray(Float64,size(A))
        vdiv!(1.0,A,B,tmp)
        return (tmp,nothing)
    end
    
    function FAhadamarddivB_inplace(value::CudaArray,aux,A::CudaArray,B::CudaArray)
        vdiv!(1.0,A,B,value)
    end


    function DAhadamarddivB(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,B::CudaArray)
    if derivativeIDX==1
        vdivupdate!(1.0,grad_c,B,grad_n)
    elseif derivativeIDX==2
        vAoverBupdate!(1.0,grad_c,A,B,grad_n)
    end
end
    
end


Derivative[FAhadamarddivB]=DAhadamarddivB
Inplace[FAhadamarddivB]=FAhadamarddivB_inplace

import Base. ./
./(A::ADnode,B::ADnode)=ADnode(FAhadamarddivB,[A B])
export ./


