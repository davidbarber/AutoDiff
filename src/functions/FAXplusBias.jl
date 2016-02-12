# f(A,x,b)=A*x+b
# Note that the bias (a column vector) gets expanded here to match the dimension of A*x

FAXplusBias(A,X,b)=(A*X+b*ones(1,size(X,2)),nothing)

function FAXplusBias_inplace(value,aux,A,X,b)
    copy!(value,A*X+b*ones(1,size(X,2)))
end

function DAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,X,b)
    if derivativeIDX==1
        axpy!(1.0,grad_c*X',grad_n)
    elseif derivativeIDX==2
        axpy!(1.0,A'*grad_c,grad_n)
    elseif derivativeIDX==3
        axpy!(1.0,sum(grad_c,2),grad_n)
    end
end


if PROC=="GPU"
#    function FAXplusBias(A::CudaArray,X::CudaArray,b::CudaArray)
#        ons=CudaArray(Float64,(1,size(X,2))); fill!(ons,1.0);        
#        #return (Fxpy(FAX(A,X)[1],FAX(b,ons)[1])[1],nothing)
#        #return (Fxpy(A*X,b*ons)[1],nothing)
#        return (A*X+b*ons,nothing)
#    end


    function FAXplusBias_inplace(value,aux,A::CudaArray{Float64},X::CudaArray{Float64},b::CudaArray{Float64})
        ons=CudaArray(Float64,(1,size(X,2))); fill!(ons,1.0)        
#        Fxpy_inplace(value,aux,FAX(A,X)[1],FAX(b,ons)[1])
        gemm!('N','N',1.0,b,ons,0.0,value)
        gemm!('N','N',1.0,A,X,1.0,value)
        free(ons)
    end

    function DAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray{Float64},X::CudaArray{Float64},b::CudaArray{Float64})
        if derivativeIDX==1
            gemm!('N','T',1.0,grad_c,X,1.0,grad_n)
        elseif derivativeIDX==2
            gemm!('T','N',1.0,A,grad_c,1.0,grad_n)
        elseif derivativeIDX==3
            ons=CudaArray(Float64,(size(X,2),1)); fill!(ons,1.0)
            gemm!('N','N',1.0,grad_c,ons,1.0,grad_n)
            free(ons)
        end
    end


    function FAXplusBias_inplace(value,aux,A::CudaArray{Float32},X::CudaArray{Float32},b::CudaArray{Float32})
        ons=CudaArray(Float32,(1,size(X,2))); fill!(ons,Float32(1.0))        
#        Fxpy_inplace(value,aux,FAX(A,X)[1],FAX(b,ons)[1])
        gemm!('N','N',Float32(1.0),b,ons,Float32(0.0),value)
        gemm!('N','N',Float32(1.0),A,X,Float32(1.0),value)
        free(ons)
    end


    function DAXplusBias(derivativeIDX,f_c,faux_c,grad_c::CudaArray{Float32},grad_n::CudaArray{Float32},A::CudaArray{Float32},X::CudaArray{Float32},b::CudaArray{Float32})
        if derivativeIDX==1
            gemm!('N','T',Float32(1.0),grad_c,X,Float32(1.0),grad_n)
        elseif derivativeIDX==2
            gemm!('T','N',Float32(1.0),A,grad_c,Float32(1.0),grad_n)
        elseif derivativeIDX==3
            ons=CudaArray(Float32,(size(X,2),1)); fill!(ons,Float32(1.0))
            gemm!('N','N',Float32(1.0),grad_c,ons,Float32(1.0),grad_n)
            free(ons)
        end
    end

end

Derivative[FAXplusBias]=DAXplusBias
Inplace[FAXplusBias]=FAXplusBias_inplace

AXplusBias(A,X,b)=ADnode(FAXplusBias,[A X b])
export AXplusBias
