# Standard rectlin layer: f(A,x,b)=rectlin(A*x+b)
# Note that the bias (a column vector) gets expanded here to match the dimension of A*x

FrectlinAXplusBias(A,X,b)=begin; a=A*X+b*ones(1,size(X,2)); return (max(a,0.),(a.>0)); end

function FrectlinAXplusBias_inplace(value,aux,A,X,b)
    a=A*X+b*ones(1,size(X,2))
    copy!(value,max(a,0.0))
    copy!(aux,(a.>0))
end

function DrectlinAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A,X,b)
    if derivativeIDX==1
        axpy!(1.0,(grad_c.*faux_c)*X',grad_n)
    elseif derivativeIDX==2
        axpy!(1.0,A'*(grad_c.*faux_c),grad_n)
    elseif derivativeIDX==3
        axpy!(1.0,sum(grad_c.*faux_c,2),grad_n)
    end
end

if GPU    
#    function FrectlinAXplusBias(A::CudaArray,X::CudaArray,b::CudaArray)       
#        return(rectlin(FAXplusBias(A,X,b)[1]),FAXplusBias(A,X,b)[1])
#        # TODO: FIX THIS UNNECESSARY DOUBLE COMPUTATION
#    end
    
    function FrectlinAXplusBias_inplace(value,aux,A::CudaArray,X::CudaArray,b::CudaArray)
        FAXplusBias_inplace(value,aux,A,X,b)
        aux=CudaArray(Float64,size(value))
        copy!(aux,value)               
        rectlin!(value,value)
    end
    
    function DrectlinAXplusBias(derivativeIDX,f_c,faux_c,grad_c,grad_n,A::CudaArray,X::CudaArray,b::CudaArray)
        tmp=CudaArray(Float64,size(grad_c)); fill!(tmp,0.0)
        #A_emult_Bg0!(grad_c,faux_c,tmp)               
        # This is a bit silly -- the CPU and GPU faux store different quantities        
        # This means that one cannot convert from CPU to GPU and then run backward pass on the GPU to get correct results -- we would have to first run a forward pass on the GPU
        A_emult_Bg0!(grad_c,f_c,tmp)               
        if derivativeIDX==1
            gemm!('N','T',1.0,tmp,X,1.0,grad_n)
        elseif derivativeIDX==2
            gemm!('T','N',1.0,A,tmp,1.0,grad_n)
        elseif derivativeIDX==3
            ons=CudaArray(Float64,(size(X,2),1)); fill!(ons,1.0)
            gemm!('N','N',1.0,tmp,ons,1.0,grad_n)
            free(ons)
        end
        free(tmp)
    end

end



Derivative[FrectlinAXplusBias]=DrectlinAXplusBias
Inplace[FrectlinAXplusBias]=FrectlinAXplusBias_inplace

rectlinAXplusBias(A,X,b)=ADnode(FrectlinAXplusBias,[A X b])
export rectlinAXplusBias


