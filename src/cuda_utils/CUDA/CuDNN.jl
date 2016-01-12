# This module is based on CuDNN v3.0
# Detail information about each types and functions can be found in CuDNN LIBRARY USer Guide v3.0
# NVIDIA CuDNN v3.0 brief summary
# Features implemented in CuDNN: 
# 1. Convolution forward and backward, include cross correlation
# 2. Pooling forward and backward
# 3. Softmax forward and backward
# 4. Neuron activations forward and backward
#     i) Rectified linear
#     ii) Sigmoid
#     iii) Hyperbolic tangent
# 5. Tensor transformation function

# This is a lower level wrapper for CuDNN
# The implementation of this module refers to following developments:
# 1. CUDA.jl (https://github.com/JuliaGPU/CUDA.jl)
# 2. cudnn.h v3.0 (NVIDIA)
# 3. mnistCUDNN.cpp (NVIDIA 2014)

#TODO:
# 1. extend cudnncheck to handle more error exceptions 
# 2. Now the wrapper only wrap up the CuDNN v3.0 might need to support elder version
include("CUDA.jl")
export CuDNN
module CuDNN

using CUDA
using Compat
# this only valid for Linux and Mac OS, may also need one for windows
const libcudnn = Libdl.find_library(["libcudnn"], ["/usr/lib/", "/usr/local/cuda/lib"])


#Enumerated types reference to cudnnStatus_t

const  CUDNN_STATUS_SUCCESS          = 0
const  CUDNN_STATUS_NOT_INITIALIZED  = 1
const  CUDNN_STATUS_ALLOC_FAILED     = 2
const  CUDNN_STATUS_BAD_PARAM        = 3
const  CUDNN_STATUS_ARCH_MISMATCH    = 4
const  CUDNN_STATUS_MAPPING_ERROR    = 5
const  CUDNN_STATUS_EXECUTION_FAILED = 6
const  CUDNN_STATUS_INTERNAL_ERROR   = 7
const  CUDNN_STATUS_NOT_SUPPORTED    = 8
const  CUDNN_STATUS_LICENSE_ERROR    = 9


#CuDNN errors

immutable CuDNNError <: Exception
  code :: Int
end
#TODO: The error exception here is only the general summary, detail might be needed.
const cudnnStatus_error = @compat(Dict(
	CUDNN_STATUS_SUCCESS => "The operation complete successfully",
	CUDNN_STATUS_NOT_INITIALIZED =>"The CuDNN library was not initialized",
	CUDNN_STATUS_ALLOC_FAILED => "Resurce allocation failed, to correct: deallocate previous allocated memory as much as possible",
	CUDNN_STATUS_BAD_PARAM =>"Incorrect value passed, to correct: ensure all the parameters being passed have valid values",
	CUDNN_STATUS_ARCH_MISMATCH =>"Feature absent from the GPU device, to correct: to compile and run the application on device with compute capabilities greater than 3.0",
	CUDNN_STATUS_MAPPING_ERROR => "An access to GPU memory space failed, to correct: unbind any previous bound textures",
	CUDNN_STATUS_EXECUTION_FAILED =>"GPU program fail to execute, to correct:ccheck the hardware, an appropriate version of the friver, and the cuDNN library are coreectly installed",
	CUDNN_STATUS_INTERNAL_ERROR =>"An internal cuDNN operation failed",
	CUDNN_STATUS_NOT_SUPPORTED =>"The functionality requested is not presently supported by cuDNN",
	CUDNN_STATUS_LICENSE_ERROR =>"The functionality requested requires license",
	))

import Base.show
show(io::IO, error::CuDNNError) = print(io, cudnnStatus_error[error.code])

macro cudnncheck(fv, argtypes, args...)
  f = eval(fv)
  quote
    _curet = ccall( ($(Meta.quot(f)), $libcudnn), Cint, $argtypes, $(args...)  )
    if round(Int, _curet) != CUDNN_STATUS_SUCCESS
      throw(CuDNNError(round(Int, _curet)))
    end
  end
end

#Check Version
function cudnnGetVersion()
version = ccall((:cudnnGetVersion,libcudnn),Csize_t,())
println("CuDNN version : $version")
return version
end

#context pointer
typealias cudaStream_t Ptr{Void} # hold Cuda Stream
export cudaStrem_t
typealias cudnnHandle_t Ptr{Void} # hold cuDNN library context
export cudnnHandle_t
function cudnnCreate()
handle = cudnnHandle_t[0]
@cudnncheck(:cudnnCreate, (Ptr{cudnnHandle_t},), handle)
return handle[1]
end

function cudnnDestroy(handle ::cudnnHandle_t)
@cudnncheck(:cudnnDestroy, (cudnnHandle_t,), handle)
end

function cudnnSetStream(handle::cudnnHandle_t,streamId::cudaStream_t)
@cudnncheck(:cudnnSetStream,(cudnnHandle_t,cudaStream_t),handle,streamId)
end

function cudnnGetStream(handle::cudnnHandle_t)
streamId = cudaStream_t[0]
@cudnncheck(:cudnnGetStream,(cudnnHandle_t,Ptr{cudaStream_t}),handle,streamId)
return streamId[1]
end

#cudnnDataType_t
CUDNN_DATA_FLOAT = 0  # 32 bits 
CUDNN_DATA_DOUBLE = 1 # 64 bits
CUDNN_DATA_HALF = 2   # 16 bits

function cudnnDataTypeCheck{T<:AbstractFloat}(datatype::Type{T})

if datatype == Float32
	return CUDNN_DATA_FLOAT
elseif datatype == Float64
	return CUDNN_DATA_DOUBLE
elseif datatype == Float16
	return CUDNN_DATA_HALF
else
    error("CUDNN does not support data type $(datatype)")
end
end

function cudnnDataTypeConvert(dateType::Cint)
if dataType == CUDNN_DATA_FLOAT
	return Float32
elseif dataType == CUDNN_DATA_DOUBLE
	return Float64
elseif dataType == CUDNN_DATA_HALF
	return Float16
else
	error("CuDNN error data type:$(datatype)")
end
end

include("Tensor.jl")
include("Filter.jl")
include("Convolution.jl")
include("Softmax.jl")
include("Pooling.jl")
include("Activation.jl")
include("Rectifier.jl")

end