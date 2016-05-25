# The implementation is follow the official ArrayFire-Python wrapper
# detail can be found in https://github.com/arrayfire/arrayfire-python 
# The structure of the wrapper is followed by the list provide by ArrayFire
# This list can be found in http://arrayfire.org/docs/group__unified__func__getavailbackends.htm#ga92a9ce85385763bfa83911cda905afe8
using Compat

const ArrayFire = Libdl.find_library(["libaf"],["/usr/lib/", "/usr/local/lib"])

# Available Backend
const DEFAULT=0
const CPU=1
const CUDA=2
const OPENCL=4
# ArrayFire pointer
typealias AFPtr Ptr{Void}

macro afcheck(fv, argtypes, args...)
  f = eval(fv)
  quote
    _curet = ccall( ($(Meta.quot(f)), $ArrayFire), Cint, $argtypes, $(args...))
    if round(Int, _curet) != AF_SUCCESS
    	err_str = UInt8[0]
    	err_len = Cint[0]
    	ccall((:af_get_last_error, $ArrayFire),Ptr{UInt32},(Int,Ptr{Cint}),err_str,err_len)
      	error = bytestring(err_str[1])
      	throw(error)
    end
  end
end


function afVersion()
	major = Cint[0]
	minor = Cint[0]
	patch = Cint[0]
	ccall((:af_get_version,ArrayFire),Ptr{UInt8},(Ptr{Cint},Ptr{Cint},Ptr{Cint}),major,minor,patch)
	info("ArrayFire version : $(major[1]).$(minor[1]).$(patch[1])")
end 

function availableBackends()
	backend = Cint[0]
	count = Cuint[0]
	@afcheck(:af_get_available_backends,(Ptr{Cint}, ),backend)
	@afcheck(:af_get_backend_count,(Ptr{Cuint}, ),count)
	if backend[1] == 0
		info("No ArrayFire Backend Available")
	elseif backend[1] == 1
		info("$(count[1]) backends avialable: CPU")
		return Dict("CPU"=>CPU)
	elseif backend[1] == 2
		info("$(count[1]) backends avialable: CUDA")
		return Dict("CUDA"=>CUDA)
	elseif backend[1] == 3
		info("$(count[1]) backends avialable: CUDA,CPU")
		return Dict("CPU"=>CPU,"CUDA"=>CUDA)
	elseif backend[1] == 4
		info("$(count[1]) backends avialable: OpenCL")
		return Dict("OpenCL"=>OPENCL)
	elseif backend[1] == 5
		info("$(count[1]) backends avialable: CPU,OpenCL")
		return Dict("CPU"=>CPU,"OpenCL"=>OPENCL)
	elseif backend[1] == 6
		info("$(count[1]) backends avialable: CUDA,OpenCL")
		return Dict("CUDA"=>CUDA,"OpenCL"=>OPENCL)
	elseif backend[1] == 7
		info("$(count[1]) backends avialable: CUDA,CPU,OpenCL")
		return Dict("CPU"=>CPU."CUDA"=>CUDA,"OpenCL"=>OPENCL)
	end
end

function setBackend(backendId::Int)
@afcheck(:af_set_backend,(Cint, ),backendId)
end


function getBackendId()
end

function getDeviceId()
end

# Error Type For ArrayFire internal use 
const AF_SUCCESS=0
# ERROR 100-199 enviroment error
const AF_NO_MEMORY_ERR=101
const AF_DRIVER_ERR=102
const AF_RUNTIME_ERR=103
# ERROR 200-299 input parameters error
const AF_INVALID_ARRAY_ERR=201
const AF_ARG_ERR=202
const AF_SIZE_ERR=203
const AF_TYPE_ERR=204
const AF_DIFF_TYPE_ERR=205
const AF_BATCH_ERR=207
const AF_DEVICE_ERR=208
# ERROR 300-399 missing software features
const AF_NOT_SUPPORTED_ERR=301
const AF_NOT_CONFIGURED_ERR=302
const AF_NONFREE_ERR=303
# ERROR 400-499 missing hardware features
const AF_NO_DBL_ERR=401
const AF_NO_GFX_ERR=402
# ERROR 500-599 specific to the heterogeneous API	   
const AF_LOAD_LIB_ERR=501
const AF_LOAD_SYM_ERR =502
const AF_ARR_BKND_MISMATCH_ERR=503
# ERROR 900-999 upstream libraries and runtime 
const AF_INTERNAL_ERR=998
const AF_UNKNOWN_ERR=999

# Basic ArrayFire Types 
const f32=0 
const c32=1
const f64=2
const c64=3
const b8=4
const s32=5
const u32=6
const u8=7
const s64=8
const u64=9
const s16=10
const u16=11

function AFTypeCheck(dtype::DataType)
	   if dtype <: Float32
	   	  return f32
	   elseif dtype <: Complex{Float32}
	   	  return c32
	   elseif dtype <: Float64 
	   	  return f64
	   elseif dtype <: Complex{Float64}
	   	  return c64
	   elseif dtype <: Bool
	      return b8
	   elseif dtype <: Int32
	      return s32
	   elseif dtype <: UInt32
	   	  return u32
	   elseif dtype <: UInt8
	   	  return u8
	   elseif dtype <: Int64
	   	  return s64
	   elseif dtype <: UInt64
	      return u64
	   elseif dtype <: Int16
	   	  return s16
	   elseif dtype <: UInt16
	   	  return u16
	   else 
	   	  throw("ArrayFire Array support type $(T)")
	   end
end

# Computational source pointer
const device=0
const host=1

# Interpolation methods
const NEAREST=0
const LINEAR=1
const BILINEAR=2
const CUBIC=3
const LOWER=4

# Edge padding 
const ZERO=0
const SYM=1

# Connectivity 
const FOUR=4
const EIGHT=8

# Convolution mode 
const DEFUALT=0
const EXPAND=1

# Convolution domain
const AUTO=0
const SPATIAL=1
const FREQ=2

#sum of absolute differences
const SAD=0
#Zero mean SAD
const ZSAD=1
#Locally scaled SAD
const LSAD=2
#Sum of squared differences
const SSD=3
#Zero mean SSD
const ZSSD=4
#Locally scaled SSD
const LSSD=5
#Normalized cross correlation 
const NCC=6
#Zero mean NCC
const ZNCC=7
#Sum of hamming distances
const SHD=8

const BT_601=601
const BT_709=709
const BT_2020=2020

#Colors 
const GRAY=0
const RGB=1
const HSV=2
const YCbCr=3

#Matrix properties
const NONE=0
const TRANSPOSE=1
const CON_TRANSPOSE=2
const UPPER_TRIAN=32
const LOWER_TRAIN=64
const DIAG_UNIT = 128
const MAT_SYM=512
const POSDEF=1024
const ORTHOG = 2048
const TRI_DIAG=4096
const BLOCK_DIAG=8192
# Note the EULCID type is equal to VECTOR_2
const VECTOR_1=0
const VECTOR_INF=1
const VECTOR_2=2
const VECTOR_P=3
const MATRIX_1=4
const MATRIX_INF=5
const MATRIX_2=6
const MATRIX_L_PQ=7
const EUCLID=2

const DEFAULT=0
const SPECTRUM=1
const COLORS=2
const RED=3
const MOOD=4
const HEAT=5
const BLUE=6

const BMP=1
const ICO=1
const JPEG=2
const JNG=3
const PNG=13
const PPM=14
const PPMRAW=15
const TIFF=18
const PSD=20
const HDR=26
const EXR=29
const JP2=31
const RAW=34

const RANSAC=0
const LMEDS=1

#Markers used for different points in graphics plots
const NONE=0
const POINT=1
const CIRCLE=2
const SQUARE=3
const TRIANGE=4
const CROSS=5
const PLUS=6
const STAR=7

include("ArrayFire/Array.jl")
include("ArrayFire/Math.jl")


