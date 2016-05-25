function matmul(leftOperant::AFArray,rightOperant::AFArray)
	result = AFPtr[0]
	leftOption = 0
	rightOption = 0
    @afcheck(:af_matmul,(Ptr{AFPtr},AFPtr,AFPtr,UInt32,UInt32),result,leftOperant.ptr,rightOperant.ptr,leftOption,rightOption)
    result = AFArray(result[1],true)
end
