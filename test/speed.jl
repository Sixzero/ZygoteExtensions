
using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.10
using BoilerplateCvikli
using ZygoteExtensions: vcat_nospread, stack1
inner = reshape(collect(1f0:660f0),3,1,:)
a = [inner.+i*10 for i in 1:2000]
# @show vcat_nospread(a)
# for j in vcat(a...)
# 	@show j
# end
# @show reshape(stack1(a),4,3)
# for j in vcat_nospread(a)
# 	@show j
# end
# @show vcat(a...)
@sizes vcat(a...)
# @code_warntype vcat_nospread(a, Val(true))
@assert all(vcat(a...) .== vcat_nospread(a))
@btime vcat($a...)
@btime vcat_nospread($a)
# @btime stack1($a)
;
#%%
using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.10
using BoilerplateCvikli
using ZygoteExtensions: get_max_size
using ToggleableAsserts
toggle(false)
strict = Val(true)
A = [randn(3,2) for i in 1:10]
get_max_size(A, )
# @code_warntype get_max_size([randn(3,2) for i in 1:10], Val(true))
@btime get_max_size(A, Val(false))
@btime get_max_size($A, $strict)
;