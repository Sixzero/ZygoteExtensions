module ZygoteExtensions

using Zygote
using Flux
using InteractiveUtils
using Distributed
using Boilerplate: sizes, @sizes, @typeof, map_array
using ToggleableAsserts
using Boilerplate


vnorm(v::AbstractVector{<:Real}) = (v ./ sum(v))
antizero(arr) = (arr.+=abs.(arr); arr./=2)

function mean_nonzero(x; dims)
  nonzero_count = Zygote.@ignore (sum(x .!= 0, dims=dims) .+ 1f-4)
  sum(x, dims=dims) ./ nonzero_count
end
# function softmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
#   max_ = maximum(x; dims)
#   # max_ = 0
#   if all(isfinite, max_)
#       # @fastmath out .= exp.(x .- max_)
#       @fastmath out .= exp.(x)
#   else
#       @fastmath @. out = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 1, 0), exp(x - max_))
#   end
#   out ./= sum(out; dims)
# end
# softmax(x; dims = 1) = softmax!(zero(x), x; dims)
softmax(x::Vector; dims) = Flux.softmax.(x; dims)
softmax(x; dims) = Flux.softmax(x; dims)
@inline softmax_dim(dims) = arr -> softmax(arr; dims)
function onehot(y::AbstractArray, classes)
  @assert size(y, 1) > 2 "Batch dimensions should be larger. why?: $(sizes(y)) and why?: $(sizes(classes))"
  classes = collect(classes)
  inp_shape = size(y)
  y = Flux.onehotbatch(Int.(reshape(y, :)), classes)
  reshape(y, length(classes), inp_shape...)
end

function grad_timer(op, args...; msg = "", kw...)
  res = @time op(args...; kw...)
  println("FW timed: $(msg) ")
  res
end
Zygote.@adjoint function grad_timer(op, args...; msg = "", kw...)
  @time res, back = Zygote._pullback(op, args...; kw...)
  println("FW timed: $(msg)")
  function pullback_fn(Δy)
    @time res_back = back(Δy)
    println("BW timed: $(msg)")
    res_back
  end
  res, pullback_fn
end
macro gtime(ex, msgs...)
  if ex.head == :call
    msg_body = isempty(msgs) ? ex : msgs[1]
    msg = string(msg_body)
    return esc(:($grad_timer($(ex.args...); msg = $(msg))))
  else
    @assert false "@gtime wrong usage."
  end
end


get_slice(i) = arr_t -> arr_t[i]
assign_eles!(to, from, i) = for (j, ele) in enumerate(from)
  # to[j] !== nothing && ele !== nothing && to[j][i] !== nothing && any(ele .!== nothing) && @show (sizes(ele), typeof(ele))
  # to[j] !== nothing && to[j][i] !== nothing && @show (sizes(to[j][i]), typeof(to[j][i]), i, j)
  to[j] !== nothing && ele !== nothing && to[j][i] !== nothing && any(ele .!== nothing) && (to[j][i] = ele)
end
lax_scan_r(op::Function, init, itr) = lax_scan((a, b) -> op(b, a), init, reverse(itr), missing_parameter!)
function lax_scan_fw(op::Function, init, itr, extras = nothing)
  inputs = itr .|> get_slice(1)
  # @show "RNN forward"
  # @show "ZYGOTE_OP!"
  # @code_warntype Zygote._pullback(op, init, inputs, extras)
  # @show "ONLY OP!"
  # @code_warntype op(init, inputs, extras)
  (state, y), back = Zygote._pullback(op, 1, init, inputs, extras)
  @assert typeof(state) == typeof(init) "next_state: $(typeof(state)) != init_state: $(typeof(init))"
  timesteps = size(itr[1], 1)
  backs = similar(1:timesteps, typeof(back))
  yt = similar(1:timesteps, typeof(y))
  state_t = similar(1:timesteps, typeof(state))
  backs[1] = back
  yt[1] = y
  state_t[1] = state
  for i = 2:timesteps
    inputs = itr .|> get_slice(i)
    (state, y), back = Zygote._pullback(op, i, state, inputs, extras)
    backs[i] = back
    yt[i] = y
    state_t[i] = state
  end
  return (state_t, yt), backs
end
function lax_scan(op::Function, init, itr, extras)
  # @code_warntype lax_scan_fw(op, init, itr, extras)
  return lax_scan_fw(op, init, itr, extras)[1]
end
Zygote.@adjoint function lax_scan(op::Function, init, itr, extras)
  timesteps::Int = size(itr[1], 1)
  @assert typeof(itr) <: Tuple "\"Input\" outter dimension must be tuple for preallocation as for now."
  @assert typeof(itr[1]) <: Array "\"Input\" outter dimension must be array for preallocation as for now."
  (state_t, yt), backs = lax_scan_fw(op, init, itr, extras)
  # @show "all FW"

  lax_scan_pullback = let itr=itr; 
    (Δy) -> begin
      # @show "RNN backward"
      Δstate_t, Δyt = Δy
      # @assert typeof(Δy[2][timesteps-1]) <: Nothing "Still only 0 or 1 overlap supported."
      d_output = typeof(Δyt[timesteps]) <: Nothing ? zero.(Δyt[timesteps-1]) : Δyt[timesteps]
      Δstate_end = Δstate_t !== nothing ? Δstate_t[end] : nothing
      ∂op, _, ∂s, ∂itri, ∂extras = backs[timesteps]((Δstate_end, d_output))
      ∂itrs = itr |> map_array(similar) # typeof(itr[1])
      ∂extrast = ∂extras
      assign_eles!(∂itrs, ∂itri, timesteps)

      for i = timesteps-1:-1:1
        Δtwoway_state = Δstate_t !== nothing && Δstate_t[i] !== nothing ? ∂s .+ Δstate_t[i] : ∂s
        ∂opi, _, ∂s, ∂itrsi, ∂extr = backs[i]((Δtwoway_state, Δyt[i]))
        assign_eles!(∂itrs, ∂itrsi, i)
        ∂extrast = Zygote.accum(∂extrast, ∂extr)
      end
      return ∂op, ∂s, ∂itrs, ∂extrast
    end
  end
  return (state_t, yt), lax_scan_pullback
end
# macro xshow(exs...)
#   blk = Expr(:block)
#   for ex in exs
#       push!(blk.args, :($(show(ex))))
#       push!(blk.args, :(println("=", repr(begin value=$(esc(ex)) end))))
#   end
#   isempty(exs) || push!(blk.args, :value)
#   return blk
# end

observe(x, msg = "") = x
Zygote.@adjoint function observe(x, msg = nothing)
  x, dy -> (println(msg !== nothing ? "$msg " : "", "$(typeof(dy)) $dy", x); (dy, nothing))
end
Zygote.@adjoint function Boilerplate.sizes(x,)
  sizes(x), dy -> ((nothing))
end

function gradient_pro(f, args...)
  y, back = Zygote.pullback(f, args...)
  return y, back(one(y))
end
struct S{T} end
get_max_size(l::Vector) = get_max_size(l, Val(true))
get_max_size(l::Vector, strict::Val{true}) = begin
  @toggled_assert let 
    is_valid = true
    for (i,l_i) in enumerate(l[2:end])
      is_valid &= all(size(l_i) .=== size(l[1]))
    end
    is_valid
  end "All sizes needs to be the same." # size($(i-1))!=size($i) $(last_s) and $(size(l_i))" # TODO Tomi these aren't existing variables in this scope... what did we want to do with there?
  size(l[1])
end
get_max_size(l::Vector, strict::Val{false}) = begin
  max_size = size(l[1])
  for (i,l_i) in enumerate(l[2:end])
    max_size = (size(l_i,j) > max_size[j] ? size(l_i,j) : max_size[j] for j in 1:ndims(l_i))
  end
  max_size
end
# vcat_nospread(l::Vector{Array{Float32,N}}; strict=Val(1)) = 
vcat_nospread(l::Vector{Array{Float32,N}}, strict::Val=Val(true)) where {N} = begin
  max_size::NTuple{N,Int64} = get_max_size(l, strict)

  data::Array{Float32, N} = zeros(Float32, length(l)*max_size[1], max_size[2:end]...)
  @inbounds for (i, d) in enumerate(l)
    s1 = size(d,1)
    for j in 0:div(length(d),s1)-1
      for k in 1:s1
        data[(i-1)*s1 + j*s1*length(l) + k] = d[j*s1+k]
      end
    end
  end
  data
end
stack1(l::Vector{Array{T,N}}, strict::Val=Val(true)) where {N, T} = begin
  max_size::NTuple{N,Int64} = get_max_size(l, strict)
  data::Array{T, N+1} = zeros(T, length(l), max_size...)
  @inbounds for (i, d) in enumerate(l)
    for j in eachindex(d)
      data[i + (j-1)*length(l)] = d[j]
    end
  end
  data
end
Zygote.@adjoint function stack1(l)
  data = zeros(Float32, length(l), size(l[1])...)
  @inbounds for (i, d) in enumerate(l)
    for j in eachindex(d)
      data[i + (j-1)*length(l)] = d[j]
    end
  end
  data, dy -> begin
    @info "Todo check this code."
    [dy[i,..] for i in 1:length(l)]
  end
end
UniversalDType = Union{Tuple, AbstractArray, Dict}
size_eq(x, y) = all(size(x) .== size(y)) ? "" : "WARNING"
xshowgrad(msg::String) = (y -> xshowgrad(y, msg))
xshowgrad(y::UniversalDType, msg::String = "") = y

Zygote.@adjoint function xshowgrad(y::UniversalDType, msg::String = "")
  function back(dy)
    println("Gshow: $(typeof(dy)) $(size(dy)) $(msg) $(size_eq(y, dy))")
    (dy, nothing)
  end
  y, back
end

macro gshow(exs...)
  blk = Expr(:block)
  for ex in exs
    push!(blk.args, :($(show(ex))))
    push!(blk.args, :(println(" = ", repr(begin
      value = $(esc(begin
        ex = xshowgrad(ex)
      end))
    end))))
  end
  isempty(exs) || push!(blk.args, :value)
  return blk
end
select(c, t, f) = c ? t : f

permutedims_by(perm::Union{Tuple, AbstractVector}) = arr -> permutedims(arr, perm)

end # module
