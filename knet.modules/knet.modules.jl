using Knet

# Global context of parameters
typealias ParamContext Array{Any, 1}
typealias ParamIndex Int
# the default parameter context
global __default_param_context__ = ParamContext()

function get_default_parameter_context()::ParamContext
   return __default_param_context__
end

#=function reset_default_parameter_context()
   __default_param_context__ = ParamContext()
end=#

global __active_parameter_context__ = get_default_parameter_context()

function get_active_parameter_context()::ParamContext
   return __active_parameter_context__
end

function set_active_parameter_context!(w)
   global __active_parameter_context__ = w
end

function register_parameter!(context::ParamContext, w)::ParamIndex
   push!(context, w)
   return ParamIndex(length(context))
end

function get_value(context, index::ParamIndex)
   context[index]
end

function set_value!(context, index::ParamIndex, w)
   param = context[index]
   copy!(param, typeof(param)(w))
end

# Parameter abstraction
type Parameter
   index::ParamIndex
   context::ParamContext
   Parameter(context::ParamContext, w) = new(register_parameter!(context, w), context)
   Parameter(w) = let context = get_active_parameter_context()
      new(register_parameter!(context, w), context)
   end
end

get_value(context, v::Parameter) = get_value(context, v.index)

get_value(v::Parameter) = get_value(get_active_parameter_context(), v.index)

set_value!(context::ParamContext, v::Parameter, w) = set_value!(context, v.index, w)

set_value!(v::Parameter, w) = set_value!(get_active_parameter_context(), v.index, w)

get_grad(v::Parameter, grads) = grads[v.index]

get_value(context, x) = x

function expand_parameter_op(expr)
   if typeof(expr) == Symbol
      Expr(:call, get_value, :context, expr)
   elseif typeof(expr) == Expr && expr.head == :call
      rator = eval(expr.args[1])
      rands = map(expand_parameter_op, expr.args[2:end])
      Expr(:call, rator, rands...)
   elseif typeof(expr) == Expr && expr.head == :.
      Expr(:call, get_value, :context, expr)
   else
      expr
   end
end

macro po(expr::Expr)
   expand_parameter_op(expr)
end

# TODO: support device ids
function gpu!(v::Parameter; dtype=Float32)
   if Knet.gpu() < 0
      warn("No available gpu, transfer failed")
      return
   end
   context = v.context
   context[v.index] = KnetArray{dtype}(context[v.index])
end

function cpu!(v::Parameter; dtype=Float32)
   context = v.context
   context[v.index] = Array{dtype}(context[v.index])
end

# Module abstraction
abstract Module

function gpu!(m::Module; o...)
   if Knet.gpu() < 0
      warn("No available gpu, transfer ignored, using cpu...")
      return
   end
   for p in parameters(m)
      gpu!(p; o...)
   end
end

function cpu!(m::Module; o...)
   for p in parameters(m)
      cpu!(p; o...)
   end
end

#Macro for calling types directly
macro mc(expr)
   if typeof(expr) == Expr && expr.head == :call
      return Expr(:call, :forward, :context, expr.args[1], expr.args[2:end]...)
   end
   return expr
end

macro run(m, args...)
   Expr(:call, forward, Expr(:call, get_active_parameter_context), m, args...)
end

function _populate_recursive(m, list, match_type)
   if isa(m, match_type)
      push!(list, m)
   end

   if isa(m, Module)
      for fn in fieldnames(m)
         _populate_recursive(getfield(m, fn), list, match_type)
      end
   elseif isa(m, Array) || isa(m, Tuple)
      for l in m
         _populate_recursive(l, list, match_type)
      end
   elseif isa(m, Associative)
      for v in values(m)
         _populate_recursive(v, list, match_type)
      end
   else
      return
   end
end

function parameters(m::Module)
   res = []
   _populate_recursive(m, res, Parameter)
   return res
end

# Contains itself
function submodules(m::Module)
   res = []
   _populate_recursive(m, res, Module)
   return res
end


function set_mode!(m::Module, mode)
   for sm in submodules(m)
      if :mode in fieldnames(sm)
         sm.mode = mode
      end
   end
end

training!(m::Module) = set_mode!(m, :train)

testing!(m::Module) = set_mode!(m, :test)


# loss: (ypred, ygold) -> scalar
import AutoGrad.grad
function grad(m::Module, loss)
   _predict(w, x) = forward(w, m, x)
   _loss(w, x, y) = loss(_predict(w, x), y)
   lossgrad = grad(_loss)
   return (x, y)->lossgrad(get_active_parameter_context(), x, y)
end

#---Basic Module Definitions

type Linear <: Module
   weight
   bias
   Linear(input::Int, output::Int; dtype=Array{Float32}, bias=true) = begin
      stdv = 1 ./ sqrt(input)
      uniform(dims) = rand(dims) * 2stdv - stdv
      w = Parameter(dtype(uniform((output, input))))
      b = bias ? Parameter(dtype(zeros((output, 1)))) : nothing
      new(w, b)
   end
end

function forward(context, l::Linear, x)
   if ndims(x) > 2
      x = mat(x)
   end
   o = @po l.weight * x
   if l.bias !== nothing
      o = @po o .+ l.bias
   end
   return o
end

type Conv4 <: Module
   cfg
   weight
   bias
   Conv4(height::Int, width::Int, input::Int, output::Int;
      bias=true, dtype=Array{Float32}, o...) = begin
         w = Parameter(dtype(randn(height, width, input, output) *
               sqrt(2.0 / (height * width * output))))
         b = bias ? Parameter(dtype(zeros(1, 1, output, 1))) : nothing
         new(o, w, b)
   end
end

function forward(context, c::Conv4, x)
   #w, b = c.weight, c.bias
   o = @po conv4(c.weight, x; mode=1, c.cfg...)
   if c.bias !== nothing
      o = @po o .+ c.bias
   end
   return o
end


abstract AbstractBatchNorm <: Module

# requires params and moments to be set
function forward(context, bn::AbstractBatchNorm, x)
   mode = bn.mode
   assert(mode in [:train, :test])
   #=assert(:gamma in fieldnames(bn))
   assert(:beta in fieldnames(bn))
   assert(:running_mean in fieldnames(bn))
   assert(:running_var in fieldnames(bn))
   assert(:affine in fieldnames(bn))=#
   tx = typeof(AutoGrad.getval(x))
   if typeof(bn.running_mean) !== tx
      bn.running_mean = tx(bn.running_mean)
   end
   if typeof(bn.running_var) !== tx
      bn.running_var = tx(bn.running_var)
   end
   red_dims = let nd = ndims(x)
      if nd == 4
         (1,2,4)
      elseif nd == 2
         (2,)
      else
         error("AbstractBatchNorm only supports 2 and 4 dimensional inputs")
      end
   end
   x_hat = nothing
   if mode === :test
      x_hat = (x .- bn.running_mean) ./ sqrt(bn.running_var .+ bn.eps)
   else
      # Do the computation
      m = 1
      for i in red_dims
         m *= size(x, i)
      end
      mu = sum(x, red_dims) ./ m
      x_mu = x .- mu
      sigma2 = sumabs2(x_mu, red_dims) ./ m
      x_hat = x_mu ./ sqrt(sigma2 .+ bn.eps)
      # Update the running stats
      bn.running_mean = bn.momentum * bn.running_mean + (1 - bn.momentum) * AutoGrad.getval(mu)
      bn.running_var = bn.momentum * bn.running_var + (1 - bn.momentum) * AutoGrad.getval(sigma2)
   end

   if bn.affine
      @po bn.gamma .* x_hat .+ bn.beta
   else
      x_hat
   end
end

type BatchNorm2 <: AbstractBatchNorm
   gamma
   beta
   running_mean
   running_var
   affine::Bool
   mode::Symbol
   eps
   momentum
   BatchNorm2(output::Int; affine=true, dtype=Array{Float32}, mode=:train, eps=1e-9, momentum=.9) =
      let dims = (output, 1)
         new(
            affine ? Parameter(dtype(ones(dims...))) : nothing,
            affine ? Parameter(dtype(zeros(dims...))) : nothing,
            dtype(zeros(dims...)),
            dtype(ones(dims...)),
            affine, mode, eps, momentum
         )
      end
end

type BatchNorm4 <: AbstractBatchNorm
   gamma
   beta
   running_mean
   running_var
   affine::Bool
   mode::Symbol
   eps
   momentum
   BatchNorm4(output::Int; affine=true, dtype=Array{Float32}, mode=:train, eps=1e-9, momentum=.9) =
      let dims = (1, 1, output, 1)
         new(
            affine ? Parameter(dtype(ones(dims...))) : nothing,
            affine ? Parameter(dtype(zeros(dims...))) : nothing,
            dtype(zeros(dims...)),
            dtype(ones(dims...)),
            affine, mode, eps, momentum
         )
      end
end


type Dropout <: Module
   keep_prob::AbstractFloat
   mode::Symbol
   noise
   Dropout(keep_prob::AbstractFloat; mode=:train) = new(keep_prob, mode, nothing)
end

function forward(context, m::Dropout, x)
   assert(m.mode in [:train, :test])
   tx = typeof(AutoGrad.getval(x))
   sizex = size(x)
   etx = eltype(AutoGrad.getval(x))
   p = etx(m.keep_prob)
   if p == 0
      tx(zeros(sizex))
   elseif p == 1
      x
   elseif m.mode === :train
      if m.noise == nothing || size(m.noise) != size(x)
         m.noise = tx(rand(etx, sizex) .<= p)
      else
         copy!(m.noise, rand(sizex) .<= p)
      end
      x .* m.noise
   else # test mode
      p .* x
   end
end

#---- Containers ----
abstract Container <: Module

function layers(c::Container)
   error("layers is not implemented for type ", typeof(c))
end


# The Sequential container
#=For instance
   cnn = Sequential(
      Conv4(3, 3, 3, 5),
      relu,
      pool,
      Conv4(3, 3, 5, 8),
      relu,
      x->pool(x;mode=2, window=size(x)[1:2]),
      Linear(...)
   )
=#
type Sequential <: Container
   layers::Array{Any, 1}
   Sequential(layers...) = new([l for l in layers])
end

function layers(s::Sequential)
   return s.layers
end

function forward(context, s::Sequential, x)
   o = x
   for l in s.layers
      if isa(l, Module)
         o = @mc l(o)
      elseif isa(l, Function)
         o = l(o)
      else
         error("Container malformed: only modules and functions are allowed")
      end
   end
   return o
end

function add!(s::Sequential, m...)
   push!(s.layers, m...)
   return s
end


#= rescon = Table(+,
   Sequential(Conv4(...), BN(...), relu, Conv4(...), BN(...), relu),
   Shortcut(...)
)=#
type Table <: Container
   op::Function
   layers::Array{Any, 1}
   Table(op::Function, layers...) = new(op, [l for l in layers])
end

function layers(table::Table)
   return table.layers
end

function forward(context, t::Table, x)
   outputs = []
   for l in t.layers
      if isa(l, Container)
         push!(outputs, @mc l(x))#forward(context, l, x))
      elseif isa(l, Function)
         push!(outputs, l(x))
      else
         error("Container malformed: only functions and modules are allowed")
      end
   end
   t.op(outputs...)
end


#---Recurrent modules---

abstract AbstractRNN <: Module

function reset_state!(rnn::AbstractRNN)
   rnn.history = []
end

function hidden_state(rnn::AbstractRNN; unbox=false)
   unbox ? AutoGrad.getval(rnn.history[end]) : rnn.history[end]
end

# The GRU layer
type GRU <: AbstractRNN
   history::Array{Any, 1}
   h2z::Linear
   i2z::Linear
   h2r::Linear
   i2r::Linear
   h2n::Linear
   i2n::Linear
   input_size::Int
   hidden_size::Int
   function GRU(input_size::Int, hidden_size::Int; bias=true)
      h2z = Linear(hidden_size, hidden_size; bias=bias)
      i2z = Linear(input_size, hidden_size; bias=bias)
      h2r = Linear(hidden_size, hidden_size; bias=bias)
      i2r = Linear(input_size, hidden_size; bias=bias)
      h2n = Linear(hidden_size, hidden_size; bias=bias)
      i2n = Linear(input_size, hidden_size; bias=bias)
      new([], h2z, i2z, h2r, i2r, h2n, i2n, input_size, hidden_size)
   end
end

# Single time step of a GRU
# TODO: checkout masking
function forward(context, m::GRU, x, mask=nothing)
   if length(m.history) == 0
      tx = typeof(AutoGrad.getval(x))
      h0 = zeros(m.hidden_size, size(x)[end])
      push!(m.history, tx(h0))
   end
   ht_1 = hidden_state(m)
   if mask != nothing
      if ndims(mask) == 1
         mask = reshape(mask, (1, size(mask)[1]))
      end
      assert(size(mask) == (1, size(x)[2]))
      # Remove error when sure about masking
      # error("Masking is not supported yet")
      #= The idea is remove the effect of padded values
      from the output and freeze their state
      but I'm not sure. =#
      z = sigm(@mc(m.i2z(x)) .+ @mc(m.h2z(ht_1))) .* mask
      r = sigm(@mc(m.i2r(x)) .+ @mc(m.h2r(ht_1))) .* mask
      n = tanh(@mc(m.i2n(x)) .+ r .* @mc(m.h2n(ht_1))) .* mask
      ht = (1 .- z) .* n .+ z .* ht_1 .+ (1 .- mask) .* ht_1
   else
      z = sigm(@mc(m.i2z(x)) .+ @mc(m.h2z(ht_1)))
      r = sigm(@mc(m.i2r(x)) .+ @mc(m.h2r(ht_1)))
      n = tanh(@mc(m.i2n(x)) .+ r .* @mc(m.h2n(ht_1)))
      ht = (1 .- z) .* n .+ z .* ht_1
   end
   push!(m.history, ht)[end]
end

# To be continued
#=type LSTM <: AbstractRNN
   history
end
...
=#
