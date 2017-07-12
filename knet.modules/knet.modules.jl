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
      return Expr(:call, :forward, :context, expr.args[1], expr.args[2])
   end
   return expr
end

macro run(m, args...)
   Expr(:call, forward, Expr(:call, get_active_parameter_context), m, args...)
end

function parameters(m::Module)
   function _parameters(m, paramlist)
      if isa(m, Module)
         for fn in fieldnames(m)
            _parameters(getfield(m, fn), paramlist)
         end
      elseif isa(m, Array) || isa(m, Tuple)
         for l in m
            _parameters(l, paramlist)
         end
      elseif isa(m, Associative)
         for v in values(m)
            _parameters(l, v)
         end
      elseif isa(m, Parameter)
         push!(paramlist, m)
      else
         return
      end
   end
   res = []
   _parameters(m, res)
   return res
end


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
   o = @po conv4(c.weight, x; c.cfg...)
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
      #foldr(*, map(d->size(x, d), 1:red_dims))
      mu = sum(x, red_dims) ./ m
      x_mu = x .- mu
      sigma2 = sumabs2(x_mu, red_dims) ./ m
      x_hat = x_mu ./ sqrt(sigma2 .+ bn.eps)
      # Update the running stats
      # println(size(bn.running_mean), size(AutoGrad.getval(mu)))
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

function set_mode!(s::Module, mode)
   fnames = fieldnames(s)
   if :mode in fnames
      s.mode = mode
   end
   for f in fnames
      fv = getfield(s, f)
      if isa(fv, Module)
         set_mode!(fv, mode)
      end
   end
end


abstract Container <: Module

function layers(c::Container)
   error("layers is not implemented for type ", typeof(c))
end

function set_mode!(s, mode)
   for l in layers(s)
      set_mode!(l, mode)
   end
end

# The sequental container
#=For instance
   cnn = Sequental(
      Conv4(3, 3, 3, 5),
      relu,
      pool,
      Conv4(3, 3, 5, 8),
      relu,
      x->pool(x;mode=2, window=size(x)[1:2]),
      Linear(...)
   )
=#
type Sequental <: Container
   layers::Array{Any, 1}
   Sequental(layers...) = new([l for l in layers])
end

function layers(s::Sequental)
   return s.layers
end

function forward(context, s::Sequental, x)
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

function add!(s::Sequental, m...)
   push!(s.layers, m...)
   return s
end


#= rescon = Table(+,
   Sequental(Conv4(...), BN(...), relu, Conv4(...), BN(...), relu),
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

function hidden_state(rnn::AbstractRNN)
   AutoGrad.getval(rnn.history[end])
end

type LSTM <: AbstractRNN
   history
end
