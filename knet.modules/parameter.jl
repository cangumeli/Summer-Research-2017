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

get_grad(v::Parameter, grads) = let g = grads[v.index]
   typeof(g) == Void ? nothing : g
end

get_value(context, x) = x

function expand_parameter_op(expr)
   if typeof(expr) == Symbol
      #=quote
         get_value($(esc(:context)), expr)
      end=#
      #=quote
         let c = context
            get_value(c, $(expr))
         end
      end=#
      esc(:(get_value(context, $expr)))
      #esc(Expr(:call, get_value, :context), expr))
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
