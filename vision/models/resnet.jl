using Knet
include("../../knet.modules/knet.modules.jl")

# TODO: support concat etc.
type Shortcut <: Module
   op
   function Shortcut(input, output, stride=1)
      use_conv = input !== output
      if use_conv
         new(Sequential(
               Conv4(1, 1, input, output; bias=false, stride=stride),
               BatchNorm4(output)))
         else
            new(nothing)
         end
      end
end

function forward(context, s::Shortcut, x)
   if s.op !== nothing
      x = @mc s.op(x)
   end
   return x
end

abstract Block <: Module

function forward(context, b::Block, x)
   o1 = @mc b.layers(x)
   o0 = @mc b.shortcut(x)
   relu(o0 .+ o1)
end

type BasicBlock <: Block
   layers::Module
   shortcut::Shortcut
   function BasicBlock(input::Int, output::Int, stride::Int=1)
      layers = Sequential(
         Conv4(3, 3, input, output; padding=1, stride=stride, bias=false),
         BatchNorm4(output),
         relu,
         Conv4(3, 3, output, output; padding=1, bias=false),
         BatchNorm4(output)
      )
      shortcut = Shortcut(input, output, stride)
      new(layers, shortcut)
   end
end

type Bottleneck <: Block
   layers::Module
   shortcut::Shortcut
   function Bottleneck(input::Int, output::Int, stride::Int=1)
      n = Int(output/4)
      layers = Sequential(
         Conv4(1, 1, input, n; bias=false),
         BatchNorm4(n),
         relu,
         Conv4(3, 3, n, n; stride=stride, bias=false, padding=1),
         BatchNorm4(n),
         relu,
         Conv4(1, 1, n, output; bias=false),
         BatchNorm4(output),
      )
      new(layers, Shortcut(input, output, stride))
   end
end

abstract ResNetBase <: Module

function forward(context, model::ResNetBase, x)
   # First conv
   o = @mc model.inp(x)
   for block in model.blocks
      # println(size(o))
      # Use array for easy surgery and checking intermadiate outputs
      o = @mc block(o)
   end
   if model.out != nothing
      o = @mc model.out(o)
   end
   return o
   # The output pooling and classification
end

type ResNetCifar <: ResNetBase
   depth::Int
   inp::Module
   blocks::Array{Block, 1}
   out::Module
   function ResNetCifar(depth::Int; nclasses=10)
      n = Int((depth-2)/6)
      inp = Sequential(
         Conv4(3, 3, 3, 16; padding=1, bias=false),
         BatchNorm4(16),
         relu
      )
      blocks = []
      input, output = 16, 16
      for i = 1:3n
         stride = Int(i>1 && i%n ==1) + 1
         output *= stride
         push!(blocks, BasicBlock(input, output, stride))
         input = output
      end
      out = Sequential(
         x->pool(x; mode=2, window=size(x)[1:2]),
         Linear(output, nclasses)
      )
      new(depth, inp, blocks, out)
   end
end

type ResNetConfig
   block::Type
   repeat::Array{Int, 1}
   channels::Array{Int, 1}
   conv_features::Int # For re-using ImageNet models, 0 for full outputs
end

ResNetConfig(block::Type, repeat::Array{Int, 1}, channels::Array{Int, 1}) =
   ResNetConfig(block, repeat, channels, 0)

type ResNet <: ResNetBase
   config::ResNetConfig
   inp::Module
   blocks::Array{Module, 1}
   out # nullable
   function ResNet(config::ResNetConfig; nclasses=1000)
      inp = Sequential(
         Conv4(7, 7, 3, 64; padding=3, stride=2),
         BatchNorm4(64),
         relu,
         x->pool(x; window=3, stride=2)
      )
      blocks = []
      input, output = 64, config.channels[1]
      stride = 1
      for (i, (r, c)) in enumerate(zip(config.repeat, config.channels))
         output = c
         push!(blocks, config.block(input, output, stride))
         for j = 2:r
            push!(blocks, config.block(output, output))
         end
         input = output
         stride = 2
         if (i + 1) == config.conv_features
            return new(config, inp, blocks, nothing)
         end
      end

      out = Sequential(
         x->pool(x; mode=2, window=size(x)[1:2]),
         Linear(input, nclasses)
      )
      new(config, inp, blocks, out)
   end
end

function create_resnet_loader()
   _loader_included = false
   function _pt(model, depth, opt)
      if ~opt
         return model
      end
      if ~_loader_included
         try # Don't crash system if weight loaders have error
            # which is likely since it has a PyCall dependency
            include("_resnet_weight_loader.jl")
            _loader_included = true
         catch e
            error("Pretrained weight loader cannot be included:\nInfo:\n", e)
         end
      end
      dirname = download_torch_weights(depth)
      load_torch_weights!(model, dirname)
      return model
   end
   return _pt
end

load_resnet = create_resnet_loader()
const basic_channels = [64, 128, 256, 512]
const bottleneck_channels = 4basic_channels

# --- ImageNet Models ---
function resnet18(;cfeatures=0, pretrained=false)
   model = ResNet(ResNetConfig(BasicBlock, [2, 2, 2, 2], basic_channels, cfeatures))
   load_resnet(model, 18, pretrained)
end

function resnet34(;cfeatures=0, pretrained=false)
   model = ResNet(ResNetConfig(BasicBlock, [3, 4, 6, 3], basic_channels, cfeatures))
   load_resnet(model, 34, pretrained)
end

function resnet50(;cfeatures=0, pretrained=false)
   model = ResNet(ResNetConfig(Bottleneck, [3, 4, 6, 3], bottleneck_channels, cfeatures))
   load_resnet(model, 50, pretrained)
end

function resnet101(;cfeatures=0, pretrained=false)
   model = ResNet(ResNetConfig(Bottleneck, [3, 4, 23, 3], bottleneck_channels, cfeatures))
   load_resnet(model, 101, pretrained)
end

function resnet152(;cfeatures=0, pretrained=false)
   model = ResNet(ResNetConfig(Bottleneck, [3, 8, 36, 3], bottleneck_channels, cfeatures))
   load_resnet(model, 152, pretrained)
end
