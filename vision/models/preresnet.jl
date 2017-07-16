include("resnet.jl")

#----- Preactivation Models -----
type ShortcutPre <: Module
   op
   function ShortcutPre(input, output, stride=1)
      use_conv = input !== output
      if use_conv
         new(Conv4(1, 1, input, output; bias=false, stride=stride))
      else
         new(nothing)
      end
   end
end

function forward(context, shortcut::ShortcutPre, x)
   if shortcut.op !== nothing
      @mc shortcut.op(x)
   else
      x
   end
end

abstract BlockPre <: Block

function forward(context, block::BlockPre, x)
   o1 = @mc block.layers(x)
   o0 = @mc block.shortcut(x)
   o1 + o0
end

type BasicBlockPre <: BlockPre
   layers::Module
   shortcut::ShortcutPre
   function BasicBlockPre(input::Int, output::Int, stride::Int=1; first=false, last=false)
      layers = first ? Sequential() : Sequential(BatchNorm4(input), relu)
      add!(layers,
         Conv4(3, 3, input, output; padding=1, stride=stride, bias=false),
         BatchNorm4(output),
         relu,
         Conv4(3, 3, output, output; padding=1, bias=false)
      )
      if last
         add!(layers,
            BatchNorm4(output),
            relu
         )
      end
      new(layers, ShortcutPre(input, output, stride))
   end
end

type BottleneckPre <: BlockPre
   layers::Module
   shortcut::ShortcutPre
   function BottleneckPre(input::Int, output::Int, stride::Int=1; first=false, last=false)
      n = Int(output/4)
      layers = first ? Sequential() : Sequential(BatchNorm4(input), relu)
      add!(layers,
         Conv4(1, 1, input, n; bias=false),
         BatchNorm4(n),
         relu,
         Conv4(3, 3, n, n; bias=false, stride=stride, padding=1),
         BatchNorm4(n),
         relu,
         Conv4(1, 1, n, output; bias=false)
      )
      if last
         add!(layers,
            BatchNorm4(output),
            relu
         )
      end
      shortcut = ShortcutPre(input, output, stride)
      new(layers, shortcut)
   end
end

type PreResNetCifar <: ResNetBase
   inp::Module
   blocks::Array{Module, 1}
   out::Module
   function PreResNetCifar(
         depth::Int,
         block::Type,
         channels::Tuple{Int, Int, Int, Int},
         nclasses=10)
      inp = Conv4(3, 3, 3, channels[1]; padding=1)
      n = Int((depth-2)/((block == BottleneckPre) ? 9 : 6))
      blocks = []
      stride = 1
      for i = 2:4
         push!(blocks, block(channels[i-1], channels[i], stride))
         for j = 2:n
            push!(blocks, block(channels[i], channels[i]; last=j==n))
         end
         stride = 2
      end
      out = Sequential(
         x->pool(x; mode=2, window=size(x)[1:2]),
         Linear(channels[end], nclasses)
      )
      new(inp, blocks, out)
   end
end

PreResNetCifar(depth::Int, nclasses=10) =
   PreResNetCifar(depth, BasicBlockPre, (16, 16, 32, 64), nclasses)
PreResNetCifarDeep(depth::Int, nclasses=10) =
   PreResNetCifar(depth, BottleneckPre, (16, 64, 128, 256), nclasses)

resnet164(nclasses=10) = PreResNetCifarDeep(164, nclasses)
resnet1001(nclasses=10) = PreResNetCifarDeep(1001, nclasses)

type PreResNet <: ResNetBase
   config::Config
   inp::Module
   blocks::Array{Module, 1}
   out::Module
   function PreResNet(config::Config; nclasses=1000)
      inp = Sequential(
         Conv4(7, 7, 3, 64; padding=3, stride=2),
         BatchNorm4(64),
         relu,
         x->pool(x; window=3, stride=2)
      )
      blocks = []
      input = 64
      stride = 1
      first = true
      for (r, c) in zip(config.repeat, config.channels)
         output = c
         push!(blocks, config.block(input, output, stride; first=first))
         for i = 2:r
            push!(blocks, config.block(output, output; last=i==r))
         end
         input = output
         stride = 2
         first = false
      end
      out = Sequential(
         x->pool(x; mode=2, window=size(x)[1:2]),
         Linear(config.channels[end], nclasses)
      )
      new(config, inp, blocks, out)
   end
end

# TODO: support pretrained model loading
preresnet18() = PreResNet(Config(BasicBlockPre, [2, 2, 2, 2], basic_channels))
preresnet34() = PreResNet(Config(BasicBlockPre, [3, 4, 6, 3], basic_channels))
preresnet50() = PreResNet(Config(BottleneckPre, [3, 4, 6, 3], bottleneck_channels))
preresnet101() = PreResNet(Config(BottleneckPre, [3, 4, 23, 3], bottleneck_channels))
preresnet152() = PreResNet(Config(BottleneckPre, [3, 8, 36, 3], bottleneck_channels))
resnet200() = PreResNet(Config(BottleneckPre, [3, 24, 36, 3], bottleneck_channels))
