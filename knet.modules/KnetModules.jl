module KnetModules
   using Knet, AutoGrad
   export
      # Parameter
      ParamContext,
      ParamIndex,
      Parameter,
      get_default_parameter_context,
      get_active_parameter_context,
      set_active_parameter_context!,
      register_parameter!,
      get_value,
      set_value!,
      get_grad,
      @po,
      gpu!,
      cpu!,
      # Module
      KnetModule,
      @mc,
      @run,
      parameters,
      submodules,
      set_mode!,
      training!,
      testing!,
      forward,
      # Simple layers
      Linear,
      Dropout,
      # Conv layers
      Conv4,
      # Normalization layers
      AbstractBatchNorm, # For type checking
      BatchNorm2,
      BatchNorm4,
      # Containers
      layers,
      Sequential,
      Table,
      add!,
      # RNNs
      AbstractRNN,
      reset_state!,
      hidden_state,
      GRU

   include("parameter.jl")
   include("knet_module.jl")
   include("basic.jl")
   include("conv.jl")
   include("norm.jl")
   include("container.jl")
   include("rnn.jl")
end
