from aerocast.models.convlstm import ConvLSTMCell, StackedConvLSTM


def build_model(model_cfg, in_channels):
    name = model_cfg["name"]
    if name == "StackedConvLSTM":
        return StackedConvLSTM(
            input_channels=in_channels,
            hidden_dims=list(model_cfg["hidden_dims"]),
            kernel_size=model_cfg["kernel_size"],
        )
    raise ValueError(f"Unknown model '{name}'. Supported: StackedConvLSTM")


__all__ = ["ConvLSTMCell", "StackedConvLSTM", "build_model"]
