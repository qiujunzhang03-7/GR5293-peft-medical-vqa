"""Helper: enable input grads on the input embeddings for gradient
checkpointing + LoRA to work. Must be called *after* PEFT wraps the model."""

def enable_input_grads(model):
    """Make sure inputs to the embedding layer require grad.

    With gradient checkpointing + frozen base model + LoRA, the activation
    chain has no leaf with requires_grad=True, so backward fails. Hooking
    the input embedding to set requires_grad on its output fixes this.
    """
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    embed = model.get_input_embeddings()
    embed.register_forward_hook(make_inputs_require_grad)
