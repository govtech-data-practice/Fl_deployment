"""SCAFFOLD: trainable-param to state_dict index mapping."""


def build_trainable_to_state_map(model):
    """Map each trainable parameter index to its state_dict index.

    model.parameters() yields only requires_grad tensors.
    model.state_dict() includes buffers (BatchNorm running_mean, etc.).
    This mapping lets SCAFFOLD gradient corrections target the right tensors.
    """
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    mapping = {}
    t = 0
    for sd_i, key in enumerate(model.state_dict().keys()):
        if key in trainable:
            mapping[t] = sd_i
            t += 1
    return mapping
