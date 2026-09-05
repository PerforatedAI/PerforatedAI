from perforatedai import globals_perforatedai as GPA


def render_original_module(module, indent=0):
    """Render a module using normal PyTorch-style formatting.

    This prints the module structure similarly to ``print(model)`` while also
    including any directly-owned parameters that were tagged with ``pai_mode``
    during preview setup.

    Parameters
    ----------
    module : nn.Module
        The module to render.
    indent : int, optional
        Current indentation depth for recursive formatting.

    Returns
    -------
    str
        Formatted string representation of the module.
    """
    children = list(module.named_children())
    extra_repr = module.extra_repr()
    direct_params = [
        (name, param)
        for name, param in module.named_parameters(recurse=False)
        if hasattr(param, "pai_mode")
    ]

    # Leaf modules can be returned directly unless they own tagged parameters.
    if not children and not direct_params:
        if extra_repr:
            return f"{module._get_name()}({extra_repr})"
        return f"{module._get_name()}()"

    lines = [f"{module._get_name()}("]
    if extra_repr:
        lines.append("  " * (indent + 1) + extra_repr)
    # Show preview-only parameter tags inline with the owning module.
    for param_name, param in direct_params:
        prefix = "  " * (indent + 1)
        lines.append(f"{prefix}({param_name}): Parameter(pai_mode={param.pai_mode})")
    for child_name, child in children:
        child_repr = render_preview_model(child, indent + 1).splitlines()
        prefix = "  " * (indent + 1)
        lines.append(f"{prefix}({child_name}): {child_repr[0]}")
        for line in child_repr[1:]:
            lines.append(f"{prefix}{line}")
    lines.append("  " * indent + ")")
    return "\n".join(lines)


def render_preview_model(module, indent=0):
    """Render a module using preview wrapper names when tagged.

    This checks ``pai_mode`` on the module and, when present, formats the module
    as if it had already been wrapped by the real perforation process.

    Parameters
    ----------
    module : nn.Module
        The module to render.
    indent : int, optional
        Current indentation depth for recursive formatting.

    Returns
    -------
    str
        Formatted preview string for the module.
    """
    pai_mode = getattr(module, "pai_mode", None)
    if pai_mode == "perforated":
        wrapper_name = "PAINeuronModule"
    elif pai_mode == "tracked":
        wrapper_name = "PAITrackedModule"
    elif pai_mode == "replaced":
        wrapper_name = "ReplacedModule"
    else:
        return render_original_module(module, indent)

    # Preserve the original module body and only swap the outer printed wrapper.
    original_lines = render_original_module(module, indent + 1).splitlines()
    if len(original_lines) == 1:
        return f"{wrapper_name}({original_lines[0]})"

    lines = [f"{wrapper_name}("]
    lines.append(f"  (main_module): {original_lines[0]}")
    for line in original_lines[1:]:
        lines.append(f"  {line}")
    lines.append(")")
    return "\n".join(lines)


def clear_pai_preview_tags(model):
    """Remove temporary preview tags from a model.

    The config preview path temporarily adds ``pai_mode`` attributes to modules
    and parameters. This function removes those tags so the original model is
    left unchanged after preview.

    Parameters
    ----------
    model : nn.Module
        The model whose preview tags should be cleared.

    Returns
    -------
    None
        This function does not return a value.
    """
    for module in model.modules():
        if hasattr(module, "pai_mode"):
            del module.pai_mode
    for param in model.parameters():
        if hasattr(param, "pai_mode"):
            del param.pai_mode


def set_perforation_targets(model):
    """Show a preview of how a model will be processed and wait for confirmation.

    This runs the lightweight preview tagging path, prints the model using
    preview wrapper names, waits for the user to confirm, then removes the
    temporary preview tags.

    Parameters
    ----------
    model : nn.Module
        The model to preview before perforation begins.

    Returns
    -------
    None
        This function does not return a value.
    """
    from perforatedai import modules_perforatedai as PA
    from perforatedai import utils_perforatedai as UPA

    # Handle the root module directly when it is itself a configured target.
    if (type(model) in GPA.pc.get_modules_to_perforate()) or (
        type(model).__name__ in GPA.pc.get_module_names_to_perforate()
    ):
        model.pai_mode = "perforated"
    elif type(model) in GPA.pc.get_modules_to_replace():
        model.pai_mode = "replaced"
    elif (type(model) in GPA.pc.get_modules_to_track()) or (
        type(model).__name__ in GPA.pc.get_module_names_to_track()
    ):
        model.pai_mode = "tracked"
    else:
        # Reuse the existing traversal logic to tag all matching submodules.
        UPA.convert_module(
            model,
            0,
            "",
            [],
            [],
            PA.PAINeuronModule,
            PA.TrackedNeuronModule,
            config_setup=True,
        )

    print(render_preview_model(model))
    input("Press Enter to confirm perforation targets...")
    clear_pai_preview_tags(model)
    GPA.pc.set_configuration_confirmed(True)
