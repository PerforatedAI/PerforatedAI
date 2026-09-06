import sys
import termios
import tty
import shutil
import json
import os
import select
import re

from perforatedai import globals_perforatedai as GPA


def dedupe_list(values):
    """Return a list with duplicates removed while preserving order.

    Parameters
    ----------
    values : list
        Input list that may contain duplicate values.

    Returns
    -------
    list
        De-duplicated list with original order preserved.
    """
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def normalize_selection_conflicts():
    """Normalize id and name selection lists and remove direct conflicts.

    Rules enforced:
    - No item can exist in both id lists.
    - No item can exist in both name lists.
    - Id conflicts and name conflicts are resolved by removing from tracked
      when also present in perforated.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return a value.
    """
    ids_perforate = dedupe_list(GPA.pc.get_module_ids_to_perforate())
    ids_track = dedupe_list(GPA.pc.get_module_ids_to_track())
    names_perforate = dedupe_list(GPA.pc.get_module_names_to_perforate())
    names_track = dedupe_list(GPA.pc.get_module_names_to_track())

    overlap_ids = set(ids_perforate) & set(ids_track)
    overlap_names = set(names_perforate) & set(names_track)

    if overlap_ids:
        ids_track = [value for value in ids_track if value not in overlap_ids]
    if overlap_names:
        names_track = [value for value in names_track if value not in overlap_names]

    GPA.pc.set_module_ids_to_perforate(ids_perforate)
    GPA.pc.set_module_ids_to_track(ids_track)
    GPA.pc.set_module_names_to_perforate(names_perforate)
    GPA.pc.set_module_names_to_track(names_track)


def get_module_entries(model):
    """Collect modules for interactive navigation.

    Parameters
    ----------
    model : nn.Module
        Root model to inspect.

    Returns
    -------
    list
        List of module entry dictionaries containing module id, module type,
        object pointer, and tree depth.
    """
    entries = []
    for name, module in model.named_modules(remove_duplicate=False):
        if name == "":
            continue
        module_id = "." + name
        depth = name.count(".")
        entries.append(
            {
                "id": module_id,
                "name": name,
                "type_name": type(module).__name__,
                "module": module,
                "depth": depth,
                "direct_param_count": sum(
                    param.numel()
                    for _param_name, param in module.named_parameters(recurse=False)
                ),
                "tree_param_count": sum(
                    param.numel()
                    for _param_name, param in module.named_parameters(recurse=True)
                ),
            }
        )
    return entries


def get_id_mode(module_id):
    """Get explicit id-based mode for a module id.

    Parameters
    ----------
    module_id : str
        Module id in dot notation (e.g. ".layer1.0.conv1").

    Returns
    -------
    str or None
        "perforated", "tracked", or None when not set by id.
    """
    if module_id in GPA.pc.get_module_ids_to_perforate():
        return "perforated"
    if module_id in GPA.pc.get_module_ids_to_track():
        return "tracked"
    return None


def get_name_mode(module_type_name):
    """Get type-name-based mode for a module type.

    Parameters
    ----------
    module_type_name : str
        Module class name (e.g. "Linear").

    Returns
    -------
    str or None
        "perforated", "tracked", or None when not set by name.
    """
    if module_type_name in GPA.pc.get_module_names_to_perforate():
        return "perforated"
    if module_type_name in GPA.pc.get_module_names_to_track():
        return "tracked"
    return None


def get_effective_mode(entry):
    """Compute the effective mode for one module entry.

    Precedence:
    1) Explicit id mode
    2) Type-name mode
    3) Replacement mode
    4) None

    Parameters
    ----------
    entry : dict
        Module entry dictionary from get_module_entries().

    Returns
    -------
    str or None
        "perforated", "tracked", "replaced", or None.
    """
    id_mode = get_id_mode(entry["id"])
    if id_mode is not None:
        return id_mode

    name_mode = get_name_mode(entry["type_name"])
    if name_mode is not None:
        return name_mode

    if type(entry["module"]) in GPA.pc.get_modules_to_replace():
        return "replaced"

    return None


def get_parent_module_id(module_id):
    """Return the direct parent module id for a dot-id module path.

    Parameters
    ----------
    module_id : str
        Module id in dot notation (e.g. ".layer1.0.conv1").

    Returns
    -------
    str or None
        Parent module id or None when there is no parent.
    """
    parts = module_id.split(".")
    if len(parts) <= 2:
        return None
    return "." + ".".join(parts[1:-1])


def build_recursive_modes(entries):
    """Build recursive inherited modes for all entries.

    Recursive rule:
    - if parent has a recursive mode, inherit it (descendant override ignored)
    - otherwise explicit id mode or explicit name mode sets this node's mode

    Parameters
    ----------
    entries : list
        Module entries from get_module_entries().

    Returns
    -------
    dict
        Map of module id to recursive mode ("perforated", "tracked", or None).
    """
    recursive_modes = {}
    for entry in entries:
        explicit_mode = get_id_mode(entry["id"])
        if explicit_mode is None:
            explicit_mode = get_name_mode(entry["type_name"])

        parent_id = get_parent_module_id(entry["id"])
        parent_mode = None
        if parent_id is not None:
            parent_mode = recursive_modes.get(parent_id)

        if parent_mode is not None:
            recursive_modes[entry["id"]] = parent_mode
        elif explicit_mode is not None:
            recursive_modes[entry["id"]] = explicit_mode
        else:
            recursive_modes[entry["id"]] = None

    return recursive_modes


def get_color_hex_for_mode(mode):
    """Map preview mode to color hex value.

    Parameters
    ----------
    mode : str or None
        Mode name.

    Returns
    -------
    str
        Six-character RGB hex string.
    """
    if mode == "perforated":
        return "00A5A5"
    if mode == "tracked":
        return "DEECED"
    return "000000"


def format_human_count(value):
    """Format integer counts into compact human-readable units.

    Parameters
    ----------
    value : int
        Non-negative integer to format.

    Returns
    -------
    str
        Formatted value using raw numbers, K, M, or B suffix.
    """
    if value < 1000:
        return str(value)

    if value < 1000000:
        scaled = value / 1000.0
        text = f"{scaled:.1f}"
        if text.endswith(".0"):
            text = text[:-2]
        return f"{text}K"

    if value < 1000000000:
        scaled = value / 1000000.0
        text = f"{scaled:.1f}"
        if text.endswith(".0"):
            text = text[:-2]
        return f"{text}M"

    scaled = value / 1000000000.0
    text = f"{scaled:.1f}"
    if text.endswith(".0"):
        text = text[:-2]
    return f"{text}B"


def get_resolved_mode(entry, recursive_modes):
    """Resolve final mode for an entry including recursive inheritance.

    Parameters
    ----------
    entry : dict
        Module entry dictionary from get_module_entries().
    recursive_modes : dict
        Map from module id to recursive inherited mode.

    Returns
    -------
    str or None
        "perforated", "tracked", "replaced", or None.
    """
    recursive_mode = recursive_modes.get(entry["id"])
    if recursive_mode is not None:
        return recursive_mode

    if type(entry["module"]) in GPA.pc.get_modules_to_replace():
        return "replaced"

    return None


def build_target_summary_line(entries, recursive_modes):
    """Build the sticky summary line for target counts and parameters.

    Parameters
    ----------
    entries : list
        Module entries from get_module_entries().
    recursive_modes : dict
        Map from module id to recursive inherited mode.

    Returns
    -------
    str
        Summary line shown in the fixed header.
    """
    perforated_targets = 0
    tracked_targets = 0
    unset_targets = 0
    total_parameters_added_per_cycle = 0
    total_model_params = 0
    seen_parameter_ids = set()
    seen_perforated_parameter_ids = set()

    for entry in entries:
        for _param_name, parameter in entry["module"].named_parameters(recurse=False):
            parameter_id = id(parameter)
            if parameter_id in seen_parameter_ids:
                continue
            seen_parameter_ids.add(parameter_id)
            total_model_params += parameter.numel()

        resolved_mode = get_resolved_mode(entry, recursive_modes)
        param_count = entry["direct_param_count"]

        if resolved_mode == "perforated":
            perforated_targets += 1
            for _param_name, parameter in entry["module"].named_parameters(recurse=False):
                parameter_id = id(parameter)
                if parameter_id in seen_perforated_parameter_ids:
                    continue
                seen_perforated_parameter_ids.add(parameter_id)
                total_parameters_added_per_cycle += parameter.numel()
        elif resolved_mode == "tracked":
            tracked_targets += 1
        elif param_count > 0:
            unset_targets += 1

    return (
        "Perforated targets - "
        f"{perforated_targets}, "
        "Tracked targets - "
        f"{tracked_targets}, "
        "unset targets which need to be set - "
        f"{unset_targets}, "
        "total parameters added per cycle - "
        f"{format_human_count(total_parameters_added_per_cycle)}, "
        "total model params - "
        f"{format_human_count(total_model_params)}"
    )


def module_has_direct_parameters(module):
    """Check whether a module directly owns any parameters.

    Parameters
    ----------
    module : nn.Module
        Module to inspect.

    Returns
    -------
    bool
        True when the module has at least one direct parameter.
    """
    for _name, _param in module.named_parameters(recurse=False):
        return True
    return False


def make_color_square(hex_color):
    """Create one colored square character using ANSI truecolor.

    Parameters
    ----------
    hex_color : str
        RGB hex color in RRGGBB format.

    Returns
    -------
    str
        Colored square character.
    """
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"\x1b[38;2;{r};{g};{b}m█\x1b[0m"


def make_inverted_text(text):
    """Render text with inverted foreground/background colors."""
    return f"\x1b[7m{text}\x1b[0m"


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text):
    """Remove ANSI color/style escape codes for width calculation."""
    return ANSI_ESCAPE_RE.sub("", text)


def get_visual_line_count(lines, terminal_columns):
    """Count rendered terminal rows, including wrapped long lines."""
    if terminal_columns < 1:
        return len(lines)

    total = 0
    for line in lines:
        plain = strip_ansi(line)
        if len(plain) == 0:
            total += 1
            continue
        total += (len(plain) + terminal_columns - 1) // terminal_columns
    return total


def render_text_with_cursor(text, cursor_index):
    """Render text with CLI-style inverted-character cursor."""
    if cursor_index < 0:
        cursor_index = 0
    if cursor_index > len(text):
        cursor_index = len(text)

    if cursor_index == len(text):
        return text + make_inverted_text(" ")

    current_char = text[cursor_index]
    return (
        text[:cursor_index]
        + make_inverted_text(current_char)
        + text[cursor_index + 1 :]
    )


def is_up_key(key):
    """Return True when key token represents Up arrow."""
    if not key.startswith("\x1b"):
        return False
    return key in ("\x1b[A", "\x1bOA") or (
        key.startswith("\x1b[") and key.endswith("A")
    )


def is_down_key(key):
    """Return True when key token represents Down arrow."""
    if not key.startswith("\x1b"):
        return False
    return key in ("\x1b[B", "\x1bOB") or (
        key.startswith("\x1b[") and key.endswith("B")
    )


def is_left_key(key):
    """Return True when key token represents Left arrow."""
    if not key.startswith("\x1b"):
        return False
    return key in ("\x1b[D", "\x1bOD") or (
        key.startswith("\x1b[") and key.endswith("D")
    )


def is_right_key(key):
    """Return True when key token represents Right arrow."""
    if not key.startswith("\x1b"):
        return False
    return key in ("\x1b[C", "\x1bOC") or (
        key.startswith("\x1b[") and key.endswith("C")
    )


def is_page_up_key(key):
    """Return True when key token represents Page Up."""
    return key == "\x1b[5~" or (key.startswith("\x1b[") and "[5" in key and key.endswith("~"))


def is_page_down_key(key):
    """Return True when key token represents Page Down."""
    return key == "\x1b[6~" or (key.startswith("\x1b[") and "[6" in key and key.endswith("~"))


def is_delete_key(key):
    """Return True when key token represents Delete."""
    return key == "\x1b[3~" or (key.startswith("\x1b[") and "[3" in key and key.endswith("~"))


def make_mode_prefix(entry, recursive_modes):
    """Build the color-square prefix for one module line.

    Rules:
    - Submodule inherited mode draws in the first marker column (S).
    - Id mode draws in the second marker column (I).
    - Name mode draws in the third marker column (A).
    - Vertical dividers are always present between columns.
    - If neither applies, show one neutral marker in the I column.

    Parameters
    ----------
    entry : dict
        Module entry dictionary from get_module_entries().

    recursive_modes : dict
        Map from module id to recursive inherited mode.

    Returns
    -------
    str
        Prefix string in S|I|A layout.
    """
    id_mode = get_id_mode(entry["id"])
    name_mode = get_name_mode(entry["type_name"])
    parent_id = get_parent_module_id(entry["id"])
    submodule_mode = None
    if parent_id is not None:
        submodule_mode = recursive_modes.get(parent_id)

    # Fixed marker columns:
    # slots[0] = inherited parent mode (S), slots[1] = id mode (I),
    # slots[2] = name mode (A)
    slots = [" ", " ", " "]

    # Submodule column shows inherited parent mode recursively.
    if submodule_mode is not None:
        slots[0] = make_color_square(get_color_hex_for_mode(submodule_mode))

    # Id-level selection occupies column 2.
    # When a parent already applies recursively, this explicit id setting is
    # ignored by design and shown in a dedicated color.
    if id_mode is not None:
        if submodule_mode is not None:
            slots[1] = make_color_square("9781E6")
        else:
            slots[1] = make_color_square(get_color_hex_for_mode(id_mode))

    # Name-level selection occupies column 3.
    # When a parent already applies recursively, this name setting is
    # ignored by design and shown in a dedicated color.
    if name_mode is not None:
        if submodule_mode is not None:
            slots[2] = make_color_square("9781E6")
        else:
            slots[2] = make_color_square(get_color_hex_for_mode(name_mode))

    if id_mode is None and name_mode is None and submodule_mode is None:
        if module_has_direct_parameters(entry["module"]):
            slots[1] = make_color_square("FD4D00")
        else:
            slots[1] = make_color_square("000000")

    return f"{slots[0]}|{slots[1]}|{slots[2]}"


def set_module_id_mode(module_id, mode):
    """Set id-based mode and enforce id-list exclusivity.

    Parameters
    ----------
    module_id : str
        Module id in dot notation.
    mode : str
        "perforated" or "tracked".

    Returns
    -------
    None
        This function does not return a value.
    """
    ids_perforate = dedupe_list(GPA.pc.get_module_ids_to_perforate())
    ids_track = dedupe_list(GPA.pc.get_module_ids_to_track())

    if mode == "perforated":
        if module_id in ids_perforate:
            ids_perforate = [value for value in ids_perforate if value != module_id]
        else:
            ids_perforate.append(module_id)
            ids_track = [value for value in ids_track if value != module_id]
    elif mode == "tracked":
        if module_id in ids_track:
            ids_track = [value for value in ids_track if value != module_id]
        else:
            ids_track.append(module_id)
            ids_perforate = [value for value in ids_perforate if value != module_id]

    GPA.pc.set_module_ids_to_perforate(ids_perforate)
    GPA.pc.set_module_ids_to_track(ids_track)


def set_module_name_mode(module_type_name, mode):
    """Set name-based mode and enforce name-list exclusivity.

    Parameters
    ----------
    module_type_name : str
        Module class name (e.g. "Linear").
    mode : str
        "perforated" or "tracked".

    Returns
    -------
    None
        This function does not return a value.
    """
    names_perforate = dedupe_list(GPA.pc.get_module_names_to_perforate())
    names_track = dedupe_list(GPA.pc.get_module_names_to_track())

    if mode == "perforated":
        if module_type_name in names_perforate:
            names_perforate = [
                value for value in names_perforate if value != module_type_name
            ]
        else:
            names_perforate.append(module_type_name)
            names_track = [value for value in names_track if value != module_type_name]
    elif mode == "tracked":
        if module_type_name in names_track:
            names_track = [value for value in names_track if value != module_type_name]
        else:
            names_track.append(module_type_name)
            names_perforate = [
                value for value in names_perforate if value != module_type_name
            ]

    GPA.pc.set_module_names_to_perforate(names_perforate)
    GPA.pc.set_module_names_to_track(names_track)


def render_module_line(entry, selected, recursive_modes):
    """Render one interactive line for a module entry.

    Parameters
    ----------
    entry : dict
        Module entry dictionary from get_module_entries().
    selected : bool
        Whether this line is currently selected by the cursor.

    recursive_modes : dict
        Map from module id to recursive inherited mode.

    Returns
    -------
    str
        Rendered text line.
    """
    selector = ">" if selected else " "
    indent = "  " * entry["depth"]
    prefix = make_mode_prefix(entry, recursive_modes)
    effective_mode = get_resolved_mode(entry, recursive_modes)
    if effective_mode is None:
        effective_mode_text = "none"
    else:
        effective_mode_text = effective_mode
    param_count_text = format_human_count(entry["tree_param_count"])

    return (
        f"{selector} {prefix} {indent}{entry['id']} "
        f"[{entry['type_name']}] mode={effective_mode_text} "
        f"params={param_count_text}"
    )


INTERNAL_CONFIG_KEYS = {
    "module_name",
    "module_type",
    "manually_set_keys",
    "loading_config_values",
    "auto_persist_config",
}

BUCKET_ORDER = [
    "Run And Persistence",
    "Target Selection",
    "Switch Strategy",
    "Optimization And Search",
    "Model Math",
    "Visualization And Logging",
    "Dashboard",
    "Experimental",
    "Other",
]

FALLBACK_BUCKET_DESCRIPTIONS = {
    "Run And Persistence": "Run identity, config file location, safety toggles, and save/load behavior.",
    "Target Selection": "Which modules and parameters are perforated, tracked, replaced, or custom-processed.",
    "Switch Strategy": "When and how dendrite expansion or switching is triggered during training.",
    "Optimization And Search": "Thresholds and retry/search controls for candidate evaluation.",
    "Model Math": "Device, dtype, activation, and tensor-shape related core math settings.",
    "Visualization And Logging": "Console verbosity, graph generation, and save behavior.",
    "Dashboard": "Live dashboard event stream and endpoint configuration.",
    "Experimental": "Flags for optional or advanced behavior not used in common flows.",
    "Other": "Global settings that do not fit another category.",
}

DESCRIPTION_FILE_NAME = "configuration_descriptions.json"
DESCRIPTION_CACHE = None


def load_description_data():
    """Load bucket and setting descriptions from local JSON."""
    global DESCRIPTION_CACHE
    if DESCRIPTION_CACHE is not None:
        return DESCRIPTION_CACHE

    description_file = os.path.join(os.path.dirname(__file__), DESCRIPTION_FILE_NAME)
    payload = {}
    try:
        with open(description_file, "r") as file_handle:
            payload = json.load(file_handle)
    except Exception:
        payload = {}

    bucket_descriptions = payload.get("buckets", {})
    setting_descriptions = payload.get("settings", {})
    DESCRIPTION_CACHE = {
        "buckets": bucket_descriptions,
        "settings": setting_descriptions,
    }
    return DESCRIPTION_CACHE


def format_setting_value(value):
    """Format setting values for compact configuration screen display."""
    if value is None:
        return "None"
    text = str(value)
    if len(text) > 140:
        return text[:137] + "..."
    return text


def classify_setting_bucket(setting_name):
    """Classify one setting into a configuration bucket."""
    if setting_name in {
        "save_name",
        "config_file",
        "configuration_confirmed",
        "testing_dendrite_capacity",
        "using_safe_tensors",
        "strict_loading",
        "checked_skipped_modules",
        "unwrapped_modules_confirmed",
        "weight_decay_accepted",
    }:
        return "Run And Persistence"

    if (
        setting_name.startswith("module_")
        or setting_name.startswith("modules_")
        or setting_name.startswith("parameter_ids")
        or setting_name in {"output_dimensions", "replacement_modules"}
    ):
        return "Target Selection"

    if setting_name in {
        "switch_mode",
        "DOING_SWITCH_EVERY_TIME",
        "DOING_HISTORY",
        "DOING_FIXED_SWITCH",
        "DOING_NO_SWITCH",
        "n_epochs_to_switch",
        "history_lookback",
        "initial_history_after_switches",
        "fixed_switch_num",
        "first_fixed_switch_num",
        "reset_best_score_on_switch",
        "param_vals_setting",
        "PARAM_VALS_BY_TOTAL_EPOCH",
        "PARAM_VALS_BY_UPDATE_EPOCH",
        "PARAM_VALS_BY_NEURON_EPOCH_START",
    }:
        return "Switch Strategy"

    if setting_name in {
        "improvement_threshold",
        "improvement_threshold_raw",
        "find_best_lr",
        "dont_give_up_unless_learning_rate_lowered",
        "max_dendrite_tries",
        "max_dendrites",
        "retain_all_dendrites",
        "global_candidates",
        "candidate_weight_initialization_multiplier",
        "candidate_weight_init_by_main",
    }:
        return "Optimization And Search"

    if setting_name in {
        "device",
        "use_cuda",
        "d_type",
        "pai_forward_function",
        "confirm_correct_sizes",
        "debugging_output_dimensions",
    }:
        return "Model Math"

    if setting_name in {
        "verbose",
        "extra_verbose",
        "silent",
        "drawing_pai",
        "drawing_extra_graphs",
        "save_old_graph_scores",
        "test_saves",
        "pai_saves",
        "library_validation_score",
        "library_extra_scores",
        "library_extra_scores_without_graphing",
    }:
        return "Visualization And Logging"

    if setting_name.startswith("dashboard_"):
        return "Dashboard"

    if setting_name in {
        "learn_dendrites_live",
        "no_extra_n_modes",
        "perforated_backpropagation",
        "weight_tying_experimental",
    }:
        return "Experimental"

    return "Other"


def get_all_global_parameters():
    """Collect all global config parameters as name -> current value."""
    values = {}

    for key, value in GPA.pc.__dict__.items():
        if callable(value):
            continue
        if not (key.startswith("_") and not key.startswith("__")):
            continue
        clean_name = key[1:]
        if clean_name in INTERNAL_CONFIG_KEYS:
            continue
        values[clean_name] = value

    for key, value in GPA.pc.__dict__.items():
        if key.startswith("_") or callable(value):
            continue
        if key in values:
            continue
        values[key] = value

    return values


def build_settings_items(expanded_buckets):
    """Build hierarchical settings items with collapsible buckets."""
    values = get_all_global_parameters()
    grouped = {bucket_name: [] for bucket_name in BUCKET_ORDER}

    for setting_name in sorted(values.keys()):
        bucket_name = classify_setting_bucket(setting_name)
        grouped.setdefault(bucket_name, []).append(setting_name)

    items = []
    for bucket_name in BUCKET_ORDER:
        settings_in_bucket = grouped.get(bucket_name, [])
        if len(settings_in_bucket) == 0:
            continue

        is_expanded = expanded_buckets.get(bucket_name, False)
        marker = "[-]" if is_expanded else "[+]"
        items.append(
            {
                "type": "bucket",
                "bucket": bucket_name,
                "is_expanded": is_expanded,
                "count": len(settings_in_bucket),
                "text": f"{marker} {bucket_name} ({len(settings_in_bucket)})",
            }
        )

        if is_expanded:
            for setting_name in settings_in_bucket:
                items.append(
                    {
                        "type": "setting",
                        "bucket": bucket_name,
                        "name": setting_name,
                        "value": values[setting_name],
                        "text": f"  {setting_name} = {format_setting_value(values[setting_name])}",
                    }
                )

    return items


def get_item_description(item):
    """Return help text for a highlighted bucket or setting item."""
    description_data = load_description_data()
    bucket_descriptions = description_data.get("buckets", {})
    setting_descriptions = description_data.get("settings", {})

    if item["type"] == "bucket":
        return bucket_descriptions.get(
            item["bucket"],
            FALLBACK_BUCKET_DESCRIPTIONS.get(
                item["bucket"], "No description available."
            ),
        )
    if item["type"] == "setting":
        if item["name"] in setting_descriptions:
            return setting_descriptions[item["name"]]
        return f"No description available for {item['name']}."
    return "No description available."


def get_item_display_name(item):
    """Return display name for help/edit messages."""
    if item["type"] == "bucket":
        return item["bucket"]
    if item["type"] == "setting":
        return item["name"]
    return "unknown"


def get_enum_options(setting_name):
    """Return enum options for known enum-like settings."""
    if setting_name == "switch_mode":
        return [
            GPA.pc.DOING_SWITCH_EVERY_TIME,
            GPA.pc.DOING_HISTORY,
            GPA.pc.DOING_FIXED_SWITCH,
            GPA.pc.DOING_NO_SWITCH,
        ]
    if setting_name == "param_vals_setting":
        return [
            GPA.pc.PARAM_VALS_BY_TOTAL_EPOCH,
            GPA.pc.PARAM_VALS_BY_UPDATE_EPOCH,
            GPA.pc.PARAM_VALS_BY_NEURON_EPOCH_START,
        ]
    return None


def apply_setting_value(setting_name, value):
    """Apply one setting value through setter when available."""
    setter = getattr(GPA.pc, f"set_{setting_name}", None)
    if setter is not None:
        setter(value)
        return True, ""

    if hasattr(GPA.pc, setting_name) and not callable(getattr(GPA.pc, setting_name)):
        setattr(GPA.pc, setting_name, value)
        return True, ""

    return False, f"Setting {setting_name} is not editable."


def parse_value_from_text(text, sample_value):
    """Parse user text into same type as sample value where possible."""
    if isinstance(sample_value, bool):
        lowered = text.strip().lower()
        if lowered in ("1", "true", "t", "yes", "y", "on"):
            return True
        if lowered in ("0", "false", "f", "no", "n", "off"):
            return False
        raise ValueError("Expected boolean value (true/false).")

    if isinstance(sample_value, int) and not isinstance(sample_value, bool):
        return int(text.strip())

    if isinstance(sample_value, float):
        return float(text.strip())

    return text


def format_list_editor_line(list_editor_state):
    """Build one-line list editor preview."""
    values = list_editor_state["values"]
    selected_index = list_editor_state["selected_index"]
    typing_active = list_editor_state.get("typing_active", False)
    edit_index = list_editor_state.get("edit_index")
    input_buffer = list_editor_state.get("input_buffer", "")
    cursor_index = list_editor_state.get("cursor_index", len(input_buffer))

    tokens = []
    for i, value in enumerate(values):
        if typing_active and i == edit_index:
            token = render_text_with_cursor(input_buffer, cursor_index)
            tokens.append(token)
            continue

        token = str(value)
        if i == selected_index and not typing_active:
            tokens.append(make_inverted_text(token))
        else:
            tokens.append(token)

    add_index = len(values)
    add_token = "Add new entry"
    if typing_active and edit_index == add_index:
        tokens.append(render_text_with_cursor(input_buffer, cursor_index))
    elif selected_index == add_index:
        tokens.append(make_inverted_text(add_token))
    else:
        tokens.append(add_token)

    line = " | ".join(tokens)
    if len(line) > 180:
        line = line[:177] + "..."
    return line


def render_configuration_line(item, selected):
    """Render one interactive line for the configuration settings screen."""
    selector = ">" if selected else " "
    return f"{selector} {item['text']}"


def get_screen_header_line(active_screen):
    """Build the fixed screen header line with active view highlighted."""
    if active_screen == 0:
        return (
            "Screen: "
            f"{make_inverted_text('Perforation targets')}  "
            "Configuration settings"
        )
    return (
        "Screen: Perforation targets  "
        f"{make_inverted_text('Configuration settings')}"
    )


def get_preview_header_lines(
    entries=None,
    recursive_modes=None,
    active_screen=0,
    confirm_dialog_active=False,
    confirm_choice_index=0,
    help_text="",
    edit_text="",
):
    """Build fixed header lines for the interactive preview screen.

    Parameters
    ----------
    None

    Returns
    -------
    list
        Header lines shown above the scrolling module list.
    """
    lines = []
    lines.append("PerforatedAI Configuration Preview")
    lines.append(get_screen_header_line(active_screen))

    if confirm_dialog_active:
        if confirm_choice_index == 0:
            lines.append(
                "Confirm save: "
                f"{make_inverted_text('save for current run')}   "
                "overwrite configuration and save for current run   "
                "go back to editing"
            )
        elif confirm_choice_index == 1:
            lines.append(
                "Confirm save: save for current run   "
                f"{make_inverted_text('overwrite configuration and save for current run')}   "
                "go back to editing"
            )
        else:
            lines.append(
                "Confirm save: save for current run   "
                "overwrite configuration and save for current run   "
                f"{make_inverted_text('go back to editing')}"
            )
    elif active_screen == 0:
        lines.append(
            "Use Up/Down to select. PageUp/PageDown scroll page-1. Left/Right switches screens. lowercase p/t set by id. uppercase P/T set by name. Enter opens save dialog"
        )
    else:
        lines.append(
            "Use Up/Down to browse settings. PageUp/PageDown scroll page-1. Left/Right switches screens. e toggles bucket open/closed. Space edits. h shows description. Enter opens save dialog"
        )
    lines.append("")
    if help_text:
        lines.append(help_text)
    elif edit_text:
        lines.append(edit_text)
    else:
        lines.append("")

    if active_screen == 0:
        lines.append(
            "Legend: "
            f"perforated={make_color_square('00A5A5')} "
            f"tracked={make_color_square('DEECED')} "
            f"ignored-individual-setting={make_color_square('9781E6')} "
            "submodule-S=inherited parent mode recursively "
            f"neither-no-params={make_color_square('000000')} "
            f"neither-with-params={make_color_square('FD4D00')} "
            "(For best results all parameters should be either tracked or perforated)"
        )
        lines.append("")
    if entries is None:
        entries = []
    if recursive_modes is None:
        recursive_modes = {}

    if active_screen == 0:
        lines.append(build_target_summary_line(entries, recursive_modes))
        lines.append("")
        lines.append("  S|I|A")
    return lines


def get_list_window_size(
    total_entries,
    entries=None,
    recursive_modes=None,
    active_screen=0,
    confirm_dialog_active=False,
    confirm_choice_index=0,
    help_text="",
    edit_text="",
    extra_footer_lines=0,
):
    """Compute how many module rows can be shown in the terminal viewport.

    Parameters
    ----------
    total_entries : int
        Number of available module entries.

    Returns
    -------
    int
        Number of rows available for the scrolling list.
    """
    terminal_size = shutil.get_terminal_size(fallback=(120, 40))
    terminal_lines = terminal_size.lines
    terminal_columns = terminal_size.columns
    header_lines = get_visual_line_count(
        get_preview_header_lines(
            entries,
            recursive_modes,
            active_screen,
            confirm_dialog_active,
            confirm_choice_index,
            help_text,
            edit_text,
        ),
        terminal_columns,
    )
    # Reserve two lines for top/bottom overflow indicators, plus one safety
    # line so exact-fit renders do not push the first header line off-screen.
    available = terminal_lines - header_lines - 2 - extra_footer_lines - 1
    if available < 1:
        return 1
    if available > total_entries:
        return total_entries
    return available


def clamp_window_start(window_start, selected_index, window_size, total_entries):
    """Clamp and adjust list window start to keep selection visible.

    Parameters
    ----------
    window_start : int
        Current first visible row index.
    selected_index : int
        Currently selected module row index.
    window_size : int
        Number of visible list rows.
    total_entries : int
        Total number of list rows.

    Returns
    -------
    int
        Updated first visible row index.
    """
    max_start = max(0, total_entries - window_size)
    if window_start > max_start:
        window_start = max_start
    if window_start < 0:
        window_start = 0

    if selected_index < window_start:
        window_start = selected_index
    elif selected_index >= window_start + window_size:
        window_start = selected_index - window_size + 1

    if window_start > max_start:
        window_start = max_start
    if window_start < 0:
        window_start = 0
    return window_start


def render_preview_screen_window(
    entries,
    selected_index,
    window_start,
    window_size,
    active_screen,
    settings_items,
    confirm_dialog_active,
    confirm_choice_index,
    help_text,
    edit_text,
    list_editor_state,
):
    """Render preview with a fixed header and scrolling module rows.

    Parameters
    ----------
    entries : list
        Module entries from get_module_entries().
    selected_index : int
        Index of the currently selected entry.
    window_start : int
        First visible row index in the module list.
    window_size : int
        Number of visible rows.

    Returns
    -------
    str
        Full screen text to print.
    """
    recursive_modes = build_recursive_modes(entries)
    lines = get_preview_header_lines(
        entries,
        recursive_modes,
        active_screen,
        confirm_dialog_active,
        confirm_choice_index,
        help_text,
        edit_text,
    )

    if active_screen == 0:
        total_lines = len(entries)
    else:
        total_lines = len(settings_items)

    window_end = min(total_lines, window_start + window_size)

    if window_start > 0:
        lines.append("^^^^ more above ^^^^")
    else:
        lines.append("")

    if active_screen == 0:
        for i in range(window_start, window_end):
            lines.append(
                render_module_line(entries[i], i == selected_index, recursive_modes)
            )
    else:
        for i in range(window_start, window_end):
            lines.append(render_configuration_line(settings_items[i], i == selected_index))

    if window_end < total_lines:
        lines.append("vvvv more below vvvv")
    else:
        lines.append("")

    if list_editor_state is not None:
        lines.append("")
        lines.append(
            "List editor: Left/Right move, Space edit selected/add new, Backspace/Delete remove selected, Enter save list+exit, Esc cancel typing"
        )
        lines.append(format_list_editor_line(list_editor_state))

    return "\n".join(lines)


def render_preview_screen(entries, selected_index):
    """Render the full interactive preview screen.

    Parameters
    ----------
    entries : list
        Module entries from get_module_entries().
    selected_index : int
        Index of the currently selected entry.

    Returns
    -------
    str
        Full screen text to print.
    """
    recursive_modes = build_recursive_modes(entries)
    window_size = get_list_window_size(
        len(entries), entries, recursive_modes, 0, False, 0, "", "", 0
    )
    window_start = clamp_window_start(0, selected_index, window_size, len(entries))
    return render_preview_screen_window(
        entries,
        selected_index,
        window_start,
        window_size,
        0,
        [],
        False,
        0,
        "",
        "",
        None,
    )


def read_single_key():
    """Read one keypress, including arrow keys, from stdin.

    Parameters
    ----------
    None

    Returns
    -------
    str
        Key token. Arrow keys are returned as escape sequences like
        "\x1b[A" and "\x1b[B". Page Up/Down are returned as
        "\x1b[5~" and "\x1b[6~".
    """
    file_descriptor = sys.stdin.fileno()
    old_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setraw(file_descriptor)
        first = os.read(file_descriptor, 1)
        if first == b"\x03":
            raise KeyboardInterrupt
        if first != b"\x1b":
            return first.decode("latin1")

        # Read escape sequences at byte level to avoid TextIO buffering issues.
        sequence = bytearray(first)

        # Distinguish plain Esc from control sequences.
        ready, _, _ = select.select([file_descriptor], [], [], 0.05)
        if not ready:
            return "\x1b"
        sequence.extend(os.read(file_descriptor, 1))

        # If this is CSI/SS3, require at least one payload byte.
        if sequence[1] in (ord("["), ord("O")):
            ready, _, _ = select.select([file_descriptor], [], [], 0.15)
            if ready:
                sequence.extend(os.read(file_descriptor, 1))

        # Drain any remaining bytes that are already arriving for this keypress.
        while True:
            last_byte = sequence[-1]
            if chr(last_byte).isalpha() or last_byte == ord("~"):
                break
            ready, _, _ = select.select([file_descriptor], [], [], 0.01)
            if not ready:
                break
            sequence.extend(os.read(file_descriptor, 1))

        return bytes(sequence).decode("latin1")
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, old_settings)


def set_perforation_targets(model):
    """Interactive configuration prompt for perforation target selection.

    This function displays modules, allows keyboard-driven selection updates,
    and writes resulting id/name selections to the global PAI config.

    Controls:
    - Up/Down arrows: move selection
    - p: perforate selected module by id
    - t: track selected module by id
    - Shift+P: perforate selected module type by name
    - Shift+T: track selected module type by name
    - Enter: confirm and continue

    Parameters
    ----------
    model : nn.Module
        Model whose modules are shown in the interactive selector.

    Returns
    -------
    None
        This function does not return a value.
    """
    previous_auto_persist = GPA.pc.__dict__.get("_auto_persist_config", True)
    GPA.pc.__dict__["_auto_persist_config"] = False
    try:
        normalize_selection_conflicts()

        entries = get_module_entries(model)
        if len(entries) == 0:
            print(model)
            input("Press Enter to confirm perforation targets...")
            GPA.pc.set_configuration_confirmed(True)
            return

        selected_index = 0
        window_start = 0
        settings_selected_index = 0
        settings_window_start = 0
        active_screen = 0
        confirm_dialog_active = False
        confirm_choice_index = 0
        help_text = ""
        help_overlay_active = False
        edit_text = ""
        expanded_buckets = {}
        list_editor_state = None

        def finalize_save(choice_index):
            if choice_index == 0:
                GPA.pc.persist_config_outputs(overwrite_config_file=False)
                return

            config_file = GPA.pc.get_config_file()
            if not config_file:
                print("\x1b[2J\x1b[H", end="")
                print("No config_file is set.")
                print(
                    "Enter a local JSON filename/path to create a reusable configuration file."
                )
                print(
                    "After this run, update your training code to call perforate_model(..., config_file='your_path.json')."
                )
                while True:
                    filename = input("Config filename/path: ").strip()
                    if filename:
                        GPA.pc.__dict__["_config_file"] = filename
                        break
                    print("Filename is required for overwrite mode.")

            GPA.pc.persist_config_outputs(overwrite_config_file=True)

        while True:
            normalize_selection_conflicts()
            settings_items = build_settings_items(expanded_buckets)

            if selected_index >= len(entries):
                selected_index = len(entries) - 1
            if settings_selected_index >= len(settings_items):
                settings_selected_index = max(0, len(settings_items) - 1)

            if active_screen == 0:
                current_total = len(entries)
                current_selected = selected_index
                current_window_start = window_start
            else:
                current_total = len(settings_items)
                current_selected = settings_selected_index
                current_window_start = settings_window_start

            recursive_modes = build_recursive_modes(entries)
            window_size = get_list_window_size(
                current_total,
                entries,
                recursive_modes,
                active_screen,
                confirm_dialog_active,
                confirm_choice_index,
                help_text,
                edit_text,
                3 if list_editor_state is not None else 0,
            )
            current_window_start = clamp_window_start(
                current_window_start, current_selected, window_size, current_total
            )

            if active_screen == 0:
                window_start = current_window_start
            else:
                settings_window_start = current_window_start

            # Clear terminal and draw the updated preview.
            print("\x1b[2J\x1b[H", end="")
            print(
                render_preview_screen_window(
                    entries,
                    current_selected,
                    current_window_start,
                    window_size,
                    active_screen,
                    settings_items,
                    confirm_dialog_active,
                    confirm_choice_index,
                    help_text,
                    edit_text,
                    list_editor_state,
                )
            )

            key = read_single_key()

            if help_overlay_active:
                help_overlay_active = False
                help_text = ""
                continue

            if confirm_dialog_active:
                if is_left_key(key):
                    confirm_choice_index = max(0, confirm_choice_index - 1)
                    continue
                if is_right_key(key):
                    confirm_choice_index = min(2, confirm_choice_index + 1)
                    continue
                if key == "\r" or key == "\n":
                    if confirm_choice_index == 2:
                        confirm_dialog_active = False
                        edit_text = ""
                        continue
                    finalize_save(confirm_choice_index)
                    GPA.pc.set_configuration_confirmed(True)
                    break
                continue

            if list_editor_state is not None:
                values = list_editor_state["values"]
                selected_list_index = list_editor_state["selected_index"]
                add_index = len(values)
                typing_active = list_editor_state.get("typing_active", False)

                if typing_active:
                    if key == "\x1b":
                        list_editor_state["typing_active"] = False
                        list_editor_state["edit_index"] = None
                        list_editor_state["input_buffer"] = ""
                        list_editor_state["cursor_index"] = 0
                        continue
                    if is_left_key(key):
                        list_editor_state["cursor_index"] = max(
                            0, list_editor_state.get("cursor_index", 0) - 1
                        )
                        continue
                    if is_right_key(key):
                        list_editor_state["cursor_index"] = min(
                            len(list_editor_state.get("input_buffer", "")),
                            list_editor_state.get("cursor_index", 0) + 1,
                        )
                        continue
                    if key == "\x7f":
                        cursor_index = list_editor_state.get("cursor_index", 0)
                        input_buffer = list_editor_state.get("input_buffer", "")
                        if cursor_index > 0:
                            list_editor_state["input_buffer"] = (
                                input_buffer[: cursor_index - 1]
                                + input_buffer[cursor_index:]
                            )
                            list_editor_state["cursor_index"] = cursor_index - 1
                        continue
                    if is_delete_key(key):
                        cursor_index = list_editor_state.get("cursor_index", 0)
                        input_buffer = list_editor_state.get("input_buffer", "")
                        if cursor_index < len(input_buffer):
                            list_editor_state["input_buffer"] = (
                                input_buffer[:cursor_index]
                                + input_buffer[cursor_index + 1 :]
                            )
                        continue
                    if key == "\r" or key == "\n":
                        edit_index = list_editor_state.get("edit_index", add_index)
                        sample_value = list_editor_state.get("sample_type")
                        if edit_index < len(values):
                            sample_value = values[edit_index]
                        try:
                            parsed = parse_value_from_text(
                                list_editor_state.get("input_buffer", ""),
                                sample_value,
                            )
                        except Exception as exc:
                            edit_text = f"Invalid list entry: {exc}"
                            list_editor_state["typing_active"] = False
                            list_editor_state["edit_index"] = None
                            list_editor_state["input_buffer"] = ""
                            continue

                        if edit_index < len(values):
                            values[edit_index] = parsed
                            list_editor_state["selected_index"] = edit_index
                        else:
                            values.append(parsed)
                            list_editor_state["selected_index"] = len(values) - 1
                        list_editor_state["typing_active"] = False
                        list_editor_state["edit_index"] = None
                        list_editor_state["input_buffer"] = ""
                        list_editor_state["cursor_index"] = 0
                        edit_text = ""
                        continue
                    if len(key) == 1 and key >= " ":
                        cursor_index = list_editor_state.get("cursor_index", 0)
                        input_buffer = list_editor_state.get("input_buffer", "")
                        list_editor_state["input_buffer"] = (
                            input_buffer[:cursor_index] + key + input_buffer[cursor_index:]
                        )
                        list_editor_state["cursor_index"] = cursor_index + 1
                        continue
                    continue

                if key == "\x1b":
                    list_editor_state = None
                    edit_text = ""
                    continue
                if is_left_key(key):
                    list_editor_state["selected_index"] = max(0, selected_list_index - 1)
                    continue
                if is_right_key(key):
                    list_editor_state["selected_index"] = min(add_index, selected_list_index + 1)
                    continue
                if key == "\x7f" or is_delete_key(key):
                    if selected_list_index < len(values):
                        del values[selected_list_index]
                        if list_editor_state["selected_index"] > len(values):
                            list_editor_state["selected_index"] = len(values)
                    continue
                if key == " ":
                    edit_index = selected_list_index
                    initial_text = ""
                    if edit_index < len(values):
                        initial_text = str(values[edit_index])
                    list_editor_state["typing_active"] = True
                    list_editor_state["edit_index"] = edit_index
                    list_editor_state["input_buffer"] = initial_text
                    list_editor_state["cursor_index"] = len(initial_text)
                    continue
                if key == "\r" or key == "\n":
                    ok, message = apply_setting_value(
                        list_editor_state["setting_name"], values
                    )
                    list_editor_state = None
                    edit_text = message if not ok else ""
                    continue
                continue

            if is_up_key(key):
                help_text = ""
                edit_text = ""
                if active_screen == 0:
                    selected_index = max(0, selected_index - 1)
                else:
                    settings_selected_index = max(0, settings_selected_index - 1)
                continue
            if is_down_key(key):
                help_text = ""
                edit_text = ""
                if active_screen == 0:
                    selected_index = min(len(entries) - 1, selected_index + 1)
                else:
                    settings_selected_index = min(
                        len(settings_items) - 1, settings_selected_index + 1
                    )
                continue
            if is_page_up_key(key):
                help_text = ""
                edit_text = ""
                page_step = max(1, window_size - 1)
                if active_screen == 0:
                    selected_index = max(0, selected_index - page_step)
                    window_start = max(0, window_start - page_step)
                else:
                    settings_selected_index = max(0, settings_selected_index - page_step)
                    settings_window_start = max(0, settings_window_start - page_step)
                continue
            if is_page_down_key(key):
                help_text = ""
                edit_text = ""
                page_step = max(1, window_size - 1)
                if active_screen == 0:
                    selected_index = min(len(entries) - 1, selected_index + page_step)
                    window_start = min(len(entries) - 1, window_start + page_step)
                else:
                    settings_selected_index = min(
                        len(settings_items) - 1, settings_selected_index + page_step
                    )
                    settings_window_start = min(
                        len(settings_items) - 1, settings_window_start + page_step
                    )
                continue
            if is_left_key(key):
                help_text = ""
                edit_text = ""
                active_screen = max(0, active_screen - 1)
                continue
            if is_right_key(key):
                help_text = ""
                edit_text = ""
                active_screen = min(1, active_screen + 1)
                continue

            if active_screen == 0:
                selected_entry = entries[selected_index]

                if key == "p":
                    set_module_id_mode(selected_entry["id"], "perforated")
                    continue
                if key == "t":
                    set_module_id_mode(selected_entry["id"], "tracked")
                    continue
                if key == "P":
                    set_module_name_mode(selected_entry["type_name"], "perforated")
                    continue
                if key == "T":
                    set_module_name_mode(selected_entry["type_name"], "tracked")
                    continue
            else:
                if key == "e" and len(settings_items) > 0:
                    item = settings_items[settings_selected_index]
                    if item["type"] == "bucket":
                        expanded_buckets[item["bucket"]] = not item["is_expanded"]
                    continue
                if key == "h" and len(settings_items) > 0:
                    item = settings_items[settings_selected_index]
                    help_text = (
                        f"HELP - {get_item_display_name(item)} - "
                        f"{get_item_description(item)}"
                    )
                    help_overlay_active = True
                    edit_text = ""
                    continue
                if key == " " and len(settings_items) > 0:
                    item = settings_items[settings_selected_index]
                    if item["type"] == "bucket":
                        expanded_buckets[item["bucket"]] = not item["is_expanded"]
                        edit_text = ""
                        continue
                    if item["type"] != "setting":
                        edit_text = ""
                        continue

                    setting_name = item["name"]
                    setting_value = item["value"]
                    enum_options = get_enum_options(setting_name)

                    if isinstance(setting_value, bool):
                        ok, message = apply_setting_value(setting_name, not setting_value)
                        edit_text = message if not ok else ""
                        continue

                    if enum_options is not None and len(enum_options) > 1:
                        try:
                            index = enum_options.index(setting_value)
                        except ValueError:
                            index = -1
                        next_value = enum_options[(index + 1) % len(enum_options)]
                        ok, message = apply_setting_value(setting_name, next_value)
                        edit_text = message if not ok else ""
                        continue

                    if isinstance(setting_value, list):
                        sample_type = None
                        if len(setting_value) > 0:
                            sample_type = setting_value[0]
                        list_editor_state = {
                            "setting_name": setting_name,
                            "values": list(setting_value),
                            "selected_index": 0,
                            "sample_type": sample_type,
                            "typing_active": False,
                            "edit_index": None,
                            "input_buffer": "",
                            "cursor_index": 0,
                        }
                        edit_text = (
                            f"EDIT - {setting_name} - list mode active"
                        )
                        continue

                    if isinstance(setting_value, (int, float)):
                        prompt = (
                            f"EDIT - {setting_name} - enter new value "
                            f"(current={setting_value}): "
                        )
                        text = input(prompt).strip()
                        if not text:
                            edit_text = ""
                            continue
                        try:
                            parsed = parse_value_from_text(text, setting_value)
                        except Exception as exc:
                            edit_text = f"Invalid value for {setting_name}: {exc}"
                            continue
                        ok, message = apply_setting_value(setting_name, parsed)
                        edit_text = message if not ok else ""
                        continue

                    edit_text = f"Setting {setting_name} edit is not supported."
                    continue

            if key == "\r" or key == "\n":
                edit_text = ""
                confirm_dialog_active = True
                confirm_choice_index = 0
                continue
    finally:
        GPA.pc.__dict__["_auto_persist_config"] = previous_auto_persist
