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


def wrap_ansi_line_on_words(line, terminal_columns):
    """Wrap one ANSI-colored line on word boundaries for terminal display.

    Falls back to a hard split only when one contiguous token is wider than
    the available width.
    """
    if terminal_columns < 1:
        return [line]

    plain = strip_ansi(line)
    if len(plain) <= terminal_columns:
        return [line]

    # Map each visible-character index to the corresponding raw-string index,
    # skipping ANSI escape sequences which have zero visual width.
    visible_to_raw = []
    raw_index = 0
    while raw_index < len(line):
        if line[raw_index] == "\x1b":
            match = ANSI_ESCAPE_RE.match(line, raw_index)
            if match is not None:
                raw_index = match.end()
                continue
        visible_to_raw.append(raw_index)
        raw_index += 1

    if len(visible_to_raw) == 0:
        return [line]

    wrapped = []
    start = 0
    visible_len = len(plain)
    while start < visible_len:
        while start < visible_len and plain[start] == " ":
            start += 1
        if start >= visible_len:
            break

        max_end = min(visible_len, start + terminal_columns)
        if max_end == visible_len:
            end = visible_len
        else:
            split_at = plain.rfind(" ", start, max_end)
            end = split_at if split_at > start else max_end

        raw_start = visible_to_raw[start]
        raw_end = len(line) if end >= visible_len else visible_to_raw[end]
        wrapped.append(line[raw_start:raw_end].rstrip())

        if end < visible_len and plain[end] == " ":
            start = end + 1
        else:
            start = end

    return wrapped if wrapped else [line]


def wrap_screen_text_for_terminal(screen_text):
    """Wrap full screen text to terminal width without mid-word breaks."""
    terminal_columns = shutil.get_terminal_size(fallback=(120, 40)).columns
    wrapped_lines = []
    for raw_line in screen_text.split("\n"):
        wrapped_lines.extend(wrap_ansi_line_on_words(raw_line, terminal_columns))
    return "\n".join(wrapped_lines)


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


def is_select_key(key):
    """Return True when key token should trigger selection/edit actions."""
    return key == " " or key == "\r" or key == "\n"


def make_mode_prefix(entry, recursive_modes):
    """Build the color-square prefix for one module line.

    Rules:
    - Submodule inherited mode draws in the first marker column (S).
    - Id mode draws in the second marker column (I).
    - Name mode draws in the third marker column (N).
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
        Prefix string in S|I|N layout.
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

RUNTIME_ONLY_CONFIG_KEYS = {"config_file"}

SETTING_COLOR_IMPACTFUL = "00A5A5"
SETTING_COLOR_SUPPORTING = "004145"
SETTING_COLOR_SUPPORTING_CONFIGURATION = "001424"
SETTING_COLOR_CONSTANT = "9892A0"
SETTING_COLOR_EXPERIMENTAL = "FD4D00"
SETTING_COLOR_PB = "00F9C9"
SETTING_COLOR_CONFIGURATION = "FFFFFF"

DESCRIPTION_FILE_NAME = "configuration_descriptions.json"
DESCRIPTION_CACHE = None

LABEL_IMPACTFUL = "Impactful hyperparameters"
LABEL_SUPPORTING = "Supporting hyperparameters"
LABEL_SUPPORTING_CONFIGURATION = "Supporting configuration parameters"
LABEL_CONSTANTS = "Constants"
LABEL_EXPERIMENTAL = "Experimental hyperparameters"
LABEL_PB = "PerforatedBP hyperparameters"
LABEL_CONFIGURATION = "Configuration Parameters"

SETTING_LABEL_PRIORITY = {
    LABEL_CONFIGURATION: 1,
    LABEL_IMPACTFUL: 2,
    LABEL_SUPPORTING: 3,
    LABEL_SUPPORTING_CONFIGURATION: 4,
    LABEL_PB: 5,
    LABEL_CONSTANTS: 6,
    LABEL_EXPERIMENTAL: 7,
}


def _normalize_label(label_text):
    """Normalize label text for stable ordering/lookups."""
    if not isinstance(label_text, str):
        return LABEL_SUPPORTING

    compact = " ".join(label_text.strip().split()).lower()
    if compact in ("impactful", "impactful hyperparameters"):
        return LABEL_IMPACTFUL
    if compact in ("supporting", "supporting hyperparameters"):
        return LABEL_SUPPORTING
    if compact in (
        "supporting configuration",
        "supporting configuration parameters",
    ):
        return LABEL_SUPPORTING_CONFIGURATION
    if compact in ("configuration", "configuration parameters"):
        return LABEL_CONFIGURATION
    if compact in ("constants", "constant"):
        return LABEL_CONSTANTS
    if compact in ("experimental", "experimental hyperparameters"):
        return LABEL_EXPERIMENTAL
    if compact in (
        "perforatedbp",
        "perforatedbp hyperparameters",
        "perforatedbp settings",
    ):
        return LABEL_PB

    return LABEL_SUPPORTING


LABEL_TO_COLOR = {
    LABEL_CONFIGURATION: SETTING_COLOR_CONFIGURATION,
    LABEL_IMPACTFUL: SETTING_COLOR_IMPACTFUL,
    LABEL_SUPPORTING: SETTING_COLOR_SUPPORTING,
    LABEL_SUPPORTING_CONFIGURATION: SETTING_COLOR_SUPPORTING_CONFIGURATION,
    LABEL_PB: SETTING_COLOR_PB,
    LABEL_CONSTANTS: SETTING_COLOR_CONSTANT,
    LABEL_EXPERIMENTAL: SETTING_COLOR_EXPERIMENTAL,
}


def load_description_data():
    """Load bucket and setting metadata from local JSON."""
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

    bucket_entries = payload.get("buckets", {})
    bucket_descriptions = {}
    setting_descriptions = {}
    setting_labels = {}
    setting_to_bucket = {}
    settings_by_bucket = {}
    bucket_order = []

    # New nested schema:
    # {
    #   "buckets": {
    #      "Bucket Name": {
    #          "description": "...",
    #          "settings": {
    #              "setting_name": {"description": "...", "label": "..."}
    #          }
    #      }
    #   }
    # }
    if isinstance(bucket_entries, dict):
        for bucket_name, bucket_payload in bucket_entries.items():
            bucket_order.append(bucket_name)
            if isinstance(bucket_payload, dict):
                bucket_descriptions[bucket_name] = bucket_payload.get("description", "")

                bucket_settings = bucket_payload.get("settings", {})
                ordered_setting_names = []
                if isinstance(bucket_settings, dict):
                    for setting_name, setting_payload in bucket_settings.items():
                        ordered_setting_names.append(setting_name)
                        setting_to_bucket[setting_name] = bucket_name
                        if isinstance(setting_payload, dict):
                            setting_descriptions[setting_name] = setting_payload.get("description", "")
                            setting_labels[setting_name] = _normalize_label(
                                setting_payload.get("label")
                            )
                        elif isinstance(setting_payload, str):
                            setting_descriptions[setting_name] = setting_payload
                            setting_labels[setting_name] = LABEL_SUPPORTING
                settings_by_bucket[bucket_name] = ordered_setting_names
            elif isinstance(bucket_payload, str):
                # Backward compatibility: old flat schema bucket description
                bucket_descriptions[bucket_name] = bucket_payload

    # Backward compatibility: old top-level flat settings object
    top_level_settings = payload.get("settings", {})
    if isinstance(top_level_settings, dict):
        for setting_name, setting_payload in top_level_settings.items():
            if setting_name in setting_descriptions:
                continue
            if isinstance(setting_payload, dict):
                setting_descriptions[setting_name] = setting_payload.get("description", "")
                setting_labels[setting_name] = _normalize_label(setting_payload.get("label"))
            elif isinstance(setting_payload, str):
                setting_descriptions[setting_name] = setting_payload
                setting_labels[setting_name] = LABEL_SUPPORTING

    DESCRIPTION_CACHE = {
        "buckets": bucket_descriptions,
        "bucket_order": bucket_order,
        "settings": setting_descriptions,
        "setting_labels": setting_labels,
        "setting_to_bucket": setting_to_bucket,
        "settings_by_bucket": settings_by_bucket,
    }
    return DESCRIPTION_CACHE


def format_setting_value(value):
    """Format setting values for compact configuration screen display."""
    if value is None:
        return "None"
    if callable(value):
        name = getattr(value, "__name__", None) or getattr(value, "__qualname__", None)
        mod = getattr(value, "__module__", None)
        if name in ("sigmoid", "relu", "tanh"):
            text = f"torch.{name}"
        elif name and mod:
            text = f"{mod}.{name}"
        elif name:
            text = str(name)
        else:
            text = repr(value)
    else:
        text = str(value)
    if len(text) > 140:
        return text[:137] + "..."
    return text


def get_setting_options_hint(setting_name):
    """Return optional inline options text for known settings."""
    return ""


def get_setting_color_hex(setting_name):
    """Choose a legend color for one setting name."""
    description_data = load_description_data()
    setting_labels = description_data.get("setting_labels", {})
    label_name = setting_labels.get(setting_name)
    if label_name is not None:
        normalized_label = _normalize_label(label_name)
        return LABEL_TO_COLOR.get(normalized_label, SETTING_COLOR_SUPPORTING)

    return SETTING_COLOR_SUPPORTING


def get_setting_label(setting_name):
    """Get label text for one setting, preferring description metadata."""
    description_data = load_description_data()
    setting_labels = description_data.get("setting_labels", {})
    if setting_name in setting_labels:
        return _normalize_label(setting_labels[setting_name])

    return LABEL_SUPPORTING


def sort_settings_for_bucket(bucket_name, setting_names):
    """Sort settings by appearance order in configuration_descriptions JSON."""
    description_data = load_description_data()
    configured_order = description_data.get("settings_by_bucket", {}).get(
        bucket_name, []
    )
    configured_index = {name: i for i, name in enumerate(configured_order)}
    fallback_base = len(configured_order) + 100000

    def _sort_key(setting_name):
        # Keep constants at the end of each bucket while preserving the
        # configured JSON order for everything else.
        is_constant = get_setting_label(setting_name) == LABEL_CONSTANTS
        if setting_name in configured_index:
            return (1 if is_constant else 0, 0, configured_index[setting_name], setting_name)
        return (1 if is_constant else 0, 1, fallback_base, setting_name)

    return sorted(setting_names, key=_sort_key)


def get_bucket_color_hex(setting_names):
    """Pick bucket marker color from highest-priority setting label."""
    if not setting_names:
        return SETTING_COLOR_SUPPORTING

    best_setting = None
    best_priority = 10**9
    for setting_name in setting_names:
        label_name = get_setting_label(setting_name)
        priority = SETTING_LABEL_PRIORITY.get(label_name, 99)
        if priority < best_priority:
            best_priority = priority
            best_setting = setting_name

    if best_setting is None:
        return SETTING_COLOR_SUPPORTING
    return get_setting_color_hex(best_setting)


def classify_setting_bucket(setting_name):
    """Classify one setting into a configuration bucket from JSON metadata."""
    description_data = load_description_data()
    bucket_name = description_data.get("setting_to_bucket", {}).get(setting_name)
    if bucket_name is not None:
        return bucket_name
    return None


def get_all_global_parameters():
    """Collect JSON-declared global config parameters as name -> current value."""
    description_data = load_description_data()
    visible_names = []
    for bucket_name in description_data.get("bucket_order", []):
        visible_names.extend(
            description_data.get("settings_by_bucket", {}).get(bucket_name, [])
        )

    values = {}

    for setting_name in visible_names:
        if setting_name in RUNTIME_ONLY_CONFIG_KEYS:
            continue
        values[setting_name] = get_global_setting_value(setting_name)

    return values


def build_settings_items(expanded_buckets):
    """Build hierarchical settings items with collapsible buckets."""
    values = get_all_global_parameters()

    items = []
    description_data = load_description_data()
    for bucket_name in description_data.get("bucket_order", []):
        settings_in_bucket = sort_settings_for_bucket(
            bucket_name,
            description_data.get("settings_by_bucket", {}).get(bucket_name, []),
        )
        if len(settings_in_bucket) == 0:
            continue

        is_expanded = expanded_buckets.get(bucket_name, False)
        marker = "[-]" if is_expanded else "[+]"
        bucket_color = make_color_square(get_bucket_color_hex(settings_in_bucket))
        display_bucket_name = bucket_name
        if bucket_name == "Target Selection":
            display_bucket_name = (
                "Target Selection (Recommended to use Perforation Targets Menu)"
            )
        items.append(
            {
                "type": "bucket",
                "bucket": bucket_name,
                "is_expanded": is_expanded,
                "count": len(settings_in_bucket),
                "text": f"{marker} {bucket_color} {display_bucket_name} ({len(settings_in_bucket)})",
            }
        )

        if is_expanded:
            for setting_name in settings_in_bucket:
                items.append(
                    {
                        "type": "setting",
                        "bucket": bucket_name,
                        "name": setting_name,
                        "value": values.get(setting_name, "<unavailable>"),
                        "text": (
                            f"  {setting_name} = "
                            f"{format_setting_value(values.get(setting_name, '<unavailable>'))}"
                            f"{get_setting_options_hint(setting_name)}"
                        ),
                    }
                )

    return items


def get_global_setting_value(setting_name):
    """Read one setting value from the global config object."""
    getter = getattr(GPA.pc, f"get_{setting_name}", None)
    if getter is not None:
        try:
            return getter()
        except Exception:
            return "<unavailable>"
    if hasattr(GPA.pc, setting_name) and not callable(getattr(GPA.pc, setting_name)):
        return getattr(GPA.pc, setting_name)
    return "<unavailable>"


def load_existing_module_settings():
    """Load existing module_settings from local config sources."""
    merged = {}

    source_paths = []
    config_file = GPA.pc.get_config_file()
    run_config = GPA.pc.get_run_config_path()
    if config_file:
        source_paths.append(config_file)
    if run_config:
        source_paths.append(run_config)

    for source_path in source_paths:
        if not source_path or not os.path.exists(source_path):
            continue
        try:
            with open(source_path, "r") as file_handle:
                payload = json.load(file_handle)
            module_settings = payload.get("module_settings", {})
            for scope_key, values in module_settings.items():
                if isinstance(values, dict):
                    merged[scope_key] = dict(values)
        except Exception:
            continue

    return merged


def get_scope_values(scope_key, module_settings_overrides, existing_module_settings):
    """Get merged module-scope values from existing data plus in-session edits."""
    values = {}
    if scope_key in existing_module_settings:
        values.update(existing_module_settings[scope_key])
    if scope_key in module_settings_overrides:
        values.update(module_settings_overrides[scope_key])
    return values


def build_module_settings_items(
    expanded_buckets,
    module_scope,
    module_settings_overrides,
    existing_module_settings,
):
    """Build bucketed items for module-customizable settings only."""
    if module_scope is None:
        return []

    scope_key = module_scope["scope_key"]
    scoped_values = get_scope_values(
        scope_key, module_settings_overrides, existing_module_settings
    )

    items = []
    description_data = load_description_data()
    customizable_setting_names = list(GPA.PAIConfig._CUSTOMIZABLE.keys())
    for bucket_name in description_data.get("bucket_order", []):
        settings_in_bucket = sort_settings_for_bucket(
            bucket_name,
            [
                setting_name
                for setting_name in customizable_setting_names
                if description_data.get("setting_to_bucket", {}).get(setting_name)
                == bucket_name
            ],
        )
        if len(settings_in_bucket) == 0:
            continue

        is_expanded = expanded_buckets.get(bucket_name, False)
        marker = "[-]" if is_expanded else "[+]"
        bucket_color = make_color_square(get_bucket_color_hex(settings_in_bucket))
        items.append(
            {
                "type": "bucket",
                "bucket": bucket_name,
                "is_expanded": is_expanded,
                "count": len(settings_in_bucket),
                "text": f"{marker} {bucket_color} {bucket_name} ({len(settings_in_bucket)})",
            }
        )

        if is_expanded:
            for setting_name in settings_in_bucket:
                if setting_name in scoped_values:
                    value = scoped_values[setting_name]
                    source = "scope"
                else:
                    value = get_global_setting_value(setting_name)
                    source = "global-default"
                items.append(
                    {
                        "type": "setting",
                        "bucket": bucket_name,
                        "name": setting_name,
                        "value": value,
                        "scope_key": scope_key,
                        "source": source,
                        "text": (
                            f"  {setting_name} = {format_setting_value(value)} "
                            f"({source})"
                            f"{get_setting_options_hint(setting_name)}"
                        ),
                    }
                )

    return items


def set_module_scoped_value(module_settings_overrides, scope_key, setting_name, value):
    """Write one module-scoped value into in-session overrides."""
    module_settings_overrides.setdefault(scope_key, {})[setting_name] = value


def get_space_scope_for_entry(entry, recursive_modes):
    """Determine module customization scope or return a warning reason."""
    parent_id = get_parent_module_id(entry["id"])
    if parent_id is not None and recursive_modes.get(parent_id) is not None:
        return None, "Cannot customize this module: it is a submodule inheriting parent mode settings."

    if entry["id"] in GPA.pc.get_module_ids_to_perforate():
        return {
            "scope_kind": "id",
            "scope_key": entry["id"],
            "module_id": entry["id"],
            "module_type": entry["type_name"],
            "scope_text": f"Applying to module id {entry['id']}",
        }, None

    if entry["type_name"] in GPA.pc.get_module_names_to_perforate():
        return {
            "scope_kind": "name",
            "scope_key": entry["type_name"],
            "module_id": entry["id"],
            "module_type": entry["type_name"],
            "scope_text": f"Applying to all modules of type {entry['type_name']}",
        }, None

    return None, "Cannot customize this module: it is not explicitly perforated by id or by name."


def persist_module_settings_updates(module_settings_overrides, overwrite_config_file=False):
    """Persist in-session module_settings updates to local config JSON files."""
    if not module_settings_overrides:
        return

    destinations = []
    run_config = GPA.pc.get_run_config_path()
    if run_config:
        destinations.append(run_config)
    if overwrite_config_file:
        config_file = GPA.pc.get_config_file()
        if config_file and config_file not in destinations:
            destinations.append(config_file)

    for destination in destinations:
        payload = {}
        if os.path.exists(destination):
            try:
                with open(destination, "r") as file_handle:
                    payload = json.load(file_handle)
            except Exception:
                payload = {}

        module_settings = payload.get("module_settings", {})
        if not isinstance(module_settings, dict):
            module_settings = {}

        for scope_key, values in module_settings_overrides.items():
            existing = module_settings.get(scope_key, {})
            if not isinstance(existing, dict):
                existing = {}
            existing.update(values)
            module_settings[scope_key] = existing

        payload["module_settings"] = module_settings

        directory = os.path.dirname(destination)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(destination, "w") as file_handle:
            json.dump(payload, file_handle, indent=2)


def get_item_description(item):
    """Return help text for a highlighted bucket or setting item."""
    description_data = load_description_data()
    bucket_descriptions = description_data.get("buckets", {})
    setting_descriptions = description_data.get("settings", {})

    if item["type"] == "bucket":
        return bucket_descriptions.get(
            item["bucket"],
            "No description available.",
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

    save_index = len(values) + 1
    save_token = "Save list and return"
    if selected_index == save_index and not typing_active:
        tokens.append(make_inverted_text(save_token))
    else:
        tokens.append(save_token)

    line = " | ".join(tokens)
    if len(line) > 180:
        line = line[:177] + "..."
    return line


def render_configuration_line(item, selected):
    """Render one interactive line for the configuration settings screen."""
    selector = ">" if selected else " "
    if item["type"] == "setting":
        color_marker = make_color_square(get_setting_color_hex(item["name"]))
        return f"{selector} {color_marker} {item['text']}"
    return f"{selector} {item['text']}"


def get_screen_header_line(active_screen):
    """Build the fixed screen header line with active view highlighted."""
    if active_screen == 0:
        return (
            "Screen: "
            f"{make_inverted_text('Perforation targets')}  "
            "Global settings  "
            "Module settings"
        )
    if active_screen == 1:
        return (
            "Screen: Perforation targets  "
            f"{make_inverted_text('Global settings')}  "
            "Module settings"
        )
    return (
        "Screen: Perforation targets  "
        "Global settings  "
        f"{make_inverted_text('Module settings')}"
    )


def get_preview_header_lines(
    entries=None,
    recursive_modes=None,
    active_screen=0,
    confirm_dialog_active=False,
    confirm_choice_index=0,
    help_text="",
    edit_text="",
    module_scope=None,
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
            "Use Up/Down to select. PageUp/PageDown to scroll. Left/Right switches to global settings. Lowercase p/t set by id. Uppercase P/T set by name. Space/Enter opens module settings for selected module (if explicitly perforated). s opens save dialog and begins training with the specified configuration."
        )
    elif active_screen == 1:
        lines.append(
            "Use Up/Down to browse settings. PageUp/PageDown to scroll. Left/Right switches screens. Space/Enter edits. h shows description. s opens save dialog and begins training with the specified configuration."
        )
        lines.append(
            "Legend: "
            f"{LABEL_CONFIGURATION}={make_color_square(SETTING_COLOR_CONFIGURATION)} "
            f"{LABEL_IMPACTFUL}={make_color_square(SETTING_COLOR_IMPACTFUL)} "
            f"{LABEL_SUPPORTING}={make_color_square(SETTING_COLOR_SUPPORTING)} "
            f"{LABEL_SUPPORTING_CONFIGURATION}={make_color_square(SETTING_COLOR_SUPPORTING_CONFIGURATION)} "
            f"{LABEL_PB}={make_color_square(SETTING_COLOR_PB)} "
            f"{LABEL_CONSTANTS}={make_color_square(SETTING_COLOR_CONSTANT)} "
            f"{LABEL_EXPERIMENTAL}={make_color_square(SETTING_COLOR_EXPERIMENTAL)}"
        )
        if not help_text and not edit_text:
            lines.append("")
    else:
        lines.append(
            "Use Up/Down to browse settings. PageUp/PageDown to scroll. Left/Right switches screens. Space/Enter edits. h shows description. s opens save dialog and begins training with the specified configuration."
        )
        lines.append(
            "Legend: "
            f"{LABEL_CONFIGURATION}={make_color_square(SETTING_COLOR_CONFIGURATION)} "
            f"{LABEL_IMPACTFUL}={make_color_square(SETTING_COLOR_IMPACTFUL)} "
            f"{LABEL_SUPPORTING}={make_color_square(SETTING_COLOR_SUPPORTING)} "
            f"{LABEL_SUPPORTING_CONFIGURATION}={make_color_square(SETTING_COLOR_SUPPORTING_CONFIGURATION)} "
            f"{LABEL_PB}={make_color_square(SETTING_COLOR_PB)} "
            f"{LABEL_CONSTANTS}={make_color_square(SETTING_COLOR_CONSTANT)} "
            f"{LABEL_EXPERIMENTAL}={make_color_square(SETTING_COLOR_EXPERIMENTAL)}"
        )
        if not help_text and not edit_text:
            lines.append("")
        if module_scope is not None:
            lines.append(module_scope["scope_text"])
        else:
            lines.append("Scope: none")
        if not help_text and not edit_text:
            lines.append("")

    if help_text:
        lines.append("")
        lines.append(help_text)
        lines.append("")
    elif edit_text:
        if edit_text.startswith("EDIT -"):
            lines.append("")
            lines.append(edit_text)
            lines.append("")
        else:
            lines.append(edit_text)

    if active_screen == 0:
        lines.append("")
        lines.append(
            "Legend: "
            f"perforated={make_color_square('00A5A5')} "
            f"tracked={make_color_square('DEECED')} "
            f"neither-no-params={make_color_square('000000')} "
            f"neither-with-params={make_color_square('FD4D00')} "
            f"ignored-individual-setting={make_color_square('9781E6')} "
            "(For best results all parameters should be either tracked or perforated)"
        )
        lines.append(
            "S - setting inherited as submodule | I - setting by id | N - setting by name of type. -- Left most takes priority"
        )
    if entries is None:
        entries = []
    if recursive_modes is None:
        recursive_modes = {}

    if active_screen == 0:
        lines.append(build_target_summary_line(entries, recursive_modes))
        lines.append("")
        lines.append("  S|I|N")
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
    module_scope=None,
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
            module_scope,
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
    module_scope=None,
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
        module_scope,
    )

    if active_screen == 0:
        total_lines = len(entries)
    else:
        total_lines = len(settings_items)

    window_end = min(total_lines, window_start + window_size)

    if window_start > 0:
        lines.append("^^^^ more above ^^^^")

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

    if list_editor_state is not None:
        lines.append("")
        lines.append(
            "List editor: Left/Right move, Space/Enter select, Backspace/Delete remove selected entry, Esc cancel typing"
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


def enter_alternate_screen():
    """Switch terminal to alternate screen buffer for interactive UI."""
    # Hide the terminal cursor while rendering the full-screen menu to avoid
    # a second editor/terminal selection rectangle overlapping the UI.
    print("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H", end="", flush=True)


def exit_alternate_screen():
    """Return terminal to normal screen buffer."""
    # Always restore the cursor when leaving the menu.
    print("\x1b[?25h\x1b[?1049l", end="", flush=True)


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
    - Space/Enter: select or edit current item
    - s: open save dialog
    - Esc: go back/cancel current inline edit

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
    entered_alt_screen = False
    try:
        enter_alternate_screen()
        entered_alt_screen = True
        normalize_selection_conflicts()

        entries = get_module_entries(model)
        if len(entries) == 0:
            print(model)
            input("Press Enter to confirm perforation targets...")
            GPA.pc.set_configuration_confirmed(True)
            return

        selected_index = 0
        window_start = 0
        global_settings_selected_index = 0
        global_settings_window_start = 0
        module_settings_selected_index = 0
        module_settings_window_start = 0
        active_screen = 0
        confirm_dialog_active = False
        confirm_choice_index = 0
        help_text = ""
        help_overlay_active = False
        edit_text = ""
        global_expanded_buckets = {}
        module_expanded_buckets = {}
        list_editor_state = None
        scalar_editor_state = None
        module_scope = None
        module_settings_overrides = {}
        existing_module_settings = load_existing_module_settings()

        def finalize_save(choice_index):
            if choice_index == 0:
                GPA.pc.persist_config_outputs(overwrite_config_file=False)
                persist_module_settings_updates(
                    module_settings_overrides,
                    overwrite_config_file=False,
                )
                return

            config_file = GPA.pc.get_config_file()
            if not config_file:
                import os

                current_save_name = GPA.pc.get_save_name() or "PAI"
                run_config_path = GPA.pc.get_run_config_path()
                relative_config_name = (
                    os.path.relpath(run_config_path, os.getcwd())
                    if run_config_path
                    else f"{current_save_name}/{current_save_name}_config.json"
                )
                print("\x1b[2J\x1b[H", end="")
                print("No config_file is set.")
                print(
                    "Enter a local JSON filename/path to create a reusable configuration file."
                )
                print(
                    f"You currently have save_name='{current_save_name}'. "
                    f"Choose the filename '{relative_config_name}' to skip this menu next time without changing your perforate_model call."
                )
                print(
                    "Alternative: set config_file in perforate_model(..., config_file='your_path.json')."
                )
                while True:
                    filename = input("Config filename/path: ").strip()
                    if filename:
                        GPA.pc.__dict__["_config_file"] = filename
                        break
                    print("Filename is required for overwrite mode.")

            GPA.pc.persist_config_outputs(overwrite_config_file=True)
            persist_module_settings_updates(
                module_settings_overrides,
                overwrite_config_file=True,
            )

        while True:
            normalize_selection_conflicts()
            if active_screen == 1:
                settings_items = build_settings_items(global_expanded_buckets)
            elif active_screen == 2:
                settings_items = build_module_settings_items(
                    module_expanded_buckets,
                    module_scope,
                    module_settings_overrides,
                    existing_module_settings,
                )
            else:
                settings_items = []

            if selected_index >= len(entries):
                selected_index = len(entries) - 1
            if active_screen == 1:
                if global_settings_selected_index >= len(settings_items):
                    global_settings_selected_index = max(0, len(settings_items) - 1)
            elif active_screen == 2:
                if module_settings_selected_index >= len(settings_items):
                    module_settings_selected_index = max(0, len(settings_items) - 1)

            if active_screen == 0:
                current_total = len(entries)
                current_selected = selected_index
                current_window_start = window_start
            elif active_screen == 1:
                current_total = len(settings_items)
                current_selected = global_settings_selected_index
                current_window_start = global_settings_window_start
            else:
                current_total = len(settings_items)
                current_selected = module_settings_selected_index
                current_window_start = module_settings_window_start

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
                module_scope,
            )
            current_window_start = clamp_window_start(
                current_window_start, current_selected, window_size, current_total
            )

            if active_screen == 0:
                window_start = current_window_start
            elif active_screen == 1:
                global_settings_window_start = current_window_start
            else:
                module_settings_window_start = current_window_start

            # Clear terminal and draw the updated preview.
            print("\x1b[2J\x1b[H", end="")
            screen_text = render_preview_screen_window(
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
                module_scope,
            )
            print(wrap_screen_text_for_terminal(screen_text))

            key = read_single_key()

            if help_overlay_active:
                help_overlay_active = False
                help_text = ""
                continue

            if confirm_dialog_active:
                if key == "\x1b":
                    confirm_dialog_active = False
                    edit_text = ""
                    continue
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

                    # Only persist configuration_confirmed=True for overwrite mode.
                    # For "save for current run", keep confirmation in-memory for this
                    # run but do not bake it into saved config JSON.
                    if confirm_choice_index == 1:
                        GPA.pc.set_configuration_confirmed(True)
                        finalize_save(confirm_choice_index)
                    else:
                        finalize_save(confirm_choice_index)
                        GPA.pc.set_configuration_confirmed(True)
                    break
                continue

            if scalar_editor_state is not None:
                input_buffer = scalar_editor_state.get("input_buffer", "")
                cursor_index = scalar_editor_state.get("cursor_index", 0)

                if key == "\x1b":
                    scalar_editor_state = None
                    edit_text = ""
                    continue
                if is_left_key(key):
                    scalar_editor_state["cursor_index"] = max(0, cursor_index - 1)
                elif is_right_key(key):
                    scalar_editor_state["cursor_index"] = min(
                        len(input_buffer), cursor_index + 1
                    )
                elif key == "\x7f":
                    if cursor_index > 0:
                        scalar_editor_state["input_buffer"] = (
                            input_buffer[: cursor_index - 1] + input_buffer[cursor_index:]
                        )
                        scalar_editor_state["cursor_index"] = cursor_index - 1
                elif is_delete_key(key):
                    if cursor_index < len(input_buffer):
                        scalar_editor_state["input_buffer"] = (
                            input_buffer[:cursor_index] + input_buffer[cursor_index + 1 :]
                        )
                elif key == "\r" or key == "\n":
                    setting_name = scalar_editor_state["setting_name"]
                    sample_value = scalar_editor_state["sample_value"]
                    try:
                        parsed = parse_value_from_text(
                            scalar_editor_state.get("input_buffer", ""), sample_value
                        )
                    except Exception as exc:
                        edit_text = f"Invalid value for {setting_name}: {exc}"
                        scalar_editor_state = None
                        continue

                    if scalar_editor_state.get("apply_to_module_scope", False):
                        set_module_scoped_value(
                            module_settings_overrides,
                            scalar_editor_state["scope_key"],
                            setting_name,
                            parsed,
                        )
                        ok, message = True, ""
                    else:
                        ok, message = apply_setting_value(setting_name, parsed)

                    scalar_editor_state = None
                    edit_text = message if not ok else ""
                    continue
                elif len(key) == 1 and key >= " ":
                    scalar_editor_state["input_buffer"] = (
                        input_buffer[:cursor_index] + key + input_buffer[cursor_index:]
                    )
                    scalar_editor_state["cursor_index"] = cursor_index + 1

                if scalar_editor_state is not None:
                    name = scalar_editor_state["setting_name"]
                    buffer_text = scalar_editor_state.get("input_buffer", "")
                    cursor = scalar_editor_state.get("cursor_index", len(buffer_text))
                    edit_text = (
                        f"EDIT - {name} - {render_text_with_cursor(buffer_text, cursor)} "
                        "(Enter save, Esc cancel)"
                    )
                continue

            if list_editor_state is not None:
                values = list_editor_state["values"]
                selected_list_index = list_editor_state["selected_index"]
                add_index = len(values)
                save_index = len(values) + 1
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
                    list_editor_state["selected_index"] = min(save_index, selected_list_index + 1)
                    continue
                if key == "\x7f" or is_delete_key(key):
                    if selected_list_index < len(values):
                        del values[selected_list_index]
                        if list_editor_state["selected_index"] > len(values) + 1:
                            list_editor_state["selected_index"] = len(values) + 1
                    continue
                if is_select_key(key):
                    if selected_list_index == save_index:
                        if list_editor_state.get("apply_to_module_scope", False):
                            if module_scope is None:
                                ok, message = False, "No module scope selected."
                            else:
                                set_module_scoped_value(
                                    module_settings_overrides,
                                    module_scope["scope_key"],
                                    list_editor_state["setting_name"],
                                    values,
                                )
                                ok, message = True, ""
                        else:
                            ok, message = apply_setting_value(
                                list_editor_state["setting_name"], values
                            )
                        list_editor_state = None
                        edit_text = message if not ok else ""
                        continue

                    edit_index = selected_list_index
                    initial_text = ""
                    if edit_index < len(values):
                        initial_text = str(values[edit_index])
                    list_editor_state["typing_active"] = True
                    list_editor_state["edit_index"] = edit_index
                    list_editor_state["input_buffer"] = initial_text
                    list_editor_state["cursor_index"] = len(initial_text)
                    continue
                continue

            if is_up_key(key):
                help_text = ""
                edit_text = ""
                if active_screen == 0:
                    selected_index = max(0, selected_index - 1)
                elif active_screen == 1:
                    global_settings_selected_index = max(
                        0, global_settings_selected_index - 1
                    )
                else:
                    module_settings_selected_index = max(
                        0, module_settings_selected_index - 1
                    )
                continue
            if is_down_key(key):
                help_text = ""
                edit_text = ""
                if active_screen == 0:
                    selected_index = min(len(entries) - 1, selected_index + 1)
                elif active_screen == 1:
                    global_settings_selected_index = min(
                        len(settings_items) - 1, global_settings_selected_index + 1
                    )
                else:
                    module_settings_selected_index = min(
                        len(settings_items) - 1, module_settings_selected_index + 1
                    )
                continue
            if is_page_up_key(key):
                help_text = ""
                edit_text = ""
                page_step = max(1, window_size - 1)
                if active_screen == 0:
                    selected_index = max(0, selected_index - page_step)
                    window_start = max(0, window_start - page_step)
                elif active_screen == 1:
                    global_settings_selected_index = max(
                        0, global_settings_selected_index - page_step
                    )
                    global_settings_window_start = max(
                        0, global_settings_window_start - page_step
                    )
                else:
                    module_settings_selected_index = max(
                        0, module_settings_selected_index - page_step
                    )
                    module_settings_window_start = max(
                        0, module_settings_window_start - page_step
                    )
                continue
            if is_page_down_key(key):
                help_text = ""
                edit_text = ""
                page_step = max(1, window_size - 1)
                if active_screen == 0:
                    selected_index = min(len(entries) - 1, selected_index + page_step)
                    window_start = min(len(entries) - 1, window_start + page_step)
                elif active_screen == 1:
                    global_settings_selected_index = min(
                        len(settings_items) - 1,
                        global_settings_selected_index + page_step,
                    )
                    global_settings_window_start = min(
                        len(settings_items) - 1,
                        global_settings_window_start + page_step,
                    )
                else:
                    module_settings_selected_index = min(
                        len(settings_items) - 1,
                        module_settings_selected_index + page_step,
                    )
                    module_settings_window_start = min(
                        len(settings_items) - 1,
                        module_settings_window_start + page_step,
                    )
                continue
            if is_left_key(key):
                help_text = ""
                edit_text = ""
                if active_screen == 0:
                    active_screen = 1
                elif active_screen == 1:
                    active_screen = 0
                else:
                    active_screen = 0
                    module_scope = None
                    module_expanded_buckets = {}
                    module_settings_selected_index = 0
                    module_settings_window_start = 0
                continue
            if is_right_key(key):
                help_text = ""
                edit_text = ""
                if active_screen == 0:
                    active_screen = 1
                elif active_screen == 1:
                    active_screen = 0
                else:
                    active_screen = 0
                    module_scope = None
                    module_expanded_buckets = {}
                    module_settings_selected_index = 0
                    module_settings_window_start = 0
                continue

            if key == "\x1b":
                help_text = ""
                edit_text = ""
                if active_screen == 2:
                    active_screen = 0
                    module_scope = None
                    module_expanded_buckets = {}
                    module_settings_selected_index = 0
                    module_settings_window_start = 0
                elif active_screen == 1:
                    active_screen = 0
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
                if is_select_key(key):
                    recursive_modes = build_recursive_modes(entries)
                    scope, warning = get_space_scope_for_entry(
                        selected_entry, recursive_modes
                    )
                    if scope is None:
                        edit_text = warning
                        continue
                    module_scope = scope
                    active_screen = 2
                    module_expanded_buckets = {}
                    module_settings_selected_index = 0
                    module_settings_window_start = 0
                    help_text = ""
                    edit_text = ""
                    continue
            elif active_screen == 1:
                if key == "e" and len(settings_items) > 0:
                    item = settings_items[global_settings_selected_index]
                    if item["type"] == "bucket":
                        global_expanded_buckets[item["bucket"]] = not item["is_expanded"]
                    continue
                if key == "h" and len(settings_items) > 0:
                    item = settings_items[global_settings_selected_index]
                    help_text = (
                        f"HELP - {get_item_display_name(item)} - "
                        f"{get_item_description(item)}"
                    )
                    help_overlay_active = True
                    edit_text = ""
                    continue
                if is_select_key(key) and len(settings_items) > 0:
                    item = settings_items[global_settings_selected_index]
                    if item["type"] == "bucket":
                        global_expanded_buckets[item["bucket"]] = not item["is_expanded"]
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
                            "apply_to_module_scope": False,
                        }
                        edit_text = (
                            f"EDIT - {setting_name} - list mode active"
                        )
                        continue

                    if isinstance(setting_value, (int, float)):
                        initial_text = str(setting_value)
                        scalar_editor_state = {
                            "setting_name": setting_name,
                            "sample_value": setting_value,
                            "input_buffer": initial_text,
                            "cursor_index": len(initial_text),
                            "apply_to_module_scope": False,
                            "scope_key": None,
                        }
                        edit_text = (
                            f"EDIT - {setting_name} - "
                            f"{render_text_with_cursor(initial_text, len(initial_text))} "
                            "(Enter save, Esc cancel)"
                        )
                        continue

                    edit_text = f"Setting {setting_name} edit is not supported."
                    continue
            else:
                if module_scope is None:
                    if key in ("h", " ", "e"):
                        edit_text = "Select a perforation target on screen 1 and press Space to choose scope."
                    continue

                if key == "e" and len(settings_items) > 0:
                    item = settings_items[module_settings_selected_index]
                    if item["type"] == "bucket":
                        module_expanded_buckets[item["bucket"]] = not item["is_expanded"]
                    continue
                if key == "h" and len(settings_items) > 0:
                    item = settings_items[module_settings_selected_index]
                    help_text = (
                        f"HELP - {get_item_display_name(item)} - "
                        f"{get_item_description(item)}"
                    )
                    help_overlay_active = True
                    edit_text = ""
                    continue
                if is_select_key(key) and len(settings_items) > 0:
                    item = settings_items[module_settings_selected_index]
                    if item["type"] == "bucket":
                        module_expanded_buckets[item["bucket"]] = not item["is_expanded"]
                        edit_text = ""
                        continue
                    if item["type"] != "setting":
                        edit_text = ""
                        continue

                    setting_name = item["name"]
                    setting_value = item["value"]
                    enum_options = get_enum_options(setting_name)
                    scope_key = module_scope["scope_key"]

                    def apply_module_setting(new_value):
                        set_module_scoped_value(
                            module_settings_overrides,
                            scope_key,
                            setting_name,
                            new_value,
                        )
                        return True, ""

                    if isinstance(setting_value, bool):
                        ok, message = apply_module_setting(not setting_value)
                        edit_text = message if not ok else ""
                        continue

                    if enum_options is not None and len(enum_options) > 1:
                        try:
                            index = enum_options.index(setting_value)
                        except ValueError:
                            index = -1
                        next_value = enum_options[(index + 1) % len(enum_options)]
                        ok, message = apply_module_setting(next_value)
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
                            "apply_to_module_scope": True,
                        }
                        edit_text = (
                            f"EDIT - {setting_name} - list mode active"
                        )
                        continue

                    if isinstance(setting_value, (int, float)):
                        initial_text = str(setting_value)
                        scalar_editor_state = {
                            "setting_name": setting_name,
                            "sample_value": setting_value,
                            "input_buffer": initial_text,
                            "cursor_index": len(initial_text),
                            "apply_to_module_scope": True,
                            "scope_key": scope_key,
                        }
                        edit_text = (
                            f"EDIT - {setting_name} - "
                            f"{render_text_with_cursor(initial_text, len(initial_text))} "
                            "(Enter save, Esc cancel)"
                        )
                        continue

                    edit_text = f"Setting {setting_name} edit is not supported."
                    continue

            if key.lower() == "s":
                edit_text = ""
                confirm_dialog_active = True
                confirm_choice_index = 0
                continue
    finally:
        if entered_alt_screen:
            exit_alternate_screen()
        GPA.pc.__dict__["_auto_persist_config"] = previous_auto_persist
