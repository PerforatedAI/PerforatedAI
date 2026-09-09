import sys
import termios
import tty
import shutil
import json
import os
import select
import re

from perforatedai import globals_perforatedai as GPA

try:
    import torch
except Exception:  # pragma: no cover - torch is a hard dependency at runtime
    torch = None


# ---------------------------------------------------------------------------
# Selection list plumbing
# ---------------------------------------------------------------------------
def dedupe_list(values):
    """Return a list with duplicates removed while preserving order."""
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def normalize_selection_conflicts():
    """Normalize id and name selection lists and remove direct conflicts."""
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

    Each entry also carries ``is_replaced_root`` (this module's class is in
    ``modules_to_replace`` and will be restructured before training) and
    ``in_replaced`` / ``replaced_root`` (this module lives under such a module,
    so its id will not exist at training time).
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

    replace_classes = tuple(GPA.pc.get_modules_to_replace())
    replaced_root_ids = []
    for entry in entries:
        entry["is_replaced_root"] = (
            len(replace_classes) > 0
            and isinstance(entry["module"], replace_classes)
        )
        if entry["is_replaced_root"]:
            replaced_root_ids.append(entry["id"])

    for entry in entries:
        entry["replaced_root"] = None
        entry["in_replaced"] = False
        for root_id in replaced_root_ids:
            if entry["id"].startswith(root_id + "."):
                entry["replaced_root"] = root_id
                entry["in_replaced"] = True
                break

    return entries


def get_id_mode(module_id):
    """Get explicit id-based mode for a module id ("perforated"/"tracked"/None)."""
    if module_id in GPA.pc.get_module_ids_to_perforate():
        return "perforated"
    if module_id in GPA.pc.get_module_ids_to_track():
        return "tracked"
    return None


def get_name_mode(module_type_name):
    """Get type-name-based mode for a module type ("perforated"/"tracked"/None)."""
    if module_type_name in GPA.pc.get_module_names_to_perforate():
        return "perforated"
    if module_type_name in GPA.pc.get_module_names_to_track():
        return "tracked"
    return None


def get_parent_module_id(module_id):
    """Return the direct parent module id for a dot-id module path."""
    parts = module_id.split(".")
    if len(parts) <= 2:
        return None
    return "." + ".".join(parts[1:-1])


def build_recursive_modes(entries):
    """Build recursive inherited modes for all entries.

    - if an ancestor has a recursive mode, inherit it (descendant override ignored)
    - otherwise an explicit id mode or explicit name mode sets this node's mode
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


def resolve_entry_modes(entries):
    """Resolve every entry's mode, source, and any overridden-by-ancestor state.

    Returns a map of module id to a dict with:
      - ``eff``:    "perforated" / "tracked" / None
      - ``source``: "id" / "type" / "inherited" / None
      - ``origin_id``: the id the mode was set on (for inherited rows)
      - ``overridden``: {"mode", "source"} when a direct setting here is ignored
    """
    resolved = {}
    by_id = {entry["id"]: entry for entry in entries}
    for entry in entries:
        module_id = entry["id"]
        parent_id = get_parent_module_id(module_id)
        parent = resolved.get(parent_id) if parent_id in by_id else None

        own_id_mode = get_id_mode(module_id)
        own_type_mode = get_name_mode(entry["type_name"])
        own_mode = own_id_mode or own_type_mode
        own_source = "id" if own_id_mode else ("type" if own_type_mode else None)

        record = {
            "eff": None,
            "source": None,
            "origin_id": module_id,
            "overridden": None,
        }
        if parent is not None and parent["eff"] is not None:
            record["eff"] = parent["eff"]
            record["source"] = "inherited"
            record["origin_id"] = parent["origin_id"]
            if own_mode is not None:
                record["overridden"] = {"mode": own_mode, "source": own_source}
        elif own_mode is not None:
            record["eff"] = own_mode
            record["source"] = own_source
            record["origin_id"] = module_id
        resolved[module_id] = record
    return resolved


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
COLOR_PERFORATE = "00A5A5"
COLOR_TRACK = "8A95A0"
COLOR_ATTENTION = "FD4D00"
COLOR_ACCENT = "D9A441"


def get_color_hex_for_mode(mode):
    """Map a resolved mode to its color hex value."""
    if mode == "perforated":
        return COLOR_PERFORATE
    if mode == "tracked":
        return COLOR_TRACK
    return "000000"


def format_human_count(value):
    """Format integer counts into compact human-readable units (K/M/B)."""
    if value < 1000:
        return str(value)
    if value < 1000000:
        scaled = value / 1000.0
        text = f"{scaled:.1f}"
        return (text[:-2] if text.endswith(".0") else text) + "K"
    if value < 1000000000:
        scaled = value / 1000000.0
        text = f"{scaled:.1f}"
        return (text[:-2] if text.endswith(".0") else text) + "M"
    scaled = value / 1000000000.0
    text = f"{scaled:.1f}"
    return (text[:-2] if text.endswith(".0") else text) + "B"


def module_has_direct_parameters(module):
    """Check whether a module directly owns any parameters."""
    for _name, _param in module.named_parameters(recurse=False):
        return True
    return False


def _ansi(text, code):
    return f"\x1b[{code}m{text}\x1b[0m"


def color_text(text, hex_color):
    """Render text in an ANSI truecolor foreground."""
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m"


def make_color_square(hex_color):
    """Create one colored block character using ANSI truecolor."""
    return color_text("█", hex_color)


def dim(text):
    """Render text dimmed."""
    return _ansi(text, "90")


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
    """Wrap one ANSI-colored line on word boundaries for terminal display."""
    if terminal_columns < 1:
        return [line]

    plain = strip_ansi(line)
    if len(plain) <= terminal_columns:
        return [line]

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
    """Render text with a CLI-style inverted-character cursor."""
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


# ---------------------------------------------------------------------------
# Key token predicates
# ---------------------------------------------------------------------------
def is_up_key(key):
    if key in ("k",):
        return True
    if not key.startswith("\x1b"):
        return False
    return key in ("\x1b[A", "\x1bOA") or (key.startswith("\x1b[") and key.endswith("A"))


def is_down_key(key):
    if key in ("j",):
        return True
    if not key.startswith("\x1b"):
        return False
    return key in ("\x1b[B", "\x1bOB") or (key.startswith("\x1b[") and key.endswith("B"))


def is_left_key(key):
    if not key.startswith("\x1b"):
        return False
    return key in ("\x1b[D", "\x1bOD") or (key.startswith("\x1b[") and key.endswith("D"))


def is_right_key(key):
    if not key.startswith("\x1b"):
        return False
    return key in ("\x1b[C", "\x1bOC") or (key.startswith("\x1b[") and key.endswith("C"))


def is_page_up_key(key):
    return key == "\x1b[5~" or (key.startswith("\x1b[") and "[5" in key and key.endswith("~"))


def is_page_down_key(key):
    return key == "\x1b[6~" or (key.startswith("\x1b[") and "[6" in key and key.endswith("~"))


def is_delete_key(key):
    return key == "\x1b[3~" or (key.startswith("\x1b[") and "[3" in key and key.endswith("~"))


def is_enter_key(key):
    return key in ("\r", "\n")


def is_tab_key(key):
    return key == "\t"


def is_select_key(key):
    """Space/Enter — used only inside the inline value editors."""
    return key in (" ", "\r", "\n")


# ---------------------------------------------------------------------------
# Mode setters
# ---------------------------------------------------------------------------
def set_module_id_mode(module_id, mode):
    """Set (or toggle off) an id-based mode, enforcing id-list exclusivity."""
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
    """Set (or toggle off) a type-name-based mode, enforcing name-list exclusivity."""
    names_perforate = dedupe_list(GPA.pc.get_module_names_to_perforate())
    names_track = dedupe_list(GPA.pc.get_module_names_to_track())

    if mode == "perforated":
        if module_type_name in names_perforate:
            names_perforate = [v for v in names_perforate if v != module_type_name]
        else:
            names_perforate.append(module_type_name)
            names_track = [v for v in names_track if v != module_type_name]
    elif mode == "tracked":
        if module_type_name in names_track:
            names_track = [v for v in names_track if v != module_type_name]
        else:
            names_track.append(module_type_name)
            names_perforate = [v for v in names_perforate if v != module_type_name]

    GPA.pc.set_module_names_to_perforate(names_perforate)
    GPA.pc.set_module_names_to_track(names_track)


def clear_module_mode(entry):
    """Clear a direct id rule on this module; if none, clear its type rule.

    Returns a short human message describing what was cleared.
    """
    module_id = entry["id"]
    type_name = entry["type_name"]

    ids_perforate = [v for v in GPA.pc.get_module_ids_to_perforate() if v != module_id]
    ids_track = [v for v in GPA.pc.get_module_ids_to_track() if v != module_id]
    if (
        module_id in GPA.pc.get_module_ids_to_perforate()
        or module_id in GPA.pc.get_module_ids_to_track()
    ):
        GPA.pc.set_module_ids_to_perforate(ids_perforate)
        GPA.pc.set_module_ids_to_track(ids_track)
        return f"cleared {module_id}"

    if (
        type_name in GPA.pc.get_module_names_to_perforate()
        or type_name in GPA.pc.get_module_names_to_track()
    ):
        GPA.pc.set_module_names_to_perforate(
            [v for v in GPA.pc.get_module_names_to_perforate() if v != type_name]
        )
        GPA.pc.set_module_names_to_track(
            [v for v in GPA.pc.get_module_names_to_track() if v != type_name]
        )
        return f"cleared the {type_name} type rule"

    return "nothing to clear on this module"


# ---------------------------------------------------------------------------
# Type rules / budget / attention
# ---------------------------------------------------------------------------
def type_rule_entries(entries):
    """Active type-name rules as (type_name, mode, live count) tuples."""
    counts = {}
    for entry in entries:
        if entry["in_replaced"]:
            continue
        counts[entry["type_name"]] = counts.get(entry["type_name"], 0) + 1

    rules = []
    for type_name in sorted(GPA.pc.get_module_names_to_perforate()):
        rules.append((type_name, "perforated", counts.get(type_name, 0)))
    for type_name in sorted(GPA.pc.get_module_names_to_track()):
        rules.append((type_name, "tracked", counts.get(type_name, 0)))
    return rules


def unset_module_entries(entries, resolved):
    """Entries that own parameters but have no resolved mode (need attention)."""
    unset = []
    for entry in entries:
        if entry["in_replaced"] or entry["is_replaced_root"]:
            continue
        if entry["direct_param_count"] <= 0:
            continue
        if resolved[entry["id"]]["eff"] is None:
            unset.append(entry)
    return unset


def build_budget_line(entries, resolved):
    """The perforation-budget line: parameter cost of the current selection."""
    added = 0
    total_model_params = 0
    seen_added = set()
    seen_total = set()

    for entry in entries:
        for _name, parameter in entry["module"].named_parameters(recurse=False):
            pid = id(parameter)
            if pid not in seen_total:
                seen_total.add(pid)
                total_model_params += parameter.numel()

        if resolved[entry["id"]]["eff"] == "perforated":
            for _name, parameter in entry["module"].named_parameters(recurse=False):
                pid = id(parameter)
                if pid not in seen_added:
                    seen_added.add(pid)
                    added += parameter.numel()

    pct = (100.0 * added / total_model_params) if total_model_params else 0.0
    return (
        f"Perforation budget:  +{format_human_count(added)} params per dendrite cycle"
        f"  ·  model {format_human_count(total_model_params)}"
        f"  (+{pct:.1f}% / cycle)"
    )


# ---------------------------------------------------------------------------
# Run settings metadata
# ---------------------------------------------------------------------------
RUNTIME_ONLY_CONFIG_KEYS = {"config_file"}

DESCRIPTION_FILE_NAME = "configuration_descriptions.json"
DESCRIPTION_CACHE = None

LABEL_IMPACTFUL = "impactful"
LABEL_SUPPORTING = "supporting"


def _normalize_label(label_text):
    """Normalize a label to one of the two tiers: impactful / supporting."""
    if not isinstance(label_text, str):
        return LABEL_SUPPORTING
    compact = " ".join(label_text.strip().split()).lower()
    if compact in ("impactful", "impactful hyperparameters"):
        return LABEL_IMPACTFUL
    return LABEL_SUPPORTING


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
                            setting_descriptions[setting_name] = setting_payload.get(
                                "description", ""
                            )
                            setting_labels[setting_name] = _normalize_label(
                                setting_payload.get("label")
                            )
                        elif isinstance(setting_payload, str):
                            setting_descriptions[setting_name] = setting_payload
                            setting_labels[setting_name] = LABEL_SUPPORTING
                settings_by_bucket[bucket_name] = ordered_setting_names
            elif isinstance(bucket_payload, str):
                bucket_descriptions[bucket_name] = bucket_payload

    DESCRIPTION_CACHE = {
        "buckets": bucket_descriptions,
        "bucket_order": bucket_order,
        "settings": setting_descriptions,
        "setting_labels": setting_labels,
        "setting_to_bucket": setting_to_bucket,
        "settings_by_bucket": settings_by_bucket,
    }
    return DESCRIPTION_CACHE


def get_setting_label(setting_name):
    """Get the tier (impactful / supporting) for one setting."""
    setting_labels = load_description_data().get("setting_labels", {})
    if setting_name in setting_labels:
        return _normalize_label(setting_labels[setting_name])
    return LABEL_SUPPORTING


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


SWITCH_MODE_NAMES = [
    "DOING_SWITCH_EVERY_TIME",
    "DOING_HISTORY",
    "DOING_FIXED_SWITCH",
    "DOING_NO_SWITCH",
]
FORWARD_FUNCTION_NAMES = ["relu", "tanh", "sigmoid"]


def _switch_mode_values():
    return [
        GPA.pc.DOING_SWITCH_EVERY_TIME,
        GPA.pc.DOING_HISTORY,
        GPA.pc.DOING_FIXED_SWITCH,
        GPA.pc.DOING_NO_SWITCH,
    ]


def get_enum_options(setting_name):
    """Return the ordered set of values for an enum-like setting, else None."""
    if setting_name == "switch_mode":
        return list(_switch_mode_values())
    if setting_name == "pai_forward_function":
        return list(FORWARD_FUNCTION_NAMES)
    return None


def format_run_setting_value(setting_name, value):
    """Display string for a run setting, naming enum values."""
    if setting_name == "switch_mode":
        try:
            return SWITCH_MODE_NAMES[list(_switch_mode_values()).index(value)]
        except (ValueError, IndexError):
            return str(value)
    if setting_name == "pai_forward_function":
        name = getattr(value, "__name__", None)
        if name in FORWARD_FUNCTION_NAMES:
            return name
        return format_setting_value(value)
    return format_setting_value(value)


def cycle_enum_setting(setting_name, current_value):
    """Compute and apply the next value for an enum setting."""
    options = get_enum_options(setting_name)
    if not options:
        return False, f"{setting_name} is not an enum setting."

    if setting_name == "pai_forward_function":
        current_name = getattr(current_value, "__name__", None)
        index = options.index(current_name) if current_name in options else -1
        next_name = options[(index + 1) % len(options)]
        next_value = getattr(torch, next_name) if torch is not None else next_name
        return apply_setting_value(setting_name, next_value)

    index = options.index(current_value) if current_value in options else -1
    next_value = options[(index + 1) % len(options)]
    return apply_setting_value(setting_name, next_value)


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


def apply_setting_value(setting_name, value):
    """Apply one setting value through a setter when available."""
    setter = getattr(GPA.pc, f"set_{setting_name}", None)
    if setter is not None:
        setter(value)
        return True, ""
    if hasattr(GPA.pc, setting_name) and not callable(getattr(GPA.pc, setting_name)):
        setattr(GPA.pc, setting_name, value)
        return True, ""
    return False, f"Setting {setting_name} is not editable."


def parse_value_from_text(text, sample_value):
    """Parse user text into the same type as a sample value where possible."""
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


def get_all_global_parameters():
    """Collect JSON-declared run settings as name -> current value."""
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


def build_run_items(expanded_buckets):
    """Build the Run settings list: bucket rows, tier-sorted settings, dividers."""
    values = get_all_global_parameters()
    description_data = load_description_data()
    items = []

    for bucket_name in description_data.get("bucket_order", []):
        names = [
            name
            for name in description_data.get("settings_by_bucket", {}).get(bucket_name, [])
            if name not in RUNTIME_ONLY_CONFIG_KEYS
        ]
        if not names:
            continue

        is_expanded = expanded_buckets.get(bucket_name, False)
        items.append(
            {
                "type": "bucket",
                "bucket": bucket_name,
                "is_expanded": is_expanded,
                "count": len(names),
            }
        )
        if not is_expanded:
            continue

        impactful = [n for n in names if get_setting_label(n) == LABEL_IMPACTFUL]
        supporting = [n for n in names if get_setting_label(n) != LABEL_IMPACTFUL]

        def _setting_item(setting_name):
            return {
                "type": "setting",
                "bucket": bucket_name,
                "name": setting_name,
                "value": values.get(setting_name, "<unavailable>"),
            }

        for setting_name in impactful:
            items.append(_setting_item(setting_name))
        if impactful and supporting:
            items.append({"type": "divider", "bucket": bucket_name})
        for setting_name in supporting:
            items.append(_setting_item(setting_name))

    return items


# ---------------------------------------------------------------------------
# Overrides (per-target settings)
# ---------------------------------------------------------------------------
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
    """Merged override values for one scope: saved data plus in-session edits."""
    values = {}
    if scope_key in existing_module_settings:
        values.update(existing_module_settings[scope_key])
    if scope_key in module_settings_overrides:
        values.update(module_settings_overrides[scope_key])
    return values


def set_module_scoped_value(module_settings_overrides, scope_key, setting_name, value):
    """Write one override value into in-session state."""
    module_settings_overrides.setdefault(scope_key, {})[setting_name] = value


def get_override_scope_for_entry(entry, resolved):
    """Determine the override scope for a target, or a reason it is blocked."""
    record = resolved[entry["id"]]
    if entry["is_replaced_root"] or entry["in_replaced"]:
        return None, (
            "Overrides are not available inside a module that will be restructured."
        )
    if record["eff"] is None:
        return None, (
            "Overrides are only for perforated modules. Perforate this one first "
            "(p), or track it if you just want it monitored."
        )
    if record["source"] == "inherited":
        origin = record["origin_id"]
        return None, (
            f"This module inherits {origin}'s mode. Open overrides on {origin} to "
            "change settings for the whole subtree."
        )
    if record["eff"] != "perforated":
        return None, "Overrides are only for perforated modules — this one is tracked."

    if record["source"] == "type":
        return (
            {"kind": "type", "scope_key": entry["type_name"]},
            None,
        )
    return ({"kind": "id", "scope_key": entry["id"]}, None)


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


CUSTOMIZABLE_SETTING_NAMES = list(GPA.PAIConfig._CUSTOMIZABLE.keys())


def get_setting_description(setting_name):
    """Help text for one setting."""
    descriptions = load_description_data().get("settings", {})
    return descriptions.get(setting_name, f"No description available for {setting_name}.")


def get_bucket_description(bucket_name):
    descriptions = load_description_data().get("buckets", {})
    return descriptions.get(bucket_name, "No description available.")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
HR = "  " + "─" * 70


def terminal_size():
    return shutil.get_terminal_size(fallback=(120, 40))


def render_tab_bar(active_screen):
    """The `[ Targets | Run settings ]` tab bar with the active screen lit."""
    targets = "Targets"
    run = "Run settings"
    if active_screen == "targets":
        targets = color_text(targets, COLOR_ACCENT)
    else:
        run = color_text(run, COLOR_ACCENT)
    return f"[ {targets} │ {run} ]" + " " * 12 + dim("[Tab] switch panel    ? help")


MODE_VERB = {"perforated": "perforate", "tracked": "track"}


def mode_verb(mode):
    """Present-tense verb form of a resolved mode for user-facing display."""
    return MODE_VERB.get(mode, str(mode))


def make_target_marker(entry, record):
    """(display, plain) marker for one target row, per D8."""
    if entry["in_replaced"]:
        return "     ", "     "

    mode = record["eff"]
    if mode is None:
        if entry["direct_param_count"] > 0:
            glyph_display = color_text("!", COLOR_ATTENTION)
            block_display = color_text("██", COLOR_ATTENTION)
            return f"{glyph_display} {block_display} ", "! ██ "
        return "     ", "     "

    block_display = make_block(mode)
    if record["source"] == "inherited":
        return f"↳ {block_display} ", "↳ ██ "
    if record["source"] == "type":
        return f"* {block_display} ", "* ██ "
    return f"  {block_display} ", "  ██ "


def make_block(mode):
    return color_text("██", get_color_hex_for_mode(mode))


def render_targets_lines(entries, visible_entries, selected_index, resolved, expanded_replaced):
    """Header + tree + footer for the Targets screen, as a list of lines."""
    need = len(unset_module_entries(entries, resolved))
    rules = type_rule_entries(entries)

    lines = [render_tab_bar("targets"), ""]
    lines.append(
        dim(
            "  perforate = add dendrites   ·   track = no dendrites, just counted"
            "   ·   every parameter needs one"
        )
    )
    lines.append("")

    hint = color_text(f"[y] type rules ({len(rules)})", COLOR_ACCENT)
    if need > 0:
        noun = "module" if need == 1 else "modules"
        verb = "needs" if need == 1 else "need"
        status = color_text(f"  ⚠ {need} {noun} still {verb} a mode", COLOR_ATTENTION)
    else:
        status = color_text("  ✓ every parameter is perforated or tracked", COLOR_PERFORATE)
    pad = max(2, 58 - len(strip_ansi(status)))
    lines.append(status + " " * pad + hint)
    lines.append(HR)

    def entry_lines(entry, selected):
        cursor = color_text("> ", "E8EEEC") if selected else "  "
        indent = "  " * entry["depth"]
        record = resolved[entry["id"]]

        if entry["is_replaced_root"]:
            child_count = sum(1 for e in entries if e["replaced_root"] == entry["id"])
            glyph = color_text("↯", COLOR_ACCENT)
            is_open = expanded_replaced.get(entry["id"], False)
            toggle = "[← collapse]" if is_open else "[→ expand]"
            return [
                f"{cursor}{glyph}   {indent}{entry['id']}  "
                f"{dim('[' + entry['type_name'] + ']')}   "
                f"{dim('will be restructured for PAI before training')}",
                "      "
                + dim(
                    f"└ {child_count} modules below — ids won't exist at "
                    f"train time · target by type (P/T), not id  "
                )
                + color_text(toggle, COLOR_ACCENT),
            ]

        marker_display, marker_plain = make_target_marker(entry, record)

        if entry["in_replaced"]:
            id_part = dim(f"{entry['id']}  [{entry['type_name']}] (id stale)")
            return [f"{cursor}{marker_display} {indent}{id_part}"]

        id_part = f"{entry['id']}  {dim('[' + entry['type_name'] + ']')}"

        tail = ""
        if record["eff"] is not None:
            mode_color = get_color_hex_for_mode(record["eff"])
            tail = color_text(f"mode={mode_verb(record['eff'])}", mode_color)
            if record["source"] == "inherited":
                tail += dim(f"  (via {record['origin_id']})")
                if record["overridden"] is not None:
                    tail += dim(
                        f"  — your {mode_verb(record['overridden']['mode'])} here is ignored"
                    )
        elif entry["direct_param_count"] > 0:
            tail = color_text("needs a mode", COLOR_ATTENTION)

        params = (
            dim(f"   params={format_human_count(entry['direct_param_count'])}")
            if entry["direct_param_count"] > 0
            else ""
        )

        lead = (
            2 + len(marker_plain) + 1 + len(indent) + len(entry["id"])
            + 2 + len(entry["type_name"]) + 2
        )
        gap = max(2, 46 - lead)
        return [
            f"{cursor}{marker_display} {indent}{id_part}" + " " * gap + tail + params
        ]

    body_lines = []
    focus = (0, 1)
    for i, entry in enumerate(visible_entries):
        rendered = entry_lines(entry, i == selected_index)
        if i == selected_index:
            focus = (len(body_lines), len(body_lines) + len(rendered))
        body_lines.extend(rendered)

    footer = [
        HR,
        dim(build_budget_line(entries, resolved)),
        "",
        dim(
            "p/t perforate·track this module   P/T whole type   x clear   "
            "h legend   ↑↓/jk move   s start"
        ),
    ]
    return lines, body_lines, footer, focus


def render_run_settings_lines(items, selected_index, describe_name):
    """Header + item body + footer for the Run settings screen."""
    lines = [
        render_tab_bar("run"),
        "",
        dim("  Run settings — run-wide configuration"),
        HR,
    ]

    body_lines = []
    for i, item in enumerate(items):
        selected = i == selected_index
        cursor = color_text("> ", "E8EEEC") if selected else "  "
        if item["type"] == "bucket":
            glyph = color_text("▾" if item["is_expanded"] else "▸", COLOR_ACCENT)
            body_lines.append(
                f"{cursor}{glyph} {item['bucket']} {dim('(' + str(item['count']) + ')')}"
            )
        elif item["type"] == "divider":
            body_lines.append("     " + dim("─ rarely changed ─"))
        else:
            value_text = format_run_setting_value(item["name"], item["value"])
            hint = ""
            if get_enum_options(item["name"]) is not None:
                hint = dim("  (Enter cycles)")
            elif isinstance(item["value"], bool):
                hint = dim("  (Enter toggles)")
            body_lines.append(f"{cursor}  {item['name']} = {value_text}{hint}")

    footer_line = dim(
        "→ expand · ← collapse   Enter edit   h describe   "
        "↑↓/jk move   Tab switch panel   s start"
    )
    if describe_name:
        footer_line = dim(f"{describe_name} — {get_setting_description(describe_name)}")
    footer = [HR, footer_line]
    focus = (selected_index, selected_index + 1)
    return lines, body_lines, footer, focus


def compose_scrolling_screen(header_lines, body_lines, footer_lines, window_start, focus=None):
    """Join a header + windowed body + footer into one screen string."""
    size = terminal_size()
    columns = size.columns
    reserved = get_visual_line_count(header_lines, columns) + get_visual_line_count(
        footer_lines, columns
    )
    available = max(1, size.lines - reserved - 3)

    total = len(body_lines)

    # Scroll the window so the focused (selected) rows stay visible.
    if focus is not None:
        focus_start, focus_end = focus
        if focus_start < window_start:
            window_start = focus_start
        if focus_end > window_start + available:
            window_start = focus_end - available

    window_start = max(0, min(window_start, max(0, total - available)))
    window_end = min(total, window_start + available)

    out = list(header_lines)
    if window_start > 0:
        out.append(dim("  ^^^^ more above ^^^^"))
    out.extend(body_lines[window_start:window_end])
    if window_end < total:
        out.append(dim("  vvvv more below vvvv"))
    out.extend(footer_lines)
    return "\n".join(out), window_start


def render_static_overlay(lines):
    return "\n".join(lines)


def render_type_rules_overlay(entries):
    rules = type_rule_entries(entries)
    rows = []
    if rules:
        for type_name, mode, count in rules:
            plain = f"{type_name} → {mode_verb(mode)}  ({count})"
            html = (
                type_name
                + " "
                + color_text(f"→ {mode_verb(mode)}", get_color_hex_for_mode(mode))
                + dim(f"  ({count})")
            )
            rows.append((plain, html))
    else:
        rows.append(("none yet — P / T on a row sets one",
                     dim("none yet — P / T on a row sets one")))

    width = max(38, max(len(plain) for plain, _ in rows) + 2)
    box = ["  " + dim("┌─ Type rules " + "─" * max(1, width - 12) + "┐")]
    for plain, html in rows:
        box.append("  " + dim("│ ") + html + " " * (width - len(plain) - 1) + dim("│"))
    box.append("  " + dim("└" + "─" * width + "┘"))
    box.append("  " + color_text("y or Esc to close", COLOR_ACCENT))
    return box


def render_overrides_overlay(scope, ov_selected_index, module_settings_overrides,
                             existing_module_settings, describe_name):
    if scope["kind"] == "id":
        title = f"Overrides — {scope['scope_key']}"
    else:
        title = f"Overrides — every {scope['scope_key']}"

    scoped = get_scope_values(
        scope["scope_key"], module_settings_overrides, existing_module_settings
    )

    lines = [render_tab_bar("targets"), "", "  " + color_text(title, COLOR_ACCENT), HR, ""]
    if not CUSTOMIZABLE_SETTING_NAMES:
        lines.append("  " + dim("No per-target settings are customizable in this build."))
    for i, name in enumerate(CUSTOMIZABLE_SETTING_NAMES):
        overridden = name in scoped
        value = scoped[name] if overridden else get_global_setting_value(name)
        cursor = color_text("> ", "E8EEEC") if i == ov_selected_index else "  "
        tag = dim("(overridden here)") if overridden else dim("(from run settings)")
        lines.append(
            f"{cursor}{name} = {format_run_setting_value(name, value)}  {tag}"
        )

    lines.append("")
    footer_line = dim(
        "Enter edit · h describe · ↑↓/jk move · Esc / x close"
    )
    if describe_name:
        footer_line = dim(f"{describe_name} — {get_setting_description(describe_name)}")
    lines.append(footer_line)
    return lines


def render_save_overlay(entries, resolved, save_selected_index, config_target_path):
    unset = unset_module_entries(entries, resolved)
    lines = ["", "  " + color_text("Save configuration", "E8EEEC"), ""]

    if unset:
        noun = "module has" if len(unset) == 1 else "modules have"
        lines.append(
            color_text(
                f"  ⚠  {len(unset)} {noun} parameters but no mode",
                COLOR_ATTENTION,
            )
        )
        lines.append("")
        for entry in unset[:5]:
            lines.append(f"      {dim(entry['id'] + '   [' + entry['type_name'] + ']')}")
        if len(unset) > 5:
            lines.append(f"      {dim('(+' + str(len(unset) - 5) + ' more)')}")
        lines.append("")
        lines.append(
            dim(
                "  These parameters get no dendrites and are not tracked. "
                "Usually a mistake."
            )
        )
        lines.append("")
        options = [
            "Keep editing",
            f"Start anyway — I know these {len(unset)} have no mode",
        ]
    else:
        lines.append(dim("  Start training with this configuration:"))
        lines.append("")
        options = [
            ("just this run", "don't touch my saved config"),
            (
                "and save it as default",
                f"write it to {config_target_path}, skip this screen next time",
            ),
            ("Keep editing", ""),
        ]

    for i, option in enumerate(options):
        marked = i == save_selected_index
        pointer = color_text("  ▸ ", "E8EEEC") if marked else "    "
        if isinstance(option, tuple):
            label, detail = option
            text = color_text(label, "E8EEEC") if marked else label
            if detail:
                text += dim(f"  — {detail}")
            lines.append(pointer + text)
        else:
            text = color_text(option, "E8EEEC") if marked else dim(option)
            lines.append(pointer + text)

    lines.append("")
    lines.append("")
    lines.append(
        dim("  ↑↓/jk move · Enter choose · Esc / x back to editing")
    )
    return lines


def render_help_overlay():
    p = lambda t: color_text(t, COLOR_PERFORATE)
    tr = lambda t: color_text(t, COLOR_TRACK)
    return [
        "",
        "  " + color_text("Keys", "E8EEEC"),
        "",
        dim("  Targets screen"),
        "   p / t     perforate / track this module",
        "   P / T     perforate / track every module of this type",
        "   x         clear this module (falls back to its type rule)",
        "   → / ←     expand / collapse a module that will be restructured (↯)",
        "   y         show / hide the type-rules list",
        "   h         show the marker & colour legend",
        "   Enter     open Overrides for a perforated module",
        "",
        dim("  Run settings screen"),
        "   → / ←     expand / collapse a bucket",
        "   Enter     edit the highlighted setting",
        "   h         describe the highlighted setting",
        "",
        dim("  Everywhere"),
        "   ↑↓ / jk   move        Tab  switch panel",
        "   s         save & start training",
        "   q         quit without configuring",
        "",
        dim("  Modes"),
        "   " + p("perforate") + " = dendrites are added here during training",
        "   " + tr("track") + "     = no dendrites, the parameters are just counted",
        "   " + dim("↳ inherited") + " = a mode set on an ancestor applies to the whole subtree",
        "   " + dim("(see the h legend on the Targets panel for how modes resolve)"),
        "",
        color_text("  ? or Esc to close", COLOR_ACCENT),
    ]


def render_legend_overlay():
    """Describe the Targets-screen markers and colours."""
    perf = color_text("██", COLOR_PERFORATE)
    trk = color_text("██", COLOR_TRACK)
    att = color_text("██", COLOR_ATTENTION)
    return [
        "",
        "  " + color_text("Targets legend", "E8EEEC"),
        "",
        dim("  Mode blocks"),
        f"   {perf}        " + color_text("perforate", COLOR_PERFORATE)
        + " — dendrites are added here during training",
        f"   {trk}        " + color_text("track", COLOR_TRACK)
        + " — no dendrites, the parameters are just counted",
        "",
        dim("  Marker prefixes"),
        f"     {perf}     mode set directly on this module (by id)",
        f"   * {perf}     set by type name — every module of this class",
        f"   ↳ {perf}     inherited from an ancestor; applies to the whole subtree",
        f"   ! {att}     has parameters but no mode — needs attention",
        "   ↯         will be restructured for PAI before training;",
        "             target the modules inside it by type (P/T), not by id",
        "   (none)    structural container with no parameters of its own",
        "",
        dim("  Which mode wins  (highest priority first)"),
        "   1. inherited — a mode on an ancestor applies to its whole",
        "      subtree; a mode set directly on a descendant is ignored",
        "      while the ancestor's mode is in effect",
        "   2. by id — a mode set on this exact module",
        "   3. by type — a mode set on the module's class name",
        "   So: inherited  >  by id  >  by type.",
        "",
        color_text("  h or Esc to close", COLOR_ACCENT),
    ]


def render_quit_overlay():
    return [
        "",
        "",
        "",
        "  " + color_text("Quit without configuring?", "E8EEEC"),
        "",
        dim("  Training will not start. Re-run to configure again, or set"),
        dim("  configuration_confirmed=True to skip this screen."),
        "",
        "  " + color_text("[y] quit", COLOR_ATTENTION) + "     " + dim("[n] keep configuring"),
    ]


# ---------------------------------------------------------------------------
# Key input
# ---------------------------------------------------------------------------
def read_single_key():
    """Read one keypress, including arrow keys, from stdin.

    Ctrl-C raises KeyboardInterrupt so the caller can exit cleanly.
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

        sequence = bytearray(first)
        ready, _, _ = select.select([file_descriptor], [], [], 0.05)
        if not ready:
            return "\x1b"
        sequence.extend(os.read(file_descriptor, 1))

        if sequence[1] in (ord("["), ord("O")):
            ready, _, _ = select.select([file_descriptor], [], [], 0.15)
            if ready:
                sequence.extend(os.read(file_descriptor, 1))

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
    """Switch terminal to alternate screen buffer for the interactive UI."""
    print("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H", end="", flush=True)


def exit_alternate_screen():
    """Return terminal to the normal screen buffer."""
    print("\x1b[?25h\x1b[?1049l", end="", flush=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _config_target_path():
    """Human path where 'save as default' would write."""
    config_file = GPA.pc.get_config_file()
    if config_file:
        return config_file
    run_config_path = GPA.pc.get_run_config_path()
    if run_config_path:
        try:
            return os.path.relpath(run_config_path, os.getcwd())
        except Exception:
            return run_config_path
    save_name = GPA.pc.get_save_name() or "PAI"
    return f"{save_name}/{save_name}_config.json"


def _move(index, delta, length):
    if length <= 0:
        return 0
    return max(0, min(length - 1, index + delta))


def set_perforation_targets(model):
    """Interactive configuration TUI for perforation targets and run settings.

    Two screens (Tab to switch): Targets (the module tree) and Run settings.
    See the `?` overlay for the full key reference.
    """
    previous_auto_persist = GPA.pc.__dict__.get("_auto_persist_config", True)
    GPA.pc.__dict__["_auto_persist_config"] = False
    entered_alt_screen = False
    quit_requested = False

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

        active_screen = "targets"
        overlay = None
        status_message = ""

        target_selected_index = 0
        target_window_start = 0
        expanded_replaced = {}

        run_selected_index = 0
        run_window_start = 0
        expanded_buckets = {}
        describe_name = ""

        ov_scope = None
        ov_selected_index = 0
        module_settings_overrides = {}
        existing_module_settings = load_existing_module_settings()

        save_selected_index = 0

        scalar_editor_state = None
        list_editor_state = None

        def finalize_save(choice_index):
            if choice_index == 0:
                GPA.pc.persist_config_outputs(overwrite_config_file=False)
                persist_module_settings_updates(
                    module_settings_overrides, overwrite_config_file=False
                )
                return

            config_file = GPA.pc.get_config_file()
            if not config_file:
                current_save_name = GPA.pc.get_save_name() or "PAI"
                relative_config_name = _config_target_path()
                exit_alternate_screen()
                print("\x1b[2J\x1b[H", end="")
                print("No config_file is set.")
                print(
                    "Enter a local JSON filename/path to create a reusable "
                    "configuration file."
                )
                print(
                    f"You currently have save_name='{current_save_name}'. Choose "
                    f"'{relative_config_name}' to skip this screen next time without "
                    "changing your perforate_model call."
                )
                while True:
                    filename = input("Config filename/path: ").strip()
                    if filename:
                        GPA.pc.__dict__["_config_file"] = filename
                        break
                    print("A filename is required to save a default configuration.")
                enter_alternate_screen()

            GPA.pc.persist_config_outputs(overwrite_config_file=True)
            persist_module_settings_updates(
                module_settings_overrides, overwrite_config_file=True
            )

        def visible_targets():
            out = []
            for entry in entries:
                if entry["in_replaced"] and not expanded_replaced.get(
                    entry["replaced_root"], False
                ):
                    continue
                out.append(entry)
            return out

        while True:
            normalize_selection_conflicts()
            resolved = resolve_entry_modes(entries)

            # ---- editors capture input first -------------------------------
            if scalar_editor_state is not None:
                _render_editor_frame(scalar_editor_state)
                key = read_single_key()
                done, status_message = _handle_scalar_editor(
                    scalar_editor_state, key, module_settings_overrides
                )
                if done:
                    scalar_editor_state = None
                continue

            if list_editor_state is not None:
                _render_editor_frame(list_editor_state)
                key = read_single_key()
                done, status_message = _handle_list_editor(
                    list_editor_state, key, module_settings_overrides
                )
                if done:
                    list_editor_state = None
                continue

            # ---- draw the current screen ---------------------------------
            print("\x1b[2J\x1b[H", end="")

            if overlay == "help":
                screen_text = render_static_overlay(render_help_overlay())
            elif overlay == "legend":
                screen_text = render_static_overlay(render_legend_overlay())
            elif overlay == "quit":
                screen_text = render_static_overlay(render_quit_overlay())
            elif overlay == "save":
                screen_text = render_static_overlay(
                    render_save_overlay(
                        entries, resolved, save_selected_index, _config_target_path()
                    )
                )
            elif overlay == "overrides":
                screen_text = render_static_overlay(
                    render_overrides_overlay(
                        ov_scope,
                        ov_selected_index,
                        module_settings_overrides,
                        existing_module_settings,
                        describe_name,
                    )
                )
            elif active_screen == "targets":
                vis = visible_targets()
                if target_selected_index >= len(vis):
                    target_selected_index = max(0, len(vis) - 1)
                header, body, footer, focus = render_targets_lines(
                    entries, vis, target_selected_index, resolved, expanded_replaced
                )
                if overlay == "typerules":
                    header = header + [""] + render_type_rules_overlay(entries) + [""]
                if status_message:
                    footer = footer + [color_text("  " + status_message, COLOR_ACCENT)]
                screen_text, target_window_start = compose_scrolling_screen(
                    header, body, footer, target_window_start, focus
                )
            else:
                items = build_run_items(expanded_buckets)
                if run_selected_index >= len(items):
                    run_selected_index = max(0, len(items) - 1)
                header, body, footer, focus = render_run_settings_lines(
                    items, run_selected_index, describe_name
                )
                if status_message:
                    footer = footer + [color_text("  " + status_message, COLOR_ACCENT)]
                screen_text, run_window_start = compose_scrolling_screen(
                    header, body, footer, run_window_start, focus
                )

            print(wrap_screen_text_for_terminal(screen_text))
            key = read_single_key()

            # ---- overlay key handling -----------------------------------
            if overlay == "help":
                if key == "?" or key == "\x1b":
                    overlay = None
                continue

            if overlay == "legend":
                if key == "h" or key == "\x1b":
                    overlay = None
                continue

            if overlay == "quit":
                if key in ("y", "Y"):
                    quit_requested = True
                    break
                if key in ("n", "N") or key == "\x1b":
                    overlay = None
                continue

            if overlay == "typerules":
                if key in ("y", "Y") or key == "\x1b":
                    overlay = None
                elif key.lower() == "s":
                    overlay = "save"
                    save_selected_index = 0
                continue

            if overlay == "save":
                unset = unset_module_entries(entries, resolved)
                option_count = 2 if unset else 3
                if is_up_key(key):
                    save_selected_index = _move(save_selected_index, -1, option_count)
                elif is_down_key(key):
                    save_selected_index = _move(save_selected_index, 1, option_count)
                elif key == "\x1b" or key == "x":
                    overlay = None
                elif is_enter_key(key):
                    if unset:
                        if save_selected_index == 0:
                            overlay = None
                        else:
                            finalize_save(0)
                            GPA.pc.set_configuration_confirmed(True)
                            break
                    else:
                        if save_selected_index == 2:
                            overlay = None
                        elif save_selected_index == 1:
                            GPA.pc.set_configuration_confirmed(True)
                            finalize_save(1)
                            break
                        else:
                            finalize_save(0)
                            GPA.pc.set_configuration_confirmed(True)
                            break
                continue

            if overlay == "overrides":
                names = CUSTOMIZABLE_SETTING_NAMES
                if key == "\x1b" or key == "x":
                    overlay = None
                    describe_name = ""
                elif is_up_key(key):
                    ov_selected_index = _move(ov_selected_index, -1, len(names))
                    describe_name = ""
                elif is_down_key(key):
                    ov_selected_index = _move(ov_selected_index, 1, len(names))
                    describe_name = ""
                elif key == "h" and names:
                    current = names[ov_selected_index]
                    describe_name = "" if describe_name == current else current
                elif is_enter_key(key) and names:
                    name = names[ov_selected_index]
                    scoped = get_scope_values(
                        ov_scope["scope_key"],
                        module_settings_overrides,
                        existing_module_settings,
                    )
                    value = scoped[name] if name in scoped else get_global_setting_value(name)
                    if isinstance(value, bool):
                        set_module_scoped_value(
                            module_settings_overrides, ov_scope["scope_key"], name, not value
                        )
                    elif get_enum_options(name) is not None:
                        options = get_enum_options(name)
                        try:
                            index = options.index(value)
                        except ValueError:
                            index = -1
                        set_module_scoped_value(
                            module_settings_overrides,
                            ov_scope["scope_key"],
                            name,
                            options[(index + 1) % len(options)],
                        )
                    elif isinstance(value, list):
                        list_editor_state = _new_list_editor(
                            name, value, scope_key=ov_scope["scope_key"]
                        )
                    elif isinstance(value, (int, float)):
                        scalar_editor_state = _new_scalar_editor(
                            name, value, scope_key=ov_scope["scope_key"]
                        )
                    else:
                        status_message = f"{name} cannot be edited here."
                elif key.lower() == "s":
                    overlay = "save"
                    save_selected_index = 0
                continue

            # ---- global keys (no overlay) -------------------------------
            if key == "?":
                overlay = "help"
                continue
            if key == "q":
                overlay = "quit"
                continue
            if key.lower() == "s":
                overlay = "save"
                save_selected_index = 0
                continue
            if is_tab_key(key):
                active_screen = "run" if active_screen == "targets" else "targets"
                status_message = ""
                describe_name = ""
                continue

            status_message = ""

            if active_screen == "targets":
                vis = visible_targets()
                if not vis:
                    continue
                if target_selected_index >= len(vis):
                    target_selected_index = len(vis) - 1
                entry = vis[target_selected_index]
                record = resolved[entry["id"]]

                if is_up_key(key):
                    target_selected_index = _move(target_selected_index, -1, len(vis))
                elif is_down_key(key):
                    target_selected_index = _move(target_selected_index, 1, len(vis))
                elif is_page_up_key(key):
                    target_selected_index = _move(target_selected_index, -10, len(vis))
                elif is_page_down_key(key):
                    target_selected_index = _move(target_selected_index, 10, len(vis))
                elif key == "y":
                    overlay = "typerules"
                elif key == "h":
                    overlay = "legend"
                elif is_right_key(key) or is_left_key(key):
                    if entry["is_replaced_root"]:
                        expanded_replaced[entry["id"]] = is_right_key(key)
                elif key in ("p", "t"):
                    if entry["is_replaced_root"] or entry["in_replaced"]:
                        status_message = (
                            f"{entry['id']} is inside a module that will be restructured "
                            "— its id won't exist at training time. Use P/T to target "
                            "by type instead."
                        )
                    else:
                        set_module_id_mode(
                            entry["id"], "perforated" if key == "p" else "tracked"
                        )
                elif key in ("P", "T"):
                    if entry["is_replaced_root"]:
                        status_message = (
                            "Set a mode on this module's children by type, not on the "
                            "container that will be restructured."
                        )
                    else:
                        set_module_name_mode(
                            entry["type_name"], "perforated" if key == "P" else "tracked"
                        )
                elif key == "x":
                    status_message = clear_module_mode(entry)
                elif is_enter_key(key):
                    scope, reason = get_override_scope_for_entry(entry, resolved)
                    if scope is None:
                        status_message = reason
                    else:
                        ov_scope = scope
                        ov_selected_index = 0
                        describe_name = ""
                        overlay = "overrides"
                continue

            # ---- run settings screen -----------------------------------
            items = build_run_items(expanded_buckets)
            if not items:
                continue
            if run_selected_index >= len(items):
                run_selected_index = len(items) - 1
            item = items[run_selected_index]

            if is_up_key(key) or is_down_key(key):
                step = -1 if is_up_key(key) else 1
                run_selected_index = _move(run_selected_index, step, len(items))
                while (
                    0 < run_selected_index < len(items) - 1
                    and items[run_selected_index]["type"] == "divider"
                ):
                    run_selected_index = _move(run_selected_index, step, len(items))
                describe_name = ""
            elif is_page_up_key(key):
                run_selected_index = _move(run_selected_index, -10, len(items))
            elif is_page_down_key(key):
                run_selected_index = _move(run_selected_index, 10, len(items))
            elif is_right_key(key):
                if item["type"] == "bucket":
                    expanded_buckets[item["bucket"]] = True
            elif is_left_key(key):
                if item["type"] == "bucket":
                    expanded_buckets[item["bucket"]] = False
                else:
                    expanded_buckets[item["bucket"]] = False
                    for idx, other in enumerate(items):
                        if other["type"] == "bucket" and other["bucket"] == item["bucket"]:
                            run_selected_index = idx
                            break
            elif key == "h" and item["type"] == "setting":
                describe_name = "" if describe_name == item["name"] else item["name"]
            elif is_enter_key(key):
                describe_name = ""
                if item["type"] == "bucket":
                    expanded_buckets[item["bucket"]] = not item["is_expanded"]
                elif item["type"] == "setting":
                    name = item["name"]
                    value = item["value"]
                    if isinstance(value, bool):
                        ok, message = apply_setting_value(name, not value)
                        status_message = message
                    elif get_enum_options(name) is not None:
                        ok, message = cycle_enum_setting(name, value)
                        status_message = message
                    elif isinstance(value, list):
                        list_editor_state = _new_list_editor(name, value, scope_key=None)
                    elif isinstance(value, (int, float)):
                        scalar_editor_state = _new_scalar_editor(name, value, scope_key=None)
                    else:
                        status_message = f"{name} cannot be edited from this screen."
            continue

    except KeyboardInterrupt:
        quit_requested = True
    finally:
        if entered_alt_screen:
            exit_alternate_screen()
        GPA.pc.__dict__["_auto_persist_config"] = previous_auto_persist

    if quit_requested:
        print("Configuration cancelled — training did not start.")
        print(
            "Re-run to configure again, or set configuration_confirmed=True "
            "(GPA.pc.set_configuration_confirmed(True)) to skip this screen."
        )
        sys.exit(0)


# ---------------------------------------------------------------------------
# Inline value editors (scalar + list)
# ---------------------------------------------------------------------------
def _new_scalar_editor(setting_name, value, scope_key):
    text = str(value)
    return {
        "kind": "scalar",
        "setting_name": setting_name,
        "sample_value": value,
        "input_buffer": text,
        "cursor_index": len(text),
        "scope_key": scope_key,
    }


def _new_list_editor(setting_name, value, scope_key):
    return {
        "kind": "list",
        "setting_name": setting_name,
        "values": list(value),
        "selected_index": 0,
        "sample_type": value[0] if value else None,
        "typing_active": False,
        "edit_index": None,
        "input_buffer": "",
        "cursor_index": 0,
        "scope_key": scope_key,
    }


def _render_editor_frame(state):
    print("\x1b[2J\x1b[H", end="")
    name = state["setting_name"]
    if state["kind"] == "scalar":
        body = render_text_with_cursor(state["input_buffer"], state["cursor_index"])
        lines = [
            "  " + color_text(f"Edit {name}", "E8EEEC"),
            "",
            f"  {body}",
            "",
            dim("  Enter save · Esc cancel"),
        ]
    else:
        lines = [
            "  " + color_text(f"Edit {name}", "E8EEEC"),
            "",
            "  " + _format_list_editor_line(state),
            "",
            dim(
                "  ←/→ move · Enter edit/select · "
                "Backspace/Delete remove · Esc cancel"
            ),
        ]
    print(wrap_screen_text_for_terminal("\n".join(lines)))


def _format_list_editor_line(state):
    values = state["values"]
    tokens = []
    for i, value in enumerate(values):
        if state["typing_active"] and i == state["edit_index"]:
            tokens.append(render_text_with_cursor(state["input_buffer"], state["cursor_index"]))
        elif i == state["selected_index"] and not state["typing_active"]:
            tokens.append(make_inverted_text(str(value)))
        else:
            tokens.append(str(value))

    add_index = len(values)
    if state["typing_active"] and state["edit_index"] == add_index:
        tokens.append(render_text_with_cursor(state["input_buffer"], state["cursor_index"]))
    elif state["selected_index"] == add_index:
        tokens.append(make_inverted_text("Add new entry"))
    else:
        tokens.append("Add new entry")

    save_index = len(values) + 1
    save_token = "Save list and return"
    tokens.append(
        make_inverted_text(save_token)
        if state["selected_index"] == save_index and not state["typing_active"]
        else save_token
    )
    return " | ".join(tokens)


def _persist_edited_value(state, module_settings_overrides, value):
    if state["scope_key"] is not None:
        set_module_scoped_value(
            module_settings_overrides, state["scope_key"], state["setting_name"], value
        )
        return ""
    ok, message = apply_setting_value(state["setting_name"], value)
    return "" if ok else message


def _handle_scalar_editor(state, key, module_settings_overrides):
    buffer_text = state["input_buffer"]
    cursor = state["cursor_index"]

    if key == "\x1b":
        return True, ""
    if is_left_key(key):
        state["cursor_index"] = max(0, cursor - 1)
        return False, ""
    if is_right_key(key):
        state["cursor_index"] = min(len(buffer_text), cursor + 1)
        return False, ""
    if key == "\x7f":
        if cursor > 0:
            state["input_buffer"] = buffer_text[: cursor - 1] + buffer_text[cursor:]
            state["cursor_index"] = cursor - 1
        return False, ""
    if is_delete_key(key):
        if cursor < len(buffer_text):
            state["input_buffer"] = buffer_text[:cursor] + buffer_text[cursor + 1 :]
        return False, ""
    if is_enter_key(key):
        try:
            parsed = parse_value_from_text(state["input_buffer"], state["sample_value"])
        except Exception as exc:
            return True, f"Invalid value for {state['setting_name']}: {exc}"
        return True, _persist_edited_value(state, module_settings_overrides, parsed)
    if len(key) == 1 and key >= " ":
        state["input_buffer"] = buffer_text[:cursor] + key + buffer_text[cursor:]
        state["cursor_index"] = cursor + 1
    return False, ""


def _handle_list_editor(state, key, module_settings_overrides):
    values = state["values"]
    add_index = len(values)
    save_index = len(values) + 1

    if state["typing_active"]:
        buffer_text = state["input_buffer"]
        cursor = state["cursor_index"]
        if key == "\x1b":
            state["typing_active"] = False
            state["edit_index"] = None
            state["input_buffer"] = ""
            state["cursor_index"] = 0
            return False, ""
        if is_left_key(key):
            state["cursor_index"] = max(0, cursor - 1)
            return False, ""
        if is_right_key(key):
            state["cursor_index"] = min(len(buffer_text), cursor + 1)
            return False, ""
        if key == "\x7f":
            if cursor > 0:
                state["input_buffer"] = buffer_text[: cursor - 1] + buffer_text[cursor:]
                state["cursor_index"] = cursor - 1
            return False, ""
        if is_delete_key(key):
            if cursor < len(buffer_text):
                state["input_buffer"] = buffer_text[:cursor] + buffer_text[cursor + 1 :]
            return False, ""
        if is_enter_key(key):
            edit_index = state["edit_index"]
            sample = state["sample_type"]
            if edit_index is not None and edit_index < len(values):
                sample = values[edit_index]
            try:
                parsed = parse_value_from_text(state["input_buffer"], sample)
            except Exception as exc:
                state["typing_active"] = False
                state["edit_index"] = None
                state["input_buffer"] = ""
                return False, f"Invalid list entry: {exc}"
            if edit_index is not None and edit_index < len(values):
                values[edit_index] = parsed
                state["selected_index"] = edit_index
            else:
                values.append(parsed)
                state["selected_index"] = len(values) - 1
            state["typing_active"] = False
            state["edit_index"] = None
            state["input_buffer"] = ""
            state["cursor_index"] = 0
            return False, ""
        if len(key) == 1 and key >= " ":
            state["input_buffer"] = buffer_text[:cursor] + key + buffer_text[cursor:]
            state["cursor_index"] = cursor + 1
        return False, ""

    selected = state["selected_index"]
    if key == "\x1b":
        return True, ""
    if is_left_key(key):
        state["selected_index"] = max(0, selected - 1)
        return False, ""
    if is_right_key(key):
        state["selected_index"] = min(save_index, selected + 1)
        return False, ""
    if key == "\x7f" or is_delete_key(key):
        if selected < len(values):
            del values[selected]
            if state["selected_index"] > len(values) + 1:
                state["selected_index"] = len(values) + 1
        return False, ""
    if is_select_key(key):
        if selected == save_index:
            return True, _persist_edited_value(
                state, module_settings_overrides, list(values)
            )
        initial_text = str(values[selected]) if selected < len(values) else ""
        state["typing_active"] = True
        state["edit_index"] = selected
        state["input_buffer"] = initial_text
        state["cursor_index"] = len(initial_text)
    return False, ""
