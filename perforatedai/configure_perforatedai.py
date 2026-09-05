import sys
import termios
import tty
import shutil

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


def get_preview_header_lines(entries=None, recursive_modes=None):
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
    lines.append("Use Up/Down to select. PageUp/PageDown scroll page-1. lowercase p/t select individual modules to perforate or track (not perforate but process as part of perforation). uppercase P/T select all modules of the same type. Enter=confirm")
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

    lines.append(build_target_summary_line(entries, recursive_modes))
    lines.append("")
    lines.append("  S|I|A")
    return lines


def get_list_window_size(total_entries):
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
    terminal_lines = shutil.get_terminal_size(fallback=(120, 40)).lines
    header_lines = len(get_preview_header_lines([], {}))
    # Reserve two lines for top/bottom overflow indicators.
    available = terminal_lines - header_lines - 2
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


def render_preview_screen_window(entries, selected_index, window_start, window_size):
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
    lines = get_preview_header_lines(entries, recursive_modes)
    window_end = min(len(entries), window_start + window_size)

    if window_start > 0:
        lines.append("^^^^ more above ^^^^")
    else:
        lines.append("")

    for i in range(window_start, window_end):
        lines.append(
            render_module_line(entries[i], i == selected_index, recursive_modes)
        )

    if window_end < len(entries):
        lines.append("vvvv more below vvvv")
    else:
        lines.append("")

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
    window_size = get_list_window_size(len(entries))
    window_start = clamp_window_start(0, selected_index, window_size, len(entries))
    return render_preview_screen_window(entries, selected_index, window_start, window_size)


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
        first = sys.stdin.read(1)
        if first == "\x03":
            raise KeyboardInterrupt
        if first == "\x1b":
            second = sys.stdin.read(1)
            if second == "[":
                tail = ""
                while True:
                    char = sys.stdin.read(1)
                    tail += char
                    if char.isalpha() or char == "~":
                        break
                return f"\x1b[{tail}"
            return first + second
        return first
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
    normalize_selection_conflicts()

    entries = get_module_entries(model)
    if len(entries) == 0:
        print(model)
        input("Press Enter to confirm perforation targets...")
        GPA.pc.set_configuration_confirmed(True)
        return

    selected_index = 0
    window_start = 0
    while True:
        normalize_selection_conflicts()
        if selected_index >= len(entries):
            selected_index = len(entries) - 1

        window_size = get_list_window_size(len(entries))
        window_start = clamp_window_start(
            window_start, selected_index, window_size, len(entries)
        )

        # Clear terminal and draw the updated preview.
        print("\x1b[2J\x1b[H", end="")
        print(
            render_preview_screen_window(
                entries, selected_index, window_start, window_size
            )
        )

        key = read_single_key()

        if key == "\x1b[A":
            selected_index = max(0, selected_index - 1)
            continue
        if key == "\x1b[B":
            selected_index = min(len(entries) - 1, selected_index + 1)
            continue
        if key == "\x1b[5~":
            page_step = max(1, window_size - 1)
            selected_index = max(0, selected_index - page_step)
            window_start = max(0, window_start - page_step)
            continue
        if key == "\x1b[6~":
            page_step = max(1, window_size - 1)
            selected_index = min(len(entries) - 1, selected_index + page_step)
            window_start = min(len(entries) - 1, window_start + page_step)
            continue

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

        if key == "\r" or key == "\n":
            break

    GPA.pc.set_configuration_confirmed(True)
