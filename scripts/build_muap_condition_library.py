import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# USER SETTINGS
# =============================================================================

# Folder produced by generate_isometric_wrist_extension_dataset.py.
DATASET_DIR = "C:/tmp/data/emg/DYN25/sim"

# Isometric files always save full MUAPs. Dynamic files only contain full MUAP
# sequences when DYNAMIC_SAVE_FULL_MUAP_SEQUENCE = True in the generator script.
# Dynamic files with empty MUAP tensors are skipped automatically.
SOURCE_SUBDIRS = ["isometric", "dynamic"]

MUSCLES = ["ECRB"]

OUTPUT_DIR = "C:/tmp/data/emg/DYN25/sim/muap_condition_library"
OUTPUT_PICKLE = "muap_condition_library.pkl"
OUTPUT_MAT = "muap_condition_library.mat"

# Visualization layout. ECRB default 186 MUs
MUS_PER_PAGE = 62
PLOT_COLUMNS = 6
PLOT_WINDOW_MS = 30.0
PLOT_UNIQUE_ANGLES = True
COLORMAP = "coolwarm"
FIG_DPI = 220
SHOW_FIGURES = False


def make_settings():
    return SimpleNamespace(
        dataset_dir=resolve_path(DATASET_DIR),
        source_subdirs=SOURCE_SUBDIRS,
        muscles=MUSCLES,
        output_dir=resolve_path(OUTPUT_DIR),
        output_pickle=OUTPUT_PICKLE,
        output_mat=OUTPUT_MAT,
        mus_per_page=MUS_PER_PAGE,
        plot_columns=PLOT_COLUMNS,
        plot_window_ms=PLOT_WINDOW_MS,
        plot_unique_angles=PLOT_UNIQUE_ANGLES,
        colormap=COLORMAP,
        fig_dpi=FIG_DPI,
        show_figures=SHOW_FIGURES,
    )


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def get_field(obj, field_name, default=None):
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    if hasattr(obj, field_name):
        return getattr(obj, field_name)
    if isinstance(obj, np.void) and field_name in obj.dtype.names:
        return obj[field_name]
    return default


def to_scalar(value, default=np.nan):
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    item = arr.reshape(-1)[0]
    try:
        return float(item)
    except (TypeError, ValueError):
        return item


def to_int(value, default=0):
    scalar = to_scalar(value, default)
    try:
        return int(scalar)
    except (TypeError, ValueError):
        return default


def to_string(value, default=""):
    if value is None:
        return default
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    if arr.dtype.kind in ("U", "S"):
        return "".join(arr.astype(str).reshape(-1)).strip()
    item = arr.reshape(-1)[0]
    return str(item)


def discover_mat_files(settings):
    mat_files = []
    for subdir in settings.source_subdirs:
        folder = settings.dataset_dir / subdir
        if not folder.exists():
            print("Skipping missing folder: {}".format(folder))
            continue
        for path in sorted(folder.glob("*.mat")):
            if path.name == settings.output_mat or "muap_condition_library" in str(path):
                continue
            mat_files.append(path)
    if not mat_files:
        raise FileNotFoundError("No .mat condition files found under {}".format(settings.dataset_dir))
    return mat_files


def extract_muscle_struct(mat, muscle):
    per_muscle = get_field(mat, "per_muscle")
    if per_muscle is None:
        return None
    return get_field(per_muscle, muscle)


def as_muap_conditions(muaps, mat, source_file):
    muaps = np.asarray(muaps)
    if muaps.size == 0:
        return [], []

    condition_type = to_string(get_field(mat, "condition_type"), "unknown")
    trial_id = to_int(get_field(mat, "trial_id"), 0)
    force_level = to_scalar(get_field(mat, "force_level_mvc"))
    fs = to_scalar(get_field(mat, "fs"), 2048.0)

    if muaps.ndim == 4:
        # [mu, row, col, time] -> one static condition.
        angle = to_scalar(get_field(mat, "angle_deg"))
        condition_muaps = muaps[None, ...]
        metadata = [{
            "source_file": str(source_file),
            "condition_type": condition_type,
            "trial_id": trial_id,
            "condition_index_in_file": 0,
            "angle_deg": angle,
            "force_level_mvc": force_level,
            "fs": fs,
        }]
        return condition_muaps.astype(np.float32), metadata

    if muaps.ndim == 5:
        # [mu, movement_step, row, col, time] -> one condition per movement step.
        condition_muaps = np.transpose(muaps, (1, 0, 2, 3, 4)).astype(np.float32)
        angles = np.asarray(get_field(mat, "angle_movement_deg"), dtype=np.float32).reshape(-1)
        if len(angles) != condition_muaps.shape[0]:
            angles = np.full(condition_muaps.shape[0], np.nan, dtype=np.float32)
        metadata = []
        for step_id in range(condition_muaps.shape[0]):
            metadata.append({
                "source_file": str(source_file),
                "condition_type": condition_type,
                "trial_id": trial_id,
                "condition_index_in_file": step_id,
                "angle_deg": float(angles[step_id]),
                "force_level_mvc": force_level,
                "fs": fs,
            })
        return condition_muaps, metadata

    print("Skipping {}: unsupported MUAP shape {}".format(source_file, muaps.shape))
    return [], []


def load_muscle_conditions(settings, muscle, mat_files):
    condition_arrays = []
    metadata = []
    skipped = []

    for mat_file in mat_files:
        mat = loadmat(mat_file, squeeze_me=True, struct_as_record=False)
        muscle_struct = extract_muscle_struct(mat, muscle)
        if muscle_struct is None:
            skipped.append((str(mat_file), "missing muscle {}".format(muscle)))
            continue

        muaps = get_field(muscle_struct, "muaps")
        condition_muaps, condition_metadata = as_muap_conditions(muaps, mat, mat_file)
        if len(condition_metadata) == 0:
            skipped.append((str(mat_file), "empty or unsupported MUAP tensor"))
            continue
        condition_arrays.append(condition_muaps)
        metadata.extend(condition_metadata)

    if not condition_arrays:
        raise ValueError(
            "No MUAP tensors found for {}. For dynamic files, regenerate data with "
            "DYNAMIC_SAVE_FULL_MUAP_SEQUENCE = True if you want dynamic MUAPs saved.".format(muscle)
        )

    muaps = np.concatenate(condition_arrays, axis=0)
    order = sort_condition_order(metadata)
    muaps = muaps[order]
    metadata = [metadata[i] for i in order]
    return muaps, metadata, skipped


def sort_condition_order(metadata):
    condition_type = np.array([m["condition_type"] for m in metadata], dtype=object)
    angle = np.array([m["angle_deg"] for m in metadata], dtype=np.float32)
    force = np.array([m["force_level_mvc"] for m in metadata], dtype=np.float32)
    trial = np.array([m["trial_id"] for m in metadata], dtype=np.int32)
    step = np.array([m["condition_index_in_file"] for m in metadata], dtype=np.int32)
    type_rank = np.array([0 if t == "isometric" else 1 for t in condition_type], dtype=np.int32)
    return np.lexsort((step, trial, np.nan_to_num(force, nan=999), np.nan_to_num(angle, nan=999), type_rank))


def select_representative_channels(muaps):
    # muaps: [condition, mu, row, col, time]
    peak_to_peak = np.ptp(muaps, axis=-1)
    score = np.max(peak_to_peak, axis=0)
    num_mus, _, n_col = score.shape
    flat = score.reshape(num_mus, -1).argmax(axis=1)
    rows = flat // n_col
    cols = flat % n_col
    channels = np.column_stack((rows, cols)).astype(np.int32)
    values = score[np.arange(num_mus), rows, cols].astype(np.float32)
    return channels, values


def build_representative_waveforms(muaps, channels):
    num_conditions, num_mus = muaps.shape[:2]
    n_time = muaps.shape[-1]
    waveforms = np.zeros((num_mus, num_conditions, n_time), dtype=np.float32)
    for mu_id, (row, col) in enumerate(channels):
        waveforms[mu_id] = muaps[:, mu_id, row, col, :]
    return waveforms


def metadata_to_arrays(metadata):
    return {
        "condition_type": np.array([m["condition_type"] for m in metadata], dtype=object),
        "source_file": np.array([m["source_file"] for m in metadata], dtype=object),
        "trial_id": np.array([m["trial_id"] for m in metadata], dtype=np.int32),
        "condition_index_in_file": np.array([m["condition_index_in_file"] for m in metadata], dtype=np.int32),
        "angle_deg": np.array([m["angle_deg"] for m in metadata], dtype=np.float32),
        "force_level_mvc": np.array([m["force_level_mvc"] for m in metadata], dtype=np.float32),
        "fs": np.array([m["fs"] for m in metadata], dtype=np.float32),
    }


def build_library(settings):
    mat_files = discover_mat_files(settings)
    library = {
        "dataset_dir": str(settings.dataset_dir),
        "source_subdirs": list(settings.source_subdirs),
        "muscles": {},
        "notes": (
            "muaps are [condition, mu, row, col, time]. Representative channels are "
            "selected by maximum peak-to-peak amplitude across all loaded conditions."
        ),
    }

    for muscle in settings.muscles:
        muaps, metadata, skipped = load_muscle_conditions(settings, muscle, mat_files)
        channels, peak_to_peak = select_representative_channels(muaps)
        representative_waveforms = build_representative_waveforms(muaps, channels)
        metadata_arrays = metadata_to_arrays(metadata)

        library["muscles"][muscle] = {
            "muaps": muaps,
            "condition_metadata": metadata,
            "condition_arrays": metadata_arrays,
            "representative_channels_0based": channels,
            "representative_peak_to_peak": peak_to_peak,
            "representative_waveforms": representative_waveforms,
            "shape_note": "muaps[condition, mu, row, col, time]; representative_waveforms[mu, condition, time]",
            "skipped_files": skipped,
        }
        print(
            "{}: loaded {} conditions, {} MUs, skipped {} files".format(
                muscle, muaps.shape[0], muaps.shape[1], len(skipped)
            )
        )

    return library


def save_library(settings, library):
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = settings.output_dir / settings.output_pickle
    with open(pickle_path, "wb") as file:
        pickle.dump(library, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved {}".format(pickle_path))

    mat_payload = {
        "dataset_dir": library["dataset_dir"],
        "source_subdirs": np.array(library["source_subdirs"], dtype=object),
        "notes": library["notes"],
    }
    for muscle, data in library["muscles"].items():
        prefix = muscle
        arrays = data["condition_arrays"]
        mat_payload["{}_muaps".format(prefix)] = data["muaps"]
        mat_payload["{}_representative_waveforms".format(prefix)] = data["representative_waveforms"]
        mat_payload["{}_representative_channels_0based".format(prefix)] = data["representative_channels_0based"]
        mat_payload["{}_representative_peak_to_peak".format(prefix)] = data["representative_peak_to_peak"]
        for key, value in arrays.items():
            mat_payload["{}_{}".format(prefix, key)] = value

    mat_path = settings.output_dir / settings.output_mat
    savemat(mat_path, mat_payload, do_compression=True)
    print("Saved {}".format(mat_path))


def condition_indices_for_plot(metadata_arrays, unique_angles=True):
    angles = metadata_arrays["angle_deg"]
    valid = np.where(np.isfinite(angles))[0]
    if len(valid) == 0:
        return np.arange(len(angles), dtype=np.int32)
    if not unique_angles:
        return valid.astype(np.int32)

    selected = []
    seen = set()
    for idx in valid:
        key = round(float(angles[idx]), 4)
        if key in seen:
            continue
        seen.add(key)
        selected.append(idx)
    return np.array(selected, dtype=np.int32)


def crop_to_window(waveforms, fs, window_ms):
    n_time = waveforms.shape[-1]
    full_ms = n_time / fs * 1000.0
    if window_ms is None or window_ms <= 0 or window_ms >= full_ms:
        return waveforms, np.arange(n_time, dtype=np.float32) / fs * 1000.0

    n_window = max(2, int(round(window_ms / 1000.0 * fs)))
    center = int(np.mean(np.argmax(np.abs(waveforms), axis=1)))
    start = max(0, center - n_window // 2)
    end = min(n_time, start + n_window)
    start = max(0, end - n_window)
    cropped = waveforms[:, start:end]
    time_ms = np.arange(cropped.shape[-1], dtype=np.float32) / fs * 1000.0
    return cropped, time_ms


def plot_muscle_pages(settings, muscle, data, fs):
    out_dir = settings.output_dir / "figures" / muscle
    out_dir.mkdir(parents=True, exist_ok=True)

    representative_waveforms = data["representative_waveforms"]
    channels = data["representative_channels_0based"]
    metadata_arrays = data["condition_arrays"]
    plot_indices = condition_indices_for_plot(metadata_arrays, settings.plot_unique_angles)
    plot_angles = metadata_arrays["angle_deg"][plot_indices]

    num_mus = representative_waveforms.shape[0]
    num_pages = int(np.ceil(num_mus / settings.mus_per_page))
    max_abs_angle = max(float(np.nanmax(np.abs(plot_angles))), 1.0)
    norm = plt.Normalize(vmin=-max_abs_angle, vmax=max_abs_angle)
    cmap = plt.get_cmap(settings.colormap)

    for page_id in range(num_pages):
        start_mu = page_id * settings.mus_per_page
        end_mu = min(num_mus, start_mu + settings.mus_per_page)
        page_mus = end_mu - start_mu
        n_cols = min(settings.plot_columns, page_mus)
        n_rows = int(np.ceil(page_mus / n_cols))

        fig_w = max(8.0, 2.0 * n_cols)
        fig_h = max(5.0, 1.45 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
        axes = axes.ravel()

        for local_id, mu_id in enumerate(range(start_mu, end_mu)):
            ax = axes[local_id]
            waveforms = representative_waveforms[mu_id, plot_indices, :]
            waveforms, time_ms = crop_to_window(waveforms, fs, settings.plot_window_ms)

            for angle, waveform in zip(plot_angles, waveforms):
                ax.plot(time_ms, waveform, color=cmap(norm(angle)), linewidth=0.9, alpha=0.9)

            ymax = np.max(np.abs(waveforms))
            if ymax > 0:
                ax.set_ylim(-1.18 * ymax, 1.18 * ymax)
            ax.axhline(0, color="0.78", linewidth=0.5)
            row, col = channels[mu_id]
            ax.set_title("MU {}  ch {},{}".format(mu_id + 1, row, col), fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        for ax in axes[page_mus:]:
            ax.axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:page_mus], fraction=0.018, pad=0.01)
        cbar.set_label("Wrist angle (deg)", fontsize=9)

        fig.suptitle(
            "{} MUAP angle-dependent variation, page {}/{}".format(muscle, page_id + 1, num_pages),
            fontsize=12,
        )
        fig.tight_layout(rect=[0.0, 0.0, 0.98, 0.97])
        fig_path = out_dir / "{}_muap_angle_variation_page_{:02d}.png".format(muscle, page_id + 1)
        fig.savefig(fig_path, dpi=settings.fig_dpi)
        print("Saved {}".format(fig_path))

        if settings.show_figures:
            plt.show()
        else:
            plt.close(fig)


def plot_all(settings, library):
    for muscle, data in library["muscles"].items():
        fs_values = data["condition_arrays"]["fs"]
        fs = float(fs_values[0]) if len(fs_values) else 2048.0
        plot_muscle_pages(settings, muscle, data, fs)


def main():
    settings = make_settings()
    library = build_library(settings)
    save_library(settings, library)
    plot_all(settings, library)


if __name__ == "__main__":
    main()
