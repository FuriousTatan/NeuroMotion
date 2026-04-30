import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NeuroMotion.MNPoollib.mn_params import ANGLE, DEPTH, MS_AREA, NUM_MUS, mn_default_settings


# =============================================================================
# USER SETTINGS
# =============================================================================

# BioMime files.
BIOMIME_CONFIG = "config.yaml"
MODEL_PTH = "./ckp/model_linear.pth"

# Output files.
OUTPUT_LIBRARY = "./res/isometric_muap_library.pkl"
FIG_OUTPUT = None  # Use None to save next to OUTPUT_LIBRARY with a .png suffix.

# One muscle only. Examples: "ECRB", "ECRL", "ECU", "FCU", "PL", "FDSI", "EDCI".
MUSCLE = "FDSI"

# Use None for the repository default MU count for that muscle, or set a smaller
# integer for quick tests.
NUM_MOTOR_UNITS = None

# Wrist flexion/extension angles in degrees. Negative = extension, positive = flexion.
# Set WRIST_ANGLES_DEG to None to use the min/max/number settings below.
WRIST_ANGLES_DEG = [-65, -45, -25, 0, 25, 45, 65]
MIN_WRIST_ANGLE_DEG = -65.0
MAX_WRIST_ANGLE_DEG = 65.0
NUM_WRIST_ANGLES = 9

# Runtime.
DEVICE = "auto"  # "auto", "cuda", or "cpu"
FS = 2048
SEED = 7  # Use None for random results.

# MU physiological assignment.
FIBRE_DENSITY = 200.0

# Representative channel selection for plotting.
# Use None/None to automatically pick the strongest row/column per MU.
ROW = None
COL = None

# Visualization.
SHOW_FIGURE = True
PLOT_COLUMNS = 8
MAX_PLOTTED_MOTOR_UNITS = None  # Use None to plot every MU.
PLOT_WINDOW_MS = 30.0  # Use 0 to show the full 96-sample MUAP.
COLORMAP = "coolwarm"
FIG_DPI = 200

# MUAP smoothing, matching mov2emg.py defaults.
LOWPASS_MUAPS = True
CUTOFF_HZ = 800.0
FILTER_ORDER = 4


MUSCLE_ALIASES = {
    "ED": ("EDI", "EDCI"),
    "EDCI": ("EDI", "EDCI"),
    "EDI": ("EDI", "EDCI"),
    "FDS": ("FDSI", "FDSI"),
    "FCU_U": ("FCU_u", "FCU"),
    "FCU_H": ("FCU_h", "FCU"),
}


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def resolve_muscle_names(muscle):
    muscle_key = muscle.upper()
    param_name_by_key = {name.upper(): name for name in NUM_MUS}
    if muscle_key in MUSCLE_ALIASES:
        return MUSCLE_ALIASES[muscle_key]
    if muscle_key not in param_name_by_key:
        valid = sorted(set(NUM_MUS.keys()) | set(MUSCLE_ALIASES.keys()))
        raise ValueError("Unknown muscle '{}'. Valid choices: {}".format(muscle, ", ".join(valid)))
    pool_muscle = param_name_by_key[muscle_key]
    return pool_muscle, pool_muscle


def make_settings():
    return SimpleNamespace(
        biomime_config=BIOMIME_CONFIG,
        model_pth=MODEL_PTH,
        output_library=OUTPUT_LIBRARY,
        fig_output=FIG_OUTPUT,
        muscle=MUSCLE,
        num_motor_units=NUM_MOTOR_UNITS,
        wrist_angles_deg=WRIST_ANGLES_DEG,
        min_wrist_angle_deg=MIN_WRIST_ANGLE_DEG,
        max_wrist_angle_deg=MAX_WRIST_ANGLE_DEG,
        num_wrist_angles=NUM_WRIST_ANGLES,
        device=DEVICE,
        fs=FS,
        seed=SEED,
        fibre_density=FIBRE_DENSITY,
        row=ROW,
        col=COL,
        show_figure=SHOW_FIGURE,
        plot_columns=PLOT_COLUMNS,
        max_plotted_motor_units=MAX_PLOTTED_MOTOR_UNITS,
        plot_window_ms=PLOT_WINDOW_MS,
        colormap=COLORMAP,
        fig_dpi=FIG_DPI,
        lowpass_muaps=LOWPASS_MUAPS,
        cutoff_hz=CUTOFF_HZ,
        filter_order=FILTER_ORDER,
    )


def get_wrist_angles(settings):
    if settings.wrist_angles_deg is not None:
        angles = np.array(settings.wrist_angles_deg, dtype=np.float32)
    else:
        angles = np.linspace(
            settings.min_wrist_angle_deg,
            settings.max_wrist_angle_deg,
            settings.num_wrist_angles,
            dtype=np.float32,
        )
    if np.any(angles < -70) or np.any(angles > 70):
        raise ValueError("Wrist flexion angles should stay in the model range, roughly [-70, 70] degrees.")
    return angles


def build_static_wrist_movement(msk, angles_deg):
    dof_names = msk.pose_basis.iloc[:, 0].tolist()
    if "flexion" not in dof_names:
        raise ValueError("The pose basis does not contain the wrist flexion DoF.")

    base_pose = msk.pose_basis["open"].astype(float).to_numpy()
    flexion_id = dof_names.index("flexion")

    rows = []
    for i, angle in enumerate(angles_deg):
        pose = base_pose.copy()
        pose[flexion_id] = angle
        rows.append(np.concatenate(([float(i)], pose)))

    mov = pd.DataFrame(rows, columns=["time", *dof_names])
    msk.load_mov(mov)
    return mov


def make_mn_config(pool_muscle, fibre_density):
    num_fb = np.round(MS_AREA[pool_muscle] * fibre_density)
    return SimpleNamespace(
        num_fb=num_fb,
        depth=DEPTH[pool_muscle],
        angle=ANGLE[pool_muscle],
        iz=[0.5, 0.1],
        len=[1.0, 0.05],
        cv=[4, 0.3],
    )


def generate_angle_conditioned_muaps(settings, angles_deg):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to generate MUAPs. Activate the NeuroMotion environment "
            "or install torch before running this script."
        ) from exc

    from BioMime.models.generator import Generator
    from BioMime.utils.basics import load_generator, update_config
    from NeuroMotion.MNPoollib.MNPool import MotoneuronPool
    from NeuroMotion.MSKlib.MSKpose import MSKModel

    pool_muscle, opensim_muscle = resolve_muscle_names(settings.muscle)
    num_mus = settings.num_motor_units if settings.num_motor_units is not None else NUM_MUS[pool_muscle]

    if settings.seed is not None:
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)

    if settings.device == "auto":
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif settings.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False. Set DEVICE = 'cpu'.")

    biomime_config = resolve_path(Path("ckp") / settings.biomime_config)
    cfg = update_config(str(biomime_config))
    model_pth = resolve_path(settings.model_pth)
    if not model_pth.exists():
        raise FileNotFoundError("BioMime checkpoint not found: {}".format(model_pth))

    print("Loading OpenSim model and extracting static wrist-angle lengths...")
    msk = MSKModel()
    movement = build_static_wrist_movement(msk, angles_deg)
    muscle_lengths = msk.mov2len(ms_labels=[opensim_muscle])
    changes = msk.len2params()

    print("Initialising {} motor units for {}...".format(num_mus, pool_muscle))
    mn_pool = MotoneuronPool(num_mus, pool_muscle, **mn_default_settings)
    properties = mn_pool.assign_properties(make_mn_config(pool_muscle, settings.fibre_density), normalise=True)

    base_num = torch.from_numpy(properties["num"]).float()
    base_depth = torch.from_numpy(properties["depth"]).float()
    base_angle = torch.from_numpy(properties["angle"]).float()
    base_iz = torch.from_numpy(properties["iz"]).float()
    base_cv = torch.from_numpy(properties["cv"]).float()
    base_length = torch.from_numpy(properties["len"]).float()

    ch_depth = changes["depth"].loc[:, [opensim_muscle] * num_mus].to_numpy(dtype=np.float32, copy=True)
    ch_cv = changes["cv"].loc[:, [opensim_muscle] * num_mus].to_numpy(dtype=np.float32, copy=True)
    ch_len = changes["len"].loc[:, [opensim_muscle] * num_mus].to_numpy(dtype=np.float32, copy=True)

    print("Loading BioMime generator on {}...".format(settings.device))
    generator = Generator(cfg.Model.Generator)
    generator = load_generator(str(model_pth), generator, settings.device)
    generator.eval()
    if settings.device == "cuda":
        generator.cuda()

    latent = torch.randn(num_mus, cfg.Model.Generator.Latent)
    if settings.device == "cuda":
        latent = latent.cuda()

    if settings.lowpass_muaps:
        nyq = 0.5 * settings.fs
        b, a = butter(settings.filter_order, settings.cutoff_hz / nyq, btype="low", analog=False)

    muaps_by_angle = []
    with torch.no_grad():
        for angle_id, angle in enumerate(tqdm(angles_deg, dynamic_ncols=True, desc="Generating angle MUAPs")):
            cond = torch.column_stack((
                base_num,
                base_depth * torch.from_numpy(ch_depth[angle_id]),
                base_angle,
                base_iz,
                base_cv * torch.from_numpy(ch_cv[angle_id]),
                base_length * torch.from_numpy(ch_len[angle_id]),
            ))
            if settings.device == "cuda":
                cond = cond.cuda()

            sim = generator.sample(num_mus, cond.float(), cond.device, latent)
            sim = sim.permute(0, 2, 3, 1).detach()
            if settings.device == "cuda":
                sim = sim.cpu()
            sim = sim.numpy()

            if settings.lowpass_muaps:
                num_mu_dim, n_row_dim, n_col_dim, n_time_dim = sim.shape
                sim = filtfilt(b, a, sim.reshape(-1, n_time_dim))
                sim = sim.reshape(num_mu_dim, n_row_dim, n_col_dim, n_time_dim)

            muaps_by_angle.append(sim.astype(np.float32))

    muaps = np.stack(muaps_by_angle, axis=1)
    representative_channels = select_representative_channels(muaps, settings.row, settings.col)

    library = {
        "muaps": muaps,
        "shape": "muaps[mu, angle, row, col, time]",
        "muscle": settings.muscle,
        "pool_muscle": pool_muscle,
        "opensim_muscle": opensim_muscle,
        "angles_deg": angles_deg,
        "angle_convention": "negative = wrist extension, positive = wrist flexion",
        "fs": settings.fs,
        "time_ms": np.arange(muaps.shape[-1], dtype=np.float32) / settings.fs * 1000.0,
        "movement": movement,
        "muscle_lengths": muscle_lengths,
        "parameter_changes": {
            "depth": changes["depth"],
            "cv": changes["cv"],
            "len": changes["len"],
        },
        "mn_properties": properties,
        "mn_default_settings": mn_default_settings,
        "fibre_density": settings.fibre_density,
        "representative_channels": representative_channels,
        "representative_channel_note": "Each MU uses the row/col channel with the largest RMS across angles and time unless ROW/COL are set.",
        "model_pth": str(model_pth),
        "seed": settings.seed,
    }
    return library


def select_representative_channels(muaps, row=None, col=None):
    num_mu, _, n_row, n_col, _ = muaps.shape
    if row is not None or col is not None:
        if row is None or col is None:
            raise ValueError("Set both ROW and COL, or set both to None.")
        if row < 0 or row >= n_row or col < 0 or col >= n_col:
            raise ValueError("ROW/COL are outside the MUAP grid shape {}x{}.".format(n_row, n_col))
        return np.tile(np.array([[row, col]], dtype=np.int32), (num_mu, 1))

    energy = np.mean(muaps ** 2, axis=(1, 4))
    flat_ids = energy.reshape(num_mu, -1).argmax(axis=1)
    rows = flat_ids // n_col
    cols = flat_ids % n_col
    return np.column_stack((rows, cols)).astype(np.int32)


def crop_to_window(waveforms, fs, window_ms):
    n_time = waveforms.shape[-1]
    full_ms = n_time / fs * 1000.0
    if window_ms is None or window_ms <= 0 or window_ms >= full_ms:
        return waveforms, np.arange(n_time) / fs * 1000.0

    n_window = max(2, int(round(window_ms / 1000.0 * fs)))
    center = int(np.mean(np.argmax(np.abs(waveforms), axis=1)))
    start = max(0, center - n_window // 2)
    end = min(n_time, start + n_window)
    start = max(0, end - n_window)

    cropped = waveforms[:, start:end]
    time_ms = np.arange(cropped.shape[-1]) / fs * 1000.0
    return cropped, time_ms


def plot_muap_library(library, output_path, settings):
    muaps = library["muaps"]
    angles = library["angles_deg"]
    channels = library["representative_channels"]
    num_mu = muaps.shape[0]
    plot_num = num_mu if settings.max_plotted_motor_units is None else min(num_mu, settings.max_plotted_motor_units)

    plot_cols = min(settings.plot_columns, plot_num)
    plot_rows = int(np.ceil(plot_num / plot_cols))
    fig_w = max(8.0, 1.85 * plot_cols)
    fig_h = max(5.0, 1.35 * plot_rows)

    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes.ravel()

    max_abs_angle = max(float(np.max(np.abs(angles))), 1.0)
    norm = plt.Normalize(vmin=-max_abs_angle, vmax=max_abs_angle)
    cmap = plt.get_cmap(settings.colormap)

    for mu_id in range(plot_num):
        ax = axes[mu_id]
        row, col = channels[mu_id]
        waveforms = muaps[mu_id, :, row, col, :]
        waveforms, time_ms = crop_to_window(waveforms, library["fs"], settings.plot_window_ms)

        for angle, waveform in zip(angles, waveforms):
            ax.plot(time_ms, waveform, color=cmap(norm(angle)), linewidth=0.9, alpha=0.9)

        ymax = np.max(np.abs(waveforms))
        if ymax > 0:
            ax.set_ylim(-1.15 * ymax, 1.15 * ymax)
        ax.axhline(0, color="0.80", linewidth=0.5)
        ax.set_title("MU {}  ch {},{}".format(mu_id + 1, row, col), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[plot_num:]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:plot_num], fraction=0.018, pad=0.01)
    cbar.set_label("Wrist angle (deg)", fontsize=9)

    title = "{} MUAP library: extension to flexion".format(library["pool_muscle"])
    if plot_num < num_mu:
        title += " (showing {} of {} MUs)".format(plot_num, num_mu)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0.0, 0.0, 0.98, 0.98])
    fig.savefig(output_path, dpi=settings.fig_dpi)
    print("Saved visualization to {}".format(output_path))

    if settings.show_figure:
        plt.show()
    else:
        plt.close(fig)


def save_library(library, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(library, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved MUAP library to {}".format(output_path))


def main():
    settings = make_settings()

    angles_deg = get_wrist_angles(settings)
    output_path = resolve_path(settings.output_library)
    if settings.fig_output is None:
        fig_output = output_path.with_suffix(".png")
    else:
        fig_output = resolve_path(settings.fig_output)

    library = generate_angle_conditioned_muaps(settings, angles_deg)
    save_library(library, output_path)
    plot_muap_library(library, fig_output, settings)


if __name__ == "__main__":
    main()
