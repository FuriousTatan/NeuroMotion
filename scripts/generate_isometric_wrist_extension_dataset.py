import pickle
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.signal import butter, filtfilt
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from NeuroMotion.MNPoollib.mn_params import ANGLE, DEPTH, MS_AREA, NUM_MUS, mn_default_settings
from NeuroMotion.MNPoollib.mn_utils import generate_emg_mu


# =============================================================================
# USER SETTINGS
# =============================================================================

# BioMime files.
BIOMIME_CONFIG = "config.yaml"
MODEL_PTH = "./ckp/model_linear.pth"

# Output.
OUTPUT_DIR = "./res/isometric_wrist_extension_ecrb"
SAVE_AGGREGATE_PICKLE = False
SAVE_SUMMARY_CSV = True
MAKE_QUICKLOOK_FIGURES = True
GENERATE_ISOMETRIC = True
GENERATE_DYNAMIC = True
ISOMETRIC_OUTPUT_SUBDIR = "isometric"
DYNAMIC_OUTPUT_SUBDIR = "dynamic"

# Start with ECRB only. To expand later, use for example:
# MUSCLES = ["ECRB", "ECRL", "ECU"]
MUSCLES = ["ECRB"]

# Static wrist positions. Negative = wrist extension, positive = wrist flexion.
WRIST_ANGLES_DEG = [-45, -30, -15, 0, 15, 30, 45]

# For now this is 10% MVC only. To add the full set later:
# FORCE_LEVELS_MVC = [0.10, 0.20, 0.30]
FORCE_LEVELS_MVC = [0.10]

# Trapezoid contraction. Ramp speed is 10% MVC per second.
HOLD_DURATION_SEC = 10.0
RAMP_SECONDS_PER_10_PERCENT_MVC = 1.0

# Dynamic contraction set. This produces 3 trials x 3 fixed force levels = 9
# dynamic files, plus one extra file with both angle and force varying.
DYNAMIC_NUM_TRIALS = 3
DYNAMIC_DURATION_SEC = 16.0
DYNAMIC_FORCE_LEVELS_MVC = [0.10, 0.20, 0.30]
DYNAMIC_MIN_WRIST_ANGLE_DEG = -45.0
DYNAMIC_MAX_WRIST_ANGLE_DEG = 45.0
DYNAMIC_MUAP_UPDATE_HZ = 2.0
DYNAMIC_MIXED_FORCE_MAX_MVC = 0.30
DYNAMIC_SAVE_FULL_MUAP_SEQUENCE = False

# Runtime.
DEVICE = "auto"  # "auto", "cuda", or "cpu"
FS = 2048
SEED = 7  # Use None for random results.

# MU physiological assignment.
FIBRE_DENSITY = 200.0
NUM_MOTOR_UNITS = None  # None uses each muscle's default. Or set e.g. {"ECRB": 50}.

# MUAP smoothing, matching mov2emg.py defaults.
LOWPASS_MUAPS = True
CUTOFF_HZ = 800.0
FILTER_ORDER = 4

# Figures.
QUICKLOOK_CHANNEL = None  # None = strongest channel. Or set (row, col), e.g. (5, 10).
QUICKLOOK_MAX_MUS = 12


MUSCLE_ALIASES = {
    "ED": ("EDI", "EDCI"),
    "EDCI": ("EDI", "EDCI"),
    "EDI": ("EDI", "EDCI"),
    "FDS": ("FDSI", "FDSI"),
    "FCU_U": ("FCU_u", "FCU"),
    "FCU_H": ("FCU_h", "FCU"),
}


def make_settings():
    return SimpleNamespace(
        biomime_config=BIOMIME_CONFIG,
        model_pth=MODEL_PTH,
        output_dir=OUTPUT_DIR,
        save_aggregate_pickle=SAVE_AGGREGATE_PICKLE,
        save_summary_csv=SAVE_SUMMARY_CSV,
        make_quicklook_figures=MAKE_QUICKLOOK_FIGURES,
        generate_isometric=GENERATE_ISOMETRIC,
        generate_dynamic=GENERATE_DYNAMIC,
        isometric_output_subdir=ISOMETRIC_OUTPUT_SUBDIR,
        dynamic_output_subdir=DYNAMIC_OUTPUT_SUBDIR,
        muscles=MUSCLES,
        wrist_angles_deg=np.array(WRIST_ANGLES_DEG, dtype=np.float32),
        force_levels_mvc=np.array(FORCE_LEVELS_MVC, dtype=np.float32),
        hold_duration_sec=HOLD_DURATION_SEC,
        ramp_seconds_per_10_percent_mvc=RAMP_SECONDS_PER_10_PERCENT_MVC,
        dynamic_num_trials=DYNAMIC_NUM_TRIALS,
        dynamic_duration_sec=DYNAMIC_DURATION_SEC,
        dynamic_force_levels_mvc=np.array(DYNAMIC_FORCE_LEVELS_MVC, dtype=np.float32),
        dynamic_min_wrist_angle_deg=DYNAMIC_MIN_WRIST_ANGLE_DEG,
        dynamic_max_wrist_angle_deg=DYNAMIC_MAX_WRIST_ANGLE_DEG,
        dynamic_muap_update_hz=DYNAMIC_MUAP_UPDATE_HZ,
        dynamic_mixed_force_max_mvc=DYNAMIC_MIXED_FORCE_MAX_MVC,
        dynamic_save_full_muap_sequence=DYNAMIC_SAVE_FULL_MUAP_SEQUENCE,
        device=DEVICE,
        fs=FS,
        seed=SEED,
        fibre_density=FIBRE_DENSITY,
        num_motor_units=NUM_MOTOR_UNITS,
        lowpass_muaps=LOWPASS_MUAPS,
        cutoff_hz=CUTOFF_HZ,
        filter_order=FILTER_ORDER,
        quicklook_channel=QUICKLOOK_CHANNEL,
        quicklook_max_mus=QUICKLOOK_MAX_MUS,
    )


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def safe_label(value):
    text = str(value).replace("-", "neg").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


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


def get_num_motor_units(settings, pool_muscle):
    if settings.num_motor_units is None:
        return NUM_MUS[pool_muscle]
    if isinstance(settings.num_motor_units, dict):
        return settings.num_motor_units.get(pool_muscle, NUM_MUS[pool_muscle])
    return int(settings.num_motor_units)


def validate_settings(settings):
    if np.any(settings.wrist_angles_deg < -70) or np.any(settings.wrist_angles_deg > 70):
        raise ValueError("Wrist flexion angles should stay roughly within [-70, 70] degrees.")
    if np.any(settings.force_levels_mvc <= 0) or np.any(settings.force_levels_mvc > 1):
        raise ValueError("FORCE_LEVELS_MVC should contain values in (0, 1].")
    if np.any(settings.dynamic_force_levels_mvc <= 0) or np.any(settings.dynamic_force_levels_mvc > 1):
        raise ValueError("DYNAMIC_FORCE_LEVELS_MVC should contain values in (0, 1].")
    if settings.dynamic_duration_sec <= 0:
        raise ValueError("DYNAMIC_DURATION_SEC must be positive.")
    if settings.dynamic_muap_update_hz <= 0:
        raise ValueError("DYNAMIC_MUAP_UPDATE_HZ must be positive.")
    if settings.dynamic_min_wrist_angle_deg < -70 or settings.dynamic_max_wrist_angle_deg > 70:
        raise ValueError("Dynamic wrist angles should stay roughly within [-70, 70] degrees.")
    if settings.quicklook_channel is not None and len(settings.quicklook_channel) != 2:
        raise ValueError("QUICKLOOK_CHANNEL must be None or a tuple/list like (row, col).")


def build_wrist_movement(msk, times_sec, angles_deg):
    dof_names = msk.pose_basis.iloc[:, 0].tolist()
    if "flexion" not in dof_names:
        raise ValueError("The pose basis does not contain the wrist flexion DoF.")
    if len(times_sec) != len(angles_deg):
        raise ValueError("times_sec and angles_deg must have the same length.")

    base_pose = msk.pose_basis["open"].astype(float).to_numpy()
    flexion_id = dof_names.index("flexion")

    rows = []
    for time_sec, angle in zip(times_sec, angles_deg):
        pose = base_pose.copy()
        pose[flexion_id] = angle
        rows.append(np.concatenate(([float(time_sec)], pose)))

    mov = pd.DataFrame(rows, columns=["time", *dof_names])
    msk.load_mov(mov)
    return mov


def build_static_wrist_movement(msk, angles_deg):
    return build_wrist_movement(msk, np.arange(len(angles_deg), dtype=np.float32), angles_deg)


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


def make_trapezoid(force_level_mvc, settings):
    ramp_sec = force_level_mvc / 0.10 * settings.ramp_seconds_per_10_percent_mvc
    ramp_n = int(round(ramp_sec * settings.fs))
    hold_n = int(round(settings.hold_duration_sec * settings.fs))

    ramp_up = np.linspace(0.0, force_level_mvc, ramp_n, endpoint=False, dtype=np.float32)
    hold = np.full(hold_n, force_level_mvc, dtype=np.float32)
    ramp_down = np.linspace(force_level_mvc, 0.0, ramp_n, endpoint=True, dtype=np.float32)
    force_profile = np.concatenate((ramp_up, hold, ramp_down)).astype(np.float32)

    hold_start = ramp_n
    ramp_down_start = ramp_n + hold_n
    ramp_end = len(force_profile) - 1
    trigger_names = np.array(["ramp_start", "hold_start", "ramp_down_start", "ramp_end"], dtype=object)
    trigger_indices_0based = np.array([0, hold_start, ramp_down_start, ramp_end], dtype=np.int32)
    trigger_indices_1based = trigger_indices_0based + 1

    return {
        "force_profile": force_profile,
        "ramp_duration_sec": np.float32(ramp_sec),
        "hold_duration_sec": np.float32(settings.hold_duration_sec),
        "total_duration_sec": np.float32(len(force_profile) / settings.fs),
        "trigger_names": trigger_names,
        "trigger_indices_0based": trigger_indices_0based,
        "trigger_indices_1based": trigger_indices_1based,
        "trigger_times_sec": trigger_indices_0based.astype(np.float32) / settings.fs,
        "interval_indices_0based": np.array([0, hold_start, ramp_down_start, len(force_profile)], dtype=np.int32),
        "interval_note": "Intervals are [ramp_start, hold_start), [hold_start, ramp_down_start), [ramp_down_start, ramp_end_exclusive).",
    }


def make_dynamic_angle_profile(settings, trial_id):
    n_steps = int(round(settings.dynamic_duration_sec * settings.dynamic_muap_update_hz))
    times_sec = np.arange(n_steps, dtype=np.float32) / settings.dynamic_muap_update_hz
    center = 0.5 * (settings.dynamic_min_wrist_angle_deg + settings.dynamic_max_wrist_angle_deg)
    amplitude = 0.5 * (settings.dynamic_max_wrist_angle_deg - settings.dynamic_min_wrist_angle_deg)
    phases = [-0.5 * np.pi, 0.0, 0.5 * np.pi]
    phase = phases[(trial_id - 1) % len(phases)]
    angles_deg = center + amplitude * np.sin(2 * np.pi * times_sec / settings.dynamic_duration_sec + phase)
    return times_sec.astype(np.float32), angles_deg.astype(np.float32)


def make_dynamic_force_profile(settings, force_level_mvc=None, vary_force=False):
    n_samples = int(round(settings.dynamic_duration_sec * settings.fs))
    times_sec = np.arange(n_samples, dtype=np.float32) / settings.fs
    if vary_force:
        # Smooth 0 -> max -> 0 force profile. Peak slope is below 10% MVC/s.
        force_profile = settings.dynamic_mixed_force_max_mvc * np.sin(np.pi * times_sec / settings.dynamic_duration_sec) ** 2
    else:
        ramp_sec = force_level_mvc / 0.10 * settings.ramp_seconds_per_10_percent_mvc
        ramp_n = int(round(ramp_sec * settings.fs))
        hold_n = n_samples - 2 * ramp_n
        if hold_n <= 0:
            raise ValueError("Dynamic duration is too short for the requested force ramp.")
        ramp_up = np.linspace(0.0, force_level_mvc, ramp_n, endpoint=False, dtype=np.float32)
        hold = np.full(hold_n, force_level_mvc, dtype=np.float32)
        ramp_down = np.linspace(force_level_mvc, 0.0, ramp_n, endpoint=True, dtype=np.float32)
        force_profile = np.concatenate((ramp_up, hold, ramp_down))
    return force_profile.astype(np.float32)


def make_dynamic_triggers(settings, force_profile, angle_times_sec, angle_profile_deg, force_level_mvc=None, vary_force=False):
    n_samples = len(force_profile)
    angle_min_id = int(np.argmin(angle_profile_deg))
    angle_max_id = int(np.argmax(angle_profile_deg))
    force_peak_id = int(np.argmax(force_profile))

    trigger_names = ["start", "angle_min", "angle_max"]
    trigger_indices = [
        0,
        int(round(angle_times_sec[angle_min_id] * settings.fs)),
        int(round(angle_times_sec[angle_max_id] * settings.fs)),
    ]
    if vary_force:
        trigger_names.append("force_peak")
        trigger_indices.append(force_peak_id)
    elif force_level_mvc is not None:
        ramp_sec = force_level_mvc / 0.10 * settings.ramp_seconds_per_10_percent_mvc
        ramp_n = int(round(ramp_sec * settings.fs))
        trigger_names.extend(["force_hold_start", "force_ramp_down_start"])
        trigger_indices.extend([ramp_n, n_samples - ramp_n])
    trigger_names.append("end")
    trigger_indices.append(n_samples - 1)

    trigger_indices_0based = np.clip(np.array(trigger_indices, dtype=np.int32), 0, n_samples - 1)
    trigger_indices_1based = trigger_indices_0based + 1
    return {
        "trigger_names": np.array(trigger_names, dtype=object),
        "trigger_indices_0based": trigger_indices_0based,
        "trigger_indices_1based": trigger_indices_1based,
        "trigger_times_sec": trigger_indices_0based.astype(np.float32) / settings.fs,
        "interval_indices_0based": np.array([0, n_samples], dtype=np.int32),
        "interval_note": "Dynamic trial interval is [start, end_exclusive); angle/force event indices are listed in trigger fields.",
    }


def spikes_to_musts(spikes, num_mus, time_samples, fs):
    musts = np.zeros((num_mus, time_samples), dtype=np.uint8)
    spike_indices = np.empty((num_mus, 1), dtype=object)
    spike_times_sec = np.empty((num_mus, 1), dtype=object)
    for mu_id, mu_spikes in enumerate(spikes):
        idx = np.array(mu_spikes, dtype=np.int32)
        idx = idx[(idx >= 0) & (idx < time_samples)]
        musts[mu_id, idx] = 1
        spike_indices[mu_id, 0] = idx
        spike_times_sec[mu_id, 0] = idx.astype(np.float32) / fs
    return musts, spike_indices, spike_times_sec


def flatten_emg_grid(emg_grid):
    n_row, n_col, n_time = emg_grid.shape
    channel_rows, channel_cols = np.indices((n_row, n_col))
    emg = emg_grid.reshape(n_row * n_col, n_time).T
    return emg.astype(np.float32), channel_rows.ravel().astype(np.int32), channel_cols.ravel().astype(np.int32)


def select_quicklook_channel(emg_grid, requested_channel):
    if requested_channel is not None:
        return int(requested_channel[0]), int(requested_channel[1])
    energy = np.mean(emg_grid ** 2, axis=2)
    flat_id = int(np.argmax(energy))
    n_col = emg_grid.shape[1]
    return flat_id // n_col, flat_id % n_col


def init_runtime(settings):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to generate this dataset. Activate the NeuroMotion "
            "environment or install torch before running this script."
        ) from exc

    from BioMime.models.generator import Generator
    from BioMime.utils.basics import load_generator, update_config
    from NeuroMotion.MNPoollib.MNPool import MotoneuronPool
    from NeuroMotion.MSKlib.MSKpose import MSKModel

    if settings.seed is not None:
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)

    if settings.device == "auto":
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif settings.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False. Set DEVICE = 'cpu'.")

    cfg = update_config(str(resolve_path(Path("ckp") / settings.biomime_config)))
    model_pth = resolve_path(settings.model_pth)
    if not model_pth.exists():
        raise FileNotFoundError("BioMime checkpoint not found: {}".format(model_pth))

    generator = Generator(cfg.Model.Generator)
    generator = load_generator(str(model_pth), generator, settings.device)
    generator.eval()
    if settings.device == "cuda":
        generator.cuda()

    return SimpleNamespace(
        torch=torch,
        Generator=Generator,
        MotoneuronPool=MotoneuronPool,
        MSKModel=MSKModel,
        cfg=cfg,
        generator=generator,
        model_pth=model_pth,
    )


def init_muscle_state(settings, runtime, pool_muscle):
    torch = runtime.torch
    num_mus = get_num_motor_units(settings, pool_muscle)
    mn_pool = runtime.MotoneuronPool(num_mus, pool_muscle, **mn_default_settings)
    mn_pool.fs = settings.fs
    properties = mn_pool.assign_properties(make_mn_config(pool_muscle, settings.fibre_density), normalise=True)

    state = {
        "mn_pool": mn_pool,
        "num_mus": num_mus,
        "properties": properties,
        "base_num": torch.from_numpy(properties["num"]).float(),
        "base_depth": torch.from_numpy(properties["depth"]).float(),
        "base_angle": torch.from_numpy(properties["angle"]).float(),
        "base_iz": torch.from_numpy(properties["iz"]).float(),
        "base_cv": torch.from_numpy(properties["cv"]).float(),
        "base_length": torch.from_numpy(properties["len"]).float(),
        "latent": torch.randn(num_mus, runtime.cfg.Model.Generator.Latent),
    }
    if settings.device == "cuda":
        state["latent"] = state["latent"].cuda()
    return state


def generate_muap_for_angle(settings, runtime, muscle_state, changes, opensim_muscle, angle_id):
    torch = runtime.torch
    num_mus = muscle_state["num_mus"]

    ch_depth = changes["depth"].loc[:, [opensim_muscle] * num_mus].to_numpy(dtype=np.float32, copy=True)
    ch_cv = changes["cv"].loc[:, [opensim_muscle] * num_mus].to_numpy(dtype=np.float32, copy=True)
    ch_len = changes["len"].loc[:, [opensim_muscle] * num_mus].to_numpy(dtype=np.float32, copy=True)

    cond = torch.column_stack((
        muscle_state["base_num"],
        muscle_state["base_depth"] * torch.from_numpy(ch_depth[angle_id]),
        muscle_state["base_angle"],
        muscle_state["base_iz"],
        muscle_state["base_cv"] * torch.from_numpy(ch_cv[angle_id]),
        muscle_state["base_length"] * torch.from_numpy(ch_len[angle_id]),
    ))
    if settings.device == "cuda":
        cond = cond.cuda()

    with torch.no_grad():
        sim = runtime.generator.sample(num_mus, cond.float(), cond.device, muscle_state["latent"])
        sim = sim.permute(0, 2, 3, 1).detach()
        if settings.device == "cuda":
            sim = sim.cpu()
        sim = sim.numpy()

    if settings.lowpass_muaps:
        nyq = 0.5 * settings.fs
        b, a = butter(settings.filter_order, settings.cutoff_hz / nyq, btype="low", analog=False)
        num_mu_dim, n_row_dim, n_col_dim, n_time_dim = sim.shape
        sim = filtfilt(b, a, sim.reshape(-1, n_time_dim))
        sim = sim.reshape(num_mu_dim, n_row_dim, n_col_dim, n_time_dim)

    return sim.astype(np.float32)


def simulate_muscle_condition(settings, muscle_state, muaps, force_profile):
    mn_pool = muscle_state["mn_pool"]
    time_samples = len(force_profile)
    _, spikes, fr, ipi = mn_pool.generate_spike_trains(force_profile)
    musts, spike_indices, spike_times_sec = spikes_to_musts(
        spikes,
        muscle_state["num_mus"],
        time_samples,
        settings.fs,
    )

    _, n_row, n_col, time_length = muaps.shape
    emg_with_tail = np.zeros((n_row, n_col, time_samples + time_length), dtype=np.float32)
    for mu_id in range(muscle_state["num_mus"]):
        emg_with_tail += generate_emg_mu(muaps[mu_id][None, ...], spikes[mu_id], time_samples).astype(np.float32)

    emg_grid = emg_with_tail[:, :, :time_samples]
    return {
        "muaps": muaps,
        "musts": musts,
        "spike_indices": spike_indices,
        "spike_times_sec": spike_times_sec,
        "fr": fr.astype(np.float32),
        "ipi": ipi.astype(np.float32),
        "emg_grid": emg_grid.astype(np.float32),
        "emg_grid_with_tail": emg_with_tail.astype(np.float32),
        "num_spikes_per_mu": musts.sum(axis=1).astype(np.int32),
        "recruitment_thresholds": mn_pool.rte.astype(np.float32).ravel(),
    }


def simulate_dynamic_muscle_condition(settings, runtime, muscle_state, changes, opensim_muscle, force_profile):
    mn_pool = muscle_state["mn_pool"]
    time_samples = len(force_profile)
    _, spikes, fr, ipi = mn_pool.generate_spike_trains(force_profile)
    musts, spike_indices, spike_times_sec = spikes_to_musts(
        spikes,
        muscle_state["num_mus"],
        time_samples,
        settings.fs,
    )

    num_steps = changes["steps"]
    saved_muaps = [] if settings.dynamic_save_full_muap_sequence else None
    emg_with_tail = None

    for step_id in range(num_steps):
        muaps = generate_muap_for_angle(settings, runtime, muscle_state, changes, opensim_muscle, step_id)
        if saved_muaps is not None:
            saved_muaps.append(muaps)

        _, n_row, n_col, time_length = muaps.shape
        if emg_with_tail is None:
            emg_with_tail = np.zeros((n_row, n_col, time_samples + time_length), dtype=np.float32)

        start = int(round(step_id * time_samples / num_steps))
        end = int(round((step_id + 1) * time_samples / num_steps))
        for mu_id in range(muscle_state["num_mus"]):
            step_spikes = [t for t in spikes[mu_id] if start <= t < end]
            if step_spikes:
                emg_with_tail += generate_emg_mu(muaps[mu_id][None, ...], step_spikes, time_samples).astype(np.float32)

    emg_grid = emg_with_tail[:, :, :time_samples]
    if saved_muaps is None:
        muap_sequence = np.empty((0,), dtype=np.float32)
    else:
        # [mu, movement_step, row, col, time]
        muap_sequence = np.transpose(np.stack(saved_muaps, axis=0), (1, 0, 2, 3, 4)).astype(np.float32)

    return {
        "muaps": muap_sequence,
        "muap_sequence_saved": np.array([[int(settings.dynamic_save_full_muap_sequence)]], dtype=np.uint8),
        "musts": musts,
        "spike_indices": spike_indices,
        "spike_times_sec": spike_times_sec,
        "fr": fr.astype(np.float32),
        "ipi": ipi.astype(np.float32),
        "emg_grid": emg_grid.astype(np.float32),
        "emg_grid_with_tail": emg_with_tail.astype(np.float32),
        "num_spikes_per_mu": musts.sum(axis=1).astype(np.int32),
        "recruitment_thresholds": mn_pool.rte.astype(np.float32).ravel(),
    }


def condition_to_mat_dict(condition):
    emg, channel_rows, channel_cols = flatten_emg_grid(condition["emg_grid"])
    out = {
        "condition_type": condition["condition_type"],
        "trial_id": np.array([[condition["trial_id"]]], dtype=np.int32),
        "emg": emg,
        "emg_grid": condition["emg_grid"],
        "emg_grid_with_tail": condition["emg_grid_with_tail"],
        "fs": np.array([[condition["fs"]]], dtype=np.float32),
        "time_sec": condition["time_sec"],
        "angle_deg": np.array([[condition["angle_deg"]]], dtype=np.float32),
        "angle_profile_deg": condition["angle_profile_deg"],
        "angle_movement_time_sec": condition["angle_movement_time_sec"],
        "angle_movement_deg": condition["angle_movement_deg"],
        "muap_update_hz": np.array([[condition["muap_update_hz"]]], dtype=np.float32),
        "force_level_mvc": np.array([[condition["force_level_mvc"]]], dtype=np.float32),
        "force_profile": condition["force_profile"],
        "trigger_names": condition["trigger_names"],
        "trigger_indices_0based": condition["trigger_indices_0based"],
        "trigger_indices_1based": condition["trigger_indices_1based"],
        "trigger_times_sec": condition["trigger_times_sec"],
        "interval_indices_0based": condition["interval_indices_0based"],
        "interval_note": condition["interval_note"],
        "channel_rows_0based": channel_rows,
        "channel_cols_0based": channel_cols,
        "muscle_names": np.array(condition["muscle_names"], dtype=object),
        "per_muscle": condition["per_muscle"],
        "model_pth": condition["model_pth"],
        "seed": np.array([[-1 if condition["seed"] is None else condition["seed"]]], dtype=np.int32),
        "shape_note": (
            "emg is [time, channel]; emg_grid is [row, col, time]; static muaps are [mu, row, col, time]; "
            "saved dynamic muaps, when enabled, are [mu, movement_step, row, col, time]; musts are [mu, time]."
        ),
    }
    return out


def save_condition_mat(condition, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(output_path, condition_to_mat_dict(condition), do_compression=True)


def plot_quicklook(condition, output_path, settings):
    row, col = select_quicklook_channel(condition["emg_grid"], settings.quicklook_channel)
    time_sec = condition["time_sec"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=False)
    axes[0].plot(time_sec, condition["force_profile"], color="black", linewidth=1.3)
    axes[0].set_ylabel("MVC")
    if np.isfinite(condition["angle_deg"]):
        title = "Angle {} deg, force {}% MVC".format(
            int(condition["angle_deg"]), int(round(condition["force_level_mvc"] * 100))
        )
    elif np.isfinite(condition["force_level_mvc"]):
        title = "{} trial {}, force {}% MVC".format(
            condition["condition_type"], condition["trial_id"], int(round(condition["force_level_mvc"] * 100))
        )
    else:
        title = "{} trial {}, force 0-{}% MVC".format(
            condition["condition_type"],
            condition["trial_id"],
            int(round(np.max(condition["force_profile"]) * 100)),
        )
    axes[0].set_title(title)
    for idx in condition["trigger_indices_0based"]:
        axes[0].axvline(idx / condition["fs"], color="0.75", linewidth=0.8)

    axes[1].plot(time_sec, condition["emg_grid"][row, col], color="#174A7E", linewidth=0.7)
    axes[1].set_ylabel("EMG")
    axes[1].set_title("Representative channel row {}, col {}".format(row, col))

    first_muscle = condition["muscle_names"][0]
    musts = condition["per_muscle"][first_muscle]["musts"]
    plot_mus = min(settings.quicklook_max_mus, musts.shape[0])
    for mu_id in range(plot_mus):
        spike_times = np.flatnonzero(musts[mu_id]) / condition["fs"]
        axes[2].vlines(spike_times, mu_id, mu_id + 0.75, color="black", linewidth=0.5)
    axes[2].set_ylabel("MU")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("{} MUSTs, first {} MUs".format(first_muscle, plot_mus))

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def simulate_dynamic_condition(
    settings,
    runtime,
    resolved_muscles,
    changes,
    movement,
    muscle_lengths,
    angle_times_sec,
    angle_profile_deg,
    force_profile,
    dynamic_triggers,
    trial_id,
    force_level,
    condition_type,
):
    time_samples = len(force_profile)
    time_sec = np.arange(time_samples, dtype=np.float32) / settings.fs
    angle_profile_emg = np.interp(time_sec, angle_times_sec, angle_profile_deg).astype(np.float32)

    total_emg_with_tail = None
    per_muscle = {}
    for label, item in resolved_muscles.items():
        muscle_result = simulate_dynamic_muscle_condition(
            settings,
            runtime,
            item["state"],
            changes,
            item["opensim_muscle"],
            force_profile,
        )
        per_muscle[label] = {
            "pool_muscle": item["pool_muscle"],
            "opensim_muscle": item["opensim_muscle"],
            "num_mus": np.array([[item["state"]["num_mus"]]], dtype=np.int32),
            "mn_properties": item["state"]["properties"],
            "latent_vectors": item["state"]["latent"].detach().cpu().numpy().astype(np.float32),
            "muscle_length_profile": muscle_lengths.loc[:, item["opensim_muscle"]].to_numpy(dtype=np.float32, copy=True),
            "muap_update_times_sec": angle_times_sec,
            "muap_update_angles_deg": angle_profile_deg,
            **muscle_result,
        }
        if total_emg_with_tail is None:
            total_emg_with_tail = muscle_result["emg_grid_with_tail"].copy()
        else:
            total_emg_with_tail += muscle_result["emg_grid_with_tail"]

    total_emg_grid = total_emg_with_tail[:, :, :time_samples]
    return {
        "condition_type": condition_type,
        "trial_id": trial_id,
        "fs": settings.fs,
        "time_sec": time_sec,
        "angle_deg": np.float32(np.nan),
        "angle_profile_deg": angle_profile_emg,
        "angle_movement_time_sec": angle_times_sec,
        "angle_movement_deg": angle_profile_deg,
        "muap_update_hz": np.float32(settings.dynamic_muap_update_hz),
        "force_level_mvc": np.float32(force_level),
        "force_profile": force_profile,
        "emg_grid": total_emg_grid.astype(np.float32),
        "emg_grid_with_tail": total_emg_with_tail.astype(np.float32),
        "muscle_names": list(settings.muscles),
        "per_muscle": per_muscle,
        "movement": movement,
        "muscle_lengths": muscle_lengths,
        "model_pth": str(runtime.model_pth),
        "seed": settings.seed,
        **dynamic_triggers,
    }


def generate_dataset(settings):
    validate_settings(settings)
    output_dir = resolve_path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = init_runtime(settings)
    print("Generating dataset on {}...".format(settings.device))

    resolved_muscles = {}
    for label in settings.muscles:
        pool_muscle, opensim_muscle = resolve_muscle_names(label)
        resolved_muscles[label] = {
            "pool_muscle": pool_muscle,
            "opensim_muscle": opensim_muscle,
            "state": init_muscle_state(settings, runtime, pool_muscle),
        }

    opensim_muscles = sorted({item["opensim_muscle"] for item in resolved_muscles.values()})
    all_conditions = []
    condition_count = 0
    summary_rows = []
    movement_records = {}
    muscle_length_records = {}

    if settings.generate_isometric:
        isometric_dir = output_dir / settings.isometric_output_subdir
        isometric_dir.mkdir(parents=True, exist_ok=True)
        msk = runtime.MSKModel()
        movement = build_static_wrist_movement(msk, settings.wrist_angles_deg)
        muscle_lengths = msk.mov2len(ms_labels=opensim_muscles)
        changes = msk.len2params()
        movement_records["isometric"] = movement
        muscle_length_records["isometric"] = muscle_lengths

        for angle_id, angle_deg in enumerate(tqdm(settings.wrist_angles_deg, desc="Isometric angles", dynamic_ncols=True)):
            angle_muaps = {}
            for label, item in resolved_muscles.items():
                angle_muaps[label] = generate_muap_for_angle(
                    settings,
                    runtime,
                    item["state"],
                    changes,
                    item["opensim_muscle"],
                    angle_id,
                )

            for force_level in settings.force_levels_mvc:
                trapezoid = make_trapezoid(float(force_level), settings)
                force_profile = trapezoid["force_profile"]
                time_samples = len(force_profile)
                total_emg_with_tail = None
                per_muscle = {}

                for label, item in resolved_muscles.items():
                    muscle_result = simulate_muscle_condition(settings, item["state"], angle_muaps[label], force_profile)
                    per_muscle[label] = {
                        "pool_muscle": item["pool_muscle"],
                        "opensim_muscle": item["opensim_muscle"],
                        "num_mus": np.array([[item["state"]["num_mus"]]], dtype=np.int32),
                        "mn_properties": item["state"]["properties"],
                        "latent_vectors": item["state"]["latent"].detach().cpu().numpy().astype(np.float32),
                        "muscle_length": muscle_lengths.loc[angle_id, item["opensim_muscle"]],
                        **muscle_result,
                    }
                    if total_emg_with_tail is None:
                        total_emg_with_tail = muscle_result["emg_grid_with_tail"].copy()
                    else:
                        total_emg_with_tail += muscle_result["emg_grid_with_tail"]

                total_emg_grid = total_emg_with_tail[:, :, :time_samples]
                condition = {
                    "condition_type": "isometric",
                    "trial_id": 0,
                    "fs": settings.fs,
                    "time_sec": np.arange(time_samples, dtype=np.float32) / settings.fs,
                    "angle_deg": np.float32(angle_deg),
                    "angle_profile_deg": np.full(time_samples, angle_deg, dtype=np.float32),
                    "angle_movement_time_sec": np.array([0.0], dtype=np.float32),
                    "angle_movement_deg": np.array([angle_deg], dtype=np.float32),
                    "muap_update_hz": np.float32(0.0),
                    "force_level_mvc": np.float32(force_level),
                    "force_profile": force_profile,
                    "emg_grid": total_emg_grid.astype(np.float32),
                    "emg_grid_with_tail": total_emg_with_tail.astype(np.float32),
                    "muscle_names": list(settings.muscles),
                    "per_muscle": per_muscle,
                    "movement": movement,
                    "muscle_lengths": muscle_lengths,
                    "model_pth": str(runtime.model_pth),
                    "seed": settings.seed,
                    **trapezoid,
                }

                force_pct = int(round(float(force_level) * 100))
                mat_name = "iso_wrist_ext_angle_{}_force_{:03d}mvc.mat".format(safe_label(int(angle_deg)), force_pct)
                mat_path = isometric_dir / mat_name
                save_condition_mat(condition, mat_path)

                if settings.make_quicklook_figures:
                    fig_path = mat_path.with_suffix(".png")
                    plot_quicklook(condition, fig_path, settings)

                condition_count += 1
                if settings.save_aggregate_pickle:
                    all_conditions.append(condition)
                summary_rows.append({
                    "condition_type": "isometric",
                    "trial_id": 0,
                    "mat_file": str(mat_path),
                    "angle_deg": float(angle_deg),
                    "force_level_mvc": float(force_level),
                    "duration_sec": float(trapezoid["total_duration_sec"]),
                    "ramp_duration_sec": float(trapezoid["ramp_duration_sec"]),
                    "hold_duration_sec": float(trapezoid["hold_duration_sec"]),
                    "muscles": ",".join(settings.muscles),
                })

    if settings.generate_dynamic:
        dynamic_dir = output_dir / settings.dynamic_output_subdir
        dynamic_dir.mkdir(parents=True, exist_ok=True)
        for trial_id in range(1, settings.dynamic_num_trials + 1):
            angle_times_sec, angle_profile_deg = make_dynamic_angle_profile(settings, trial_id)
            msk = runtime.MSKModel()
            movement = build_wrist_movement(msk, angle_times_sec, angle_profile_deg)
            muscle_lengths = msk.mov2len(ms_labels=opensim_muscles)
            changes = msk.len2params()
            movement_records["dynamic_trial_{}".format(trial_id)] = movement
            muscle_length_records["dynamic_trial_{}".format(trial_id)] = muscle_lengths

            for force_level in tqdm(settings.dynamic_force_levels_mvc, desc="Dynamic trial {}".format(trial_id), dynamic_ncols=True):
                force_profile = make_dynamic_force_profile(settings, float(force_level), vary_force=False)
                dynamic_triggers = make_dynamic_triggers(
                    settings,
                    force_profile,
                    angle_times_sec,
                    angle_profile_deg,
                    force_level_mvc=float(force_level),
                )
                condition = simulate_dynamic_condition(
                    settings,
                    runtime,
                    resolved_muscles,
                    changes,
                    movement,
                    muscle_lengths,
                    angle_times_sec,
                    angle_profile_deg,
                    force_profile,
                    dynamic_triggers,
                    trial_id=trial_id,
                    force_level=float(force_level),
                    condition_type="dynamic_fixed_force",
                )
                force_pct = int(round(float(force_level) * 100))
                mat_name = "dyn_trial_{:02d}_force_{:03d}mvc.mat".format(trial_id, force_pct)
                mat_path = dynamic_dir / mat_name
                save_condition_mat(condition, mat_path)
                if settings.make_quicklook_figures:
                    plot_quicklook(condition, mat_path.with_suffix(".png"), settings)

                condition_count += 1
                if settings.save_aggregate_pickle:
                    all_conditions.append(condition)
                ramp_sec = float(force_level) / 0.10 * settings.ramp_seconds_per_10_percent_mvc
                summary_rows.append({
                    "condition_type": "dynamic_fixed_force",
                    "trial_id": trial_id,
                    "mat_file": str(mat_path),
                    "angle_deg": np.nan,
                    "force_level_mvc": float(force_level),
                    "duration_sec": float(settings.dynamic_duration_sec),
                    "ramp_duration_sec": float(ramp_sec),
                    "hold_duration_sec": float(settings.dynamic_duration_sec - 2 * ramp_sec),
                    "muscles": ",".join(settings.muscles),
                })

        mixed_trial_id = settings.dynamic_num_trials + 1
        angle_times_sec, angle_profile_deg = make_dynamic_angle_profile(settings, mixed_trial_id)
        msk = runtime.MSKModel()
        movement = build_wrist_movement(msk, angle_times_sec, angle_profile_deg)
        muscle_lengths = msk.mov2len(ms_labels=opensim_muscles)
        changes = msk.len2params()
        movement_records["dynamic_mixed"] = movement
        muscle_length_records["dynamic_mixed"] = muscle_lengths
        force_profile = make_dynamic_force_profile(settings, vary_force=True)
        dynamic_triggers = make_dynamic_triggers(settings, force_profile, angle_times_sec, angle_profile_deg, vary_force=True)
        condition = simulate_dynamic_condition(
            settings,
            runtime,
            resolved_muscles,
            changes,
            movement,
            muscle_lengths,
            angle_times_sec,
            angle_profile_deg,
            force_profile,
            dynamic_triggers,
            trial_id=mixed_trial_id,
            force_level=np.nan,
            condition_type="dynamic_varying_angle_force",
        )
        mat_path = dynamic_dir / "dyn_trial_mixed_angle_force_000_to_030mvc.mat"
        save_condition_mat(condition, mat_path)
        if settings.make_quicklook_figures:
            plot_quicklook(condition, mat_path.with_suffix(".png"), settings)

        condition_count += 1
        if settings.save_aggregate_pickle:
            all_conditions.append(condition)
        summary_rows.append({
            "condition_type": "dynamic_varying_angle_force",
            "trial_id": mixed_trial_id,
            "mat_file": str(mat_path),
            "angle_deg": np.nan,
            "force_level_mvc": np.nan,
            "duration_sec": float(settings.dynamic_duration_sec),
            "ramp_duration_sec": np.nan,
            "hold_duration_sec": np.nan,
            "muscles": ",".join(settings.muscles),
        })

    if settings.save_summary_csv:
        pd.DataFrame(summary_rows).to_csv(output_dir / "condition_summary.csv", index=False)

    if settings.save_aggregate_pickle:
        resolved_muscle_info = {}
        for label, item in resolved_muscles.items():
            resolved_muscle_info[label] = {
                "pool_muscle": item["pool_muscle"],
                "opensim_muscle": item["opensim_muscle"],
                "num_mus": item["state"]["num_mus"],
                "mn_properties": item["state"]["properties"],
                "latent_vectors": item["state"]["latent"].detach().cpu().numpy().astype(np.float32),
            }
        aggregate = {
            "settings": settings,
            "resolved_muscles": resolved_muscle_info,
            "movement_records": movement_records,
            "muscle_length_records": muscle_length_records,
            "conditions": all_conditions,
            "summary": pd.DataFrame(summary_rows),
        }
        with open(output_dir / "isometric_wrist_extension_dataset.pkl", "wb") as file:
            pickle.dump(aggregate, file, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done. Saved {} conditions to {}".format(condition_count, output_dir))


def main():
    settings = make_settings()
    generate_dataset(settings)


if __name__ == "__main__":
    main()
