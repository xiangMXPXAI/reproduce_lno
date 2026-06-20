import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as scio
import torch
from PIL import Image, ImageDraw, ImageFont


FORWARD_DIR = Path(__file__).resolve().parent
PYTORCH_DIR = FORWARD_DIR.parent
REPO_DIR = PYTORCH_DIR.parent
sys.path.insert(0, str(FORWARD_DIR))

from module.model import LNO, LNO_single, LNO_triple  # noqa: E402


DATASET_TO_CONFIG = {
    "Darcy": "LNO_Darcy",
    "Airfoil": "LNO_Airfoil",
    "Elasticity": "LNO_Elasticity",
    "Plasticity": "LNO_Plasticity",
    "Pipe": "LNO_Pipe",
}

DATASET_CN = {
    "Darcy": "Darcy 渗流",
    "Airfoil": "Airfoil 翼型流场",
    "Elasticity": "Elasticity 随机单胞弹性",
    "Plasticity": "Plasticity 塑性时空场",
    "Pipe": "Pipe 管道流",
}


def strip_jsonc(text):
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return text


def load_config(name):
    path = FORWARD_DIR / "configs" / f"{name}.jsonc"
    with path.open("r", encoding="utf-8") as f:
        return json.loads(strip_jsonc(f.read()))


def latest_checkpoint(exp_name):
    ckpt_dir = FORWARD_DIR / "experiments" / exp_name / "checkpoint"
    pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: int(p.stem))
    if not pts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return pts[-1]


def rel_l2(pred, target):
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    return torch.linalg.vector_norm(pred - target, ord=2, dim=1) / (
        torch.linalg.vector_norm(target, ord=2, dim=1) + 1e-12
    )


def make_grid_2d(nx, ny):
    coords = []
    for x1 in np.linspace(0, 1, nx):
        for x2 in np.linspace(0, 1, ny):
            coords.append([x1, x2])
    return np.reshape(np.array(coords, dtype=np.float32), (nx, ny, 2))


def make_grid_3d(nx, ny, nt):
    coords = []
    for x1 in np.linspace(0, 1, nx):
        for x2 in np.linspace(0, 1, ny):
            for x3 in np.linspace(0, 1, nt):
                coords.append([x1, x2, x3])
    return np.reshape(np.array(coords, dtype=np.float32), (nx, ny, nt, 3))


def mean_std_from_last_dim(arr):
    flat = np.reshape(arr, (-1, arr.shape[-1])).astype(np.float64)
    mean = flat.mean(axis=0).astype(np.float32)
    std = (flat.std(axis=0) + 1e-8).astype(np.float32)
    return mean, std


class Stats:
    def __init__(self, x_mean=None, x_std=None, y1_mean=None, y1_std=None, y2_mean=None, y2_std=None):
        self.x_mean = x_mean
        self.x_std = x_std
        self.y1_mean = y1_mean
        self.y1_std = y1_std
        self.y2_mean = y2_mean
        self.y2_std = y2_std

    def apply_x(self, x):
        return (x - self.x_mean) / self.x_std if self.x_mean is not None else x

    def apply_y1(self, y):
        return (y - self.y1_mean) / self.y1_std if self.y1_mean is not None else y

    def apply_y2(self, y):
        return (y - self.y2_mean) / self.y2_std if self.y2_mean is not None else y

    def inverse_y2_torch(self, y):
        if self.y2_mean is None:
            return y
        mean = torch.as_tensor(self.y2_mean, dtype=y.dtype, device=y.device)
        std = torch.as_tensor(self.y2_std, dtype=y.dtype, device=y.device)
        return y * std + mean


def package_sample(x, y1, y2, stats, normalize):
    if normalize:
        x_model = stats.apply_x(x.astype(np.float32))
        y1_model = stats.apply_y1(y1.astype(np.float32))
        y2_model = stats.apply_y2(y2.astype(np.float32))
    else:
        x_model = x.astype(np.float32)
        y1_model = y1.astype(np.float32)
        y2_model = y2.astype(np.float32)
    return {
        "x_model": x_model,
        "y1_model": y1_model,
        "y2_model": y2_model,
        "x_phys": x.astype(np.float32),
        "y1_phys": y1.astype(np.float32),
        "y2_phys": y2.astype(np.float32),
    }


def load_darcy(max_samples):
    x_base = make_grid_2d(241, 241)
    train = scio.loadmat(PYTORCH_DIR / "Darcy_241" / "piececonst_r241_N1024_smooth1.mat", variable_names=["coeff", "sol"])
    val = scio.loadmat(PYTORCH_DIR / "Darcy_241" / "piececonst_r241_N1024_smooth2.mat", variable_names=["coeff", "sol"])

    coeff_train = train["coeff"].astype(np.float32)[..., None]
    sol_train = train["sol"].astype(np.float32)[..., None]
    x_mean, x_std = mean_std_from_last_dim(x_base)
    coeff_mean, coeff_std = mean_std_from_last_dim(coeff_train)
    sol_mean, sol_std = mean_std_from_last_dim(sol_train)
    stats = Stats(
        x_mean=x_mean,
        x_std=x_std,
        y1_mean=np.concatenate([x_mean, coeff_mean]),
        y1_std=np.concatenate([x_std, coeff_std]),
        y2_mean=sol_mean,
        y2_std=sol_std,
    )

    samples = []
    count = min(max_samples, val["coeff"].shape[0])
    for i in range(count):
        x = x_base
        y1_raw = val["coeff"][i].astype(np.float32)[..., None]
        y1 = np.concatenate([x, y1_raw], axis=-1)
        y2 = val["sol"][i].astype(np.float32)[..., None]
        samples.append(package_sample(x, y1, y2, stats, normalize=True))
    return samples, stats


def load_airfoil(max_samples):
    q = np.load(PYTORCH_DIR / "airfoil" / "naca" / "NACA_Cylinder_Q.npy", mmap_mode="r")
    x_phys = np.load(PYTORCH_DIR / "airfoil" / "naca" / "NACA_Cylinder_X.npy", mmap_mode="r")
    y_phys = np.load(PYTORCH_DIR / "airfoil" / "naca" / "NACA_Cylinder_Y.npy", mmap_mode="r")
    x_base = make_grid_2d(221, 51)

    samples = []
    total_used = 1200
    val_start = total_used - 200
    count = min(max_samples, 200)
    for i in range(count):
        idx = val_start + i
        x = x_base
        geom = np.stack([x_phys[idx], y_phys[idx]], axis=-1).astype(np.float32)
        y1 = np.concatenate([x, geom], axis=-1)
        y2 = q[idx, 4].astype(np.float32)[..., None]
        samples.append(package_sample(x, y1, y2, Stats(), normalize=False))
    return samples, Stats()


def load_elasticity(max_samples):
    xy = np.load(PYTORCH_DIR / "elasticity" / "Meshes" / "Random_UnitCell_XY_10.npy")
    sigma = np.load(PYTORCH_DIR / "elasticity" / "Meshes" / "Random_UnitCell_sigma_10.npy")
    xy = np.transpose(xy, (2, 0, 1)).astype(np.float32)
    sigma = np.transpose(sigma, (1, 0)).astype(np.float32)[..., None]

    train_x = xy[:1000]
    train_y2 = sigma[:1000]
    x_mean, x_std = mean_std_from_last_dim(train_x)
    y2_mean, y2_std = mean_std_from_last_dim(train_y2)
    stats = Stats(x_mean=x_mean, x_std=x_std, y1_mean=x_mean, y1_std=x_std, y2_mean=y2_mean, y2_std=y2_std)

    samples = []
    val_x = xy[-1000:]
    val_y2 = sigma[-1000:]
    count = min(max_samples, val_x.shape[0])
    for i in range(count):
        x = val_x[i]
        y1 = x.copy()
        y2 = val_y2[i]
        samples.append(package_sample(x, y1, y2, stats, normalize=True))
    return samples, stats


def load_plasticity(max_samples):
    mat = scio.loadmat(PYTORCH_DIR / "plasticity" / "plas_N987_T20.mat", variable_names=["input", "output"])
    inp = mat["input"].astype(np.float32)
    out = mat["output"].astype(np.float32)
    x_base = make_grid_3d(101, 31, 20)

    input_train = inp[:900]
    output_train = out[:900]
    x_mean, x_std = mean_std_from_last_dim(x_base)
    input_mean = input_train.reshape(-1, 1).mean(axis=0).astype(np.float32)
    input_std = (input_train.reshape(-1, 1).std(axis=0) + 1e-8).astype(np.float32)
    y2_mean, y2_std = mean_std_from_last_dim(output_train)
    stats = Stats(
        x_mean=x_mean,
        x_std=x_std,
        y1_mean=np.concatenate([x_mean, input_mean]),
        y1_std=np.concatenate([x_std, input_std]),
        y2_mean=y2_mean,
        y2_std=y2_std,
    )

    samples = []
    count = min(max_samples, 87)
    for i in range(count):
        idx = 900 + i
        y1_scalar = inp[idx, :, None]
        y1_scalar = np.repeat(y1_scalar, 31, axis=1)
        y1_scalar = np.repeat(y1_scalar[:, :, None], 20, axis=2)[..., None].astype(np.float32)
        y1 = np.concatenate([x_base, y1_scalar], axis=-1)
        y2 = out[idx]
        samples.append(package_sample(x_base, y1, y2, stats, normalize=True))
    return samples, stats


def load_pipe(max_samples):
    q = np.load(PYTORCH_DIR / "pipe" / "Pipe_Q.npy", mmap_mode="r")
    x_phys = np.load(PYTORCH_DIR / "pipe" / "Pipe_X.npy", mmap_mode="r")
    y_phys = np.load(PYTORCH_DIR / "pipe" / "Pipe_Y.npy", mmap_mode="r")
    x_base = make_grid_2d(129, 129)

    samples = []
    val_start = 2310 - 200
    count = min(max_samples, 200)
    for i in range(count):
        idx = val_start + i
        x = x_base
        geom = np.stack([x_phys[idx], y_phys[idx]], axis=-1).astype(np.float32)
        y1 = np.concatenate([x, geom], axis=-1)
        y2 = q[idx, 0].astype(np.float32)[..., None]
        samples.append(package_sample(x, y1, y2, Stats(), normalize=False))
    return samples, Stats()


LOADERS = {
    "Darcy": load_darcy,
    "Airfoil": load_airfoil,
    "Elasticity": load_elasticity,
    "Plasticity": load_plasticity,
    "Pipe": load_pipe,
}


def build_model(config, sample, device):
    x_dim = sample["x_model"].shape[-1]
    y1_dim = sample["y1_model"].shape[-1]
    y2_dim = sample["y2_model"].shape[-1]
    model_attr = {"time": False}
    mcfg = config["model"]
    if mcfg["name"] == "LNO":
        model = LNO(
            mcfg["n_block"],
            mcfg["n_mode"],
            mcfg["n_dim"],
            mcfg["n_head"],
            mcfg["n_layer"],
            x_dim,
            y1_dim,
            y2_dim,
            mcfg["attn"],
            mcfg["act"],
            model_attr,
        )
    elif mcfg["name"] == "LNO_single":
        model = LNO_single(
            mcfg["n_block"],
            mcfg["n_mode"],
            mcfg["n_dim"],
            mcfg["n_head"],
            mcfg["n_layer"],
            x_dim,
            y1_dim,
            y2_dim,
            mcfg["attn"],
            mcfg["act"],
            model_attr,
        )
    elif mcfg["name"] == "LNO_triple":
        model = LNO_triple(
            mcfg["n_block"],
            mcfg["n_mode"],
            mcfg["n_dim"],
            mcfg["n_head"],
            mcfg["n_layer"],
            x_dim,
            y1_dim,
            y2_dim,
            mcfg["attn"],
            mcfg["act"],
            model_attr,
        )
    else:
        raise NotImplementedError(mcfg["name"])
    return model.to(device).eval()


def load_weights(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    clean_state = {}
    for key, value in state.items():
        clean_state[key[7:] if key.startswith("module.") else key] = value
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing}, unexpected={unexpected}")


def predict_samples(model, samples, stats, device):
    rows = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            x = torch.from_numpy(sample["x_model"]).float().reshape(1, -1, sample["x_model"].shape[-1]).to(device)
            y1 = torch.from_numpy(sample["y1_model"]).float().reshape(1, -1, sample["y1_model"].shape[-1]).to(device)
            y2_model = torch.from_numpy(sample["y2_model"]).float().reshape(1, -1, sample["y2_model"].shape[-1]).to(device)
            pred_model = model(x, y1)
            if stats.y2_mean is not None:
                pred_phys = stats.inverse_y2_torch(pred_model)
                target_phys = torch.from_numpy(sample["y2_phys"]).float().reshape(1, -1, sample["y2_phys"].shape[-1]).to(device)
            else:
                pred_phys = pred_model
                target_phys = y2_model
            metric = rel_l2(pred_phys.cpu(), target_phys.cpu()).item()
            pred_np = pred_phys.reshape(sample["y2_phys"].shape).cpu().numpy()
            rows.append({"index": i, "rel_l2": metric, "pred": pred_np, "target": sample["y2_phys"]})
    return rows


def setup_style():
    return {
        "bg": (247, 248, 251),
        "card": (255, 255, 255),
        "ink": (17, 24, 39),
        "muted": (102, 112, 133),
        "line": (215, 220, 229),
        "blue": (37, 99, 235),
        "purple": (124, 58, 237),
    }


def get_font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


PALETTES = {
    "turbo": [(35, 23, 115), (24, 119, 210), (34, 181, 174), (103, 210, 78), (237, 214, 50), (241, 117, 36), (164, 22, 34)],
    "viridis": [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
    "inferno": [(0, 0, 4), (49, 10, 92), (126, 37, 83), (203, 71, 54), (249, 142, 8), (252, 255, 164)],
    "mako": [(11, 18, 32), (25, 52, 107), (34, 117, 169), (65, 182, 196), (222, 244, 232)],
}


def palette_lut(name):
    colors = np.array(PALETTES.get(name, PALETTES["viridis"]), dtype=np.float32)
    anchors = np.linspace(0, 1, len(colors))
    t = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(t, anchors, colors[:, c]).astype(np.uint8)
    return lut


def value_range(arr, robust=True):
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    if robust:
        vmin, vmax = np.percentile(finite, [1, 99])
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def colorize(arr, cmap="viridis", robust=True):
    arr = np.squeeze(np.asarray(arr, dtype=np.float32))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    vmin, vmax = value_range(arr, robust)
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    idx = (norm * 255).astype(np.uint8)
    idx_display = np.flipud(idx.T)
    rgb = palette_lut(cmap)[idx_display]
    return Image.fromarray(rgb, "RGB"), vmin, vmax


def draw_text(draw, xy, text, font, fill, max_width=None):
    if max_width is None:
        draw.text(xy, text, font=font, fill=fill)
        return
    words = list(text)
    lines = []
    line = ""
    for ch in words:
        candidate = line + ch
        if draw.textlength(candidate, font=font) <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = ch
    if line:
        lines.append(line)
    x, y = xy
    for line in lines[:2]:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + 4


def draw_card_base(canvas, box, title, subtitle):
    style = setup_style()
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=style["card"], outline=style["line"], width=1)
    draw.text((x0 + 20, y0 + 18), title, font=get_font(25, bold=True), fill=style["ink"])
    if subtitle:
        draw_text(draw, (x0 + 20, y0 + 54), subtitle, get_font(15), style["muted"], max_width=(x1 - x0 - 40))


def paste_colorbar(canvas, box, cmap, vmin, vmax):
    draw = ImageDraw.Draw(canvas)
    x0, y0, x1, y1 = box
    lut = palette_lut(cmap)
    h = y1 - y0
    grad = np.zeros((h, 14, 3), dtype=np.uint8)
    for i in range(h):
        grad[i, :, :] = lut[int((1 - i / max(h - 1, 1)) * 255)]
    canvas.paste(Image.fromarray(grad, "RGB"), (x0, y0))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=4, outline=(230, 233, 240), width=1)
    draw.text((x0 - 2, y0 - 20), f"{vmax:.2g}", font=get_font(12), fill=(102, 112, 133))
    draw.text((x0 - 2, y1 + 5), f"{vmin:.2g}", font=get_font(12), fill=(102, 112, 133))


def heatmap_panel(canvas, box, arr, title, subtitle, cmap="viridis", robust=True):
    draw_card_base(canvas, box, title, subtitle)
    x0, y0, x1, y1 = box
    img_box = (x0 + 20, y0 + 98, x1 - 44, y1 - 22)
    img, vmin, vmax = colorize(arr, cmap, robust)
    img = img.resize((img_box[2] - img_box[0], img_box[3] - img_box[1]), Image.Resampling.BICUBIC)
    canvas.paste(img, (img_box[0], img_box[1]))
    ImageDraw.Draw(canvas).rectangle(img_box, outline=(229, 233, 241), width=1)
    paste_colorbar(canvas, (x1 - 30, img_box[1], x1 - 16, img_box[3]), cmap, vmin, vmax)


def scale_points(xy, box, pad=20):
    xy = np.asarray(xy, dtype=np.float32)
    x0, y0, x1, y1 = box
    px0, py0, px1, py1 = x0 + pad, y0 + pad, x1 - pad, y1 - pad
    xmin, xmax = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
    ymin, ymax = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))
    if xmax - xmin < 1e-12:
        xmax = xmin + 1.0
    if ymax - ymin < 1e-12:
        ymax = ymin + 1.0
    sx = (px1 - px0) / (xmax - xmin)
    sy = (py1 - py0) / (ymax - ymin)
    scale = min(sx, sy)
    used_w = (xmax - xmin) * scale
    used_h = (ymax - ymin) * scale
    ox = px0 + (px1 - px0 - used_w) / 2
    oy = py0 + (py1 - py0 - used_h) / 2
    pts = np.column_stack([ox + (xy[:, 0] - xmin) * scale, oy + used_h - (xy[:, 1] - ymin) * scale])
    return pts


def scatter_panel(canvas, box, xy, values, title, subtitle, cmap="viridis"):
    draw_card_base(canvas, box, title, subtitle)
    x0, y0, x1, y1 = box
    plot_box = (x0 + 20, y0 + 98, x1 - 44, y1 - 22)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(plot_box, fill=(252, 253, 255), outline=(229, 233, 241), width=1)
    values = np.squeeze(values).astype(np.float32)
    vmin, vmax = value_range(values, robust=True)
    idx = (np.clip((values - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8)
    colors = palette_lut(cmap)[idx]
    pts = scale_points(xy, plot_box, pad=12)
    for (x, y), color in zip(pts, colors):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=tuple(int(c) for c in color))
    paste_colorbar(canvas, (x1 - 30, plot_box[1], x1 - 16, plot_box[3]), cmap, vmin, vmax)


def mesh_panel(canvas, box, geom, title, subtitle):
    draw_card_base(canvas, box, title, subtitle)
    style = setup_style()
    x0, y0, x1, y1 = box
    plot_box = (x0 + 20, y0 + 98, x1 - 20, y1 - 22)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(plot_box, fill=(252, 253, 255), outline=(229, 233, 241), width=1)
    x = geom[..., 0]
    y = geom[..., 1]
    pts_all = scale_points(np.column_stack([x.reshape(-1), y.reshape(-1)]), plot_box, pad=14).reshape(*x.shape, 2)
    step0 = max(1, x.shape[0] // 34)
    step1 = max(1, x.shape[1] // 24)
    for i in range(0, x.shape[0], step0):
        draw.line([tuple(p) for p in pts_all[i, :, :]], fill=(96, 165, 250), width=1)
    for j in range(0, x.shape[1], step1):
        draw.line([tuple(p) for p in pts_all[:, j, :]], fill=(167, 139, 250), width=1)
    for p in pts_all[::step0, ::step1].reshape(-1, 2):
        draw.ellipse((p[0] - 1.2, p[1] - 1.2, p[0] + 1.2, p[1] + 1.2), fill=(17, 24, 39))


def plot_dataset(dataset, sample, pred_row, output_dir):
    style = setup_style()
    canvas = Image.new("RGB", (2200, 680), style["bg"])
    draw = ImageDraw.Draw(canvas)
    target = pred_row["target"]
    pred = pred_row["pred"]
    err = np.abs(pred - target)
    rel = pred_row["rel_l2"]
    draw.text((36, 26), f"{DATASET_CN[dataset]} | 输入-输出与权重预测可视化", font=get_font(36, bold=True), fill=style["ink"])
    draw.text((38, 75), "LNO PyTorch ForwardProblem | validation sample 0 | generated by visualize_forward.py", font=get_font(17), fill=style["muted"])
    boxes = []
    margin = 34
    gap = 22
    top = 122
    card_w = (2200 - margin * 2 - gap * 3) // 4
    card_h = 510
    for i in range(4):
        x0 = margin + i * (card_w + gap)
        boxes.append((x0, top, x0 + card_w, top + card_h))

    if dataset == "Darcy":
        heatmap_panel(canvas, boxes[0], sample["y1_phys"][..., 2], "输入", "渗透率/系数场 a(x)", cmap="mako")
        heatmap_panel(canvas, boxes[1], target[..., 0], "真实输出", "压力/势函数解 u(x)", cmap="turbo")
        heatmap_panel(canvas, boxes[2], pred[..., 0], "LNO 预测", f"checkpoint relative L2 = {rel:.4e}", cmap="turbo")
        heatmap_panel(canvas, boxes[3], err[..., 0], "绝对误差", "|prediction - target|", cmap="inferno", robust=False)
    elif dataset in {"Airfoil", "Pipe"}:
        geom = sample["y1_phys"][..., 2:4]
        channel_name = "Q 第 5 通道" if dataset == "Airfoil" else "Q 第 1 通道"
        mesh_panel(canvas, boxes[0], geom, "输入", "参考网格 + 真实物理坐标几何")
        heatmap_panel(canvas, boxes[1], target[..., 0], "真实输出", channel_name, cmap="turbo")
        heatmap_panel(canvas, boxes[2], pred[..., 0], "LNO 预测", f"checkpoint relative L2 = {rel:.4e}", cmap="turbo")
        heatmap_panel(canvas, boxes[3], err[..., 0], "绝对误差", "|prediction - target|", cmap="inferno", robust=False)
    elif dataset == "Elasticity":
        xy = sample["x_phys"].reshape(-1, 2)
        mesh_canvas = np.zeros((xy.shape[0],), dtype=np.float32)
        scatter_panel(canvas, boxes[0], xy, mesh_canvas, "输入", "随机单胞非结构点坐标", cmap="mako")
        scatter_panel(canvas, boxes[1], xy, target.reshape(-1), "真实输出", "应力 sigma", cmap="turbo")
        scatter_panel(canvas, boxes[2], xy, pred.reshape(-1), "LNO 预测", f"checkpoint relative L2 = {rel:.4e}", cmap="turbo")
        scatter_panel(canvas, boxes[3], xy, err.reshape(-1), "绝对误差", "|prediction - target|", cmap="inferno")
    elif dataset == "Plasticity":
        t_idx = target.shape[2] // 2
        c_idx = 0
        heatmap_panel(canvas, boxes[0], sample["y1_phys"][:, :, t_idx, 3], "输入", f"一维输入场广播到空间-时间, t={t_idx}", cmap="viridis")
        heatmap_panel(canvas, boxes[1], target[:, :, t_idx, c_idx], "真实输出", f"输出通道 {c_idx}, t={t_idx}", cmap="turbo")
        heatmap_panel(canvas, boxes[2], pred[:, :, t_idx, c_idx], "LNO 预测", f"checkpoint relative L2 = {rel:.4e}", cmap="turbo")
        heatmap_panel(canvas, boxes[3], err[:, :, t_idx, c_idx], "绝对误差", "|prediction - target|", cmap="inferno", robust=False)
    else:
        raise NotImplementedError(dataset)

    path = output_dir / f"{dataset}_prediction.png"
    canvas.save(path)
    return path


def plot_metric_summary(metrics, output_dir):
    style = setup_style()
    names = list(metrics.keys())
    vals = [metrics[k]["mean_rel_l2"] for k in names]
    colors = ["#2563eb", "#7c3aed", "#0891b2", "#ea580c", "#16a34a"]
    canvas = Image.new("RGB", (1400, 760), style["bg"])
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 38), "Forward benchmark checkpoint evaluation", font=get_font(38, bold=True), fill=style["ink"])
    draw.text((52, 90), "Mean relative L2 on evaluated validation samples, log scale", font=get_font(18), fill=style["muted"])
    plot = (90, 160, 1320, 650)
    draw.rounded_rectangle(plot, radius=18, fill=style["card"], outline=style["line"])
    px0, py0, px1, py1 = 150, 220, 1260, 570
    vals_arr = np.array(vals, dtype=np.float64)
    log_vals = np.log10(np.maximum(vals_arr, 1e-12))
    lo = math.floor(float(log_vals.min()) - 0.2)
    hi = math.ceil(float(log_vals.max()) + 0.2)
    if hi == lo:
        hi = lo + 1
    for tick in range(lo, hi + 1):
        y = py1 - (tick - lo) / (hi - lo) * (py1 - py0)
        draw.line((px0, y, px1, y), fill=(232, 236, 244), width=1)
        draw.text((96, y - 10), f"1e{tick}", font=get_font(14), fill=style["muted"])
    bar_gap = 34
    bar_w = (px1 - px0 - bar_gap * (len(names) - 1)) / len(names)
    for i, (name, val, lv) in enumerate(zip(names, vals, log_vals)):
        x0 = px0 + i * (bar_w + bar_gap)
        x1 = x0 + bar_w
        y = py1 - (lv - lo) / (hi - lo) * (py1 - py0)
        color = tuple(int(colors[i % len(colors)].lstrip("#")[j : j + 2], 16) for j in (0, 2, 4))
        draw.rounded_rectangle((x0, y, x1, py1), radius=10, fill=color)
        draw.text((x0 + 4, y - 30), f"{val:.3e}", font=get_font(15), fill=style["ink"])
        draw.text((x0 + 6, py1 + 18), name, font=get_font(17, bold=True), fill=style["ink"])
    path = output_dir / "forward_metrics_summary.png"
    canvas.save(path)
    return path


def evaluate_dataset(dataset, max_samples, device, output_dir):
    config_name = DATASET_TO_CONFIG[dataset]
    config = load_config(config_name)
    checkpoint = latest_checkpoint(config_name)
    print(f"[{dataset}] loading data...")
    samples, stats = LOADERS[dataset](max_samples)
    model = build_model(config, samples[0], device)
    load_weights(model, checkpoint, device)
    print(f"[{dataset}] loaded checkpoint {checkpoint.name}; evaluating {len(samples)} sample(s) on {device}...")
    rows = predict_samples(model, samples, stats, device)
    rels = [row["rel_l2"] for row in rows]
    fig_path = plot_dataset(dataset, samples[0], rows[0], output_dir)
    result = {
        "dataset": dataset,
        "config": config_name,
        "checkpoint": str(checkpoint.relative_to(REPO_DIR)),
        "num_eval_samples": len(rows),
        "mean_rel_l2": float(np.mean(rels)),
        "std_rel_l2": float(np.std(rels)),
        "min_rel_l2": float(np.min(rels)),
        "max_rel_l2": float(np.max(rels)),
        "visualization": str(fig_path.relative_to(REPO_DIR)),
    }
    del model, samples, rows
    gc.collect()
    return result


def write_report(metrics, output_dir):
    report = output_dir / "forward_visualization_report.md"
    lines = [
        "# Forward 数据集权重评估与可视化报告",
        "",
        "本报告由 `ForwardProblem/visualize_forward.py` 自动生成。指标为验证样本上的相对 L2，绘图展示第一个验证样本的输入、真实输出、模型预测和绝对误差。",
        "",
        "| Dataset | Checkpoint | Eval samples | Mean rL2 | Std | Visualization |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for name, item in metrics.items():
        lines.append(
            f"| {name} | `{item['checkpoint']}` | {item['num_eval_samples']} | "
            f"{item['mean_rel_l2']:.6e} | {item['std_rel_l2']:.6e} | `{item['visualization']}` |"
        )
    lines.extend(
        [
            "",
            "说明：当前机器的 `computervision` 环境是 CPU 版 PyTorch，因此脚本默认使用 CPU 推理。若安装 CUDA 版 PyTorch，可追加 `--device cuda`。",
            "Darcy、Elasticity、Plasticity 使用训练集统计量做标准化和反标准化；Airfoil、Pipe 与原训练代码一致，不做标准化。",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LNO checkpoints and create forward-problem visualizations.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_TO_CONFIG.keys()), choices=list(DATASET_TO_CONFIG.keys()))
    parser.add_argument("--max-eval-samples", type=int, default=5, help="Number of validation samples evaluated per dataset.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=str(REPO_DIR / "outputs" / "forward_visualization"))
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for dataset in args.datasets:
        metrics[dataset] = evaluate_dataset(dataset, args.max_eval_samples, device, output_dir)
        print(
            f"[{dataset}] mean rL2={metrics[dataset]['mean_rel_l2']:.6e}, "
            f"figure={metrics[dataset]['visualization']}"
        )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = plot_metric_summary(metrics, output_dir)
    report_path = write_report(metrics, output_dir)
    print(f"[done] metrics: {metrics_path}")
    print(f"[done] summary figure: {summary_path}")
    print(f"[done] report: {report_path}")


if __name__ == "__main__":
    main()
