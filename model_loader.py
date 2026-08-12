"""
model_loader.py — Auto-detect hardware & load YOLOv8 model
============================================================
Priority logic:
  - Intel CPU  → OpenVINO (NPU > GPU > CPU) → fallback .pt
  - Non-Intel  → PyTorch .pt (CUDA if available, else CPU)

Model is loaded ONCE at server startup, not per-request.
"""

import os
import logging

log = logging.getLogger(__name__)

# ── File paths ─────────────────────────────────────────────────
OPENVINO_XML  = os.path.join("models", "best_openvino_model", "best.xml")
PT_MODEL_PATH = os.path.join("models", "yolo", "best.pt")

# ── Result container ───────────────────────────────────────────
class ModelInfo:
    """Holds loaded model + runtime metadata."""

    def __init__(self):
        self.backend: str = "none"          # "openvino" | "pytorch"
        self.device: str  = "none"          # e.g. "NPU", "GPU", "CPU", "cuda", "cpu"
        self.cpu_brand: str = "unknown"
        self.cpu_name: str  = "unknown"
        self.available_ov_devices: list = []

        # OpenVINO handles
        self.compiled_model = None
        self.input_layer    = None
        self.output_layer   = None

        # PyTorch handle
        self.pt_model = None

    @property
    def is_loaded(self) -> bool:
        return self.compiled_model is not None or self.pt_model is not None

    @property
    def display_label(self) -> str:
        """Human-readable badge for the frontend."""
        if self.backend == "openvino":
            return f"Intel {self.cpu_name.split('(R)')[-1].strip()} · OpenVINO [{self.device}]"
        elif self.backend == "pytorch":
            dev = self.device.upper()
            brand = "AMD" if "amd" in self.cpu_brand.lower() else self.cpu_brand.title()
            return f"{brand} · PyTorch [{dev}]"
        return "Model tidak termuat"


# ── Internal helpers ───────────────────────────────────────────

def _detect_cpu() -> tuple[str, str]:
    """Return (brand_lower, full_name) using py-cpuinfo."""
    try:
        import cpuinfo
        info  = cpuinfo.get_cpu_info()
        name  = info.get("brand_raw", "Unknown CPU")
        brand = info.get("vendor_id_raw", "").lower()   # e.g. "GenuineIntel", "AuthenticAMD"
        # Normalise: GenuineIntel → "intel"
        if "intel" in brand or "intel" in name.lower():
            brand = "intel"
        elif "amd" in brand or "amd" in name.lower():
            brand = "amd"
        return brand, name
    except Exception as e:
        log.warning(f"py-cpuinfo gagal: {e}")
        return "unknown", "Unknown CPU"


def _load_openvino(info: ModelInfo) -> bool:
    """Try loading OpenVINO model with best available device. Returns True on success."""
    try:
        from openvino import Core
        core = Core()
        available = core.available_devices
        info.available_ov_devices = available
        log.info(f"OpenVINO device tersedia: {available}")

        # Device priority: NPU > CPU > GPU
        # NOTE: GPU is placed LAST because OpenVINO GPU defaults to FP16 precision
        # which causes the YOLOv8 model output to be numerically corrupt
        # (all confidences become 100%, bounding boxes are invalid).
        # CPU uses FP32 by default and gives correct results.
        chosen = "CPU"
        for pref in ["NPU", "CPU", "GPU"]:
            if pref in available:
                chosen = pref
                break

        # Build compile config — always enforce FP32 to avoid precision issues
        config = {"INFERENCE_PRECISION_HINT": "f32"}

        model = core.read_model(OPENVINO_XML)
        try:
            info.compiled_model = core.compile_model(model, chosen, config)
            log.info(f"Compiled dengan precision: FP32 pada device: {chosen}")
        except Exception:
            # Fallback: compile tanpa config (beberapa device tidak support hint ini)
            info.compiled_model = core.compile_model(model, chosen)
            log.info(f"Compiled tanpa precision hint pada device: {chosen}")

        info.input_layer  = info.compiled_model.input(0)
        info.output_layer = info.compiled_model.output(0)
        info.device       = chosen
        info.backend      = "openvino"
        return True

    except Exception as e:
        log.warning(f"OpenVINO load gagal: {e}")
        return False


def _load_pytorch(info: ModelInfo, prefer_cuda: bool = False) -> bool:
    """Try loading .pt model with PyTorch. Returns True on success."""
    try:
        from ultralytics import YOLO
        import torch

        device = "cpu"
        if prefer_cuda and torch.cuda.is_available():
            device = "cuda"

        model = YOLO(PT_MODEL_PATH)
        model.to(device)
        info.pt_model = model
        info.device   = device
        info.backend  = "pytorch"
        return True

    except Exception as e:
        log.warning(f"PyTorch load gagal: {e}")
        return False


# ── Public API ─────────────────────────────────────────────────

def load_model() -> ModelInfo:
    """
    Detect hardware, load the best available model.
    Called ONCE at server startup.
    """
    info = ModelInfo()

    # ── 1. Detect CPU ─────────────────────────────────────────
    brand, name = _detect_cpu()
    info.cpu_brand = brand
    info.cpu_name  = name

    log.info("=" * 60)
    log.info(f"CPU terdeteksi: {name}")
    log.info(f"Brand         : {brand.upper()}")

    # ── 2. Choose strategy ────────────────────────────────────
    if brand == "intel":
        log.info("Strategi      : Intel -> OpenVINO (dengan fallback ke .pt)")

        ov_ok = _load_openvino(info)

        if ov_ok:
            log.info(
                f"Model digunakan: OpenVINO ({OPENVINO_XML}), device: {info.device}"
            )
        else:
            log.warning("OpenVINO gagal - beralih ke fallback PyTorch (.pt)")
            pt_ok = _load_pytorch(info, prefer_cuda=False)
            if pt_ok:
                log.info(f"Model digunakan: PyTorch fallback ({PT_MODEL_PATH}), device: {info.device}")
            else:
                log.error(
                    "Error: Semua model gagal dimuat! "
                    "Pastikan best.pt atau best_openvino_model tersedia."
                )
    else:
        # AMD / ARM / unknown → PyTorch
        log.info(f"Strategi      : Non-Intel ({brand.upper()}) -> PyTorch (.pt)")
        pt_ok = _load_pytorch(info, prefer_cuda=True)

        if pt_ok:
            log.info(f"Model digunakan: PyTorch ({PT_MODEL_PATH}), device: {info.device}")
        else:
            log.error(
                "Error: Model gagal dimuat! "
                "Pastikan best.pt tersedia di folder models/yolo/."
            )

    log.info("=" * 60)
    return info


def run_inference(info: ModelInfo, blob) -> object:
    """
    Run inference with whichever backend is loaded.
    `blob` must be numpy array shape (1,3,H,W) float32 for OpenVINO,
    or the raw image (numpy BGR) when using PyTorch path.

    Returns raw output tensor / result depending on backend.
    Raises RuntimeError if no model loaded.
    """
    if not info.is_loaded:
        raise RuntimeError("Model tidak termuat.")

    if info.backend == "openvino":
        return info.compiled_model({info.input_layer: blob})[info.output_layer]
    else:
        raise NotImplementedError(
            "PyTorch inference harus dipanggil langsung via info.pt_model.predict()"
        )
