"""
model_loader.py — Auto-detect hardware & load YOLOv8 model
============================================================
Priority logic:
  - Intel CPU  → OpenVINO INT8 (NPU > GPU > CPU) → fallback .pt
  - Non-Intel  → PyTorch .pt (CUDA if available, else CPU)

Model INT8 (quantized) is used for Intel hardware:
  - NPU  : Optimal — INT8 native, low power, highest efficiency
  - GPU  : Good    — INT8 accelerated via OpenVINO
  - CPU  : Fallback — INT8 gives ~2x speedup vs FP32

Model is loaded ONCE at server startup, not per-request.
"""

import os
import logging

log = logging.getLogger(__name__)

# ── File paths ─────────────────────────────────────────────────
# INT8 model — optimized for Intel NPU/GPU/CPU via OpenVINO quantization
OPENVINO_INT8_XML = os.path.join("models", "openvino_int8", "best.xml")
# Fallback FP32 model (legacy path, kept for backward compatibility)
OPENVINO_FP32_XML = os.path.join("models", "best_openvino_model", "best.xml")
# Active model path: prefer INT8, fall back to FP32 if not found
OPENVINO_XML  = OPENVINO_INT8_XML if os.path.exists(OPENVINO_INT8_XML) else OPENVINO_FP32_XML
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
        self.quantization: str = "none"     # "int8" | "fp32" | "none"

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
            quant_tag = f" INT8" if self.quantization == "int8" else ""
            cpu_short = self.cpu_name.split('(R)')[-1].strip()
            return f"Intel {cpu_short} · OpenVINO{quant_tag} [{self.device}]"
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
    """
    Try loading OpenVINO model with best available device.
    INT8 model → Device priority: NPU > GPU > CPU
    - NPU: native INT8 acceleration, lowest power, best for always-on monitoring
    - GPU: hardware INT8 via Intel iGPU
    - CPU: software INT8, ~2x faster than FP32
    Returns True on success.
    """
    try:
        from openvino import Core
        core = Core()
        available = core.available_devices
        info.available_ov_devices = available
        log.info(f"OpenVINO device tersedia: {available}")

        # Determine if we're using INT8 model
        is_int8 = (OPENVINO_XML == OPENVINO_INT8_XML)
        quant_label = "INT8" if is_int8 else "FP32"
        log.info(f"Model path    : {OPENVINO_XML} ({quant_label})")

        model = core.read_model(OPENVINO_XML)

        # Device priority: NPU > GPU > CPU
        # INT8 model is fully compatible with all three devices.
        # NPU is prioritized because INT8 is its native precision → best perf/watt.
        devices_to_try = [dev for dev in ["NPU", "GPU", "CPU"] if dev in available]
        if not devices_to_try:
            devices_to_try = ["CPU"]  # absolute fallback

        compiled = None
        chosen_device = None

        for dev in devices_to_try:
            try:
                # Build device-specific performance config
                config = {}
                if dev == "NPU":
                    # NPU: INT8 native, enable throughput mode for continuous stream
                    config = {
                        "PERFORMANCE_HINT": "THROUGHPUT",
                        "CACHE_DIR":        "./models/.ov_cache",
                    }
                elif dev == "GPU":
                    # GPU: INT8 acceleration, enable caching
                    config = {
                        "PERFORMANCE_HINT": "LATENCY",
                        "CACHE_DIR":        "./models/.ov_cache",
                    }
                elif dev == "CPU":
                    # CPU: INT8 runtime, latency-optimized for single-stream
                    config = {
                        "PERFORMANCE_HINT":       "LATENCY",
                        "INFERENCE_NUM_THREADS":  "0",  # auto-detect optimal threads
                    }

                compiled = core.compile_model(model, dev, config)
                chosen_device = dev
                log.info(f"✓ Compiled OpenVINO {quant_label} model pada device: {dev}")
                break
            except Exception as e:
                log.warning(f"Gagal compile OpenVINO pada device {dev} (with config): {e}")
                # Retry without config
                try:
                    compiled = core.compile_model(model, dev)
                    chosen_device = dev
                    log.info(f"✓ Compiled OpenVINO {quant_label} (tanpa config) pada device: {dev}")
                    break
                except Exception as e2:
                    log.warning(f"Gagal compile fallback OpenVINO pada device {dev}: {e2}")

        if compiled is None:
            log.warning("Semua device OpenVINO gagal dicompile.")
            return False

        info.compiled_model = compiled
        info.input_layer    = compiled.input(0)
        info.output_layer   = compiled.output(0)
        info.device         = chosen_device
        info.backend        = "openvino"
        info.quantization   = "int8" if is_int8 else "fp32"
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
        int8_available = os.path.exists(OPENVINO_INT8_XML)
        log.info(f"Strategi      : Intel -> OpenVINO {'INT8' if int8_available else 'FP32'} (NPU>GPU>CPU, fallback ke .pt)")
        log.info(f"INT8 model    : {'TERSEDIA ✓' if int8_available else 'TIDAK ADA — pakai FP32'}")

        ov_ok = _load_openvino(info)

        if ov_ok:
            log.info(
                f"Model digunakan: OpenVINO {info.quantization.upper()} ({OPENVINO_XML}), device: {info.device}"
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
