"""GPU detection and capability probing for NVIDIA CUDA acceleration."""

import subprocess
import json
import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    """Detected GPU capabilities."""
    available: bool = False
    gpu_name: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    vram_total_mb: int = 0
    vram_free_mb: int = 0

    # FFmpeg CUDA support
    hwaccel_supported: bool = False
    cuvid_decoders: List[str] = field(default_factory=list)
    cuda_filters: List[str] = field(default_factory=list)
    nvenc_encoders: List[str] = field(default_factory=list)

    # Codec → CUVID decoder mapping (e.g. "h264" → "h264_cuvid")
    codec_decoder_map: Dict[str, str] = field(default_factory=dict)

    def supports_codec(self, codec_name: str) -> bool:
        """Check if GPU can decode a given codec."""
        if not codec_name:
            return False
        normalized = codec_name.lower().strip()
        return normalized in self.codec_decoder_map

    def get_decoder(self, codec_name: str) -> Optional[str]:
        """Get the CUVID decoder name for a codec, or None."""
        if not codec_name:
            return None
        return self.codec_decoder_map.get(codec_name.lower().strip())

    def summary(self) -> str:
        if not self.available:
            return "No NVIDIA GPU detected — using CPU processing"
        parts = [
            f"GPU: {self.gpu_name}",
            f"Driver: {self.driver_version}",
            f"VRAM: {self.vram_total_mb} MB",
            f"CUDA decoders: {', '.join(self.cuvid_decoders) or 'none'}",
            f"CUDA filters: {', '.join(self.cuda_filters) or 'none'}",
        ]
        return " | ".join(parts)


# ── Singleton cache ──────────────────────────────────────────────

_cached_gpu_info: Optional[GPUInfo] = None


def detect_gpu() -> GPUInfo:
    """Detect GPU capabilities (cached after first call)."""
    global _cached_gpu_info
    if _cached_gpu_info is not None:
        return _cached_gpu_info

    info = GPUInfo()
    _creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0

    # ── 1. nvidia-smi ─────────────────────────────────────────────
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            timeout=10,
            creationflags=_creation_flags,
        )
        if smi.returncode == 0:
            line = smi.stdout.decode("utf-8", errors="replace").strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                info.available = True
                info.gpu_name = parts[0]
                info.driver_version = parts[1]
                info.vram_total_mb = int(float(parts[2]))
                info.vram_free_mb = int(float(parts[3]))
    except Exception:
        pass

    if not info.available:
        _cached_gpu_info = info
        return info

    # ── 2. FFmpeg hwaccels ────────────────────────────────────────
    try:
        hw = subprocess.run(
            ["ffmpeg", "-hwaccels"],
            capture_output=True,
            timeout=10,
            creationflags=_creation_flags,
        )
        output = hw.stdout.decode("utf-8", errors="replace")
        if "cuda" in output.lower():
            info.hwaccel_supported = True
    except Exception:
        pass

    if not info.hwaccel_supported:
        _cached_gpu_info = info
        return info

    # ── 3. CUVID decoders ─────────────────────────────────────────
    try:
        dec = subprocess.run(
            ["ffmpeg", "-decoders"],
            capture_output=True,
            timeout=10,
            creationflags=_creation_flags,
        )
        for line in dec.stdout.decode("utf-8", errors="replace").splitlines():
            match = re.search(r"(\w+_cuvid)\s", line)
            if match:
                decoder_name = match.group(1)
                info.cuvid_decoders.append(decoder_name)
                # Map codec name to decoder, e.g. "h264_cuvid" → key "h264"
                codec = decoder_name.replace("_cuvid", "")
                info.codec_decoder_map[codec] = decoder_name
    except Exception:
        pass

    # Also add common aliases
    alias_map = {
        "avc": "h264",
        "avc1": "h264",
        "h265": "hevc",
    }
    for alias, canonical in alias_map.items():
        if canonical in info.codec_decoder_map:
            info.codec_decoder_map[alias] = info.codec_decoder_map[canonical]

    # ── 4. CUDA filters ──────────────────────────────────────────
    try:
        flt = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True,
            timeout=10,
            creationflags=_creation_flags,
        )
        for line in flt.stdout.decode("utf-8", errors="replace").splitlines():
            match = re.search(r"(\w+_cuda)\b", line)
            if match:
                info.cuda_filters.append(match.group(1))
            # Also catch hwupload_cuda / hwdownload
            for kw in ("hwupload_cuda", "hwupload", "hwdownload"):
                if kw in line and kw not in info.cuda_filters:
                    info.cuda_filters.append(kw)
    except Exception:
        pass

    # ── 5. NVENC encoders (for potential future use) ──────────────
    try:
        enc = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            timeout=10,
            creationflags=_creation_flags,
        )
        for line in enc.stdout.decode("utf-8", errors="replace").splitlines():
            match = re.search(r"(\w+_nvenc)\s", line)
            if match:
                info.nvenc_encoders.append(match.group(1))
    except Exception:
        pass

    print(f"[GPU] {info.summary()}")
    _cached_gpu_info = info
    return info


def get_gpu_info() -> GPUInfo:
    """Return cached GPU info (runs detection on first call)."""
    return detect_gpu()
