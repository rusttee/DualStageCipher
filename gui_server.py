"""Local, dependency-free HTML GUI for the paper-locked cipher project.

The server binds only to 127.0.0.1 and serves no external JavaScript/CSS.
First-run dataset preparation is performed in a background thread.  Once the
OFFLINE_READY marker validates, normal encryption/decryption runs are local-only.
"""
from __future__ import annotations

import contextlib
import io
import json
import mimetypes
import os
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from common import ROOT, OUTPUT, SIPI_CATALOG, CipherConfig, download_sipi
from dataset_manager import ensure_offline_datasets, all_local_files_present, medical_video_paths, prepared_volume_paths
from image_cipher import encrypt_image, decrypt_image
from video_cipher import encrypt_video, decrypt_video
from volume3d_cipher import encrypt_volume, decrypt_volume

WEB = ROOT / "web"
EXT_OUT = OUTPUT / "extended"
EXT_OUT.mkdir(parents=True, exist_ok=True)


def paper_config() -> CipherConfig:
    return CipherConfig(
        y01=0.55427, mu01=20.389429,
        y02=0.54324, mu02=20.385437,
        y03=0.56543, mu03=20.389429,
        k=20.0,
        discard_first_100=True,
        quantization_mode="abs_scale",
        quantization_scale=1e15,
        color_preprocess="official",
    )


def _stamp(tag: str) -> Path:
    p = EXT_OUT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _latest(pattern: str) -> Path | None:
    items = sorted(EXT_OUT.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return items[0] if items else None


class BufferLog(io.TextIOBase):
    def __init__(self):
        self._parts: list[str] = []
        self._lock = threading.Lock()
    def write(self, s):
        if s:
            with self._lock:
                self._parts.append(str(s))
                # Keep memory bounded but retain useful tail.
                if sum(map(len, self._parts)) > 1_000_000:
                    self._parts = ["".join(self._parts)[-700_000:]]
        return len(s)
    def flush(self):
        return None
    def text(self):
        with self._lock:
            return "".join(self._parts)


SETUP = {"state": "queued", "log": BufferLog(), "error": None}
JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()


def _setup_worker():
    SETUP["state"] = "running"
    try:
        with contextlib.redirect_stdout(SETUP["log"]), contextlib.redirect_stderr(SETUP["log"]):
            ensure_offline_datasets()
        SETUP["state"] = "ready"
    except Exception as exc:
        SETUP["state"] = "error"
        SETUP["error"] = repr(exc)
        SETUP["log"].write(f"\nERROR: {exc}\n")


def start_setup():
    if all_local_files_present():
        SETUP["state"] = "ready"
        SETUP["log"].write("Medical dataset cache: READY. No network access is needed for normal runs.\n")
        return
    threading.Thread(target=_setup_worker, daemon=True).start()


def _resolve_input(operation: str, source: str) -> Path:
    if source.startswith("paper:"):
        if operation != "0":
            raise ValueError("Paper image identifiers are valid only for image encryption.")
        return download_sipi(source.split(":", 1)[1])
    p = Path(source).expanduser()
    if not p.is_absolute():
        # Allow project-relative paths from the GUI.
        p2 = ROOT / p
        if p2.exists():
            p = p2
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _job_worker(job_id: str, req: dict):
    job = JOBS[job_id]
    job["state"] = "running"
    log: BufferLog = job["_log"]
    try:
        # Only one heavy operation at a time. This also makes stdout redirection safe.
        with RUN_LOCK, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            op = str(req.get("operation"))
            source = _resolve_input(op, str(req.get("source", "")))
            trace_index = max(0, int(req.get("trace_index", 0) or 0))
            config = paper_config()
            trace_path: Path | None = None
            primary: Path | None = None

            if op == "0":
                out = _stamp("image_encrypt")
                primary = encrypt_image(source, out, config, show_steps=False)
                trace_path = out / "web_dashboard" / "trace_dashboard.json"
            elif op == "1":
                out = _stamp("image_decrypt")
                decrypt_image(source, out, show_steps=False)
                primary = out / "06_recovered_image.png"
                trace_path = out / "web_dashboard" / "trace_dashboard.json"
            elif op == "2":
                out = _stamp("video_encrypt")
                primary = out / "encrypted_video.mkv"
                trace_dir = out / f"frame_{trace_index:06d}_trace"
                encrypt_video(source, primary, config, trace_frame=trace_index, trace_dir=trace_dir, show_steps=False)
                trace_path = trace_dir / "web_dashboard" / "trace_dashboard.json"
            elif op == "3":
                out = _stamp("video_decrypt")
                primary = out / "decrypted_video.mkv"
                trace_dir = out / f"frame_{trace_index:06d}_trace"
                decrypt_video(source, primary, trace_frame=trace_index, trace_dir=trace_dir, show_steps=False)
                trace_path = trace_dir / "web_dashboard" / "trace_dashboard.json"
            elif op == "4":
                out = _stamp("volume3d_encrypt")
                primary = out / "encrypted_volume.npz"
                trace_dir = out / f"slice_{trace_index:03d}_trace"
                encrypt_volume(source, primary, config, bool(req.get("depth_permutation", False)),
                               trace_slice=trace_index, trace_dir=trace_dir, show_steps=False)
                trace_path = trace_dir / "web_dashboard" / "trace_dashboard.json"
            elif op == "5":
                out = _stamp("volume3d_decrypt")
                primary = out / "decrypted_volume.npy"
                trace_dir = out / f"slice_{trace_index:03d}_trace"
                decrypt_volume(source, primary, trace_slice=trace_index, trace_dir=trace_dir, show_steps=False)
                trace_path = trace_dir / "web_dashboard" / "trace_dashboard.json"
            else:
                raise ValueError("Operation must be 0..5")

            job["output_dir"] = str(out)
            job["primary"] = str(primary) if primary else None
            job["trace_path"] = str(trace_path) if trace_path and trace_path.exists() else None
            if trace_path and not trace_path.exists():
                log.write(f"\nDetailed trace was not created at {trace_path}. The requested frame/slice may be outside the source range.\n")
        job["state"] = "done"
    except Exception as exc:
        job["state"] = "error"
        job["error"] = str(exc)
        log.write(f"\nERROR: {exc}\n")


def start_job(req: dict) -> str:
    if SETUP["state"] != "ready":
        raise RuntimeError("Offline dataset preparation has not completed yet.")
    with JOB_LOCK:
        if any(j.get("state") in {"queued", "running"} for j in JOBS.values()):
            raise RuntimeError("Another encryption/decryption job is already running.")
        jid = uuid.uuid4().hex[:12]
        JOBS[jid] = {"state": "queued", "created": time.time(), "_log": BufferLog(), "output_dir": None,
                     "trace_path": None, "primary": None, "error": None}
    threading.Thread(target=_job_worker, args=(jid, req), daemon=True).start()
    return jid


def options_payload() -> dict:
    images = [{"label": f"{image_id} — {spec['name']} (paper dataset)", "value": f"paper:{image_id}"}
              for image_id, spec in SIPI_CATALOG.items()]
    videos = [{"label": f"{p.name} — full medical echocardiography clip", "value": str(p)}
              for p in medical_video_paths() if p.exists()]
    volumes = [{"label": f"{p.name} — full prepared AOMIC T1 brain MRI", "value": str(p)}
               for p in prepared_volume_paths() if p.exists()]
    latest = {
        "image": str(_latest("encrypted_image.npz") or ""),
        "video": str(_latest("encrypted_video.mkv") or ""),
        "volume": str(_latest("encrypted_volume.npz") or ""),
    }
    return {"images": images, "videos": videos, "volumes": volumes, "latest": latest,
            "offline_ready": all_local_files_present()}


def _safe_project_path(raw: str) -> Path:
    p = Path(raw).expanduser().resolve()
    root = ROOT.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise PermissionError("Only artifacts inside this project may be served by the local GUI.")
    return p


class Handler(BaseHTTPRequestHandler):
    server_version = "NatureCipherGUI/4.0"

    def log_message(self, fmt, *args):
        # Keep terminal clean; processing logs are shown in the GUI.
        return

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(data)

    def _file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(404); return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                return self._file(WEB / "index.html")
            if u.path.startswith("/static/"):
                name = Path(u.path[len("/static/"):]).name
                return self._file(WEB / name)
            if u.path == "/api/setup":
                return self._json({"state": SETUP["state"], "log": SETUP["log"].text(), "error": SETUP["error"]})
            if u.path == "/api/options":
                return self._json(options_payload())
            if u.path == "/api/job":
                jid = parse_qs(u.query).get("id", [""])[0]
                j = JOBS.get(jid)
                if not j: return self._json({"error": "Unknown job"}, 404)
                return self._json({k: (v.text() if k == "_log" else v) for k,v in j.items() if k != "_log"} | {"log": j["_log"].text()})
            if u.path == "/api/trace":
                raw = parse_qs(u.query).get("path", [""])[0]
                p = _safe_project_path(raw)
                if p.name != "trace_dashboard.json":
                    raise PermissionError("Only trace_dashboard.json may be loaded through this endpoint.")
                return self._json(json.loads(p.read_text(encoding="utf-8")))
            if u.path == "/artifact":
                raw = parse_qs(u.query).get("path", [""])[0]
                return self._file(_safe_project_path(raw))
            self.send_error(404)
        except PermissionError as exc:
            self._json({"error": str(exc)}, 403)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/start":
            self.send_error(404); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            jid = start_job(req)
            self._json({"job_id": jid})
        except Exception as exc:
            self._json({"error": str(exc)}, 400)


def launch(host: str = "127.0.0.1", port: int = 0, open_browser: bool = True):
    start_setup()
    httpd = ThreadingHTTPServer((host, port), Handler)
    actual_port = httpd.server_address[1]
    url = f"http://{host}:{actual_port}/"
    print("=" * 78)
    print("PAPER-LOCKED CHAOTIC CIPHER — LOCAL HTML GUI")
    print(url)
    print("The GUI is local-only. Press Ctrl+C here to stop it.")
    print("=" * 78)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    launch()
