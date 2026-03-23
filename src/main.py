import flet as ft
import sounddevice as sd
from kokoro_onnx import Kokoro
import datetime
import os
import subprocess
import threading
from pathlib import Path

# ── Active server processes ───────────────────────────────────────────────────
_processes: dict[str, subprocess.Popen] = {}
_log_threads: list[threading.Thread] = []

def main(page: ft.Page):
    page.title = "IRIS Control Center"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width = 960
    page.window.height = 800
    page.window.resizable = False
    page.window.maximizable = False
    page.views[0].padding = ft.Padding.all(0)

    log_view = ft.ListView(expand=True, auto_scroll=True, spacing=4)

    def write_log(message: str, is_error: bool = False):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        color = "#ff6b6b" if is_error else "#BDD1D9"
        log_view.controls.append(
            ft.Text(f"[{timestamp}] {message}", color=color, font_family="JetBrains Mono")
        )
        page.update()

    try:
        tts = Kokoro("models/tts/kokoro-v1.0.onnx", "models/tts/voices-v1.0.bin")
        write_log("Kokoro TTS model loaded successfully.")
    except Exception as e:
        tts = None
        write_log(f"Failed to load Kokoro TTS: {e}", is_error=True)

    def play_voice(is_female: bool):
        if not tts:
            write_log("Cannot play voice: TTS engine is not initialized.", is_error=True)
            return
        try:
            dropdown = female_voice_dd if is_female else male_voice_dd
            character = dropdown.text.split("(")[0].strip() if dropdown.text else "Iris"
            lang = "en-gb" if "British" in (dropdown.text or "") else "en-us"
            write_log(f"Generating audio for {character} ({lang})...")
            samples, sample_rate = tts.create(
                f"Hello! I am {character}. It is nice to meet you.",
                voice=dropdown.value,
                speed=1.0,
                lang=lang,
            )
            write_log("Playing audio...")
            sd.play(samples, sample_rate)
            sd.wait()
            write_log("Playback complete.")
        except Exception as e:
            write_log(f"Audio playback error: {e}", is_error=True)

    # ── Document list ─────────────────────────────────────────────────────────

    doc_list_view = ft.ListView(expand=True, scroll=ft.ScrollMode.AUTO)

    def remove_document(e, tile_ref):
        doc_list_view.controls.remove(tile_ref)
        write_log(f"Removed document: {tile_ref.title.value}")
        check_ready()
        page.update()

    def add_tile(file: ft.FilePickerFile):
        tile = ft.ListTile(
            title=ft.Text(file.name, weight=ft.FontWeight.BOLD),
            subtitle=ft.Text(file.path),
            leading=ft.Icon(ft.Icons.FOLDER_OUTLINED),
        )
        del_btn = ft.IconButton(
            icon=ft.Icons.DELETE,
            on_click=lambda e, t=tile: remove_document(e, t)
        )
        tile.trailing = del_btn
        doc_list_view.controls.append(tile)

    async def pick_files():
        files = await ft.FilePicker().pick_files(
            allow_multiple=True,
            allowed_extensions=["pdf", "docx", "png", "jpg"]
        )
        if files:
            for f in files:
                add_tile(f)
                write_log(f"Added document: {f.name}")
            check_ready()
            page.update()

    # ── Ready check ───────────────────────────────────────────────────────────

    def check_ready():
        ready = all([
            webpage_port_tf.value,
            vlm_port_tf.value,
            provenance_port_tf.value,
            avatar_port_tf.value,
            len(doc_list_view.controls) > 0,
        ])
        start_switch.disabled = not ready
        page.update()

    # ── Server management ─────────────────────────────────────────────────────

    def _stream_output(name: str, proc: subprocess.Popen, color: str):
        """Read stdout/stderr from a subprocess and pipe to the log view."""
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if line:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                log_view.controls.append(
                    ft.Text(
                        f"[{timestamp}] [{name}] {line}",
                        color=color,
                        font_family="JetBrains Mono",
                        size=12,
                    )
                )
                # Throttle UI updates to avoid flooding
                if len(log_view.controls) % 5 == 0:
                    try:
                        page.update()
                    except Exception:
                        pass
        try:
            page.update()
        except Exception:
            pass

    def start_servers():
        global _processes, _log_threads
        _log_threads.clear()

        # Collect docs as a space-separated list of paths for the RAG indexer
        doc_paths = " ".join(
            f'"{t.subtitle.value}"' for t in doc_list_view.controls
        )

        servers = [
            {
                "name": "KIOSK",
                "color": "#74c0fc",
                "cmd": [
                    "./.venv/Scripts/python.exe", "-m", "http.server",
                    webpage_port_tf.value,
                    "--directory", f"{os.path.join(os.getcwd(), "avatar")}"
                ],
            },
            {
                "name": "AVATAR",
                "color": "#a9e34b",
                "cmd": ["./.venv/Scripts/python.exe", "scripts/assistant/avatar.py", "--port", avatar_port_tf.value],
            },
            {
                "name": "PROVENANCE",
                "color": "#ffd43b",
                "cmd": [
                    "./.venv/Scripts/uvicorn.exe", "server:app",
                    "--host", "0.0.0.0",
                    "--port", provenance_port_tf.value,
                    '--app-dir', f"{os.path.join(os.getcwd(), "scripts", "provenance_checker")}"
                ],
            },
            {
                "name": "VLM",
                "color": "#da77f2",   # purple — informational only
                "cmd": [
                    './server/llama-server.exe', '--alias', 'qwen3-vl-instruct',
                    '--model', f"models/vlm/Qwen3VL-2B-Instruct-Q4_K_M.gguf",
                    '--mmproj', f"models/vlm/mmproj-Qwen3VL-2B-Instruct-F16.gguf",
                    '--n-gpu-layers', '99', '--ctx-size', '32768',
                    '--port', vlm_port_tf.value, '--no-mmap'
                ],
                "note": f"VLM expected on port {vlm_port_tf.value} — start llama-server manually.",
            },
            # .\llama-server.exe --alias qwen3-vl-instruct --model C:\Users\owen\Desktop\llamacpp\ggufs\Qwen3VL-4B-Instruct-Q4_K_M.gguf --mmproj C:/Users/owen/Desktop/llamacpp/ggufs/mmproj-Qwen3VL-4B-Instruct-F16.gguf
            # --n-gpu-layers 99 --ctx-size 32768 --port 8001 --no-mmap
        ]

        for srv in servers:
            if srv.get("cmd") is None:
                write_log(f"[{srv['name']}] {srv['note']}")
                continue

            try:
                proc = subprocess.Popen(
                    srv["cmd"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                )
                _processes[srv["name"]] = proc
                write_log(f"[{srv['name']}] Started (PID {proc.pid})")

                t = threading.Thread(
                    target=_stream_output,
                    args=(srv["name"], proc, srv["color"]),
                    daemon=True,
                )
                t.start()
                _log_threads.append(t)

            except FileNotFoundError as exc:
                write_log(f"[{srv['name']}] Failed to start: {exc}", is_error=True)

    def stop_servers():
        for name, proc in _processes.items():
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                write_log(f"[{name}] Stopped.")
        _processes.clear()

    def toggle_system(e):
        if e.control.value:
            write_log("─" * 48)
            write_log("IRIS System ONLINE — starting servers...")
            start_servers()
        else:
            write_log("─" * 48)
            write_log("IRIS System OFFLINE — stopping servers...")
            stop_servers()

    def update_index(e):
        write_log("Refreshing document index...")
        check_ready()

    # ── Controls ──────────────────────────────────────────────────────────────

    start_switch = ft.Switch(
        label="Toggle IRIS System",
        label_position=ft.LabelPosition.LEFT,
        on_change=toggle_system,
        label_text_style=ft.TextStyle(weight=ft.FontWeight.BOLD),
        disabled=True,
    )

    female_voice_dd = ft.Dropdown(
        border_color=ft.Colors.WHITE_54,
        expand=True,
        value="bf_isabella",
        options=[
            ft.DropdownOption(key="af_alloy",    text="Alloy"),
            ft.DropdownOption(key="af_aoede",    text="Aoede"),
            ft.DropdownOption(key="af_bella",    text="Bella"),
            ft.DropdownOption(key="af_heart",    text="Heart"),
            ft.DropdownOption(key="af_jessica",  text="Jessica"),
            ft.DropdownOption(key="af_kore",     text="Kore"),
            ft.DropdownOption(key="af_nicole",   text="Nicole"),
            ft.DropdownOption(key="af_nova",     text="Nova"),
            ft.DropdownOption(key="af_river",    text="River"),
            ft.DropdownOption(key="af_sarah",    text="Sarah"),
            ft.DropdownOption(key="af_sky",      text="Sky"),
            ft.DropdownOption(key="bf_alice",    text="Alice (British)"),
            ft.DropdownOption(key="bf_emma",     text="Emma (British)"),
            ft.DropdownOption(key="bf_isabella", text="Isabella (British)"),
            ft.DropdownOption(key="bf_lily",     text="Lily (British)"),
        ],
    )

    male_voice_dd = ft.Dropdown(
        expand=True,
        border_color=ft.Colors.WHITE_54,
        value="am_echo",
        options=[
            ft.DropdownOption(key="am_adam",    text="Adam"),
            ft.DropdownOption(key="am_echo",    text="Echo"),
            ft.DropdownOption(key="am_eric",    text="Eric"),
            ft.DropdownOption(key="am_fenrir",  text="Fenrir"),
            ft.DropdownOption(key="am_liam",    text="Liam"),
            ft.DropdownOption(key="am_michael", text="Michael"),
            ft.DropdownOption(key="am_puck",    text="Puck"),
            ft.DropdownOption(key="am_onyx",    text="Onyx"),
            ft.DropdownOption(key="am_santa",   text="Santa"),
            ft.DropdownOption(key="bm_daniel",  text="Daniel (British)"),
            ft.DropdownOption(key="bm_fable",   text="Fable (British)"),
            ft.DropdownOption(key="bm_george",  text="George (British)"),
            ft.DropdownOption(key="bm_lewis",   text="Lewis (British)"),
        ],
    )

    webpage_port_tf = ft.TextField(
        label="Kiosk Webpage Port",
        value="5050",
        border_color=ft.Colors.WHITE_54,
        expand=True,
        height=80,
        on_change=lambda e: check_ready(),
    )
    vlm_port_tf = ft.TextField(
        label="VLM Server Port",
        value="8001",
        border_color=ft.Colors.WHITE_54,
        expand=True,
        on_change=lambda e: check_ready(),
    )
    provenance_port_tf = ft.TextField(
        label="Provenance Site Port",
        value="4321",
        border_color=ft.Colors.WHITE_54,
        expand=True,
        on_change=lambda e: check_ready(),
    )
    avatar_port_tf = ft.TextField(
        label="Avatar WebSocket Port",
        value="8080",
        border_color=ft.Colors.WHITE_54,
        expand=True,
        on_change=lambda e: check_ready(),
    )

    # ── Layout ────────────────────────────────────────────────────────────────

    page.add(
        ft.SafeArea(
            expand=True,
            content=ft.Container(
                expand=True,
                bgcolor="#30302E",
                content=ft.Column(
                    expand=True,
                    alignment=ft.MainAxisAlignment.START,
                    horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                    controls=[
                        # Header
                        ft.Container(
                            bgcolor="#262626",
                            padding=ft.Padding.all(16),
                            content=ft.Row(
                                expand=True,
                                alignment=ft.MainAxisAlignment.CENTER,
                                controls=[
                                    ft.Image(src="icon.png", width=48, height=48),
                                    ft.Text("IRIS Control Center", size=32, weight=ft.FontWeight.BOLD),
                                    ft.Container(
                                        expand=True,
                                        alignment=ft.Alignment.CENTER_RIGHT,
                                        content=start_switch,
                                    ),
                                ],
                            ),
                        ),
                        # Section labels
                        ft.Container(
                            padding=ft.Padding.symmetric(horizontal=16),
                            content=ft.Row(
                                controls=[
                                    ft.Text("Configuration", size=20, weight=ft.FontWeight.W_600, expand=True),
                                    ft.Row(
                                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                        expand=True,
                                        controls=[
                                            ft.Text("Embedded Documents", size=20, weight=ft.FontWeight.W_600),
                                            ft.Row([
                                                ft.IconButton(icon=ft.Icons.ADD_OUTLINED, on_click=pick_files),
                                                ft.IconButton(icon=ft.Icons.UPDATE_OUTLINED, on_click=update_index),
                                            ]),
                                        ],
                                    ),
                                ],
                            ),
                        ),
                        # Main body
                        ft.Container(
                            padding=ft.Padding.all(16),
                            expand=True,
                            content=ft.Row(
                                expand=True,
                                controls=[
                                    # Left — config
                                    ft.Column(
                                        spacing=16,
                                        expand=True,
                                        controls=[
                                            ft.Text("Ports", weight=ft.FontWeight.W_500),
                                            ft.Container(
                                                expand=True,
                                                content=ft.Column(
                                                    expand=True,
                                                    controls=[
                                                        webpage_port_tf,
                                                        vlm_port_tf,
                                                        provenance_port_tf,
                                                        avatar_port_tf,
                                                    ],
                                                ),
                                            ),
                                            ft.Row([
                                                ft.Row(expand=True, controls=[
                                                    ft.Text("Female Voice", weight=ft.FontWeight.W_500, expand=True),
                                                    ft.IconButton(ft.Icons.PLAY_ARROW, on_click=lambda e: play_voice(True)),
                                                ]),
                                                ft.Row(expand=True, controls=[
                                                    ft.Text("Male Voice", weight=ft.FontWeight.W_500, expand=True),
                                                    ft.IconButton(ft.Icons.PLAY_ARROW, on_click=lambda e: play_voice(False)),
                                                ]),
                                            ]),
                                            ft.Container(content=ft.Row(controls=[female_voice_dd, male_voice_dd])),
                                        ],
                                    ),
                                    # Right — documents
                                    doc_list_view,
                                ],
                            ),
                        ),
                        # Log panel
                        ft.Container(
                            bgcolor="#0d1117",
                            margin=ft.Margin.only(top=0, left=16, right=16, bottom=16),
                            padding=ft.Padding.all(16),
                            height=230,
                            border_radius=8,
                            shadow=ft.BoxShadow(spread_radius=0.2, blur_radius=1, offset=ft.Offset(y=2)),
                            content=log_view,
                        ),
                    ],
                ),
            ),
        )
    )

    # Seed with demo docs
    write_log("System initialized. Awaiting user input...")
    add_tile(ft.FilePickerFile(0, "BROCHURE.pdf", size=652000, path="C:/Users/owen/Downloads/BROCHURE.pdf"))
    add_tile(ft.FilePickerFile(1, "OSD manual FINAL.pdf", size=652000, path="C:/Users/owen/Downloads/OSD manual FINAL.pdf"))
    add_tile(ft.FilePickerFile(2, "BatStateU-Citizens-Charter-2025-01-Final.pdf", size=652000, path="C:/Users/owen/Downloads/BatStateU-Citizens-Charter-2025-01-Final.pdf"))
    check_ready()

    write_log(f"Currently in: {os.getcwd()}")

    # Clean up on window close
    def on_disconnect(e):
        stop_servers()

    page.on_disconnect = on_disconnect

ft.run(main)