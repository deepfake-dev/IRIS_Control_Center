import flet as ft
import sounddevice as sd
from kokoro_onnx import Kokoro
import datetime
import os
import sqlite3
import subprocess
import threading
import flet_code_editor as fce
from mappings import *

import pandas as pd
import socket

import lancedb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import LanceDB
from langchain_experimental.text_splitter import SemanticChunker

# ── Active server processes ───────────────────────────────────────────────────
_processes: dict[str, subprocess.Popen] = {}
_log_threads: list[threading.Thread] = []

def ControlPage(page: ft.Page):
    db_path = os.getenv("DATABASE_URL")
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

    def show_content(e, tile: ft.ListTile, row: dict):
        dialog = ft.AlertDialog(
            title=ft.Text(tile.title.value),
            content=fce.CodeEditor(
                language=fce.CodeLanguage.MARKDOWN,
                code_theme=fce.CodeTheme.SOLARIZED_DARK,
                expand=True,
                read_only=True,
                value=row["request_md"]
            ),
            alignment=ft.Alignment.CENTER,
        )
        page.show_dialog(dialog)
    
    def mark_merge(e, tile: ft.ListTile, row: dict):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE requests 
            SET status_merged = ? 
            WHERE id = ?
        ''', (1, int(row["id"])))

        page.pop_dialog()
        
        if cursor.rowcount == 0:
            page.show_dialog(ft.SnackBar(ft.Text("Failed to update nonexistent.")))
        else:
            page.show_dialog(ft.SnackBar(ft.Text(f"Success: Request {row["id"]} updated to status {row["id"]}.")))
            
        conn.commit()
        conn.close()

        retrieve_embeddings()
    
    def ask_merge(e, tile: ft.ListTile, row: dict):
        modal_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Please confirm"),
            content=ft.Text("Do you really want to mark this row as merged? You need to rebuild index after to reflect the changes."),
            actions=[
                ft.TextButton("Yes", on_click=lambda e, t=tile, r=row: mark_merge(e, t, r)),
                ft.TextButton("No", on_click=lambda e: page.pop_dialog()),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        page.show_dialog(modal_dialog)

    def add_tile(row: dict):
        name = f"Entry: {info_type_mapping[row["info_type"]]} | {office_mapping[row["office"]]} | {target_mapping[row["target"]]}"
        tile = ft.ListTile(
            title=ft.Text(name, weight=ft.FontWeight.BOLD),
            subtitle=ft.Text("Merged" if bool(row["status_merged"]) else "Waiting"),
            leading=ft.Icon(ft.Icons.FOLDER_OUTLINED),
        )

        view_btn = ft.IconButton(
            icon=ft.Icons.VISIBILITY_OUTLINED,
            on_click=lambda e, t=tile, r=row: show_content(e, t, r)
        )

        controls = [view_btn]
        width = 32

        if not bool(row["status_merged"]):
            controls.append(
                ft.IconButton(
                    icon=ft.Icons.MERGE_OUTLINED,
                    on_click=lambda e, t=tile, r=row: ask_merge(e, t, r)
                )
            )
            width = 72

        tile.trailing = ft.Row(controls, width=width)
        doc_list_view.controls.append(tile)

    def retrieve_embeddings(e = None):
        doc_list_view.controls = []
        conn = sqlite3.connect(db_path)
        
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM requests 
            ORDER BY status_merged ASC, id DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        for row in results:
            result = dict(row)
            add_tile(result)

    # ── Ready check ───────────────────────────────────────────────────────────

    def check_ready():
        vlm_ready = False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0) # 1 second time out
            try:
                # connect_ex returns 0 if the connection was successful
                result = s.connect_ex(('127.0.0.1', int(vlm_port_tf.value)))
                if result == 0:
                    vlm_ready = True
                    write_log(f"VLM is ready at port {int(vlm_port_tf.value)}.")
                else:
                    vlm_ready = False
                    write_log(f"VLM is NOT ready.")
            except Exception as e:
                vlm_ready = False
                write_log(f"VLM is ready: {e}")

        ready = all([
            webpage_port_tf.value,
            vlm_port_tf.value,
            provenance_port_tf.value,
            avatar_port_tf.value,
            len(doc_list_view.controls) > 0,
            vlm_ready
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
        write_log("Starting safe database update... This may take a minute.")
        
        build_progress.visible = True
        build_button.disabled = True
        page.update()
        
        def _build_task():
            try:
                # 1. Fetch from SQLite
                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM requests WHERE status_merged = 1")
                rows = cursor.fetchall()
                conn.close()

                if not rows:
                    write_log("No merged documents found to index.", is_error=True)
                    return

                # =========================================================
                # NEW: Check LanceDB to see what has already been built
                # =========================================================
                lance_db_path = os.getenv("LANCEDB_URL")
                db = lancedb.connect(lance_db_path)
                table_name = "batstateu_rag_nomic"

                df_old = None
                existing_doc_ids = set()

                if table_name in db.table_names():
                    table = db.open_table(table_name)
                    df_old = table.to_pandas()
                    
                    if "doc_id" in df_old.columns:
                        # Grab all the unique doc_ids currently in LanceDB
                        existing_doc_ids = set(df_old['doc_id'].fillna('').astype(str).unique())

                # Filter the rows: Only keep documents that are NOT in LanceDB yet
                rows_to_build = [r for r in rows if str(r["id"]) not in existing_doc_ids]

                if not rows_to_build:
                    write_log("✅ All merged documents are already indexed. Nothing new to build!")
                    return

                write_log(f"Found {len(rows_to_build)} NEW merged documents. Chunking...")
                
                # =========================================================
                
                # 2. Chunking
                embedder = HuggingFaceEmbeddings(
                    model_name="nomic-ai/nomic-embed-text-v1.5",
                    model_kwargs={'device': 'cpu', 'trust_remote_code': True}
                )
                chunker = SemanticChunker(embedder)
                
                new_chunks = []
                
                # Iterate ONLY over the new rows
                for row in rows_to_build:
                    md_text = row["request_md"]
                    doc_id_str = str(row["id"])
                    
                    meta = {
                        "doc_id": doc_id_str,
                        "source": info_type_mapping.get(row["info_type"], "Unknown"),
                        "office": office_mapping.get(row["office"], "Unknown"),
                        "target": target_mapping.get(row["target"], "Unknown")
                    }
                    new_chunks.extend(chunker.create_documents([md_text], metadatas=[meta]))

                write_log(f"Generated {len(new_chunks)} new chunks. Embedding them now...")

                # 3. Generate vectors manually ONLY for the new chunks
                texts = [chunk.page_content for chunk in new_chunks]
                vectors = embedder.embed_documents(texts)
                
                new_records = []
                for i, chunk in enumerate(new_chunks):
                    record = {
                        "vector": vectors[i],
                        "text": chunk.page_content,
                    }
                    record.update(chunk.metadata) # Add metadata fields
                    new_records.append(record)

                # 4. Safe Pandas Merge
                if df_old is not None:
                    write_log("Merging new records with existing database...")
                    df_new = pd.DataFrame(new_records)
                    
                    # Simply concat the new records to the old ones (no overlaps exist because we filtered them)
                    merged_df = pd.concat([df_old, df_new], ignore_index=True)
                    db.create_table(table_name, data=merged_df, mode="overwrite")
                else:
                    write_log("Creating new database table...")
                    db.create_table(table_name, data=new_records)
                
                write_log("✅ Database updated! Your other information was preserved.")

            except Exception as exc:
                write_log(f"Index rebuild failed: {exc}", is_error=True)
                
            finally:
                build_progress.visible = False
                build_button.disabled = False
                try:
                    page.update()
                except Exception:
                    pass

        threading.Thread(target=_build_task, daemon=True).start()

    # ── Controls ──────────────────────────────────────────────────────────────

    build_progress = ft.ProgressRing(width=16, height=16, stroke_width=2, visible=False)
    build_button = ft.IconButton(icon=ft.Icons.BUILD_OUTLINED, on_click=update_index)

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

    widget = ft.SafeArea(
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
                                        ft.Text("Embedding Documents", size=20, weight=ft.FontWeight.W_600),
                                        ft.Row([
                                            ft.IconButton(icon=ft.Icons.REFRESH_OUTLINED, on_click=retrieve_embeddings),
                                            build_progress,
                                            build_button,
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

    # Seed with demo docs
    write_log("System initialized. Awaiting user input...")
    retrieve_embeddings()
    check_ready()

    write_log(f"Currently in: {os.getcwd()}")

    def on_disconnect(e):
        stop_servers()

    page.on_disconnect = on_disconnect

    return ft.View(
        route="/control",
        bgcolor="#262626",
        padding=0,
        controls=[
            ft.SafeArea(
                expand=True,
                content=widget
            )
        ]
    )