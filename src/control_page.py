import flet as ft
import sounddevice as sd
from kokoro_onnx import Kokoro
import datetime
import os
import subprocess
import threading
import tempfile
import socket
import pandas as pd

import flet_code_editor as fce
from mappings import *

# Vector DB & Embeddings
import lancedb
from langchain_huggingface import HuggingFaceEmbeddings

# Supabase
from supabase import create_client, Client

# Docling & Tokenizer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.datamodel.base_models import InputFormat
from transformers import AutoTokenizer

import pyarrow as pa
from sentence_transformers import CrossEncoder

# ── Active server processes ───────────────────────────────────────────────────
_processes: dict[str, subprocess.Popen] = {}
_log_threads: list[threading.Thread] = []

def ControlPage(page: ft.Page):
    # Supabase Setup
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    def get_supabase() -> Client | None:
        if not supabase_url or not supabase_key:
            return None
        return create_client(supabase_url, supabase_key)

    log_view = ft.ListView(expand=True, auto_scroll=True, spacing=4)

    def write_log(message: str, is_error: bool = False):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        color = "#ff6b6b" if is_error else "#BDD1D9"
        log_view.controls.append(
            ft.Text(f"[{timestamp}] {message}", color=color, font_family="JetBrains Mono", selectable=True)
        )

        if len(log_view.controls) % 5 == 0:
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

    doc_list_view = ft.ListView(expand=4, scroll=ft.ScrollMode.AUTO)

    def show_content(e, tile: ft.ListTile, row: dict):
        dialog = ft.AlertDialog(
            title=ft.Text(tile.title.value),
            content=fce.CodeEditor(
                language=fce.CodeLanguage.MARKDOWN,
                code_theme=fce.CodeTheme.SOLARIZED_DARK,
                expand=True,
                read_only=True,
                value=row.get("request", "")
            ),
            alignment=ft.Alignment.CENTER,
        )
        page.show_dialog(dialog)
    
    def mark_merge(e, tile: ft.ListTile, row: dict):
        supabase = get_supabase()
        if not supabase:
            page.show_dialog(ft.SnackBar(ft.Text("Database connection error.")))
            return

        try:
            # Now calling the updated markdown RPC
            response = supabase.rpc("mark_markdown_merged", {"p_id": int(row["id"])}).execute()
            
            page.pop_dialog()
            
            if response.data is True:
                page.show_dialog(ft.SnackBar(ft.Text(f"Success: Document {row['id']} updated to merged.")))
            else:
                page.show_dialog(ft.SnackBar(ft.Text("Failed to update nonexistent row.")))
        except Exception as ex:
            page.pop_dialog()
            page.show_dialog(ft.SnackBar(ft.Text(f"Error: {ex}")))

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
        info_type = row.get("info_type", "")
        office = row.get("office", "")
        target = row.get("target", "")
        status_merged = bool(row.get("status", 0))

        name = f"Entry: {info_type_mapping.get(info_type, 'Unknown')} | {office_mapping.get(office, 'Unknown')} | {target_mapping.get(target, 'Unknown')}"
        tile = ft.ListTile(
            title=ft.Text(name, weight=ft.FontWeight.BOLD),
            subtitle=ft.Text("Merged" if status_merged else "Waiting"),
            leading=ft.Icon(ft.Icons.FOLDER_OUTLINED),
        )

        view_btn = ft.IconButton(
            icon=ft.Icons.VISIBILITY_OUTLINED,
            on_click=lambda e, t=tile, r=row: show_content(e, t, r)
        )

        controls = [view_btn]
        width = 32

        if not status_merged:
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
        supabase = get_supabase()
        
        if not supabase:
            write_log("Failed to load documents: Supabase connection failed.", is_error=True)
            return
            
        try:
            # Now calling the updated markdown RPC
            response = supabase.rpc("get_all_markdowns").execute()
            results = response.data
            
            for row in results:
                add_tile(row)
                
            page.update()
        except Exception as ex:
            write_log(f"Error retrieving from Supabase: {ex}", is_error=True)

    # ── Ready check ───────────────────────────────────────────────────────────

    def check_ready():
        vlm_ready = False
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                result = s.connect_ex(('127.0.0.1', int(vlm_port_tf.value)))
                if result == 0:
                    vlm_ready = True
                    write_log(f"VLM is ready at port {int(vlm_port_tf.value)}.")
                else:
                    vlm_ready = False
                    write_log(f"VLM is NOT ready.")
            except Exception as e:
                vlm_ready = False
                write_log(f"VLM check error: {e}")

        ready = all([
            webpage_port_tf.value,
            vlm_port_tf.value,
            provenance_port_tf.value,
            avatar_port_tf.value,
            top_k_tf.value,
            initial_retrieval_k_tf.value,
            wakeword_thresh_tf.value,
            stt_silence_thresh_tf.value,
            minimum_audio_level_tf.value,
            len(doc_list_view.controls) > 0,
            vlm_ready
        ])
        start_switch.disabled = not ready
        page.update()

    # ── Server management ─────────────────────────────────────────────────────

    def _stream_output(name: str, proc: subprocess.Popen, color: str):
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
                        selectable=True
                    )
                )
                if len(log_view.controls) % 5 == 0:
                    page.update()
        try:
            page.update()
        except Exception:
            pass

    def start_servers():
        global _processes, _log_threads
        _log_threads.clear()

        print("")

        os.environ["PROVENANCE_PORT"] = str(provenance_port_tf.value)

        servers = [
            {
                "name": "KIOSK",
                "color": "#74c0fc",
                "cmd": [
                    "./.venv/Scripts/python.exe", "scripts/https_kiosk.py",
                    webpage_port_tf.value,
                    f"{os.path.join(os.getcwd(), 'avatar_aang')}"
                ],
            },
            {
                "name": "AVATAR",
                "color": "#a9e34b",
                "cmd": ["./.venv/Scripts/python.exe", "scripts/assistant/iris_server.py",
                        "--port", avatar_port_tf.value, 
                        "--api-key", os.getenv("GEMINI_API_KEY"),
                        "--male-voice", male_voice_dd.value,
                        "--female-voice", female_voice_dd.value,
                        "--top-k", top_k_tf.value,
                        "--initial-retrieval-k", initial_retrieval_k_tf.value,
                        "--wakeword-threshold", wakeword_thresh_tf.value,
                        "--stt-silence-threshold", stt_silence_thresh_tf.value,
                        "--minimum-audio-level", minimum_audio_level_tf.value
                    ],
            },
        ]

        for srv in servers:
            if srv.get("cmd") is None:
                write_log(f"[{srv['name']}] {srv.get('note', '')}")
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
        write_log("Starting safe database update from Supabase... This may take a minute.")
        
        build_progress.visible = True
        build_button.disabled = True
        page.update()
        
        def _build_task():
            try:
                # 1. Fetch from Supabase
                supabase = get_supabase()
                if not supabase:
                    write_log("Supabase credentials missing.", is_error=True)
                    return
                
                write_log("Fetching merged markdowns from Supabase securely...")
                response = supabase.rpc("get_merged_markdowns").execute()
                rows = response.data

                if not rows:
                    write_log("No merged documents found in Supabase to index.", is_error=True)
                    return

                # Check LanceDB
                lance_db_path = os.getenv("LANCEDB_URL")
                db = lancedb.connect(lance_db_path)
                table_name = "batstateu_info"

                df_old = None
                existing_doc_ids = set()

                if table_name in db.table_names():
                    table = db.open_table(table_name)
                    df_old = table.to_pandas()
                    if "doc_id" in df_old.columns:
                        existing_doc_ids = set(df_old['doc_id'].fillna('').astype(str).unique())

                # Filter for new/updated documents
                rows_to_build = [r for r in rows if str(r["id"]) not in existing_doc_ids]

                if not rows_to_build:
                    write_log("✅ All merged documents are already indexed. Nothing new to build!")
                    return

                write_log(f"Found {len(rows_to_build)} NEW merged documents. Processing with Docling...")
                
                # ==========================================
                # 2. Advanced Docling Extraction & Chunking
                # ==========================================
                
                # A. Enable Table Structure Recognition
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options = TableStructureOptions(mode="accurate")
                
                converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
                
                # C. Tune the Hybrid Chunker with Overlap
                nomic_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
                chunker = HybridChunker(
                    tokenizer=nomic_tokenizer, 
                    max_tokens=512, 
                    merge_peers=True,
                    overlap_tokens=64  # CRITICAL: Prevents cutting thoughts in half
                )
                
                new_chunks = []
                temp_files = []
                file_metadata = {}
                
                # Prepare temporary files for multiprocessing
                for row in rows_to_build:
                    md_text = row.get("request", "")
                    doc_id_str = str(row["id"])
                    
                    meta = {
                        "doc_id": doc_id_str,
                        "source": info_type_mapping.get(row.get("info_type"), "Unknown"),
                        "office": office_mapping.get(row.get("office"), "Unknown"),
                        "target": target_mapping.get(row.get("target"), "Unknown")
                    }

                    temp_md = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
                    temp_md.write(md_text)
                    temp_md.close() # Close to allow Docling to open it safely

                    temp_files.append(temp_md.name)
                    file_metadata[temp_md.name] = (doc_id_str, meta)

                try:
                    # B. Batch Document Processing
                    write_log(f"Running batch conversion on {len(temp_files)} files...")
                    
                    # Removed num_threads. Added raises_on_error=False so if one file fails, 
                    # it doesn't crash the entire ingestion batch.
                    conv_results = list(converter.convert_all(temp_files, raises_on_error=False))

                    # Chunk the results
                    for i, conv_result in enumerate(conv_results):
                        doc_id_str, meta = file_metadata[temp_files[i]]
                        chunk_iter = chunker.chunk(conv_result.document)

                        for c_idx, chunk in enumerate(chunk_iter):
                            new_chunks.append({
                                "chunk_id": f"{doc_id_str}_c{c_idx}", 
                                "text": chunk.text,
                                **meta
                            })
                finally:
                    # Clean up temp files
                    for f in temp_files:
                        if os.path.exists(f):
                            os.remove(f)

                write_log(f"Generated {len(new_chunks)} hybrid chunks. Initializing NGARAG Pipeline...")

                # ==========================================
                # 3. Initialize Models
                # ==========================================
                write_log("Loading Embedding Model...")
                embedder = HuggingFaceEmbeddings(
                    model_name="nomic-ai/nomic-embed-text-v1.5",
                    model_kwargs={'device': 'cpu', 'trust_remote_code': True}
                )
                
                write_log("Loading NLI Gatekeeper...")
                gatekeeper = CrossEncoder("cross-encoder/nli-deberta-v3-small")
                
                write_log("Embedding chunks...")
                texts = [chunk["text"] for chunk in new_chunks]
                vectors = embedder.embed_documents([f"search_document: {t}" for t in texts])

                # 4. Define Schema & Open Table
                schema = pa.schema([
                    pa.field("chunk_id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), 768)),
                    pa.field("doc_id", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("office", pa.string()),
                    pa.field("target", pa.string())
                ])

                if table_name in db.table_names():
                    table = db.open_table(table_name)
                else:
                    write_log("Creating new LanceDB table...")
                    table = db.create_table(table_name, schema=schema, mode="create")

                # ==========================================
                # 5. Batched NGARAG Intelligent Upsert
                # ==========================================
                write_log("Executing NLI-Gated Insertion (Batched for Performance)...")
                appended_count, dropped_count, overwritten_count = 0, 0, 0
                similarity_threshold = 0.6  
                
                records_to_add = []
                ids_to_delete = []

                if len(table) == 0:
                    # If DB is completely empty, append everything directly
                    records_to_add = [{"vector": vectors[i], **c} for i, c in enumerate(new_chunks)]
                    appended_count = len(records_to_add)
                else:
                    gatekeeper_pairs = []
                    gatekeeper_meta = []

                    # Pass 1: Vector Search Collection
                    for i, chunk_data in enumerate(new_chunks):
                        vector = vectors[i]
                        record = chunk_data.copy()
                        record["vector"] = vector

                        results = table.search(vector).limit(1).to_pandas()

                        if results.empty or results['_distance'][0] > similarity_threshold:
                            records_to_add.append(record)
                            appended_count += 1
                        else:
                            closest_match = results.iloc[0]
                            # Queue for the Gatekeeper batch
                            gatekeeper_pairs.append((closest_match['text'], chunk_data['text']))
                            gatekeeper_meta.append({
                                'old_id': closest_match['chunk_id'],
                                'record': record
                            })

                    # Pass 2: Mass Matrix Prediction (Eliminates the O(N) bottleneck)
                    if gatekeeper_pairs:
                        scores = gatekeeper.predict(gatekeeper_pairs, batch_size=32)
                        predictions = scores.argmax(axis=1)

                        for pred, meta in zip(predictions, gatekeeper_meta):
                            if pred == 1:
                                dropped_count += 1
                            elif pred == 0:
                                ids_to_delete.append(meta['old_id'])
                                records_to_add.append(meta['record'])
                                overwritten_count += 1
                            else:
                                records_to_add.append(meta['record'])
                                appended_count += 1

                # Execute Database Mutations
                if ids_to_delete:
                    for old_id in ids_to_delete:
                        table.delete(f"chunk_id = '{old_id}'")
                        
                if records_to_add:
                    table.add(records_to_add)

                write_log(f"NGARAG Complete: {appended_count} Appended, {dropped_count} Dropped, {overwritten_count} Overwritten.")

                # ==========================================
                # 6. Database Indexing & Optimizations
                # ==========================================
                write_log("Building Full-Text Search (FTS) index...")
                table.create_fts_index("text", replace=True)
                
                # Advanced Vector Indexing (IVF-PQ) - Triggers once DB is sufficiently large
                if len(table) > 1000:
                    try:
                        write_log("Optimizing vector clusters (IVF-PQ)...")
                        table.create_index(metric="L2", vector_column_name="vector", num_partitions=256, num_sub_vectors=96)
                    except Exception as idx_err:
                        write_log(f"Vector index optimization skipped: {idx_err}")
                
                write_log("✅ Database updated and Hybrid Search is ready!")

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
        value="am_fenrir",
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
        value="7040",
        border_color=ft.Colors.WHITE_54,
        expand=True,
        on_change=lambda e: check_ready(),
    )
    top_k_tf = ft.TextField(
        label="Top K",
        value="5",
        border_color=ft.Colors.WHITE_54,
        expand=True,
        on_change=lambda e: check_ready(),
        tooltip="How many of the ranked relevant searches to provide to the VLM context?"
    )
    initial_retrieval_k_tf = ft.TextField(
        label="Initial Retrieval K",
        value="12",
        border_color=ft.Colors.WHITE_54,
        expand=True,
        on_change=lambda e: check_ready(),
        tooltip="How many relevant items to look for in the database?"
    )
    wakeword_thresh_tf = ft.TextField(
        label="Wakeword Trigger Accuracy Threshold",
        value="0.2",
        border_color=ft.Colors.WHITE_54,
        expand=True,
        on_change=lambda e: check_ready(),
    )
    stt_silence_thresh_tf = ft.TextField(
        label="STT Silence Threshold",
        value="0.75",
        border_color=ft.Colors.WHITE_54,
        expand=True,
        on_change=lambda e: check_ready(),
    )
    minimum_audio_level_tf = ft.TextField(
        label="Minimum Audio Level",
        value="1500",
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
                                ft.Text("Configuration", size=20, weight=ft.FontWeight.W_600, expand=6),
                                ft.Row(
                                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                    expand=4,
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
                                ft.Row(
                                    expand=6,
                                    controls=[
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
                                        ft.Column(
                                            spacing=16,
                                            expand=True,
                                            controls=[
                                                ft.Text("RAG Parameters", weight=ft.FontWeight.W_500),
                                                ft.Container(
                                                    expand=True,
                                                    content=ft.Column(
                                                        expand=True,
                                                        controls=[
                                                            initial_retrieval_k_tf,
                                                            top_k_tf,
                                                            ft.Text("Model Sensitivity", weight=ft.FontWeight.W_500),
                                                            wakeword_thresh_tf,
                                                            stt_silence_thresh_tf,
                                                            minimum_audio_level_tf
                                                        ],
                                                    ),
                                                ),
                                            ],
                                        ),
                                    ]
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