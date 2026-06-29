"""
Findings-aware clickable Tkinter table window for selection (D6).
"""
from dataclasses import dataclass
from typing import Optional
from src.layer_05_visualizer.catalog import CatalogEntry, _parse_tasks

@dataclass
class PickerRow:
    video_id: str
    short_id: str
    star: int
    task: str
    layers: str
    audio: str
    video_found: str

def build_picker_rows(catalog: dict[str, CatalogEntry]) -> list[PickerRow]:
    rows = []
    for vid, c in catalog.items():
        star = c.num_layers_with_findings
        tasks = _parse_tasks(c.manifest_entry)
        # collapse newlines — Ego4D task labels can be multi-line, which breaks table cells
        task_label = " ".join(str(tasks[0].get("task_label", "unknown")).split()) if tasks else "none"

        finding_layers = sorted(l for l, st in c.findings.items() if st == "finding")
        layers_str = " ".join(finding_layers) if finding_layers else "-"

        # BMP-safe text only: the macOS system Tk (8.5) can't render astral emoji like 🔊.
        audio_str = {True: "yes", False: "no", None: "?"}[c.has_audio]
        vid_str = "yes" if c.video_path else "no"
        short_id = vid[:8]

        rows.append(PickerRow(
            video_id=vid,
            short_id=short_id,
            star=star,
            task=task_label,
            layers=layers_str,
            audio=audio_str,
            video_found=vid_str
        ))
    return sorted(rows, key=lambda r: r.star, reverse=True)

def filter_rows(rows: list[PickerRow], query: str) -> list[PickerRow]:
    if not query:
        return rows
    q = query.lower()
    if q == "noaudio":
        return [r for r in rows if r.audio == "no"]
    
    res = []
    for r in rows:
        text = f"{r.star} {r.task} {r.layers} {r.audio} {r.video_found} {r.video_id}".lower()
        if q in text:
            res.append(r)
    return res

def sort_rows(rows: list[PickerRow], column: str, descending: bool) -> list[PickerRow]:
    # Stable sort. `star` sorts numerically; the rest are case-insensitive text.
    keys = {
        "star": lambda r: r.star,
        "task": lambda r: r.task.lower(),
        "layers": lambda r: r.layers.lower(),
        "audio": lambda r: r.audio.lower(),
        "video": lambda r: r.video_found.lower(),
        "id": lambda r: r.video_id.lower(),
    }
    key = keys.get(column)
    if key is None:
        return rows
    return sorted(rows, key=key, reverse=descending)

# Internal column ids (ASCII — never use emoji as a Tk column identifier) mapped
# to their display heading text.
_COLUMNS = [
    ("star",   "★ find"),
    ("task",   "Task"),
    ("layers", "Layers w/ findings"),
    ("audio",  "Audio"),
    ("video",  "Video"),
    ("id",     "video_id"),
]
_COL_WIDTHS = {"star": 60, "task": 300, "layers": 170, "audio": 60, "video": 60, "id": 110}


def pick_video_terminal(catalog: dict[str, CatalogEntry]) -> Optional[CatalogEntry]:
    """Pure-terminal selector. No GUI, no dependency — works anywhere a prompt does.

    Shows a numbered, findings-sorted list; you type a number to pick, `/text` to
    filter, `/clear` to reset the filter, or `q` to quit. This is the default and
    most robust selector (the macOS system Tk 8.5 GUI cannot render Treeview rows).
    """
    rows = build_picker_rows(catalog)
    if not rows:
        print("No clips found under the scan roots. Try --scan-root <dir> or --list.")
        return None

    flt = ""
    while True:
        shown = filter_rows(rows, flt) if flt else rows
        print()
        print(f"{'#':>3}  {'★':>2}  {'task':<30}  {'layers w/ findings':<20}  {'aud':<3}  {'video_id':<8}")
        print("-" * 78)
        for i, r in enumerate(shown):
            task = (r.task[:29] + "…") if len(r.task) > 30 else r.task
            print(f"{i:>3}  {r.star:>2}  {task:<30}  {r.layers:<20}  {r.audio:<3}  {r.short_id:<8}")
        if not shown:
            print("  (no rows match the current filter)")
        print("-" * 78)

        prompt = (
            f"Filter={flt!r} · showing {len(shown)}/{len(rows)}\n"
            "Enter a # to visualize · /<text> to filter · /clear · q to quit: "
        )
        try:
            choice = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo clip selected.")
            return None

        if choice.lower() in ("q", "quit", "exit"):
            return None
        if choice.startswith("/"):
            arg = choice[1:].strip()
            flt = "" if arg in ("clear", "") else arg
            continue
        if choice.isdigit():
            i = int(choice)
            if 0 <= i < len(shown):
                return catalog[shown[i].video_id]
            print(f"  '{i}' is out of range (0..{len(shown) - 1}).")
            continue
        # bare text is treated as a filter for convenience
        flt = choice


def pick_video(catalog: dict[str, CatalogEntry]) -> Optional[CatalogEntry]:
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("Select a clip to visualize")
    root.geometry("900x600")

    # macOS ships an old system Tk (8.5) whose default `aqua` ttk theme does NOT
    # render Treeview rows (the window comes up blank). `clam` is bundled on every
    # platform and renders rows + custom colors reliably — this is the core fix.
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("Treeview", rowheight=24, font=("TkDefaultFont", 12))
    style.configure("Treeview.Heading", font=("TkDefaultFont", 12, "bold"))

    if not catalog:
        messagebox.showerror(
            "No clips found",
            "The catalog is empty — no clips were discovered under the scan roots.\n"
            "Run with --list to inspect discovery, or add --scan-root <dir>.",
            parent=root,
        )
        root.destroy()
        return None

    result = {"entry": None}
    all_rows = build_picker_rows(catalog)
    state = {"rows": all_rows[:], "sort_desc": {}}

    col_ids = [c[0] for c in _COLUMNS]

    # ---- top: filter box -------------------------------------------------
    top = ttk.Frame(root)
    top.pack(fill=tk.X, padx=10, pady=(10, 4))
    ttk.Label(top, text="Filter:").pack(side=tk.LEFT)
    filter_var = tk.StringVar()
    ttk.Entry(top, textvariable=filter_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
    ttk.Label(top, text="(click a header to sort · double-click a row to render)").pack(side=tk.RIGHT)

    # ---- table -----------------------------------------------------------
    mid = ttk.Frame(root)
    mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
    tree = ttk.Treeview(mid, columns=col_ids, show="headings", selectmode="browse")
    vsb = ttk.Scrollbar(mid, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def populate(rows):
        tree.delete(*tree.get_children())
        for r in rows:
            tree.insert(
                "", tk.END, iid=r.video_id,
                values=(r.star, r.task, r.layers, r.audio, r.video_found, r.short_id),
            )

    def handle_sort(col):
        desc = not state["sort_desc"].get(col, False)
        state["sort_desc"][col] = desc
        state["rows"] = sort_rows(state["rows"], col, desc)
        populate(state["rows"])

    for col_id, heading in _COLUMNS:
        tree.heading(col_id, text=heading, command=lambda c=col_id: handle_sort(c))
        anchor = tk.CENTER if col_id in ("star", "audio", "video") else tk.W
        tree.column(col_id, width=_COL_WIDTHS[col_id], anchor=anchor, stretch=(col_id == "task"))

    populate(state["rows"])

    # ---- detail panel ----------------------------------------------------
    detail = tk.Text(root, height=7, wrap=tk.WORD)
    detail.pack(fill=tk.X, padx=10, pady=4)
    detail.insert(tk.END, "Select a row to see its summary…")
    detail.config(state=tk.DISABLED)

    def on_select(_event=None):
        sel = tree.selection()
        if not sel or sel[0] not in catalog:
            return
        detail.config(state=tk.NORMAL)
        detail.delete("1.0", tk.END)
        detail.insert(tk.END, catalog[sel[0]].summary_text)
        detail.config(state=tk.DISABLED)

    tree.bind("<<TreeviewSelect>>", on_select)

    def on_filter(*_):
        state["rows"] = filter_rows(all_rows, filter_var.get())
        populate(state["rows"])

    filter_var.trace_add("write", on_filter)

    # ---- buttons ---------------------------------------------------------
    def do_render(_event=None):
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Click a clip in the table first.", parent=root)
            return
        result["entry"] = catalog[sel[0]]
        root.destroy()

    def do_cancel(_event=None):
        result["entry"] = None
        root.destroy()

    tree.bind("<Double-1>", do_render)
    tree.bind("<Return>", do_render)

    btns = ttk.Frame(root)
    btns.pack(fill=tk.X, padx=10, pady=(4, 10))
    ttk.Button(btns, text="Render selected", command=do_render).pack(side=tk.RIGHT, padx=5)
    ttk.Button(btns, text="Cancel", command=do_cancel).pack(side=tk.RIGHT, padx=5)

    root.protocol("WM_DELETE_WINDOW", do_cancel)
    root.update_idletasks()
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.mainloop()

    return result["entry"]
