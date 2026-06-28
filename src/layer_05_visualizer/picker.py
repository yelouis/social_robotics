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
        task_label = tasks[0].get("task_label", "unknown") if tasks else "none"
        
        finding_layers = [l for l, st in c.findings.items() if st == "finding"]
        layers_str = " ".join(finding_layers) if finding_layers else "—"
        
        has_audio = c.manifest_entry.get("has_audio", True)
        audio_str = "🔊" if has_audio else "··"
        vid_str = "✓" if c.video_path else "x"
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
        return [r for r in rows if r.audio == "··"]
    
    res = []
    for r in rows:
        text = f"{r.star} {r.task} {r.layers} {r.audio} {r.video_found} {r.video_id}".lower()
        if q in text:
            res.append(r)
    return res

def sort_rows(rows: list[PickerRow], column: str, descending: bool) -> list[PickerRow]:
    # Stable sort
    if column == "★":
        return sorted(rows, key=lambda r: r.star, reverse=descending)
    elif column == "task":
        return sorted(rows, key=lambda r: r.task.lower(), reverse=descending)
    elif column == "layers":
        return sorted(rows, key=lambda r: r.layers.lower(), reverse=descending)
    elif column == "audio":
        return sorted(rows, key=lambda r: r.audio.lower(), reverse=descending)
    elif column == "video":
        return sorted(rows, key=lambda r: r.video_found.lower(), reverse=descending)
    elif column == "id":
        return sorted(rows, key=lambda r: r.video_id.lower(), reverse=descending)
    return rows

def pick_video(catalog: dict[str, CatalogEntry]) -> Optional[CatalogEntry]:
    import tkinter as tk
    from tkinter import ttk
    
    root = tk.Tk()
    root.title("Select a clip to visualize")
    root.geometry("800x600")
    
    result = None
    all_rows = build_picker_rows(catalog)
    current_rows = all_rows[:]
    sort_states = {}
    
    top_frame = tk.Frame(root)
    top_frame.pack(fill=tk.X, padx=10, pady=5)
    
    tk.Label(top_frame, text="filter:").pack(side=tk.LEFT)
    filter_var = tk.StringVar()
    filter_entry = tk.Entry(top_frame, textvariable=filter_var, name="filter_entry")
    filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    tk.Label(top_frame, text="sort: ▼★ (click a header to re-sort)").pack(side=tk.RIGHT)
    
    columns = ("★", "task", "layers", "audio", "video", "id")
    tree = ttk.Treeview(root, columns=columns, show="headings", name="treeview")
    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def populate_tree(rows):
        for item in tree.get_children():
            tree.delete(item)
        for r in rows:
            tree.insert("", tk.END, values=(r.star, r.task, r.layers, r.audio, r.video_found, r.short_id), iid=r.video_id)
            
    def handle_sort(col):
        desc = sort_states.get(col, True)
        sort_states[col] = not desc
        nonlocal current_rows
        current_rows = sort_rows(current_rows, col, desc)
        populate_tree(current_rows)
        
    for col in columns:
        tree.heading(col, text=col, command=lambda c=col: handle_sort(c))
        if col in ("★", "audio", "video"):
            tree.column(col, width=50, anchor=tk.CENTER)
        elif col == "id":
            tree.column(col, width=100)
        elif col == "layers":
            tree.column(col, width=150)
        else:
            tree.column(col, width=300)
            
    populate_tree(current_rows)
    
    # Detail panel
    detail_text = tk.Text(root, height=8, wrap=tk.WORD, name="summary_text")
    detail_text.pack(fill=tk.X, padx=10, pady=5)
    detail_text.insert(tk.END, "Select a row to see summary...")
    detail_text.config(state=tk.DISABLED)
    
    def on_select(event):
        sel = tree.selection()
        if not sel:
            return
        vid = sel[0]
        if vid in catalog:
            detail_text.config(state=tk.NORMAL)
            detail_text.delete(1.0, tk.END)
            detail_text.insert(tk.END, catalog[vid].summary_text)
            detail_text.config(state=tk.DISABLED)
            
    tree.bind("<<TreeviewSelect>>", on_select)
    
    def on_filter(*args):
        nonlocal current_rows
        current_rows = filter_rows(all_rows, filter_var.get())
        populate_tree(current_rows)
        
    filter_var.trace_add("write", on_filter)
    
    btn_frame = tk.Frame(root)
    btn_frame.pack(fill=tk.X, padx=10, pady=10)
    
    def do_render(event=None):
        sel = tree.selection()
        if sel:
            nonlocal result
            result = catalog[sel[0]]
            root.destroy()
            
    def do_cancel():
        nonlocal result
        result = None
        root.destroy()
        
    tree.bind("<Double-1>", do_render)
    
    tk.Button(btn_frame, text="Cancel", command=do_cancel, name="cancel_btn").pack(side=tk.RIGHT, padx=5)
    tk.Button(btn_frame, text="Render selected", command=do_render, name="render_btn").pack(side=tk.RIGHT, padx=5)
    
    root.protocol("WM_DELETE_WINDOW", do_cancel)
    
    # Start app
    root.mainloop()
    
    return result
