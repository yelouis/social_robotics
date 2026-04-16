# Project Overview: Modular Social-Affective Filter (SAF)

## 📌 Grounding Document for AI Generation
**Read This First**: If you are an AI assistant or coding agent attached to this project workspace, this is your foundational grounding document. By referencing this file, you understand the core goal, the hardware limitations, and the architectural dependencies of all sub-modules in the `saf_project/docs` directory.

---

## 🎯 Global Project Goal
Develop an end-to-end pipeline to **generate proxy social reward signals for Vision-Language-Action (VLA) pre-training**.

### Source Dataset: Charades-Ego
Charades-Ego is a **paired-view** dataset of daily indoor activities: the same actor performs the same scripted task twice — once recorded from a **first-person (egocentric)** camera and once from a **third-person (observer)** camera. It contains **7,860 videos**, **157 action classes** (verb+object combinations like "Opening a door", "Holding a cup"), and **68,536 temporal annotations**.

### Why Charades-Ego for Social Reward?
Charades-Ego is **not** a human-robot interaction dataset. We use it as a **proxy pre-training source**: the premise is that a human performing everyday manipulation tasks (pouring, opening, placing, tidying) exhibits natural body language that encodes comfort, focus, and surprise. These signals transfer to downstream robotics settings where a robot must detect whether a nearby human is comfortable with its actions. The paired first-person / third-person views are uniquely valuable — the **egocentric** clip provides action context while the **third-person** clip exposes the actor's face and full-body posture for affective analysis.

### Output
A quantified `Social Reward` ($R_s$) **Metadata & Derived Signal Dataset** structured in Parquet format, ready for the Hugging Face Embodied Data Leaderboard. To comply with the AI2 Charades-Ego license (non-commercial, no redistribution), the dataset contains zero raw video — only derived annotations and computed signals.

## 💻 Critical Hardware Constraint (The "16GB Rule")
**DO NOT** write monolithic data processing loops. **DO NOT** suggest loading multiple LLMs/VLMs simultaneously.
The explicit hardware target is Apple Silicon with **16GB of Unified Memory (M1 Pro)**. 
Therefore, the pipeline must strictly utilize a **Serialized Batch Process (The Waterfall Filter)**. We aggressively discard irrelevant clips using computationally cheap heuristics (CSV action-class metadata) before running surviving clips through expensive Temporal VLMs (like Qwen2.5-VL via 4-bit quantization).

---

## 🧩 The Module Ecosystem (The Waterfall Filter)
The project is strictly separated into 7 discrete modules to maintain engineering modularity and protect system memory.

1. **[01_dataset_acquisition.md] (Data Acquisition)**
   * **Role**: Data Source. Downloads **both** the Charades-Ego videos (egocentric) **and** the original Charades videos (third-person pairs) from AI2's public S3 bucket via `wget`. Builds a unified manifest mapping each egocentric clip to its paired third-person clip via the `charades_video` CSV column.
2. **[02_attention_module.md] (Node 1 — Very Low Compute, CPU-only)**
   * **Role**: The Entry Gate. Pure metadata filter — no video I/O. Validates that a clip contains meaningful **object-manipulation activity** (robot-assistable tasks) using temporal action annotations from the Charades-Ego CSV. Clips with insufficient annotated activity coverage are immediately discarded.
3. **[03_engagement_module.md] (Node 2 — High Compute, VLM)**
   * **Role**: Cognitive Extraction. Uses a 4-bit Quantized VLM on the **third-person paired video** to label the actor's observable psychological state as exactly one of: `[Focused, Neutral, Startled]`. The third-person view is essential because the actor's face and full body are visible.
4. **[04_multimodal_perception_module.md] (Node 3 — Medium Compute, Pose)**
   * **Role**: Kinematic Validation. Runs a Pose tracker on the **third-person paired video** to calculate Peak Velocity ($V_{peak}$) of upper-body landmarks. A physical "Flinch" acts as a hard override, indicating a safety-relevant startle response regardless of the VLM's facial analysis.
5. **[05_appropriateness_judge.md] (The Brain)**
   * **Role**: Synthesis. Compiles data from Nodes 1, 2, and 3 into the final formula: $R_s = (Attention \times State\_Weight) - (Flinch \times 2.0)$. Formats the finalist clips into `.parquet`.
6. **[06_sanity_checks_and_workflows.md] (Infrastructure Protections)**
   * **Role**: The Shield. Because VLMs hallucinate or crash inside 16GB limits, this module injects visual debug overlays, distribution checks (preventing flatlined scores), and a temporal filter discarding impossible human reaction speeds ($<100ms$).
7. **[07_huggingface_export.md] (The Deployment)**
   * **Role**: Pipeline Exit. Pushes the final `.parquet` derived metadata and a generated Dataset Card (`README.md`) to the public Hugging Face Hub. Includes a `load_dataset.py` hydration script that guides end-users to download the original videos directly from AI2's public S3 URLs (no credentials required).

---

## 🛠️ AI Development Directives
When tasked to write code for any of these modules:
1. Always parse the corresponding `AI Task Breakdown` file located in this `docs/` folder first.
2. Adhere strictly to the "Action" steps and "Success Criteria" defined for that specific node.
3. Ensure the module saves its intermediary state to disk cleanly (e.g., JSON maps), allowing the pipeline to be resumed if the VLM causes an Out-Of-Memory (OOM) operating system crash.
4. **Egocentric vs. Third-Person**: Modules 02 (Attention) operates on CSV metadata only. Modules 03 (Engagement VLM) and 04 (Flinch Pose) **must** use the third-person paired video where the actor's face and body are visible. Never attempt to analyze the actor's own face from their egocentric camera.
