# Project Overview: Modular Social-Affective Filter (SAF)

## 📌 Grounding Document for AI Generation
**Read This First**: If you are an AI assistant or coding agent attached to this project workspace, this is your foundational grounding document. By referencing this file, you understand the core goal, the hardware limitations, and the architectural dependencies of all sub-modules in the `saf_project/docs` directory.

---

## 🎯 Global Project Goal
Develop an end-to-end pipeline to **generate high-fidelity social reward signals for Vision-Language-Action (VLA) training**. 
This is achieved by passing raw human-robot interaction video (like Ego4D) through a sophisticated "Waterfall Filter" that extracts "Appropriate Human Reactions." 
The final output is a quantified `Social Reward` ($R_s$) **Metadata & Derived Signal Dataset** structured in Parquet format, accompanied by a `load_dataset.py` hydration script, ready for the Hugging Face Embodied Data Leaderboard. To strictly comply with Ego4D licensing, the dataset contains absolutely zero raw video clips.

## 💻 Critical Hardware Constraint (The "16GB Rule")
**DO NOT** write monolithic data processing loops. **DO NOT** suggest loading multiple LLMs/VLMs simultaneously.
The explicit hardware target is Apple Silicon with **16GB of Unified Memory (M1 Pro)**. 
Therefore, the pipeline must strictly utilize a **Serialized Batch Process (The Waterfall Filter)**. We aggressively discard irrelevant data using computationally cheap heuristics (Metadata/Pose) before running the surviving, finalized clips through expensive Temporal VLMs (like Qwen2.5-VL via 4-bit quantization).

---

## 🧩 The Module Ecosystem (The Waterfall Filter)
The project is strictly separated into 7 discrete modules to maintain engineering modularity and protect system memory.

1. **[01_dataset_acquisition.md]** 
   * **Role**: Data Source. Uses the Ego4D CLI to retrieve ONLY the interaction/social metadata and clips, saving local disk space.
2. **[02_attention_module.md] (Node 1 - Very Low Compute)**
   * **Role**: The Entry Gate. Validates that the human bystander is actually watching the action using 2D spatial metadata (Gaze intersecting Object of Interest bounding boxes). If gaze $< 60\%$, the video is thrown out.
3. **[03_engagement_module.md] (Node 2 - High Compute)**
   * **Role**: Cognitive Extraction. Uses a 4-bit Quantized VLM strictly on passing clips to label the human's psychological state as exactly one of: `[Focused, Neutral, Startled]`.
4. **[04_multimodal_perception_module.md] (Node 3 - Medium Compute)**
   * **Role**: Kinematic Validation. Implements a Pose tracker to calculate Peak Velocity ($V_{peak}$) of the human's body. A physical "Flinch" acts as a hard override, indicating a safety violation regardless of the VLM's facial analysis.
5. **[05_appropriateness_judge.md] (The Brain)**
   * **Role**: Synthesis. Compiles data from Modules 2, 3, and 4 into the final formula: $R_s = (Attention \times State\_Weight) - (Flinch \times 2.0)$. Formats the finalist clips into `.parquet`.
6. **[06_sanity_checks_and_workflows.md] (Infrastructure Protections)**
   * **Role**: The Shield. Because VLMs hallucinate or crash inside 16GB limits, this module injects visual debug overlays, distribution checks (preventing flatlined 0.5 scores), and a temporal filter discarding impossible human reaction speeds ($<100ms$).
7. **[07_huggingface_export.md] (The Deployment)**
   * **Role**: Pipeline Exit. Automatically authenticated script pushing the final `.parquet` derived metadata and a generated Dataset Card (`README.md`) to the public Hugging Face Hub. It also includes the `load_dataset.py` hydration script to maintain compliance with Ego4D distribution rules.

---

## 🛠️ AI Development Directives
When tasked to write code for any of these modules:
1. Always parse the corresponding `AI Task Breakdown` file located in this `docs/` folder first.
2. Adhere strictly to the "Action" steps and "Success Criteria" defined for that specific node.
3. Ensure the module saves its intermediary state to disk cleanly (e.g., JSON maps), allowing the pipeline to be resumed if the VLM causes an Out-Of-Memory (OOM) operating system crash.
