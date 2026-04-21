# AI Task Breakdown: Social Feature Layers Architecture

## Objective
This document outlines the core operational paradigm of the project. We extract affective and social context out of the POV video datasets. We utilize a highly extensible "Layer Architecture" where social features are investigated by independent python scripts, each appending its generated metadata safely to the project state.

---

## 🏗️ The "Ongoing Layers" Paradigm
As the repository is updated, we will continuously brainstorm and add independent layers. Each Layer should be developed with these restrictions in mind:
1. **Never modify the original video chunks.**
2. **Never overwrite another Layer's dataset ledger columns/JSON keys.** 
3. **Execute against the `filtered_manifest.json` provided by Node 02.**

### Conceptual Example 1: The `Flinch` Layer
- **Goal**: Measure physical startle responses and abrupt kinesthetic shifts from the POV wearer toward another human.
- **Implementation**: Runs PyTorch/Pose-Tracking on the video. If the velocity of body/arm movement jumps dramatically in correlation to another actor, it sets `flinch_detected: true` and logs the timestamp.

### Conceptual Example 2: The `Engagement/Eye-Contact` Layer
- **Goal**: Evaluate if the other human in the FOV is visibly focused on the camera wearer or acting distracted.
- **Implementation**: Runs facial/gaze tracking. Estimates the pitch/yaw of the other person's face towards the camera's centroid. Logs `attention_score: 0.85`.

---

## Output Integration
Each layer is designed to output its own `.json` chunk, or it writes into a centralized SQLite/Pandas `result_file` database instance using the `video_id` as the primary key. This is done to ensure the system is completely horizontal—adding a new "Empathy Layer" does not require rewriting the Flinch layer.
