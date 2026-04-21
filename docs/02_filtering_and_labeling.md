# AI Task Breakdown: Video Filtering and Labeling

## Objective
To ensure computational efficiency and relevance, not all downloaded videos make it to the advanced Social Feature Layers. This module performs two critical filtration passes to weed out unusable data from first-person (POV) perspectives.

---

## 1. The Social Presence Filter
Since our project relies on investigating social-affective interactions, a video of a person entirely alone is useless. 

**Criteria**: 
The video **must** contain more than one person. Since this is an egocentric/POV video, the person whose perspective is being shown (the camera wearer) *does not count* towards this total. There must be at least one other visible human actor present in the frame.

**Task Requirements**:
- Evaluate the raw videos from the initial manifest.
- Drop any video where zero exterior individuals are found.
- *Potential Implementation Idea*: Utilize standard lightweight Object Detection (e.g. YOLOv8) on heavily downsampled frames to check for bounding boxes labeled `person`.

---

## 2. Contextual Task Labeling
A social interaction has fundamentally different mechanics if the POV person is reading a book versus cooking a meal with a knife. For downstream layers to accurately process features (like "Flinch" physics), we must define the task context.

**Criteria**:
If the initial dataset metadata does not already contain accurate labels for what the POV actor is doing, we must actively generate one. If no clear task is taking place, the video is dropped.

**Task Requirements**:
- For clips that successfully pass the Social Presence Filter, evaluate what the POV persona is physically doing.
- Write a VLM processing layer that observes interval frames from the video to classify the Action/Task.
- *Drop Rule*: If the VLM states "Idling," "No clear task," or "Ambiguous activity", the video does not possess enough context and must be discarded from the manifest.

## Output
This module outputs a `filtered_manifest.json` that the downstream **Social Feature Extraction Layers** will consume. Only clips that survived both the Social Presence check and Contextual Task extraction make it onto this ledger.
