# AI Task Breakdown: Reasonable Emotion Layer (03b)

## Objective
The **Reasonable Emotion Layer** leverages the affective reactions of bystanders to deduce the success, failure, or general outcome of an action performed by the POV actor. 

Crucially, this layer analyzes **emotion transitions and durations** over time (e.g., a fleeting shock turning into sustained applause) purely within the dynamically bounded temporal moment of the action. It then outputs both granular temporal slices and a **Late-Stage Weighted Average** representing final task success.

---

## 📥 Input Requirements
- **`filtered_manifest.json`** (required): This layer iterates through the `identified_tasks` array provided by Node 02. For each task, it retrieves:
  - **Contextual Task Label** (`task_label`): To know what action the bystander is reacting to.
  - **Temporal Alignment** (`task_reaction_window_sec`): Derived dynamically via Optical Flow peak detection (+ VLM refinement) and Action Velocity profiling, this is the exact timestamp boundary within which we sample bystander reactions for a given task.
- **Cross-layer (optional)**: `03a_attention_result.json` — Used to weight emotion readings. A bystander who isn't even looking at the POV actor (attention score < 0.3) during the reaction window is flagged as unreliable.

---

## 🛠️ Implementation Strategy

Because a single video might contain multiple tasks, this sequence occurs iteratively for *each* task inside the `identified_tasks` array.

### Step 1: Expectation Generation (Gemma 4)
Using the extracted Contextual Task (e.g., "Juggling apples"), prompt a locally running LLM (**Gemma 4** via Ollama) to generate baseline emotional expectations. This lightweight local LLM inference is well-suited for the **24GB RAM Mac mini M4 Pro**, allowing concurrent tracking arrays without swapping.

**Structured Prompt Template:**
```text
You are analyzing bystander reactions to this task: "{task_label}"

Generate sets of emotions a bystander would display if:
1. The outcome is POSITIVE (successful, impressive).
2. The outcome is NEGATIVE (failed, dangerous).

Respond in EXACTLY this JSON format:
{
  "positive_emotions": ["emotion1", "emotion2"],
  "negative_emotions": ["emotion1", "emotion2"],
  "neutral_baseline": ["bored", "neutral", "distracted"]
}
```

### Step 2: Temporal Sampling & Pairwise Chunking
We track the bystander strictly within the `task_reaction_window_sec` for the current task.
1. Run a SOTA emotion model (e.g., Py-Feat) on the bounding box faces at **3-5 FPS** across the reaction window.
2. Form an Emotion Timeseries (e.g., `Neutral (0.5s) -> Shock (0.5s) -> Joy (2.0s)`).
3. **Chunking into Pairs**: Break the sequence into consecutive Pairwise Transitions. Emotion logic is evaluated strictly one transition at a time to anchor LLM reasoning.
   - Transition 1: `Neutral -> Shock`
   - Transition 2: `Shock -> Joy`

### Step 3: Accumulated History Evaluation (Gemma 4)
Each emotional transition pair is fed sequentially to Gemma 4. However, to maintain chronological coherence, the prompt **cumulatively injects previously analyzed pairs as historical context**.

**Evaluation Prompt:**
```text
Task: "{task_label}"
Predicted Positive Emotions: {positive_emotions}
Predicted Negative Emotions: {negative_emotions}

Previous Reaction History (for context):
{accumulated_history} 
*(e.g., "1. Neutral -> Shock (Classified: Anticipation/Neutral)")*

Current Transition being Evaluated:
During the climax of the task, the bystander's emotion transitioned explicitly from {emotion_start} to {emotion_end}. 

Considering the history, does this new transition indicate a positive or negative task execution?

Respond in EXACTLY this JSON format:
{
  "classified_direction": "positive" | "negative" | "neutral",
  "reasoning": "Briefly map this specific transition to the task outcome context."
}
```

### Step 4: Temporal Slices & Overarching Task Scalar
For every evaluated pair, we extract a slice scalar: `slice_success_scalar = magnitude * direction_sign` (where positive=1, negative=-1, neutral=0).

#### Final Video-Level Aggregate Score (Per Task)
Instead of just leaving researchers with raw temporal slices, we calculate a final overarching score representing the outcome of that specific task.

We use a **Late-Stage Weighted Average**:
Emotions resolve over time. A "shock" reflex at the beginning of a window is less indicative of the final success than the "joy" at the end of the window. Therefore, we weight the `slice_success_scalar` by both its **duration** and its **chronological position** (how late it occurs relative to the task climax).

$$ \text{Task Success Score} = \frac{\sum_{i=1}^{n} (S_i \times D_i \times W_i)}{\sum_{i=1}^{n} (D_i \times W_i)} $$

Where:
- $S_i$ = The `slice_success_scalar` of slice $i$
- $D_i$ = Duration of slice $i$ in seconds
- $W_i$ = Chronological Weight in seconds: $W_i = \max(0.1,\; t_{\text{start}_i} - t_{\text{climax}})$. The `max(0.1, ...)` floor prevents the first slice from being zeroed out entirely. Later slices mathematically overpower early reflexes.

*(Note: If there are multiple bystanders, this final score is averaged across all bystander scores for that task, weighted by their 03a attention span.)*

---

## 📤 Output Schema and Integration
The layer outputs a structured array mapping the bystander's emotional journey *per task* within the video.

**Example Output Data (`03b_reasonable_emotion_result.json`):**
```json
{
  "video_id": "ego4d_clip_10293",
  "layer": "03b_reasonable_emotion",
  "tasks_analyzed": [
    {
      "task_id": "t_01",
      "task_label": "Juggling apples",
      "task_reaction_window_sec": [6.2, 8.2],
      "per_person": [
        {
          "person_id": 0,
          "temporal_slices": [
            {
              "slice_id": 1,
              "window_sec": [6.2, 6.7],
              "transition_pair": ["neutral", "surprise"],
              "terminal_magnitude": 0.85,
              "classified_direction": "neutral",
              "slice_success_scalar": 0.0
            },
            {
              "slice_id": 2,
              "window_sec": [6.7, 8.2],
              "transition_pair": ["surprise", "joy"],
              "terminal_magnitude": 0.92,
              "classified_direction": "positive",
              "slice_success_scalar": 0.92
            }
          ],
          "late_stage_weighted_success_score": 0.81
        }
      ],
      "task_aggregate_score": 0.81
    }
  ]
}
```

## Verification & Validation Check
To ensure the LLM reasoning is chronologically sound and empirically reliable:
- **Singular Video Test**: Run the emotion layer for a specific video ID. Dump the exact prompt string sent to Gemma 4 and its exact JSON return to the console. Manually review the `classified_direction` logical mapping against the input `transition_pair` to verify the prompt is unbroken.
- **Batch Test**: Run the step on an entire `filtered_manifest.json` batch. Parse the final output and check the total distribution of `task_aggregate_score` values. If >95% of the values are strictly exactly positive or exactly negative, review the Gemma 4 temperature settings as the model may have collapsed into a predictable output path. Performance profiling should verify that processing scales continuously on the **Mac mini M4 Pro (24GB RAM)**.
