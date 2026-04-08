# AI Task Breakdown: The Engagement Module

## Objective
Develop the logic to extract human "Cognitive State" (Focused, Neutral, Startled) from filtered video clips using a Temporal Vision-Language Model.

## Agent Instructions: Step-by-Step Tasks

### Task 1: VLM Integration Setup
- **Action**: Create a new Python file `analyze_engagement.py`.
- **Action**: Integrate a Temporal VLM API or local loading script (e.g., Qwen2.5-VL using HuggingFace `transformers` or vLLM). 
- **Constraint**: Account for the 16GB unified memory constraint by explicitly loading the model using 4-bit quantization (e.g., with bitsandbytes) `load_in_4bit=True` if running locally.

### Task 2: Prompt Engineering & Execution
- **Action**: Implement a function `generate_engagement_prompt(t_start, t_end)` that returns a highly specific strict-JSON formatting prompt:
  *"Analyze the human's face from {t_start} to {t_end}. Categorize their cognitive state into exactly one of three categories: [Focused], [Neutral], or [Startled]. Return your response in JSON format as {"state": "..."}"*
- **Action**: Write the inference function `run_vlm_inference(clip_path, prompt)` that feeds the relevant processed video frames and prompt to the VLM.

### Task 3: Output Parsing
- **Action**: Write a robust parser `parse_vlm_response(response_text)` that parses the JSON output to strictly classify into `FOCUSED`, `NEUTRAL`, or `STARTLED`. 
- **Action**: Implement error handling: Catch `JSONDecodeError` or unmapped responses (hallucinations) and assign them to an `UNKNOWN` or `FAILED` state.

### Task 4: Data Formatting
- **Action**: Save the output into an internal format (e.g., returning a dictionary mapped to clip IDs) to be passed efficiently to the subsequent modules.
- **Success Criteria**: The VLM categorizes a batch of video clips into the three designated cognitive states without causing Mac OOM (Out of Memory) crashes or swapping freezes.
