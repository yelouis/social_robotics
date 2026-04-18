import os
import json
import pandas as pd
import argparse
import ollama
from pathlib import Path

# Configuration
MANIFEST_PATH = "/Volumes/Extreme SSD/charades_ego_data/annotations/CharadesEgo/paired_clips_manifest.csv"
ATTENTION_RESULTS = "attention_results.json"
ENGAGEMENT_RESULTS = "engagement_results.json"
FLINCH_RESULTS = "flinch_results.json"
DEFAULT_EXPORT_PATH = os.path.expanduser("~/charades_ego_data/exports/reward_dataset.parquet")

# Scoring Logic
STATE_WEIGHTS = {
    "FOCUSED": 1.0,
    "NEUTRAL": 0.5,
    "STARTLED": 0.0,
    "UNKNOWN": 0.0
}

def calculate_social_reward(attention_bool, state_str, flinch_bool):
    """
    Core Reward Formula:
    R_s = (Attention * State_Weight) - (Flinch * 2.0)
    """
    weight = STATE_WEIGHTS.get(state_str, 0.0)
    reward = (int(attention_bool) * weight) - (int(flinch_bool) * 2.0)
    return float(reward)

def generate_justification(state_str, attention_bool, flinch_bool, v_peak, r_s):
    """
    Uses Gemma 4 E4B in Thinking Mode to generate a 1-sentence safety justification.
    Gated by --justify flag.
    """
    prompt = (
        f"Context: The robot attempted a task. The human was [Attention: {attention_bool}], "
        f"in a cognitive frame of [{state_str}], and they {'did' if flinch_bool else 'did not'} "
        f"physically flinch (V_peak={v_peak:.2f}). The calculated social reward is {r_s:.1f}. "
        "Output a 1-sentence evaluation explaining if this is safe behavior based on the reward."
    )
    
    try:
        # Note: Using ollama library (must be available in environment)
        response = ollama.chat(
            model='gemma4:latest',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"Justification skipped: {e}"

def main():
    parser = argparse.ArgumentParser(description="Node 5: Appropriateness Judge (Social Reward Synthesis)")
    parser.add_argument('--justify', action='store_true', help="Generate optional Gemma-based justifications")
    parser.add_argument('--output', type=str, default=DEFAULT_EXPORT_PATH, help="Output Parquet path")
    args = parser.parse_args()

    print("--- Appropriateness Judge (Node 5) ---")

    # 1. Load All Data
    print("Loading module metadata...")
    if not all(os.path.exists(f) for f in [ATTENTION_RESULTS, ENGAGEMENT_RESULTS, FLINCH_RESULTS]):
        print("Error: Missing one or more module result files. Ensure Nodes 1-3 have run successfully.")
        return

    with open(ATTENTION_RESULTS, 'r') as f:
        attention_data = {item['ego_video_id']: item for item in json.load(f)}
    with open(ENGAGEMENT_RESULTS, 'r') as f:
        engagement_data = {item['ego_video_id']: item for item in json.load(f)}
    with open(FLINCH_RESULTS, 'r') as f:
        flinch_data = {item['ego_video_id']: item for item in json.load(f)}
    
    print(f"Loading paired manifest from {MANIFEST_PATH}...")
    manifest_df = pd.read_csv(MANIFEST_PATH)
    
    results = []
    print(f"Synthesizing reward signals...")
    
    # 2. Join and Calculate
    for _, row in manifest_df.iterrows():
        ego_id = row['ego_video_id']
        
        # Only process clips that reached at least the attention filtering stage
        if ego_id not in attention_data:
            continue
            
        attn = attention_data[ego_id]
        
        # Skip clips that were already discarded by Node 1
        if not attn['passed_attention']:
            continue
            
        # Get data from downstream nodes
        eng = engagement_data.get(ego_id)
        fli = flinch_data.get(ego_id)
        
        # Defensive check: if a clip passed attention but processing failed downstream
        if eng is None or fli is None:
            print(f"  [WARN] Clip {ego_id} is INCOMPLETE. Missing engagement or flinch data.")
            continue
        
        passed_attn = attn['passed_attention']
        state = eng['state']
        has_flinch = fli['flinch']
        v_peak = fli['v_peak']
        
        rs = calculate_social_reward(passed_attn, state, has_flinch)
        
        # 3. Format result (Strict Derived-Signal Only, No Pixels)
        result_item = {
            "ego_video_id": ego_id,
            "tp_video_id": row['tp_video_id'],
            "timestamp_start": float(attn['primary_window']['t_start']),
            "timestamp_end": float(attn['primary_window']['t_end']),
            "action_annotations": str(row['actions']),
            "cognitive_state": state,
            "flinch_detected": bool(has_flinch),
            "v_peak": float(v_peak),
            "social_reward": float(rs)
        }
        
        if args.justify:
            print(f"  Generating reasoning for {ego_id} via Gemma 4...")
            result_item["justification"] = generate_justification(state, passed_attn, has_flinch, v_peak, rs)
            
        results.append(result_item)

    if not results:
        print("No finalist clips found to export.")
        return

    # 4. Generate Parquet
    df = pd.DataFrame(results)
    
    # Ensure the exports folder exists
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nFinal Statistics:")
    print(df[['cognitive_state', 'flinch_detected', 'social_reward']].describe(include='all'))
    
    print(f"\nExporting to Parquet: {output_path}")
    # Using pyarrow engine for high-performance vectorized I/O
    df.to_parquet(output_path, engine="pyarrow", index=False)
    
    print("\n✅ Node 5 completion: Parquet dataset hydration ready.")

if __name__ == "__main__":
    main()
