import torch
import json
import numpy as np
import os
from policy import policy_from_checkpoint_path

# --- CONFIGURATION ---
# Point this to your actual .pth file
CHECKPOINT_PATH = "../checkpoints/league_ppo (misc rewards)/checkpoint_7100000.pth"

def export_to_json():
    print(f"Loading {CHECKPOINT_PATH}...")
    # Load the policy
    policy, _ = policy_from_checkpoint_path(CHECKPOINT_PATH)
    policy.eval()

    weights = {}

    def save_layer(prefix, layer):
        # Save Weight (w) and Bias (b) as lists
        weights[f"{prefix}_w"] = layer.weight.detach().cpu().numpy().tolist()
        if layer.bias is not None:
            weights[f"{prefix}_b"] = layer.bias.detach().cpu().numpy().tolist()
        else:
            weights[f"{prefix}_b"] = [0.0] * layer.out_features

    # 1. Plan Network (Linear -> ReLU -> Linear -> ReLU)
    #    Structure: [Linear, ReLU, Linear, ReLU]
    save_layer("plan_0", policy.plan_net[0])
    save_layer("plan_2", policy.plan_net[2])

    # 2. Action Network (Linear -> ReLU -> Linear -> ReLU)
    #    Structure: [Linear, ReLU, Linear, ReLU]
    save_layer("action_0", policy.action_net[0])
    save_layer("action_2", policy.action_net[2])

    # 3. Output Heads
    save_layer("head_left", policy.head_left)
    save_layer("head_right", policy.head_right)
    save_layer("head_jump", policy.head_jump)

    # Save to file
    output_file = "model_weights.json"
    with open(output_file, "w") as f:
        json.dump(weights, f)

    print(f"Success! Saved weights to {output_file}")

if __name__ == "__main__":
    export_to_json()
