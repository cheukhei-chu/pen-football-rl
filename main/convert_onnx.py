import torch
import torch.nn as nn
import numpy as np
import os
from policy import policy_from_checkpoint_path, CurriculumMLPPolicy

# --- Configuration ---
CHECKPOINT_PATH = "../checkpoints/league_ppo (misc rewards)/checkpoint_7100000.pth"
OUTPUT_ONNX_PATH = "../docs/league_ppo (misc rewards) checkpoint_7100000.onnx"
INPUT_OBS_SHAPE = (1, 12)  # (Batch, obs_dim)
# Note: The policy's embed_dim is 3.

class CurriculumPolicyExportWrapper(nn.Module):
    """Wraps the CurriculumMLPPolicy to expose all necessary inputs for ONNX tracing."""
    def __init__(self, policy: CurriculumMLPPolicy):
        super().__init__()
        # Copy the original policy's components
        self.policy = policy
        self.plan_net = policy.plan_net
        self.action_net = policy.action_net
        self.head_left = policy.head_left
        self.head_right = policy.head_right
        self.head_jump = policy.head_jump
        self.value_head = policy.value_head

        # We also need the drill_to_index map if we want to support dynamic task changes

    def forward(self, obs: torch.Tensor):
        """
        Custom forward pass that only traces the 'Drill Mode' logic,
        as that is what is used when the policy is *not* None/Planning.
        """
        plan = self.plan_net(obs)

        # 3. Action Network (uses obs subset + task embedding)
        # Relevant obs features: [0, 1, 2, 3, 8, 9, 10, 11]
        obs_subset = obs[:, [0, 1, 2, 3, 8, 9, 10, 11]]     # (batch, 8)

        # Concatenate and pass through action network trunk
        x = self.action_net(torch.cat([obs_subset, plan], dim=-1)) # (batch, 128)

        # 4. Heads
        return (
            self.head_left(x),
            self.head_right(x),
            self.head_jump(x),
            self.value_head(x),
        )

def convert_curriculum_policy_to_onnx(checkpoint_path, output_onnx_path, obs_shape):
    # 1. Load the policy
    print(f"Loading checkpoint from: {checkpoint_path}")
    policy, _ = policy_from_checkpoint_path(checkpoint_path)
    assert isinstance(policy, CurriculumMLPPolicy), "Loaded policy is not CurriculumMLPPolicy."
    policy.eval()

    # 2. Wrap the policy for ONNX export
    export_model = CurriculumPolicyExportWrapper(policy)

    # 3. Define dummy inputs
    dummy_obs = torch.randn(*obs_shape, dtype=torch.float32)
    # Dummy inputs for the curriculum setting. Use a fixed index (e.g., 3 for 'shoot_left')
    # The batch dimension is 1 for the index.
    dummy_idx = torch.tensor([3], dtype=torch.long) # shape (1,)
    dummy_par = torch.tensor([[0.5]], dtype=torch.float32) # shape (1, 1)

    # 4. Perform the export
    input_names = ["obs", "drill_idx", "par_val"]
    output_names = ["left_logits", "right_logits", "jump_logits", "value"]

    print(f"Starting ONNX export of the Curriculum Worker Network...")
    torch.onnx.export(
        export_model,
        dummy_obs,
        output_onnx_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,
    )

    print(f"Successfully exported ONNX model to {output_onnx_path}")


if __name__ == "__main__":
    full_checkpoint_path = os.path.join(os.path.dirname(__file__), CHECKPOINT_PATH)
    # Ensure this script is run from a location where the relative path to the checkpoint works
    convert_curriculum_policy_to_onnx(full_checkpoint_path, OUTPUT_ONNX_PATH, INPUT_OBS_SHAPE)
