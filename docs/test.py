import onnxruntime
sess = onnxruntime.InferenceSession("league_ppo (misc rewards) checkpoint_7100000.onnx") # Use your exact filename
for i, input_meta in enumerate(sess.get_inputs()):
    print(f"Input {i}: {input_meta.name}, Shape: {input_meta.shape}")
