from ctransformers import AutoModelForCausalLM
import time

# ==== SETTINGS ====
# This must be the full path to your downloaded GGUF file on the SSD.
MODEL_PATH = "/home/shreesha/ssd/gemma-2-2b-it-Q4_K_M.gguf" 

# ==== LOAD THE MODEL ON CPU ====
print(f"Loading model from: {MODEL_PATH}")
# gpu_layers=0 ensures this runs entirely on the CPU.
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    model_type="gemma2",
    gpu_layers=0  
)
print("✅ Model loaded successfully on the CPU.")

# ==== RUN INFERENCE ====
prompt = "Convert this sentence to a Linux command: 'Find all text files in my documents folder that contain the word invoice'"

print(f"\nPrompt: {prompt}")

start_time = time.time()
response = llm(prompt)
end_time = time.time()

print(f"\nGenerated Command: {response.strip()}")
print(f"\n(Time taken: {end_time - start_time:.2f} seconds)")
