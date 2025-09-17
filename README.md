
# Voice-Controlled Terminal Assistant for Blind Accessibility

A hands-free, conversational AI assistant that allows users to interact with a Linux terminal using only their voice. This project is designed to run efficiently on edge devices, specifically the NVIDIA Jetson Orin Nano, to provide a powerful accessibility tool.



---

## Key Features

* **Voice-Only Interaction:** Control your terminal with natural language commands.
* **Real-Time & Responsive:** A low-latency architecture ensures the assistant feels immediate and conversational.
* **Edge-Optimized:** Designed to run entirely on-device (NVIDIA Jetson), ensuring privacy and offline capability.
* **Intelligent Command Generation:** Powered by a quantized Gemma 2B model to translate complex requests into accurate shell commands.
* **Full Feedback Loop:** The assistant not only executes commands but also vocalizes the results.

---

## System Architecture

The assistant is built on a modular, three-part pipeline that strategically allocates tasks between the Jetson's CPU and GPU to maximize performance and stability.

<img width="952" height="479" alt="image" src="https://github.com/user-attachments/assets/d5b29990-6899-4ee6-b275-4204b3ff1745" />


---

## Technology Stack

* **Language Model (The Brain):** Google Gemma 2B (Instruction-Tuned), quantized to 4-bit GGUF.
* **LLM Runner:** `llama-cpp-python` for efficient CPU-based inference.
* **Speech-to-Text (The Ears):** `faster-whisper` for GPU-accelerated transcription.
* **Text-to-Speech (The Mouth):** `piper-tts` for lightweight, CPU-based voice generation.
* **Audio Handling:** `sounddevice` for microphone input.
* **Hardware:** NVIDIA Jetson Orin Nano (8GB recommended).
* **Environment:** Conda with Python 3.10.


## Setup and Installation (on Jetson)

1.  **Create a Conda Environment:**
    ```bash
    conda create --name blind_assistant python=3.10
    conda activate blind_assistant
    ```

2.  **Install PyTorch:**
    Install the official NVIDIA wheel for your specific JetPack version. Find the correct command on the [NVIDIA Jetson forums](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).

3.  **Install Dependencies:**
    ```bash
    # Install the core libraries
    pip install faster-whisper piper-tts sounddevice

    # Install llama-cpp-python with CUDA support by building from source
    CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
    ```

4.  **Download the Model:**
    Obtain the `gemma-2-2b-it.q4_k_m.gguf` model file and place it in your project directory.

---

## Usage

1.  Activate the conda environment: `conda activate blind_assistant`.
2.  Run the main application script: `python3 main.py`.
3.  The application will start listening for voice commands. Speak clearly and wait for the spoken response.

**Example Commands:**
* "List all the files in this directory."
* "Create a new folder named 'project reports'."
* "Tell me my current location in the filesystem."
````
