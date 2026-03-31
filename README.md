# VideoAI: LLM-Driven Video Director

An optimized AI video generation pipeline designed specifically for RTX 3090 (24GB VRAM) and similar hardware.

## Why this architecture?
Generating long, complex narratives directly using Text-to-Video models often causes severe VRAM overload and morphological nightmares (the model blends concepts together). This pipeline solves the problem by:
1. **The Director (Gemini):** Using an LLM to split a complex prompt into multiple 2-second "shots".
2. **Chunked Generation:** Forcing the local `CogVideoX` model to only generate exactly 2 seconds of video at a time, protecting VRAM and forcing visual adherence.
3. **Automated Assembly:** Using `moviepy` to automatically stitch the shots together into a continuous final clip.

## Prerequisites & Installation

It is recommended to use a Python virtual environment to prevent dependency conflicts.

```bash
# 1. (Optional but recommended) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install the newly added dependencies
pip install google-genai pydantic moviepy

# 3. (Assuming you already have these, but just in case) Core ML libraries
pip install diffusers transformers accelerate torch torchvision
```

> **Note for Mac/Linux users:** `moviepy` relies on `ffmpeg` under the hood to stitch the videos. If you run into any ffmpeg-related errors upon generation, you can install it system-wide using `brew install ffmpeg`.

## Running the Pipeline

1. **Set your Gemini API Key:** The script requires Gemini to perform the director role. You can set it as an environment variable in your terminal:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
2. **Execute:**
   ```bash
   python main.py
   ```
   *Note: If you didn't set the API key in your environment, the script will pause and interactively ask you to paste it before continuing.*

## Output Workflow
1. The script reaches out to Gemini to split your `master_prompt`.
2. It loops through the new prompts, generating `clip_part_1.mp4`, `clip_part_2.mp4`, etc.
3. It seamlessly concatenates them into `mein_eigenes_ki_video_director.mp4` and deletes the temporary parts.
