import os
import json
import timeit
import torch
from google import genai
from pydantic import BaseModel, Field
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Pydantic schema for structured Gemini output
class Shot(BaseModel):
    shot_number: int = Field(description="Sequential shot number (e.g., 1, 2, 3)")
    prompt: str = Field(description="Detailed image prompt for the AI video generation model capturing this shot.")

class VideoScript(BaseModel):
    shots: list[Shot] = Field(description="List of consecutive shots that compose the entire scene.")

def generate_script_with_gemini(story_prompt: str, api_key: str) -> list[str]:
    print("Generating script from the story prompt via Gemini...")
    client = genai.Client(api_key=api_key)
    
    system_instruction = (
        "You are an expert film director breaking down a complicated story into a sequence of individual 2-second "
        "camera shots for an AI video generator. Each shot must explicitly re-state subject descriptions "
        "(e.g., 'a man in comic style', 'a yellow traffic light') so the generator doesn't lose context. "
        "Do not use generic pronouns. Keep it strictly to the current action. "
        "Return exactly 3 to 4 simple, distinct 2-second shots."
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=story_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=VideoScript,
            temperature=0.2,
        ),
    )
    
    # Using the native SDK parsing feature
    script_data = response.parsed
    print("\n🎬 Generated Shot Sequence:")
    prompts = []
    for shot in script_data.shots:
        print(f"  Shot {shot.shot_number}: {shot.prompt}")
        prompts.append(shot.prompt)
    
    return prompts

def stitch_videos(video_files: list[str], output_path: str):
    print(f"\nStitching {len(video_files)} clips together...")
    clips = []
    for v in video_files:
        clips.append(VideoFileClip(v))
    
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path, fps=24, logger=None)
    
    # Ensure memory is released
    for clip in clips:
        clip.close()
    final_clip.close()
    
    print(f"Successfully saved final stitched video at {output_path}")

def main():
    start = timeit.default_timer()
    print("Starting pipeline.")
    
    # 1. Make sure user configured their API key
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        # Prompt if env var is missing to make debugging easier
        gemini_api_key = input("Please enter your GEMINI_API_KEY (or set it as an env variable): ").strip()

    master_prompt = (
        "A man looking on his smartphone. He does not recognise that he did not press the traffic light button. "
        "Another man arrives. He looks at the man and notices the situation. He presses the button instead. "
        "The traffic light turns green and only the conscious man starts walking. "
        "Produce this in comic optic and in a relatively simple style (Tom & Jerry like)."
    )
    
    # 2. Split script using Gemini
    shot_prompts = generate_script_with_gemini(master_prompt, gemini_api_key)
    
    # 3. Setup CogVideoX pipeline
    model_id = "THUDM/CogVideoX-2b"
    print(f"\nLoading local video model ({model_id})...")
    pipe = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )
    
    # Memory optimizations for RTX 3090 / 24GB GPUs
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    
    generated_video_files = []
    
    # 4. Generation Loop (49 frames = ~2 seconds)
    print(f"\nStarting grouped generation of {len(shot_prompts)} distinct 2-second clips...")
    for idx, shot_prompt in enumerate(shot_prompts):
        print(f"\n--- Generating Clip {idx+1}/{len(shot_prompts)} ---")
        print(f"Prompt: {shot_prompt}")
        
        with torch.autocast("cuda"):
            video_frames = pipe(
                prompt=shot_prompt,
                num_inference_steps=50,
                num_frames=49,  # 49 is generally stable for 2 seconds chunking
                guidance_scale=6.0
            ).frames[0]
            
        filename = f"clip_part_{idx+1}.mp4"
        export_to_video(video_frames, filename, fps=24)
        generated_video_files.append(filename)
        print(f"Saved {filename}")

    # 5. Stitch Videos together seamlessly
    stitch_videos(generated_video_files, "mein_eigenes_ki_video_director.mp4")
    
    # Cleanup chunk files to keep directory clean (Optional)
    for f in generated_video_files:
        try:
            os.remove(f)
        except Exception:
            pass
            
    end = timeit.default_timer()
    print(f"\nFinished pipeline. Total runtime: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()