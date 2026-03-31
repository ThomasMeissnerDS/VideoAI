import timeit
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

def main():
    start = timeit.timeit()
    print(f"Starting pipeline at: {start}")
    model_id = "THUDM/CogVideoX-2b"

    # 1. Wir laden das Modell weiterhin in 16-bit
    pipe = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )

    # 2. DER RETTER FÜR 24GB KARTEN:
    # Anstatt enable_model_cpu_offload() nutzen wir das aggressivere VRAM-Management.
    # Dies lädt immer nur exakt das Teil-Modell in den VRAM, das in der Sekunde gebraucht wird.
    pipe.enable_sequential_cpu_offload()

    # 3. VAE Tiling aktivieren (Gegen den Absturz am Ende!)
    # Das teilt das Video im letzten Schritt in "Kacheln" auf und berechnet
    # sie nacheinander statt alle auf einmal. Das spart extrem viel VRAM.
    pipe.vae.enable_tiling()

    # Optional, aber empfohlen für NVIDIA RTX Karten:
    # pipe.enable_xformers_memory_efficient_attention() # Setzt voraus, dass xformers installiert ist

    prompt = (
        "A man looking on his smartphone. He does not recognise that he did not press the traffic light button."
        "Another man arrives. He looks at the man and notices the situation. He presses the button instead."
        "The traffic light turns green and only the conscious man starts walking."
        "Produce this in comic optic and in a relatively simple style (Tom & Jerry like) and use professional cut scenes."
    )

    print("Generiere Video... Das kann einige Minuten dauern!")

    # 4. Memory-Optimierung während der Generierung
    with torch.autocast("cuda"):
        video_frames = pipe(
            prompt=prompt,
            num_inference_steps=50,
            num_frames=240,
            guidance_scale=6.0
        ).frames[0]

    export_to_video(video_frames, "mein_eigenes_ki_video.mp4", fps=24)
    print("Video erfolgreich gespeichert!")
    end = timeit.timeit()
    print(f"Finished pipeline at: {end}")
    print(f"Total runtime: {end - start}")

if __name__ == "__main__":
    main()