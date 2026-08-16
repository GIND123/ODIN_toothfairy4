import modal

app = modal.App("b2t-smoke")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, gpu="A10G", timeout=300)
def gpu_probe() -> str:
    import subprocess
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
