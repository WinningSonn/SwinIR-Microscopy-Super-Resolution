import psutil
import torch

def get_system_stats():
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory
        used_mem = torch.cuda.memory_allocated(0)
        gpu_load = f"{(used_mem / total_mem) * 100:.1f}%"
    else:
        gpu_name = "No GPU Detected"
        gpu_load = "N/A"

    return {
        "CPU Usage": f"{cpu_usage:.1f}%",
        "RAM Usage": f"{ram_usage:.1f}%",
        "Disk Usage": f"{disk_usage:.1f}%",
        "GPU Load": gpu_load,
        "GPU Name": gpu_name,
    }
