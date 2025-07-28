import runpod
import time  
import torch

def print_gpu_info():
    if not torch.cuda.is_available():
        print("No GPUs available.")
        return
    num_gpus = torch.cuda.device_count()
    print(f"GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")

def handler(event):
    print(f"Worker Start")
    input = event['input']
    
    prompt = input.get('prompt')  
    seconds = input.get('seconds', 0)  

    print(f"Received prompt: {prompt}")
    print(f"Sleeping for {seconds} seconds...")
    print_gpu_info()
    
    # Replace the sleep code with your Python function to generate images, text, or run any machine learning workload
    time.sleep(seconds)  
    
    return prompt 

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
