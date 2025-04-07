import requests
import json
import time
import sys

# API URL - default to localhost if not provided
api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

# Test prompt
prompt = "A circle transforming into a square and then a triangle"
complexity = 2

print(f"Sending animation request to {api_url}...")
print(f"Prompt: '{prompt}'")
print(f"Complexity: {complexity}")

# Send animation request
try:
    response = requests.post(
        f"{api_url}/generate",
        json={
            "prompt": prompt,
            "animate": True,
            "complexity": complexity
        },
        timeout=10
    )

    response.raise_for_status()
    result = response.json()
    job_id = result["job_id"]
    print(f"\nJob submitted with ID: {job_id}")
    
    # Poll for status
    print("\nPolling for job status...")
    while True:
        status_response = requests.get(f"{api_url}/status/{job_id}")
        status = status_response.json()
        
        print(f"Status: {json.dumps(status, indent=2)}")
        
        if status.get("success") or status.get("error"):
            break
            
        print("Waiting for job to complete...")
        time.sleep(3)
    
    if status.get("success") and status.get("video_url"):
        print(f"\n✅ Video ready at: {status['video_url']}")
    elif status.get("error"):
        print(f"\n❌ Error generating animation: {status['error']}")
    else:
        print("\n❓ Unknown status")
        
except requests.exceptions.RequestException as e:
    print(f"\n❌ Error communicating with API: {str(e)}")
except Exception as e:
    print(f"\n❌ Unexpected error: {str(e)}")