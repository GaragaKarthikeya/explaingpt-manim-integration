import requests
import json
import time
import sys
import argparse
from urllib.parse import urlparse

def print_banner():
    """Print a nice banner for the Manim API client."""
    print("\n" + "=" * 60)
    print("              MANIM ANIMATION API CLIENT")
    print("=" * 60)

def validate_complexity(value):
    """Validate complexity level input."""
    ivalue = int(value)
    if ivalue < 1 or ivalue > 3:
        raise argparse.ArgumentTypeError("Complexity must be between 1-3")
    return ivalue

def get_user_input():
    """Get prompt and complexity from command line args or user input."""
    parser = argparse.ArgumentParser(description='Generate animations with Manim API')
    parser.add_argument('--url', default="http://localhost:8000", help='API URL (default: http://localhost:8000)')
    parser.add_argument('--prompt', help='Animation prompt')
    parser.add_argument('--complexity', type=validate_complexity, choices=[1, 2, 3], 
                        help='Animation complexity level (1-3)')
    
    args = parser.parse_args()
    
    # Get API URL
    api_url = args.url
    
    # Get prompt from args or user input
    if args.prompt:
        prompt = args.prompt
    else:
        print("\nEnter your animation prompt (what would you like to visualize?):")
        prompt = input("> ")
        
        if not prompt.strip():
            print("Error: Prompt cannot be empty. Exiting.")
            sys.exit(1)
    
    # Get complexity from args or user input
    if args.complexity:
        complexity = args.complexity
    else:
        print("\nSelect complexity level (1-3):")
        print("  1 - Simple: Basic visualization")
        print("  2 - Moderate: More detailed visualization")
        print("  3 - Complex: Comprehensive visualization with details")
        
        while True:
            try:
                complexity = int(input("Enter complexity [2]: ") or "2")
                if 1 <= complexity <= 3:
                    break
                else:
                    print("Please enter a number between 1 and 3.")
            except ValueError:
                print("Please enter a valid number.")
    
    return api_url, prompt, complexity

def generate_animation(api_url, prompt, complexity):
    """Send animation request and poll for results."""
    print(f"\nSending animation request to {api_url}...")
    print(f"Prompt: '{prompt}'")
    print(f"Complexity: {complexity}")
    
    try:
        # Send animation request
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
        
        # Poll for status with progress display
        print("\nPolling for job status...")
        iteration = 0
        spinner = ['|', '/', '-', '\\']
        
        while True:
            try:
                status_response = requests.get(f"{api_url}/status/{job_id}")
                status_response.raise_for_status()
                status = status_response.json()
                
                # Clear previous line
                sys.stdout.write('\r' + ' ' * 80)
                sys.stdout.flush()
                
                # Print spinner and status
                status_emoji = "⏳"
                if status.get("success"):
                    status_emoji = "✅"
                elif status.get("error"):
                    status_emoji = "❌"
                    
                sys.stdout.write(f"\r{status_emoji} {spinner[iteration % 4]} Status: {status.get('status', 'Processing')}")
                sys.stdout.flush()
                
                if status.get("success") or status.get("error"):
                    print("\n")  # Add some space after completion
                    break
                    
                time.sleep(2)
                iteration += 1
                
            except requests.exceptions.RequestException as e:
                print(f"\n❌ Error checking job status: {str(e)}")
                time.sleep(5)
        
        if status.get("success") and status.get("video_url"):
            video_url = status["video_url"]
            print(f"✅ Animation Complete!")
            print(f"\n🎬 Video available at: {video_url}")
            
            # Determine if the URL is local or remote
            parsed_url = urlparse(video_url)
            if not parsed_url.netloc or parsed_url.netloc == "localhost":
                print("\nNote: This appears to be a local URL. Access it from your browser on this machine.")
            else:
                print("\nNote: You can access this URL from any browser.")
                
            return video_url
        elif status.get("error"):
            print(f"\n❌ Error generating animation: {status['error']}")
            return None
        else:
            print("\n❓ Unknown status")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error communicating with API: {str(e)}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return None

def main():
    """Main function."""
    print_banner()
    api_url, prompt, complexity = get_user_input()
    video_url = generate_animation(api_url, prompt, complexity)
    
    if video_url:
        print(f"\nThank you for using the Manim Animation API!")

if __name__ == "__main__":
    main()
