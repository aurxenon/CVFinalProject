import os
import requests
from tqdm import tqdm
# Set environment variable BEFORE importing huggingface_hub
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import HfApi, hf_hub_download

def download_dataset():
    # Repository information
    repo_id = "masterda/SloMoBlur"
    repo_type = "dataset"
    
    # Output directory
    output_dir = "./Dataset/SloMoBlur"
    
    # # Use Chinese mirror (HF-Mirror)
    # mirror_endpoint = 'https://hf-mirror.com'
    # print(f"Using Chinese mirror: {mirror_endpoint}")
    # print(f"HF_ENDPOINT environment variable: {os.environ.get('HF_ENDPOINT', 'Not set')}")
    
    # # Initialize API with mirror endpoint
    # api = HfApi(endpoint=mirror_endpoint)
    api = HfApi()
    
    # Directories to download
    # directories = ["0_9999/blurred", "0_9999/groundtruth"]
    directories = ["0_9999/groundtruth"]

    # Create local directories
    for directory in directories:
        full_path = os.path.join(output_dir, directory)
        os.makedirs(full_path, exist_ok=True)
    
    # Get all files in repository
    print("Fetching file list...")
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    
    # Download files from each directory
    for directory in directories:
        print(f"\nDownloading files from {directory}...")
        
        # Get files in this directory
        directory_files = [f for f in repo_files if f.startswith(directory + "/")]
        print(f"Found {len(directory_files)} files in {directory}")
        
        # Download each file
        for i, file_path in enumerate(tqdm(directory_files, desc=f"Downloading {directory}")):
            try:
                # Try using hf_hub_download first
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file_path,
                        repo_type=repo_type,
                        local_dir=output_dir,
                        local_dir_use_symlinks=False
                    )
                except Exception as e1:
                    # If that fails, try manual download from mirror
                    print(f"\nTrying manual download for {file_path}...")
                    # Construct mirror URL
                    # Format: https://hf-mirror.com/datasets/{repo_id}/resolve/main/{file_path}
                    mirror_url = f"https://hf-mirror.com/datasets/{repo_id}/resolve/main/{file_path}"
                    
                    # Local file path
                    local_file_path = os.path.join(output_dir, file_path)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    
                    # Download with requests
                    response = requests.get(mirror_url, stream=True)
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get('content-length', 0))
                    with open(local_file_path, 'wb') as f, tqdm(
                        desc=os.path.basename(file_path),
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                bar.update(len(chunk))
                    print(f"✓ Downloaded: {file_path}")
                    
            except Exception as e:
                print(f"❌ Error downloading {file_path}: {e}")
                continue
    
    print("\nDownload complete!")
    print(f"Files saved to: {output_dir}")

# Run the download
download_dataset()