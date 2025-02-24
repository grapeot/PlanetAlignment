#!/usr/bin/env python3
"""
Combine alignment visualization images into a video using ffmpeg.
"""

import os
import re
from datetime import datetime
import subprocess

def natural_sort_key(s):
    """
    Generate a key for natural sorting of strings containing dates.
    """
    # Extract date from filename
    match = re.search(r'(\d{8})_\d{4}', s)
    if match:
        return datetime.strptime(match.group(1), '%Y%m%d')
    return datetime.min

def main():
    # Get all PNG files in the alignment_viz directory
    image_dir = 'alignment_viz'
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    # Sort files by date
    image_files.sort(key=natural_sort_key)
    
    print(f"Found {len(image_files)} images")
    print("Creating video...")
    
    # Get year range from filenames
    start_year = natural_sort_key(image_files[0]).year
    end_year = natural_sort_key(image_files[-1]).year
    
    # Create output filename
    output_file = f"planetary_alignment_{start_year}"
    if start_year != end_year:
        output_file += f"-{end_year}"
    output_file += ".mp4"
    
    # Create ffmpeg command
    # -framerate 60: Set input framerate to 60 fps (1 day = 1 frame)
    # -pattern_type glob: Use glob pattern for input
    # -i: Input pattern
    # -c:v libx264: Use H.264 codec
    # -preset medium: Balance between encoding speed and compression
    # -crf 23: Constant Rate Factor for quality (lower = better, 23 is default)
    # -pix_fmt yuv420p: Standard pixel format for compatibility
    cmd = [
        'ffmpeg',
        '-framerate', '60',  # Increased to 60fps for smoother playback
        '-pattern_type', 'glob',
        '-i', f'{image_dir}/*.png',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file if exists
        output_file
    ]
    
    try:
        print("Running ffmpeg command...")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Video creation complete!")
        print(f"Output: {output_file}")
    except subprocess.CalledProcessError as e:
        print("Error creating video:")
        print(f"ffmpeg stderr: {e.stderr}")
        raise

if __name__ == "__main__":
    main() 