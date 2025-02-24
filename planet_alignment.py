#!/usr/bin/env python3
"""
Test script for visualizing planetary positions and alignment scores.
Creates two projections with visibility information based on altitude/azimuth coordinates.
"""

import numpy as np
from skyfield.api import load, wgs84, utc
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from skyfield.api import Topos
import json
from pathlib import Path
from skyfield import almanac
from sklearn.linear_model import RANSACRegressor
import multiprocessing as mp
from tqdm import tqdm
import fcntl
import tempfile
from queue import Empty

# Constants
BEIJING_LAT = 39.9042
BEIJING_LON = 116.4074
TIMEZONE_OFFSET = 8  # Beijing is UTC+8
CIVIL_TWILIGHT_ALTITUDE = -6  # Civil twilight is when sun is 6° below horizon
VISIBILITY_CACHE_FILE = 'planet_visibility_cache.jsonl'
SCORES_CACHE_FILE = 'alignment_scores_cache.json'

# Global cache for visibility data
visibility_cache = {}

# Global variables to store Skyfield objects
ts = None
eph = None
earth = None
beijing = None
planets = None
sun = None

def init_worker():
    """Initialize global variables for each worker process"""
    global ts, eph, earth, beijing, planets, sun
    
    # Load required data
    ts = load.timescale()
    eph = load('de421.bsp')
    
    # Define observer location (Beijing)
    earth = eph['earth']
    beijing = earth + Topos(BEIJING_LAT, BEIJING_LON)
    
    # Define all planets (except Earth) and Sun
    planets = {
        'mercury': eph['mercury'],
        'venus': eph['venus'],
        'mars': eph['mars'],
        'jupiter': eph['jupiter barycenter'],
        'saturn': eph['saturn barycenter'],
        'uranus': eph['uranus barycenter'],
        'neptune': eph['neptune barycenter']
    }
    sun = eph['sun']

def safe_read_json(filename):
    """Safely read a JSON file with file locking"""
    if not os.path.exists(filename):
        return {}
        
    with open(filename, 'r') as f:
        # Get an exclusive lock
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}
        finally:
            # Release the lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def safe_write_json(data, filename):
    """Safely write data to a JSON file with file locking"""
    # Write to a temporary file first
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(filename))
    try:
        with os.fdopen(temp_fd, 'w') as temp_file:
            # Get an exclusive lock
            fcntl.flock(temp_file.fileno(), fcntl.LOCK_EX)
            json.dump(data, temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            # Lock is automatically released when file is closed
    except:
        os.unlink(temp_path)
        raise
        
    # Atomic rename
    os.rename(temp_path, filename)

def load_visibility_cache():
    """
    Load visibility cache from JSONL file.
    Returns a dictionary with dates as keys and visibility info as values.
    """
    cache = {}
    if os.path.exists(VISIBILITY_CACHE_FILE):
        with open(VISIBILITY_CACHE_FILE, 'r') as f:
            # Get a shared lock for reading
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        date_str = data['date']
                        cache[date_str] = data['visibility']
                    except json.JSONDecodeError:
                        continue
            finally:
                # Release the lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return cache

def save_visibility_to_cache(date, visibility_data):
    """
    Save visibility data to memory cache and update the cache file.
    """
    global visibility_cache
    date_str = date.strftime('%Y-%m-%d')
    visibility_cache[date_str] = visibility_data
    
    # Convert visibility data to JSON serializable format
    serializable_data = {
        'date': date_str,
        'visibility': {k: bool(v) for k, v in visibility_data.items()}  # Ensure booleans are JSON serializable
    }
    
    # Append to file with locking
    with open(VISIBILITY_CACHE_FILE, 'a') as f:
        # Get an exclusive lock
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(serializable_data, f)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        finally:
            # Release the lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def check_visibility(time):
    """
    Check visibility of all planets at given time.
    Returns a dictionary with planet names as keys and tuples of
    (is_visible, alt, az, ra, dec, lon, lat) as values.
    """
    visibility = {}
    
    # Check each planet
    for name, body in planets.items():
        # Get horizontal coordinates (alt/az)
        apparent = beijing.at(time).observe(body).apparent()
        alt, az, _ = apparent.altaz()
        
        # Get equatorial coordinates
        ra, dec, _ = apparent.radec()
        
        # Get ecliptic coordinates
        lat, lon, _ = apparent.ecliptic_latlon('date')
        lon_deg = float(lon.degrees)
        lat_deg = float(lat.degrees)
        
        # Normalize longitude to [0, 360] range
        while lon_deg < 0:
            lon_deg += 360
        while lon_deg >= 360:
            lon_deg -= 360
        
        # Planet is visible if it's above horizon
        is_visible = alt.degrees > 0
        
        visibility[name] = (is_visible, alt.degrees, az.degrees, 
                          ra.hours, dec.degrees, lon_deg, lat_deg)
    
    return visibility

def get_sunrise_sunset(date):
    """
    Get sunrise and sunset times for a given date in Beijing.
    Returns (sunrise_time, sunset_time) in UTC.
    """
    # Create time span for the day
    t0 = ts.from_datetime(date.replace(hour=0))
    t1 = ts.from_datetime(date.replace(hour=23, minute=59))
    
    # Create observer
    observer = wgs84.latlon(BEIJING_LAT, BEIJING_LON)
    
    # Find sunrise and sunset events
    f = almanac.sunrise_sunset(eph, observer)
    times, events = almanac.find_discrete(t0, t1, f)
    
    sunrise = None
    sunset = None
    
    for t, e in zip(times, events):
        if e == 1:  # Sunrise
            sunrise = t
        else:  # Sunset
            sunset = t
    
    return sunrise, sunset

def check_daily_visibility(date):
    """
    Check visibility for each hour between sunset+15min and sunrise-15min.
    Returns a dictionary with planet names as keys and boolean values.
    Uses cache if available.
    """
    date_str = date.strftime('%Y-%m-%d')
    
    # Try to load from cache first
    cache = load_visibility_cache()
    if date_str in cache:
        return cache[date_str]
    
    # Get sunrise and sunset times
    sunrise, sunset = get_sunrise_sunset(date)
    if sunrise is None or sunset is None:
        # If we can't get sunrise/sunset times, mark all planets as not visible
        return {name: False for name in planets}
    
    # Add 15 minutes to sunset and subtract 15 minutes from sunrise
    sunset_time = sunset.utc_datetime() + timedelta(minutes=15)
    sunrise_time = sunrise.utc_datetime() - timedelta(minutes=15)
    
    # If sunset_time is after midnight, it belongs to the next day
    if sunset_time.date() > date.date():
        sunset_time = sunset_time.replace(
            year=date.year, month=date.month, day=date.day, hour=23, minute=59
        )
    
    # If sunrise_time is before midnight, it belongs to the previous day
    if sunrise_time.date() < date.date():
        sunrise_time = sunrise_time.replace(
            year=date.year, month=date.month, day=date.day, hour=0, minute=0
        )
    
    # Initialize visibility dictionary
    daily_visibility = {name: False for name in planets}
    
    # Generate hourly timestamps between sunset+15min and sunrise-15min
    current_time = sunset_time
    while current_time <= sunrise_time:
        # Convert to Skyfield time
        time = ts.from_datetime(current_time)
        
        # Check visibility at this hour
        visibility = check_visibility(time)
        
        # Update visibility - if a planet is visible at any hour, mark it as visible
        for name, (is_visible, *_) in visibility.items():
            daily_visibility[name] = daily_visibility[name] or is_visible
        
        # Move to next hour
        current_time += timedelta(hours=1)
    
    # Save to cache
    save_visibility_to_cache(date, daily_visibility)
    
    return daily_visibility

def calculate_positions(time):
    """
    Calculate positions and visibility for all planets.
    Returns visibility info and positions in both coordinate systems.
    """
    visibility = check_visibility(time)
    
    # Extract coordinates from visibility info
    equatorial = {name: (info[3], info[4]) for name, info in visibility.items()}
    ecliptic = {name: (info[5], info[6]) for name, info in visibility.items()}
    
    return visibility, equatorial, ecliptic

def calculate_alignment_score(positions):
    """
    Calculate alignment score using RANSAC with 10-degree tolerance.
    Returns (score, line_segments, inlier_mask).
    Score is from 0-100, where 100 is the best alignment.
    The score is weighted as follows:
    - 40% based on the absolute number of aligned planets (max 7)
    - 30% based on how well they are aligned (perpendicular distance to line)
    - 30% based on how concentrated they are in longitude (longitude range)
    """
    # Extract coordinates (already centered)
    lons = np.array([lon for lon, _ in positions.values()])
    lats = np.array([lat for _, lat in positions.values()])
    
    if len(positions) < 2:  # Need at least 2 points for a line
        return 0.0, [], None
    
    # Reshape for RANSAC
    X = lons.reshape(-1, 1)
    y = lats
    
    try:
        # Initialize and fit RANSAC
        ransac = RANSACRegressor(
            residual_threshold=10.0,  # 10 degree tolerance
            min_samples=2,  # Minimum points needed for a line
            max_trials=1000,  # More trials for better fit
            random_state=42  # For reproducibility
        )
        ransac.fit(X, y)
        
        # Get inlier mask and model parameters
        inlier_mask = ransac.inlier_mask_
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        
        # Calculate perpendicular distances for inliers only
        distances = np.abs(y - (slope * X.ravel() + intercept)) / np.sqrt(1 + slope**2)
        inlier_distances = distances[inlier_mask]
        
        # Calculate components of the score
        
        # 1. Inlier count score (40% weight)
        # Scale from 0 to 100 based on absolute number of inliers (max 7)
        inlier_count = np.sum(inlier_mask)
        inlier_score = 100 * (inlier_count / 7.0)  # 7 is max possible planets
        
        # 2. Distance score (30% weight)
        # Mean distance is best when close to 0 and worst when close to tolerance
        # Normalize to 0-100 range where 100 means perfect alignment (0 distance)
        mean_distance = np.mean(inlier_distances) if len(inlier_distances) > 0 else 10.0
        distance_score = 100 * (1 - min(mean_distance / 10.0, 1.0))
        
        # 3. Longitude range score (30% weight)
        # Calculate the range of ecliptic longitudes for inlier planets
        # Score is best (100) when all planets are within 60° range
        # Score decreases linearly to 0 when range reaches 120°
        inlier_lons = lons[inlier_mask]
        # Handle the case where longitudes cross the 180/-180 boundary
        lon_diffs = np.abs(np.subtract.outer(inlier_lons, inlier_lons))
        lon_diffs = np.minimum(lon_diffs, 360 - lon_diffs)
        lon_range = np.max(lon_diffs) if len(inlier_lons) > 0 else 120.0
        lon_range_score = 100 * (1 - min(lon_range / 120.0, 1.0))
        
        # Combine scores with weights
        final_score = 0.4 * inlier_score + 0.3 * distance_score + 0.3 * lon_range_score
        
        # Generate line points for visualization
        extension = 20  # degrees
        t_min = np.min(lons) - extension
        t_max = np.max(lons) + extension
        t = np.linspace(t_min, t_max, 1000)
        lat_line = slope * t + intercept
        
        return final_score, [(t, lat_line)], inlier_mask
    except:
        # If RANSAC fails, return zero score
        return 0.0, [], None

def create_visualizations(date):
    """
    Create visualizations for a given date with visibility information.
    """
    # Set up the figure with two subplots and a score plot
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 18))
    
    # Create grid spec for layout with fixed margins
    gs = plt.GridSpec(3, 1, height_ratios=[8, 8, 3], hspace=0.4)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
    
    # Create subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Calculate positions and visibility
    time = ts.from_datetime(date)
    visibility, equatorial, ecliptic = calculate_positions(time)
    
    # Get daily visibility
    daily_vis = check_daily_visibility(date)
    
    # Calculate mean longitude considering circular nature
    lons = np.array([lon for lon, _ in ecliptic.values()])
    mean_lon = np.degrees(np.arctan2(
        np.mean(np.sin(np.radians(lons))),
        np.mean(np.cos(np.radians(lons)))
    ))
    
    # Transform coordinates to be centered
    centered_ecliptic = {}
    centered_equatorial = {}
    for name in planets:
        # Get original coordinates
        lon, lat = ecliptic[name]
        ra, dec = equatorial[name]
        
        # Center longitude around mean
        centered_lon = ((lon - mean_lon + 180) % 360) - 180
        centered_ecliptic[name] = (centered_lon, lat)
        
        # Center RA around corresponding mean (convert mean_lon to hours)
        mean_ra = (mean_lon / 15.0) % 24
        centered_ra = ((ra - mean_ra + 12) % 24) - 12
        centered_equatorial[name] = (centered_ra, dec)
    
    # Calculate alignment score using only visible planets
    visible_ecliptic = {name: pos for name, pos in centered_ecliptic.items() 
                       if daily_vis[name]}
    
    score = 0.0
    line_segments = []
    ransac_inliers = None
    if len(visible_ecliptic) >= 2:
        score, line_segments, ransac_inliers = calculate_alignment_score(visible_ecliptic)
    
    # Create a list of visible planet names for RANSAC inlier/outlier tracking
    visible_planets = list(visible_ecliptic.keys())
    
    # Add title with alignment score and visibility info
    visible_count = sum(daily_vis.values())
    inlier_count = np.sum(ransac_inliers) if ransac_inliers is not None else 0
    fig.suptitle(
        f'Planetary Positions on {date.strftime("%Y-%m-%d %H:%M")} UTC\n'
        f'Alignment Score: {score:.1f}/100 ({inlier_count} aligned planets out of {visible_count} visible)\n'
        f'Mean Longitude: {mean_lon:.1f}°', 
        fontsize=14, y=0.95
    )
    
    # Colors and alpha for planets based on visibility and RANSAC
    colors = {
        'mercury': 'gray',
        'venus': 'yellow',
        'mars': 'red',
        'jupiter': 'orange',
        'saturn': 'gold',
        'uranus': 'cyan',
        'neptune': 'blue'
    }
    
    # Plot 1: Centered RA/DEC
    ax1.set_title('Centered Equatorial Coordinates (RA/DEC)')
    ax1.set_xlabel('Relative Right Ascension (hours)')
    ax1.set_ylabel('Declination (degrees)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-12, 12)
    ax1.set_ylim(-90, 90)
    
    for name, (ra, dec) in centered_equatorial.items():
        # Determine if planet is a RANSAC inlier
        is_inlier = False
        if daily_vis[name] and ransac_inliers is not None and name in visible_planets:
            idx = visible_planets.index(name)
            is_inlier = ransac_inliers[idx]
        
        # Set color and alpha based on visibility and RANSAC
        alpha = 1.0 if daily_vis[name] and is_inlier else 0.3
        color = colors[name] if daily_vis[name] and is_inlier else 'gray'
        
        # Add label with visibility and alignment status
        status = ["not visible"] if not daily_vis[name] else ["visible"]
        if daily_vis[name]:
            if is_inlier:
                status.append("aligned")
            else:
                status.append("outlier")
        label = f"{name.capitalize()} ({', '.join(status)})"
        
        ax1.scatter(ra, dec, c=color, s=100, alpha=alpha, label=label)
        
        # Show coordinates in annotation
        original_ra = equatorial[name][0]
        status_text = ", ".join(status)
        label = f'{name.capitalize()}\n(RA={original_ra:.1f}h, DEC={dec:.1f}°)\n({status_text})'
        ax1.annotate(label, (ra, dec), xytext=(5, 5), textcoords='offset points',
                    color=color, alpha=alpha)
    
    # Plot 2: Centered Ecliptic
    ax2.set_title('Centered Ecliptic Coordinates (Longitude/Latitude)')
    ax2.set_xlabel('Relative Ecliptic Longitude (degrees)')
    ax2.set_ylabel('Ecliptic Latitude (degrees)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-10, 10)
    
    # Plot alignment line for visible planets
    if line_segments:
        for lon_line, lat_line in line_segments:
            ax2.plot(lon_line, lat_line, 'w--', alpha=0.5, label='Best fit line')
    
    for name, (lon, lat) in centered_ecliptic.items():
        # Determine if planet is a RANSAC inlier
        is_inlier = False
        if daily_vis[name] and ransac_inliers is not None and name in visible_planets:
            idx = visible_planets.index(name)
            is_inlier = ransac_inliers[idx]
        
        # Set color and alpha based on visibility and RANSAC
        alpha = 1.0 if daily_vis[name] and is_inlier else 0.3
        color = colors[name] if daily_vis[name] and is_inlier else 'gray'
        
        # Add label with visibility and alignment status
        status = ["not visible"] if not daily_vis[name] else ["visible"]
        if daily_vis[name]:
            if is_inlier:
                status.append("aligned")
            else:
                status.append("outlier")
        label = f"{name.capitalize()} ({', '.join(status)})"
        
        ax2.scatter(lon, lat, c=color, s=100, alpha=alpha, label=label)
        
        # Show coordinates in annotation
        original_lon = ecliptic[name][0]
        status_text = ", ".join(status)
        label = f'{name.capitalize()}\n(λ={original_lon:.1f}°, β={lat:.1f}°)\n({status_text})'
        ax2.annotate(label, (lon, lat), xytext=(5, 5), textcoords='offset points',
                    color=color, alpha=alpha)
    
    # Plot 3: Alignment Score Time Series
    ax3.set_title('Alignment Score Over Time', pad=10)
    
    # Load all scores from cache or calculate if needed
    scores_cache = safe_read_json(SCORES_CACHE_FILE)
    if date.strftime('%Y-%m-%d') in scores_cache:
        day_score = scores_cache[date.strftime('%Y-%m-%d')]
    else:
        # Calculate score for visible planets
        visible_ecliptic = {name: pos for name, pos in centered_ecliptic.items() 
                          if daily_vis[name]}
        
        if len(visible_ecliptic) >= 2:
            day_score, _, _ = calculate_alignment_score(visible_ecliptic)
        else:
            day_score = 0.0
        
        # Save score to cache
        scores_cache[date.strftime('%Y-%m-%d')] = day_score
    
    # Convert dates and scores to lists for plotting, sorted by date
    sorted_items = sorted(scores_cache.items(), key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))
    dates_plot = [datetime.strptime(d, '%Y-%m-%d').date() for d, _ in sorted_items]
    scores_plot = [s for _, s in sorted_items]
    
    # Plot the time series
    ax3.plot(dates_plot, scores_plot, 'w-', alpha=0.5, linewidth=1)
    
    # Calculate y-axis range
    min_score = min(scores_plot)
    max_score = max(scores_plot)
    y_margin = (max_score - min_score) * 0.1  # Add 10% margin
    y_min = max(0, min_score - y_margin)  # Don't go below 0
    y_max = min(100, max_score + y_margin)  # Don't exceed 100
    
    # Highlight current date
    current_date = date.date()
    current_score = scores_cache[date.strftime('%Y-%m-%d')]
    ax3.scatter([current_date], [current_score], color='red', s=100, zorder=5)
    ax3.axvline(current_date, color='red', linestyle='--', alpha=0.5)
    
    # Add score label
    ax3.text(current_date, current_score, f' Score: {current_score:.1f}',
             color='white', fontweight='bold', verticalalignment='bottom')
    
    # Customize the plot
    ax3.grid(True, alpha=0.2)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Alignment Score')
    ax3.set_ylim(y_min, y_max)
    
    # Format x-axis to show dates nicely
    ax3.tick_params(axis='x', rotation=45)
    
    # Add legends with fixed position
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Save with fixed size
    os.makedirs('alignment_viz', exist_ok=True)
    output_file = f'alignment_viz/alignment_{date.strftime("%Y%m%d_%H%M")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches=None)
    plt.close()
    
    return output_file, day_score

def process_visibility_chunk(dates):
    """
    Process a chunk of dates for visibility calculation.
    """
    # Initialize worker if not already initialized
    if ts is None:
        init_worker()
        
    results = {}
    for date in dates:
        try:
            date_str = date.strftime('%Y-%m-%d')
            sunrise, sunset = get_sunrise_sunset(date)
            
            if sunrise is None or sunset is None:
                # If we can't get sunrise/sunset times, mark all planets as not visible
                results[date_str] = {name: False for name in planets}
                continue
            
            # Add 15 minutes to sunset and subtract 15 minutes from sunrise
            sunset_time = sunset.utc_datetime() + timedelta(minutes=15)
            sunrise_time = sunrise.utc_datetime() - timedelta(minutes=15)
            
            # If sunset_time is after midnight, it belongs to the next day
            if sunset_time.date() > date.date():
                sunset_time = sunset_time.replace(
                    year=date.year, month=date.month, day=date.day, hour=23, minute=59
                )
            
            # If sunrise_time is before midnight, it belongs to the previous day
            if sunrise_time.date() < date.date():
                sunrise_time = sunrise_time.replace(
                    year=date.year, month=date.month, day=date.day, hour=0, minute=0
                )
            
            # Initialize visibility dictionary
            daily_visibility = {name: False for name in planets}
            
            # Check visibility at each hour
            current_time = sunset_time
            while current_time <= sunrise_time:
                # Convert to Skyfield time
                time = ts.from_datetime(current_time)
                
                # Check visibility at this hour
                visibility = check_visibility(time)
                
                # Update visibility - if a planet is visible at any hour, mark it as visible
                for name, (is_visible, *_) in visibility.items():
                    daily_visibility[name] = daily_visibility[name] or is_visible
                
                # Move to next hour
                current_time += timedelta(hours=1)
            
            results[date_str] = daily_visibility
            
        except Exception as e:
            print(f"Error processing date {date}: {str(e)}")
            continue
            
    return results

def process_alignment_chunk(dates):
    """
    Process a chunk of dates for alignment score calculation.
    """
    # Initialize worker if not already initialized
    if ts is None:
        init_worker()
        
    results = {}
    for date in dates:
        try:
            # Get daily visibility
            date_str = date.strftime('%Y-%m-%d')
            daily_vis = visibility_cache[date_str]
            
            # Calculate positions
            time = ts.from_datetime(date)
            visibility, equatorial, ecliptic = calculate_positions(time)
            
            # Calculate mean longitude considering circular nature
            lons = np.array([lon for lon, _ in ecliptic.values()])
            mean_lon = np.degrees(np.arctan2(
                np.mean(np.sin(np.radians(lons))),
                np.mean(np.cos(np.radians(lons)))
            ))
            
            # Transform coordinates to be centered
            centered_ecliptic = {}
            for name in planets:
                lon, lat = ecliptic[name]
                # Center longitude around mean
                centered_lon = ((lon - mean_lon + 180) % 360) - 180
                centered_ecliptic[name] = (centered_lon, lat)
            
            # Calculate alignment score using only visible planets
            visible_ecliptic = {name: pos for name, pos in centered_ecliptic.items() 
                              if daily_vis[name]}
            
            score = 0.0
            if len(visible_ecliptic) >= 2:
                score, _, _ = calculate_alignment_score(visible_ecliptic)
            
            results[date_str] = score
            
        except Exception as e:
            print(f"\nError processing date {date}: {str(e)}")
            continue
    return results

def process_visualization_chunk(dates):
    """
    Process a chunk of dates for visualization generation.
    """
    # Initialize worker if not already initialized
    if ts is None:
        init_worker()
        
    for date in dates:
        try:
            output_file, _ = create_visualizations(date)
        except Exception as e:
            print(f"\nError processing date {date}: {str(e)}")
            continue

def main():
    """
    Generate visualizations for 2010-2030 using parallel processing.
    Split into three steps:
    1. Collect visibility data using parallel processing
    2. Calculate alignment scores
    3. Generate visualizations
    """
    print("Step 1: Collecting visibility data for 2010-2030...")
    
    # Initialize main process
    init_worker()
    
    # Generate dates for 2010-2030
    start_date = datetime(2010, 1, 1, tzinfo=utc)
    end_date = datetime(2030, 12, 31, tzinfo=utc)
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    # Split dates into chunks for parallel processing
    num_cores = mp.cpu_count()
    chunk_size = len(dates) // num_cores + 1
    date_chunks = [dates[i:i + chunk_size] for i in range(0, len(dates), chunk_size)]
    
    print(f"Processing {len(dates)} dates using {num_cores} CPU cores...")
    
    # Clear existing visibility cache file
    if os.path.exists(VISIBILITY_CACHE_FILE):
        os.remove(VISIBILITY_CACHE_FILE)
    
    # Step 1: Process visibility data in parallel
    with mp.Pool(num_cores, initializer=init_worker) as pool:
        # Process chunks with progress bar
        with tqdm(total=len(dates), desc="Collecting visibility data") as pbar:
            for chunk_results in pool.imap_unordered(process_visibility_chunk, date_chunks):
                # Save results to cache
                for date_str, daily_vis in chunk_results.items():
                    visibility_cache[date_str] = daily_vis
                    # Convert to JSON serializable format
                    serializable_data = {
                        'date': date_str,
                        'visibility': {k: bool(v) for k, v in daily_vis.items()}
                    }
                    # Append to cache file
                    with open(VISIBILITY_CACHE_FILE, 'a') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        try:
                            json.dump(serializable_data, f)
                            f.write('\n')
                            f.flush()
                            os.fsync(f.fileno())
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                pbar.update(len(chunk_results))
    
    print("\nStep 2: Calculating alignment scores...")
    
    # Load visibility data from cache file
    print("Loading visibility data from cache...")
    visibility_cache.clear()
    visibility_cache.update(load_visibility_cache())
    print(f"Loaded visibility data for {len(visibility_cache)} dates")
    
    # Initialize scores cache
    scores_cache = {}
    
    # Step 2: Calculate alignment scores in parallel
    with mp.Pool(num_cores, initializer=init_worker) as pool:
        with tqdm(total=len(dates), desc="Calculating alignment scores") as pbar:
            for chunk_results in pool.imap_unordered(process_alignment_chunk, date_chunks):
                # Update scores cache
                scores_cache.update(chunk_results)
                pbar.update(len(chunk_results))
    
    # Save all scores to cache file
    print("\nSaving alignment scores to cache file...")
    safe_write_json(scores_cache, SCORES_CACHE_FILE)
    
    print("\nStep 3: Generating visualizations...")
    
    # Step 3: Generate visualizations in parallel
    with mp.Pool(num_cores, initializer=init_worker) as pool:
        with tqdm(total=len(dates), desc="Generating visualizations") as pbar:
            for _ in pool.imap_unordered(process_visualization_chunk, date_chunks):
                pbar.update(chunk_size)
    
    # After processing all dates, create a video
    print("\nCreating video from generated images...")
    from make_video import main as make_video
    make_video()

if __name__ == "__main__":
    main() 