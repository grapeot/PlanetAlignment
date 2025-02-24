# Planetary Alignment Visualization

English | [中文](README_zh.md)

A Python project to visualize and analyze planetary alignments, with a special focus on finding and visualizing the best planetary alignments in 2025 from Beijing's perspective.

## Motivation

Planetary alignments are rare astronomical events where multiple planets appear to line up in the night sky. While true geometric alignment in space is extremely rare, the apparent alignment from Earth's perspective creates spectacular astronomical views and has captured human imagination throughout history.

This project aims to:
- Find the best planetary alignments between 2010 and 2030
- Visualize the alignments using both equatorial and ecliptic coordinate systems
- Generate videos showing the planetary movements around alignment dates
- Provide quantitative measures of alignment quality

## How It Works

### Core Algorithm

The project uses several key algorithms to analyze and visualize planetary alignments:

1. **Position Calculation**
   - Uses the Skyfield library for high-precision planetary position calculations
   - Converts heliocentric coordinates to topocentric coordinates for Beijing's perspective
   - Accounts for light-time correction and aberration effects

2. **Alignment Score Calculation**
   - Employs RANSAC (Random Sample Consensus) to find the best-fit line through visible planets
   - Scores alignments based on three components:
     - Number of aligned planets (40% weight)
     - Quality of alignment (perpendicular distance to best-fit line, 30% weight)
     - Concentration around the ecliptic (latitude dispersion, 30% weight)

3. **Visibility Analysis**
   - Calculates sunrise and sunset times for Beijing
   - Determines planet visibility during night hours
   - Caches visibility data for efficient processing

4. **Visualization**
   - Creates dual-projection visualizations:
     - Equatorial coordinates (RA/Dec)
     - Ecliptic coordinates (longitude/latitude)
   - Generates time series of alignment scores
   - Produces video animations of planetary movements

### Example Visualization

![Example of planetary alignment visualization](example.png)

The visualization includes three panels:
- Top: Equatorial coordinate view (RA/Dec) showing planet positions relative to the celestial equator
- Middle: Ecliptic coordinate view showing planet positions relative to the ecliptic plane, with the best-fit alignment line
- Bottom: Time series of alignment scores, with the current date highlighted

## Setup

### Requirements

The project requires Python 3.8+ and the following main dependencies:
- skyfield >= 1.46.0 (astronomical calculations)
- numpy >= 1.24.0 (numerical computations)
- matplotlib >= 3.8.0 (visualization)
- moviepy == 1.0.3 (video generation)
- scikit-learn >= 1.3.0 (RANSAC algorithm)
- tqdm >= 4.66.0 (progress bars)

### Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download astronomical data:
   The project requires the DE421 ephemeris file from JPL for planetary calculations. When you run the script for the first time, Skyfield will automatically download the required `de421.bsp` file (about 17MB) to your local cache directory:
   - Linux/Mac: `~/.cache/skyfield/`
   - Windows: `C:\Users\<username>\AppData\Local\skyfield\`

   Alternatively, you can manually download it from:
   ```
   https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de421.bsp
   ```
   and place it in the project directory.

### Usage

1. Run the main script to analyze alignments and generate visualizations:
   ```bash
   python planet_alignment.py
   ```

2. The script will:
   - Calculate planetary positions and visibility
   - Generate alignment scores
   - Create visualization images in the `alignment_viz` directory
   - Produce a video showing the planetary movements

### Output

- Individual frame images in `alignment_viz/`
- Alignment score cache in `alignment_scores_cache.json`
- Visibility data cache in `planet_visibility_cache.jsonl`
- Final video showing planetary movements

## Notes

- All calculations are performed from Beijing's perspective (39.9042°N, 116.4074°E)
- The project uses parallel processing to speed up calculations
- Caching is implemented to avoid redundant calculations
- The visualization includes both visible and non-visible planets for completeness 