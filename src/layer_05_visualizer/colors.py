"""
Color palette and utilities for the visualizer.
Provides stable colors for person IDs.
"""

# A distinct, bright palette for genuine tracks.
PALETTE = [
    "#4FC3F7", # Light Blue
    "#FF8A65", # Deep Orange
    "#81C784", # Light Green
    "#BA68C8", # Purple
    "#FFD54F", # Amber
    "#4DB6AC", # Teal
    "#F06292", # Pink
    "#AED581", # Lime
    "#7986CB", # Indigo
    "#A1887F", # Brown
]

PHANTOM_COLOR = "#9E9E9E" # Grey for negative-id phantoms

def get_person_color_hex(person_id: int) -> str:
    """
    Return a stable hex color for a given person_id.
    Negative IDs are phantoms and get the PHANTOM_COLOR.
    """
    if person_id < 0:
        return PHANTOM_COLOR
    
    # Hash the ID into the palette
    return PALETTE[hash(str(person_id)) % len(PALETTE)]

def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string like '#4FC3F7' to a BGR tuple for cv2."""
    hex_color = hex_color.lstrip('#')
    # Parse RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r) # cv2 uses BGR
