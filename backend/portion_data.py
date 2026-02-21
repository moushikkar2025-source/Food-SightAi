"""
FoodSight AI - Portion Size Database
=====================================

Standard portion sizes and multipliers for nutrition calculation.
"""

# Standard multipliers for portion sizes
PORTION_MULTIPLIERS = {
    "small": 0.5,
    "medium": 1.0,  # Standard serving
    "large": 1.5,
    "extra_large": 2.0
}

# Standard size descriptions (can be customized per dish type if needed)
PORTION_SIZES = {
    "small": "Small (Half serving)",
    "medium": "Medium (Standard Serving)",
    "large": "Large (1.5x Standard)",
    "extra_large": "Family Pack (2x)"
}

def get_portion_multiplier(size):
    """Get multiplier for a portion size."""
    return PORTION_MULTIPLIERS.get(size, 1.0)
