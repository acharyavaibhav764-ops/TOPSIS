"""
data_feeds/entity_resolution.py — Centralized entity normalization.
Maps various country aliases and ISO-3166 alpha-3 codes to their standard
pipeline names to ensure perfect aggregation before scoring.
"""

ISO3_TO_NAME = {
    "USA": "United States",
    "GBR": "United Kingdom",
    "SGP": "Singapore",
    "AUS": "Australia",
    "CHE": "Switzerland",
    "HKG": "Hong Kong",
    "JPN": "Japan",
    "IND": "India",
    "ARE": "United Arab Emirates",
    "CAN": "Canada",
    "FRA": "France",
    "DEU": "Germany",
    "NLD": "Netherlands",
    "KOR": "South Korea",
    "BRA": "Brazil",
}

ALIASES = {
    "UAE": "United Arab Emirates",
    "UK": "United Kingdom",
    "US": "United States",
}

def normalize_entity_name(identifier: str) -> str:
    """
    Normalize a country identifier (ISO-3 or alias) to its standard pipeline name.
    """
    identifier = identifier.strip()
    
    if identifier in ISO3_TO_NAME:
        return ISO3_TO_NAME[identifier]
        
    if identifier in ALIASES:
        return ALIASES[identifier]
        
    return identifier
