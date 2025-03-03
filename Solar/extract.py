import re

def extract_temperature(row):
    """Extracts temperature from a NOAA ASOS row."""
    match = re.search(r'(?<!\d)(-?\d{2,3})00(?!\d)', row)  # Match temperatures like 2300, -500
    if match:
        return int(match.group(1)) / 100.0  # Convert to decimal
    return None

def extract_visibility(row):
    """Extracts visibility (in statute miles) from a NOAA ASOS row."""
    match = re.search(r'(\d+\.?\d*)SM', row)  # Match values like 10.0SM
    if match:
        return float(match.group(1))  # Convert to float
    return None

def extract_humidity(row):
    """Extracts humidity percentage from a NOAA ASOS row."""
    match = re.findall(r'\b(\d{1,2})\b', row)  # Find all standalone 1-2 digit numbers
    for value in match:
        value = int(value)
        if 0 < value <= 100:  # Valid humidity range
            return value
    return None