import numpy as np
import pandas as pd
import datetime
import math
from math import cos, asin, sin, sqrt, atan2, pi, degrees, radians, floor

def solar_intensity(lat=40.713, lng=-74.006, date_time=None):
    if date_time is None:
        date_time = datetime.datetime.now()
    
    day_of_year = date_time.timetuple().tm_yday
    declination_angle = degrees(asin(sin(radians(23.45)) * sin(2 * pi/365 * (day_of_year - 81))))

    # Equation of Time (EoT) correction
    e_o_t = 9.87 * sin(2 * 2 * pi / 365 * (day_of_year - 81)) - \
            7.53 * cos(2 * pi / 365 * (day_of_year - 81)) - \
            1.5 * sin(2 * pi / 365 * (day_of_year - 81))

    # Local Standard Time Meridian
    gmt_diff = floor(lng / 15)
    lstm = 15.0 * gmt_diff
    tc = 4.0 * (lng - lstm) + e_o_t  # Time correction factor (minutes)
    
    # Convert to Local Solar Time
    local_solar_time = date_time + datetime.timedelta(minutes=tc)
    lst_dec = local_solar_time.hour + local_solar_time.minute / 60.0 + local_solar_time.second / 3600.0

    hour_angle = 15.0 * (lst_dec - 12.0)

    # Solar elevation angle
    elevation_angle = degrees(asin(sin(radians(declination_angle)) * sin(radians(lat)) +
                                   cos(radians(declination_angle)) * cos(radians(lat)) * cos(radians(hour_angle))))

    # Zenith angle for airmass calculation
    rad_from_vert = pi / 2 - radians(elevation_angle)

    if rad_from_vert < pi / 2:
        air_mass = 1 / (cos(rad_from_vert) + 0.50572 * (96.07995 - degrees(rad_from_vert)) ** (-1.6364))
    else:
        air_mass = None

    # Solar constant adjusted for Earth's orbit
    e_0 = 1367 * (1 + 0.033 * cos(2 * pi * (day_of_year - 3) / 365))

    if air_mass:
        intensity = cos(rad_from_vert) * e_0 * 0.7 ** (air_mass ** 0.678)
        return max(intensity, 0)  # Ensure non-negative values
    else:
        return 0

def integrate(intensity_list):
    integral = 0
    for index in range(len(intensity_list) - 1):
        delta_t = float((intensity_list[index + 1]['t'] - intensity_list[index]['t']) / 3600.0)  # Hours
        avg_intensity = (intensity_list[index + 1]['intensity'] + intensity_list[index]['intensity']) / 2.0
        integral += delta_t * avg_intensity
    return integral / 1000.0  # Convert to kWh/m²

# Generate test data for a full day
lat, lng = 40.713, -74.006
date = datetime.datetime(2025, 3, 3)

test_list = []
for hour in range(0, 24):
    time_point = date + datetime.timedelta(hours=hour)
    intensity = solar_intensity(lat, lng, time_point)
    test_list.append({"t": hour * 3600, "intensity": intensity})

# Compute total solar energy received
solar_energy = integrate(test_list)
print(f"Total Solar Energy: {solar_energy:.4f} kWh/m²")
