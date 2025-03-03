import requests
from xml.etree import cElementTree as ElementTree

ALL_STATIONS_URL = "https://www.ndbc.noaa.gov/activestations.xml"

def get_all_stations_xml():
    """
    Bulk fetch all the data from NWS' stations endpoint.
    """
    r = requests.get(ALL_STATIONS_URL)
    r.raise_for_status()
    return r.content

def parse_all_stations_xml(xml_raw):
    """
    Parse stations from the NWS all stations XML file.

    :param bytes xml_raw: xml file content straight from NWS
    :rtype: Iterable[ElementTree.Element]
    """
    root = ElementTree.fromstring(xml_raw)
    for station in root.findall("station"):
        yield station

def extract_lat_lon(station_tree):
    """
    Extracts a dict from a station ETree with station_id, lat, lon.
    
    :param ElementTree.Element station_tree: Station object
    :rtype: dict
    """ 
    station_id = station_tree.get('id')
    latitude = station_tree.get('lat')
    longitude = station_tree.get('lon')

    if station_id is None or latitude is None or longitude is None:
        return None  # Skip if any required field is missing

    return {
        "station_id": station_id,
        "lat_lon": (float(latitude), float(longitude))
    }

if __name__ == "__main__":
    stations = parse_all_stations_xml(get_all_stations_xml())

    for _ in range(5):
        station_data = extract_lat_lon(next(stations, None))
        if station_data:
            print(station_data)