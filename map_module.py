from shapely.geometry import Polygon
from geopy.geocoders import Nominatim

def get_coordinates_from_place(place_name):
    """
    Get latitude and longitude for a given place name.
    Raises ValueError if location not found.
    """
    geolocator = Nominatim(user_agent="agrismart")
    location = geolocator.geocode(place_name)

    if location:
        return location.latitude, location.longitude
    else:
        raise ValueError(f"Location '{place_name}' not found. Try a valid area or city name.")


def get_region_from_place(place_name):
    """
    Returns region/state and country for the given location.
    """
    geolocator = Nominatim(user_agent="agrismart")
    location = geolocator.geocode(place_name, addressdetails=True)

    if location and "address" in location.raw:
        address = location.raw["address"]
        state = address.get("state", "Unknown")
        country = address.get("country", "Unknown")
        district = address.get("county", address.get("state_district", "Unknown"))
        return district, state, country
    else:
        raise ValueError(f"Region not found for '{place_name}'")


def get_land_area(lat, lon):
    """
    Simulates farmland boundary area around a GPS coordinate.
    Returns land area in acres.
    """
    coords = [
        (lon, lat),
        (lon + 0.001, lat),
        (lon + 0.001, lat + 0.001),
        (lon, lat + 0.001)
    ]

    poly = Polygon(coords)
    area_sq_m = poly.area * (111_000**2)  # approx conversion m²
    area_acres = round(area_sq_m / 4046.85642, 2)  # convert to acres

    return area_acres


def get_coordinates_region_area(place_name):
    """
    Combined function returning:
    coordinates, district, state, country, area
    """
    lat, lon = get_coordinates_from_place(place_name)
    district, state, country = get_region_from_place(place_name)
    area = get_land_area(lat, lon)

    return {
        "coordinates": (lat, lon),
        "district": district,
        "state": state,
        "country": country,
        "land_area_acres": area
    }


# Demo usage
if __name__ == "__main__":
    place = input("Enter farm location: ")
    data = get_coordinates_region_area(place)
    print("\n--- Location Details ---")
    print(f"Coordinates  : {data['coordinates']}")
    print(f"District     : {data['district']}")
    print(f"State        : {data['state']}")
    print(f"Country      : {data['country']}")
    print(f"Land Area    : {data['land_area_acres']} acres")
