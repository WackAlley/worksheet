import datetime
from typing import List

# Part 1
from pyhafas import HafasClient
from pyhafas.profile import DBProfile
from pyhafas.types.fptf import Leg

client = HafasClient(DBProfile())

# Part 2
#locations = client.locations("Frankfurt")
#best_found_location = locations[0]
#print(best_found_location)  # <class 'pyhafas.types.fptf.Station'>({'id': '008005556', 'name': 'Siegburg/Bonn', 'latitude': 50.794051, 'longitude': 7.202616})

# Part 3
departures: List[Leg] = client.departures(
    station=8000105,
    date=datetime.datetime.now(),
    max_trips=5,
    products={
        'long_distance_express': True,
        'regional_express': False,
        'regional': False,
        'suburban': False,
        'bus': False,
        'ferry': False,
        'subway': False,
        'tram': False,
        'taxi': False
    }
)
