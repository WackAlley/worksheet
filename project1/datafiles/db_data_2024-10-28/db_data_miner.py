from pyhafas import HafasClient
from pyhafas.profile import DBProfile
import pandas as pd 
from datetime import datetime, date, timedelta
from time import sleep
from os import mkdir



def request_data(start_time):
    #requests data and returns it as pandas data frame
    return pd.DataFrame((stationBoardLeg.__dict__ for stationBoardLeg in client.departures(station='8098105',
         date=start_time, max_trips=1e5)))\
        .astype({'id': 'string', 'name' : 'string', 'station' : 'string',  'direction' : 'string', 'platform' : 'string'})\
        .set_index('id')


def seconds_until_midnight():
    #returns number of seconds remainining until midnight (00:00 O'Clock = 12 AM)
    return ((datetime.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0) - datetime.now()).seconds


intervall_minitues = 10

client = HafasClient(DBProfile())
first_day = True

while True:
    # if first day start immediately, else wait until midnight
    start_time_today = datetime.now()
    current_dir = f'--will_be_overwritten_with_date--'
    if first_day:
        current_dir = f'{start_time_today}'
    else:
        sleep(seconds_until_midnight() + 1)
        start_time_today = datetime.now() # start time changed a lot mabe, because of long sleep, update start time
        current_dir = f'{date.today()}'
        
    mkdir(current_dir)
    mkdir(f'{current_dir}/request_results')
    updated_data = request_data(start_time_today)
    updated_data = updated_data[updated_data['platform'].str.isnumeric()]
    updated_data.to_feather(f"{current_dir}/initial_data_t={start_time_today}.feather", compression_level=6) # changed was before: updated_data.to_feather(f"{current_dir}/initial_data_t={start_time_today}.feather", compression_level=6)

    # collect data every intervall_minitues minutes
    for _ in range(int(seconds_until_midnight() / 60 / intervall_minitues)-1): # one safety intervall period to be extra sure to finish before midnight in addition int(...) rounds off
        #request:
        request_time =  datetime.now()
        new_data = request_data(start_time_today)
        #save request (backup):
        new_data.to_feather(f"{current_dir}/request_results/request_at_t+{(request_time - start_time_today).seconds}s.feather", compression_level=6)
        #update:
        updated_data.update(new_data[new_data['platform'].str.isnumeric()])

        #wait rest of intervall
        time_to_wait = timedelta(minutes = intervall_minitues) - (datetime.now() - request_time)
        sleep(time_to_wait.seconds)

    #save result
    updated_data = updated_data.astype({'platform' : 'int16'}) #all platforms are integegers (since they were filtered by .str.isnumeric())
    updated_data.to_feather(f"{current_dir}/result.feather", compression_level=6)
    updated_data.to_csv(f"{current_dir}/result.csv")

    first_day = False

