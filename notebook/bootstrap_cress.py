# used to get all needed data
from bisect import bisect_left
import os
import progressbar
import pandas
import requests


def get_data_from_api(cycle, path):
    from config import CRESS_API_TOKEN
    headers = {
        'Authorization': 'Token %s' % CRESS_API_TOKEN,
        'Content-Type':'application/json'
    }
    url = 'https://cress.space/v1/{0}/?cycle={1}'.format(path, cycle)
    s = requests.Session()
    response = s.get(url, headers=headers)
    resp_json = response.json()
    while resp_json.get('next'):
        yield resp_json.get('results')
        response = s.get(resp_json.get('next'), headers=headers)
        resp_json = response.json()
    yield resp_json.get('results')


def sensors(cycle):
    filename = 'data/cycle_{}.csv'.format(cycle)
    if not os.path.exists(filename):
        print("download sensor data")
        with open(filename, 'w') as fp:
            keys = ['sensor_type', 'value_type', 'position', 'unit', 'value', 'created', 'seconds_from_cycle_start']
            fp.write('"' + '","'.join(keys) + '"')
            fp.write('\n')
            bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
            for index, chunk in enumerate(get_data_from_api(cycle, 'sensor')):
                for item in chunk:
                    fp.write('"' + '","'.join([str(item[i]) for i in keys]) + '"')
                    fp.write('\n')
                bar.update(index)


def photos(cycle):
    filename = 'data/photo_cycle_{}.csv'.format(cycle)
    if not os.path.exists(filename):
        print("download photo list")
        with open(filename, 'w') as fp:
            keys = ['id', 'photo', 'created', 'seconds_from_cycle_start']
            fp.write('"' + '","'.join(keys) + '"')
            fp.write('\n')
            bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
            for index, chunk in enumerate(get_data_from_api(cycle, 'photo')):
                for item in chunk:
                    fp.write('"' + '","'.join([str(item[i]) for i in keys]) + '"')
                    fp.write('\n')
                bar.update(index)


def return_closest(lst, number):
    pos = bisect_left(lst, number)
    if pos == 0:
        return lst[0]
    if pos == len(lst):
        return lst[-1]
    before = lst[pos - 1]
    after = lst[pos]
    if after - number < number - before:
        return after
    return before


def enrich_photo_csv(cycle):
    filename = "data/photo_cycle_{}_enriched.csv".format(cycle)
    if os.path.exists(filename):
        return

    df_sensors = pandas.read_csv("data/cycle_{}.csv".format(cycle))
    df_water = df_sensors[(df_sensors['sensor_type']=='FC28') & (df_sensors['value_type']=='watermark')]

    df_photos = pandas.read_csv("data/photo_cycle_{}.csv".format(cycle))
    df_photos['watermark'] = 0
    df_photos['watermark_seconds_from_start'] = 0

    # correlate - find for every photo the next water sensor value
    bar = progressbar.ProgressBar(max_value=len(df_photos))
    water_list_seconds = sorted(df_water['seconds_from_cycle_start'].values)
    for idx, photo_ds in bar(df_photos.iterrows()):
        a = return_closest(water_list_seconds, photo_ds['seconds_from_cycle_start'])
        df_photos.loc[idx, 'watermark'] = (df_water[df_water['seconds_from_cycle_start'] == a]['value']).values[0]
        df_photos.loc[idx, 'watermark_seconds_from_start'] = a

    df_photos[['watermark', 'watermark_seconds_from_start']] = df_photos[['watermark', 'watermark_seconds_from_start']].astype(int)
    df_photos.to_csv(filename)


def get_csv_files(cycle):
    # early out if nothing needs to be done
    filename = "data/photo_cycle_{}_enriched.csv".format(cycle)
    if os.path.exists(filename):
        return
    # download all from api (needs api key!)
    sensors(cycle)
    photos(cycle)
    enrich_photo_csv(cycle)
