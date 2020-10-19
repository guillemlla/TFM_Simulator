from Sim_Modules.Scenario import Scenario
from Sim_Modules.UserGroup import UserGroup
from Sim_Modules.BaseStation import BaseStation
from NN_Models.Adapter import Adapter
from NN_Models.Predictor import Predictor
import json

BASE_STATION_FILE = 'jsons/baseStations.json'


def load_jsons(env, base_stations_active, ia_active=False, pipe_messages=None):
    with open('jsons/existing_probabilities.json', 'r') as f:
        users_existing_probabilities = json.load(f)

    with open('jsons/scenario.json', 'r') as f:
        scenario = Scenario(f)

    with open(BASE_STATION_FILE, 'r') as f:
        base_stations_obj = json.load(f)
        base_stations = []
        for baseStation in base_stations_obj:
            base_stations.append(
                BaseStation(env, scenario.channel_losses, base_stations_active, None, None,
                            baseStation, ia_active, pipe_messages))

    with open('jsons/usersGroup.json', 'r') as f:
        user_group_json = json.load(f)
        user_groups = []
        for user_group in user_group_json:
            user_groups.append(UserGroup(user_group, env, scenario.size, base_stations,
                                         users_existing_probabilities[user_group['existing_probabilities']],
                                         scenario.max_concurrent_users))

    scenario.user_groups = user_group
    scenario.base_stations = base_stations

    return user_groups, base_stations, scenario


def load_base_station_no_channel_error(env):
    with open(BASE_STATION_FILE, 'r') as f:
        base_stations_obj = json.load(f)
        for baseStation in base_stations_obj:
            base_station = BaseStation(env, None, True, None, None, baseStation, None, None)
    return base_station


def load_base_station_ia_enabled(env):
    with open(BASE_STATION_FILE, 'r') as f:
        base_stations_obj = json.load(f)
        for baseStation in base_stations_obj:
            base_station = BaseStation(env, None, True, Predictor(4), Adapter(4),  baseStation, True, None)
    return base_station
