from Sim_Modules import json_parser
import simpy
import random
import logging
from datetime import datetime
import Constants

dateTime = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
logging.basicConfig(filename='logs/log_fase_0_' + dateTime + '.log', level=logging.DEBUG)

RANDOM_SEED = 42
SIM_TIME = Constants.MINUTE * 20

random.seed(RANDOM_SEED)
env = simpy.Environment()


def count_time(env):
    while True:
        yield env.timeout(SIM_TIME / 1000)
        logging.debug((env.now / SIM_TIME) * 100)


env.process(count_time(env))

pipe_packets = simpy.Store(env)

user_groups, baseStations, scenario = json_parser.load_jsons(env, True, False, None)

env.run(until=SIM_TIME)


def print_results(base_station):
    statistics = base_station.get_slices_statistics()
    for slice_id in base_station.slice_ids:
        print(statistics[slice_id])
        logging.info(statistics[slice_id])


for baseStation in baseStations:
    print_results(baseStation)

dateTime = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
logging.info('END AT ' + dateTime)
