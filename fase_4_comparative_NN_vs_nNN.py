from Sim_Modules import json_parser
import simpy
import random
import logging
from datetime import datetime
from Sim_Modules import Packet
import pandas as pd
import numpy as np
import Constants

dateTime = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
logging.basicConfig(filename='logs/log_fase_4_' + dateTime + '.log', level=logging.DEBUG)


def count_time(env):
    while True:
        yield env.timeout(SIM_TIME / 1000)
        logging.debug((env.now / SIM_TIME) * 100)


def re_route_packets(pipe, base_stations):
    while True:
        packet = yield pipe.get()
        for base_station in base_stations:
            base_station.send_message_no_channel_error(packet)


def send_messages(env, pipe):
    chunksize = 10 ** 5
    for df in pd.read_csv('data/packet_data.csv', skiprows=0, chunksize=chunksize):
        for packet in df.values:
            next_packet = np.array(packet).tolist()
            if env.now == next_packet[0]:
                packet = Packet.Packet(int(next_packet[0]), next_packet[5], int(next_packet[4]), int(next_packet[3]),
                                       int(next_packet[1]), None, env.now)
                pipe.put(packet)
            elif env.now < next_packet[0]:
                yield env.timeout(next_packet[0] - env.now)
            elif env.now > next_packet[0]:
                raise Exception("Paquetes desordenados")


def print_results(base_station):
    statistics = base_station.get_slices_statistics()
    for slice_id in base_station.slice_ids:
        logging.debug(statistics[slice_id])


RANDOM_SEED = 42
SIM_TIME = Constants.DAY

random.seed(RANDOM_SEED)
env = simpy.Environment()

base_station_ia = json_parser.load_base_station_ia_enabled(env)
base_station_no_ia = json_parser.load_base_station_no_channel_error(env)

pipe_packets = simpy.Store(env)

packets_from_csv = True

if packets_from_csv:
    env.process(send_messages(env, pipe_packets))
else:
    user_groups, baseStations, scenario = json_parser.load_jsons(env, False, False, pipe_packets)

env.process(re_route_packets(pipe_packets, [base_station_ia, base_station_no_ia]))
env.process(count_time(env))
env.run(until=SIM_TIME)

logging.debug("Printing IA result:")
print_results(base_station_ia)
logging.debug("Printing NO IA result:")
print_results(base_station_no_ia)
