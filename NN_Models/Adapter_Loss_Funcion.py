import numpy as np
np.random.seed(123)
from Sim_Modules.BaseStation import BaseStation
import json
import simpy
import random
from Sim_Modules import Packet
import logging
import pandas as pd
import torch.nn.functional as F


def create_target(output, weights):
    target_np = output.clone().detach().numpy()[0]
    target = output.clone()

    for i in range(0, len(weights)):
        if target_np[i] == 0:
            suma = np.sum([target_np[j] for j in range(0, int(len(weights)))])
            if suma == 0:
                target[0][i] = 1 / len(weights)
            else:
                target[0][i] = suma / ((len(weights)) - 1)
        else:
            target[0][i] = target[0][i] + (1 - target[0][i]) * weights[i]

    return target


def loss_function(adaptor_output_queue, adaptor_output_pps, packets_filename):

    RANDOM_SEED = 42
    logging.debug("Begin simulator")
    filename = packets_filename[0]

    packets = pd.read_csv('data/packet_data/' + filename, skiprows=0)

    SIM_TIME = int((packets.values[-1][0] - packets.values[0][0]).item())
    FROM = int(packets.values[0][0].item())

    random.seed(RANDOM_SEED)
    env = simpy.Environment()

    with open('jsons/baseStations.json', 'r') as f:
        base_stations_obj = json.load(f)
        for baseStation in base_stations_obj:
            base_station = BaseStation(env, None, True, None, None, baseStation)

    params_queue = adaptor_output_queue.clone().detach().numpy()
    params_pps = adaptor_output_pps.clone().detach().numpy()
    base_station.adapt_parameters(params_queue[0], params_pps[0])

    def send_messages(environment):
        chunk_size = 10 ** 7
        for df in pd.read_csv('data/packet_data/' + filename , skiprows=0, chunksize=chunk_size):
            for packet in df.values:
                next_packet = np.array(packet).tolist()
                if environment.now == next_packet[0] - FROM:
                    packet = Packet.Packet(int(next_packet[0]), next_packet[5], int(next_packet[4]), int(next_packet[3]), int(next_packet[1]), None, environment.now)
                    base_station.send_message_no_channel_error(packet)
                elif environment.now < next_packet[0] - FROM:
                    yield environment.timeout(next_packet[0] - FROM - environment.now)
                elif environment.now > next_packet[0] - FROM:
                    raise Exception("Paquetes desordenados")


    env.process(send_messages(env))

    env.run(until=SIM_TIME)

    results = base_station.get_slices_statistics()

    total_packets_lost_queue = 0
    total_packets_lost_time = 0

    logging.debug("CALCULATING LOSS")

    for result in results.values():
        total_packets_lost_time += result['packets_time_error']
        total_packets_lost_queue += result['packets_queue_full']

    weights_queue = []
    weights_pps = []
    for result in results.values():
        if total_packets_lost_time == 0 or result['packets_received'] == 0:
            weights_pps.append(0)
        else:
            weights_pps.append((result['packets_time_error'] * result['packets_time_error']) /
                               (total_packets_lost_time * result['packets_received']))
        if total_packets_lost_queue == 0 or result['packets_received'] == 0:
            weights_queue.append(0)
        else:
            weights_queue.append((result['packets_queue_full'] * result['packets_queue_full']) / (total_packets_lost_queue * result['packets_received']))

    target_pps = create_target(adaptor_output_pps, weights_pps)
    target_queue = create_target(adaptor_output_queue, weights_queue)

    loss_pps = F.mse_loss(adaptor_output_pps, target_pps)
    loss_queue = F.mse_loss(adaptor_output_queue, target_queue)

    logging.debug("LOSS CALCULATED PPS output: " + str(adaptor_output_pps) + " target: " + str(target_pps))
    logging.debug("LOSS CALCULATED QUEUE output: " + str(adaptor_output_queue) + " target: " + str(target_queue))
    return loss_queue, loss_pps
