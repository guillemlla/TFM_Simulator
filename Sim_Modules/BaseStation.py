import simpy
from scipy.spatial import distance
from Sim_Modules.PacketResult import PacketResult
import random
import numpy
from utils import CSVHelper
import Constants
import math
import torch
import logging


class Slice(object):
    def __init__(self, slice_id, env, capacity, pps, dps, time_req_mils):
        self.slice_id = slice_id
        self.seconds_per_packet = 1 / pps
        self.seconds_per_data = 1 / dps
        self.env = env
        self.capacity = capacity
        self.pipe = simpy.Store(self.env, self.capacity)
        self.packets_received = 0
        self.packets_time_error = 0
        self.packets_queue_full = 0
        self.packets_inserted_queue = 0
        self.time_req_mils = time_req_mils
        env.process(self.run_slice())

    def put(self, packet):
        self.packets_received += 1
        if not self.pipe:
            raise RuntimeError('There are no output pipes.')
        if self.pipe.items.__len__() == self.capacity:
            # print('packet lost')
            self.packets_queue_full += 1
            return PacketResult.QUEUE_FULL
        else:
            self.pipe.put(packet)
            self.packets_inserted_queue += 1
            return PacketResult.PACKET_OK

    def run_slice(self):
        while True:
            packet = yield self.pipe.get()
            if self.env.now - packet.creation_time > Constants.MILLISECOND * self.time_req_mils:
                # Discarted because time
                self.packets_time_error += 1
            yield self.env.timeout(self.calculate_timeout(packet))

    def calculate_timeout(self, packet):
        return self.seconds_per_packet * Constants.SECOND  # + self.seconds_per_data * packet.size

    def print_result(self):
        return '        {Slice Id: ' + str(self.slice_id) + ' Packets processed: ' + str(self.packets_inserted_queue -
                                                                                         self.packets_time_error) + '}'


class BaseStation:
    def __init__(self, env, channel_losses, active, predictor, adapter, base_station_json, ia_active=False,
                 pipe_messages=None):
        self.coordinates = []
        self.range = 0
        self.pps = 0
        self.dps = 0
        self.queue_size = 0
        self.slice_ids = []
        self.mappings = None
        self.predictor = None
        self.adapter = None
        self.id = 0
        self.latency_requirement_slices_ids = None
        self.__dict__ = base_station_json
        self.env = env
        self.channel_losses = channel_losses
        self.slices = {}
        self.packet_counter = numpy.zeros(len(self.slice_ids))
        self.active = active
        self.ia_active = ia_active
        self.predictor = predictor
        self.adapter = adapter
        self.pipe_messages = pipe_messages
        req_milis_dict = {
            True: 0.5,
            False: 5
        }
        if self.active:
            for slice_id in self.slice_ids:
                self.slices[slice_id] = Slice(slice_id, env, int(self.queue_size / len(self.slice_ids)),
                                              self.pps / self.slice_ids.__len__(),
                                              self.dps / self.slice_ids.__len__(),
                                              req_milis_dict[slice_id in self.latency_requirement_slices_ids])
        if ia_active:
            self.predictor.load_model()
            self.adapter.load_model()
            env.process(self.run_ia())

    def run_ia(self):
        while True:
            current_timestamp = self.env.now
            yield self.env.timeout(Constants.MINUTE * 20)
            time_day = current_timestamp % Constants.DAY
            time_sin = float(math.sin(2 * math.pi * time_day / Constants.DAY))
            time_cos = float(math.cos(2 * math.pi * time_day / Constants.DAY))
            self.packet_counter = self.packet_counter / self.packet_counter.sum(0)
            self.packet_counter = self.packet_counter.astype(float)
            logging.debug("PAQ PER SIM: " + str(self.packet_counter))
            self.packet_counter = numpy.insert(self.packet_counter, 0, time_cos, axis=0)
            self.packet_counter = numpy.insert(self.packet_counter, 0, time_sin, axis=0)

            output_predictor = self.predictor.network(torch.from_numpy(self.packet_counter).float())
            logging.debug("OUTPUT")
            output_adapter_queue = self.adapter.network_queue(output_predictor)
            out_adapter_pps = self.adapter.network_pps(output_predictor)
            self.adapt_parameters(output_adapter_queue.detach().numpy(), out_adapter_pps.detach().numpy())
            self.packet_counter = numpy.zeros(len(self.slice_ids))

    def calculate_channel_losses(self, coordinates, bs_range):
        dist = distance.euclidean(self.coordinates, coordinates) / bs_range
        if dist >= self.channel_losses[0][0]:
            for distance_error in self.channel_losses:
                if dist <= distance_error[0]:
                    return distance_error[1]
            return 1
        else:
            return 0

    def send_message_with_channel_error(self, packet):
        try:
            packet_error = self.calculate_channel_losses(packet.coordinates_from, self.range)
        except:
            print("ERROR")

        if self.ia_active:
            self.packet_counter[self.slice_ids.index(self.mappings[str(packet.type)])] += 1

        if random.uniform(0, 1) > packet_error:
            if self.active:
                return self.slices[self.mappings[str(packet.type)]].put(packet)
            elif self.pipe_messages is not None:
                self.pipe_messages.put(packet)
            else:
                CSVHelper.write_packet(packet, self.mappings[str(packet.type)])

        else:
            return PacketResult.CHANNEL_ERROR

    def adapt_parameters(self, output_adapter_queue, output_adapter_pps):
        if len(output_adapter_queue) != len(self.slice_ids) or len(output_adapter_pps) != len(self.slice_ids):
            raise Exception("Wrong number of slices in the output of adapter")

        for i in range(0, len(self.slice_ids)):
            try:
                if output_adapter_queue[i] < 0 or output_adapter_pps[i] < 0:
                    raise Exception(
                        "Error con los parametros pps o capacity: pps: " + str(output_adapter_pps[i]) + " cap: " + str(
                            output_adapter_queue[i]))

                self.slices[self.slice_ids[i]].seconds_per_packet = (1 / (output_adapter_pps[i] * self.pps))
                self.slices[self.slice_ids[i]].capacity = output_adapter_queue[i] * self.queue_size

                logging.debug("Parametros slice " + str(self.slice_ids[i]) + " pps: " + str(
                    output_adapter_pps[i]) + " cap: " + str(output_adapter_queue[i]))
            except:
                raise Exception("Error con los putos indices i: " + str(i) + " self.slice_ids: " + str(self.slice_ids))

    def send_message_no_channel_error(self, packet):
        if self.ia_active:
            self.packet_counter[self.slice_ids.index(self.mappings[str(packet.type)])] += 1

        return self.slices[self.mappings[str(packet.type)]].put(packet)

    def get_slices_statistics(self):
        statistics = {}
        for slice_id in self.slice_ids:
            statistics[slice_id] = {
                'packets_received': self.slices[slice_id].packets_received,
                'packets_time_error': self.slices[slice_id].packets_time_error,
                'packets_queue_full': self.slices[slice_id].packets_queue_full,
                'packets_inserted_queue': self.slices[slice_id].packets_inserted_queue
            }
        return statistics
