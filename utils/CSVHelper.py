import csv
import pandas as pd
import Constants
import math
import numpy
import torch
import pickle
import logging

TIME_INCREMENT = Constants.MINUTE * 20


def create_csv_file():
    with open('data/packet_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "BaseStationId", "SliceId", "UserId", "PacketTypeId", "Size"])


def write_packet(packet, slice_id):
    with open('data/packet_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([packet.timestamp, packet.to_id, slice_id, packet.to_id, packet.type, packet.size])


def generate_predictor_data():
    chunksize = 10 ** 5
    current_timestamp = -1
    slices_values = numpy.zeros(Constants.TOTAL_SLICES)
    dictionary_slicesId = dict(zip(Constants.SLICES, range(Constants.TOTAL_SLICES)))
    data = []
    for df in pd.read_csv("data/packet_data.csv", skiprows=0, chunksize=chunksize):
        if current_timestamp == -1:
            current_timestamp = df.values[0][0]
        for value in df.values:
            if current_timestamp + TIME_INCREMENT < value[0]:
                append_to_predictor_dataset(data, current_timestamp, slices_values)
                current_timestamp = value[0]
                slices_values = numpy.zeros(Constants.TOTAL_SLICES)
            slices_values[dictionary_slicesId[value[2]]] += 1

    append_to_predictor_dataset(data, current_timestamp, slices_values)
    return data, Constants.TOTAL_SLICES

def append_to_predictor_dataset(data, current_timestamp, slices_values):
    time_day = current_timestamp % Constants.DAY
    time_sin = float(math.sin(2 * math.pi * time_day / Constants.DAY))
    time_cos = float(math.cos(2 * math.pi * time_day / Constants.DAY))
    slices_values = slices_values / slices_values.sum(0)
    slices_values = slices_values.astype(float)
    slices_values = numpy.insert(slices_values, 0, time_cos, axis=0)
    slices_values = numpy.insert(slices_values, 0, time_sin, axis=0)
    data.append(slices_values)


def append_to_adaptor_dataset(data, slices_values):

    slices_values = slices_values / slices_values.sum(0)
    slices_values = slices_values.astype(float)
    data.append(slices_values)


def generate_predictor_dataset():
    logging.debug("Generating predictor Dataset")
    data, num_slices = generate_predictor_data()

    ids = data[0:len(data) - 1]
    labels = data[1:len(data)]

    labels = list(map(lambda x: x[-num_slices:], labels))

    logging.debug("Saving predictor Dataset")

    file = {'dataset': Dataset(ids, labels), 'num_slices': num_slices}

    with open('predictor.dataset', 'wb') as predictor_dataset_file:
        pickle.dump(file, predictor_dataset_file)

    logging.debug("Predictor Dataset Saved")


def generate_adapter_dataset():
    logging.debug("Generating adapter Dataset")
    data, packets_per_increment, num_slices = generate_adapter_data()

    logging.debug("Saving adapter Dataset")

    file = {'dataset': Dataset(data, packets_per_increment), 'num_slices': num_slices}

    with open('adapter.dataset', 'wb') as adapter_dataset_file:
        pickle.dump(file, adapter_dataset_file)

    logging.debug("Adapter Dataset Saved")


def generate_adapter_data():
    chunksize = 10 ** 5
    current_packets = []
    packets_filaname_per_increment = []
    data = []
    slices_values = numpy.zeros(Constants.TOTAL_SLICES)
    dictionary_slicesId = dict(zip(Constants.SLICES, range(Constants.TOTAL_SLICES)))
    current_timestamp = -1
    i = 1
    for df in pd.read_csv('data/packet_data.csv', chunksize=chunksize):
        if current_timestamp == -1:
            current_timestamp = df.values[0][0]

        for pack in df.values:
            if current_timestamp + TIME_INCREMENT < pack[0]:
                append_to_adaptor_dataset(data, slices_values)
                save_packets_to_file('packet_' + str(i) + '.csv', current_packets)
                packets_filaname_per_increment.append('packet_' + str(i) + '.csv')
                current_timestamp = pack[0]
                current_packets = []
                i += 1
                slices_values = numpy.zeros(Constants.TOTAL_SLICES)
            current_packets.append(pack)
            slices_values[dictionary_slicesId[pack[2]]] += 1
    append_to_adaptor_dataset(data, slices_values)
    save_packets_to_file('packet_' + str(i) + '.csv', current_packets)
    packets_filaname_per_increment.append('packet_' + str(i) + '.csv')
    return data, packets_filaname_per_increment, Constants.TOTAL_SLICES


def save_packets_to_file(file_name, data):
    with open('data/packet_data/' + file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        y = self.labels[index]
        return [ID, y]
