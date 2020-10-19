from scipy.spatial import distance
from Sim_Modules.PacketResult import PacketResult
from Sim_Modules.MobiltyPattern import MobilityPattern
import random
import Constants


class User:

    def __init__(self, env, user_id, mobility_pattern, packet_type_id, packet_size, burst_traffic_prob, burst_length, normal_timeout, burst_timeout, speed, color, limit, base_stations):
        self.user_id = user_id
        self.speed = speed * (1 + random.gauss(0, 0.1))
        self.mobility_pattern = mobility_pattern
        self.packet_type_id = packet_type_id
        self.packet_size = packet_size
        self.burst_traffic_prob = burst_traffic_prob
        self.burst_length = burst_length
        self.normal_timeout = normal_timeout
        self.burst_timeout = burst_timeout
        self.color = color
        self.limit = limit
        self.base_stations = base_stations
        self.coordinates = self.create_coordinates()
        self.old_coordinates = []
        self.min_path = None
        self.max_path = None
        self.base_station = None
        self.coordinates_hist_x = []
        self.coordinates_hist_y = []
        self.number_moves = 0
        self.direction = 0
        self.bound = 0
        self.env = env
        self.packets_correct = 0
        self.packets_lost = 0
        self.packets_not_send = 0
        self.set_mobility_pattern()
        self.active = True
        if self.max_path > 0:
            env.process(self.move())
        else:
            self.coordinates_hist_x = self.coordinates[0]
            self.coordinates_hist_y = self.coordinates[1]

    def create_coordinates(self):
        return [random.randint(0, self.limit[0]), random.randint(0, self.limit[1])]

    def set_mobility_pattern(self):
        if self.mobility_pattern == MobilityPattern.CAR.value:
            self.min_path = 5
            self.max_path = 100
        elif self.mobility_pattern == MobilityPattern.PEDESTRIAN.value:
            self.min_path = 1
            self.max_path = 10
        elif self.mobility_pattern == MobilityPattern.STILL.value:
            self.max_path = 0
            self.min_path = 0

    def isInRange(self, coordinates, range):
        return distance.euclidean(self.coordinates, coordinates) <= range

    def move(self):
        while self.active:
            self.coordinates_hist_x.append(self.coordinates[0])
            self.coordinates_hist_y.append(self.coordinates[1])
            if self.number_moves == 0:
                self.number_moves = random.randint(self.min_path, self.max_path)
                direction_old = self.direction
                self.direction = 0 if random.random() > 0.5 else 1
                if direction_old != self.direction:
                    self.bound = 1 if random.random() > 0.5 else -1

            if not self.able_to_move(self.direction, self.bound == 1):
                self.number_moves = 0
            else:
                self.coordinates[self.direction] += (self.speed * self.bound)
                self.number_moves -= 1

            yield self.env.timeout(Constants.MOVE_TIME)

    def able_to_move(self, axis, non_negative):
        return (non_negative and self.coordinates[axis] + self.speed < self.limit[axis]) or (
                not non_negative and self.coordinates[axis] - self.speed > 0)

    def run_user(self):
        while self.active:
            if self.base_station is None or not self.isInRange(self.base_station.coordinates, self.base_station.range):
                self.base_station = None
                for base_station in self.base_stations:
                    if self.isInRange(base_station.coordinates, base_station.range):
                        self.base_station = base_station
                self.packets_not_send += 1
                yield self.env.timeout(Constants.MOVE_TIME)
            else:
                # Send packet
                self.send_packet()

                if random.random() > self.burst_traffic_prob:
                    # Normal Traffic
                    yield self.env.timeout(random.randint(self.normal_timeout[0], self.normal_timeout[1]))
                else:
                    # Burst Traffic
                    for _ in range(random.randint(self.burst_length[0], self.burst_length[1]) - 1):
                        yield self.env.timeout(random.randint(self.burst_timeout[0], self.burst_timeout[1]))
                        self.send_packet()
                    yield self.env.timeout(random.randint(self.burst_timeout[0], self.burst_timeout[1]))

    def send_packet(self):
        packet = Packet(self.env.now, random.uniform(self.packet_size[0], self.packet_size[1]),
                        self.packet_type_id, self.user_id, self.base_station.id, self.coordinates, self.env.now)
        result = self.base_station.send_message_with_channel_error(packet)
        if result == PacketResult.PACKET_OK:
            self.packets_correct = self.packets_correct + 1
        else:
            self.packets_lost = self.packets_lost + 1

    def print_result(self):
        return '{User Id: ' + str(self.user_id) + ' Packets send: ' + str(self.packets_correct) + ' Packets Lost: ' + str(
            self.packets_lost) + ' Packets not Send: ' + str(self.packets_not_send) +  '}'


class Packet:
    def __init__(self, timestamp, size, packet_type, from_id, to_id, coordinates_from, creation_time):
        self.timestamp = timestamp
        self.size = size
        self.coordinates_from = coordinates_from
        self.type = packet_type
        self.from_id = from_id
        self.to_id = to_id
        self.creation_time = creation_time

    def to_string(self):
        return '{ From:' + str(self.from_id) + ' To: ' + str(self.to_id) + ' Type: ' + str(self.type) + ' Size: ' + str(
            self.size) + '}'
