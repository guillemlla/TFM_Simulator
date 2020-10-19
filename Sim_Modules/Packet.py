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
