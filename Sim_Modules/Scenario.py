import json


class Scenario:
    def __init__(self, json_file):
        self.size = None
        self.channel_losses = None
        self.__dict__ = json.load(json_file)
        self.user_groups = None
        self.base_stations = None

