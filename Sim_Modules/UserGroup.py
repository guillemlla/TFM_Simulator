import random
from Sim_Modules.User import User
import Constants


class UserGroup:
    def __init__(self, user_group, env, map_limit, base_stations, existing_probabilities, max_concurrent_users):
        self.user_group_name = None
        self.users_percentage = None
        self.mobility_pattern = None
        self.packet_type_id = None
        self.packet_size = None
        self.burst_traffic_prob = None
        self.burst_length = None
        self.normal_timeout = None
        self.burst_timeout = None
        self.color = None
        self.speed = None
        self.__dict__ = user_group
        self.existing_probabilities = existing_probabilities
        self.base_stations = base_stations
        self.map_limit = map_limit
        self.env = env
        self.users = []
        self.num_users_per_hour = []
        self.last_hour = 23
        self.last_prob = 0
        self.total_users = 1
        self.max_users = int(max_concurrent_users * (self.users_percentage / 100))
        print("User group: " + self.user_group_name + " number max users:" +  str(self.max_users))
        env.process(self.checkNumberUsers())

    def checkNumberUsers(self):
        while True:
            self.last_hour = self.last_hour + 1 if self.last_hour < 23 else 0

            if self.last_prob != self.existing_probabilities[self.last_hour]:
                num_users = self.max_users * (float(self.existing_probabilities[self.last_hour]) + random.gauss(self.last_prob / 10, self.last_prob / 10))
                diff = round(num_users - self.users.__len__())
                if diff > 0:
                    for _ in range(diff):
                        new_user = User(self.env, self.user_group_name + "_" + str(self.total_users),
                                        self.mobility_pattern, self.packet_type_id, self.packet_size,
                                        self.burst_traffic_prob, self.burst_length, self.normal_timeout, self.burst_timeout, self.speed, self.color,
                                        self.map_limit, self.base_stations)
                        self.env.process(new_user.run_user())
                        self.users.append(new_user)
                        self.total_users += 1
                elif diff < 0 and self.users.__len__() > 0:
                    for _ in range(abs(diff)):
                        num = random.randint(0, self.users.__len__() - 1) if self.users.__len__() > 1 else 0
                        self.users[num].active = False
                        self.users.__delitem__(num)
                self.last_prob = self.existing_probabilities[self.last_hour]
            self.num_users_per_hour.append(self.users.__len__())
            yield self.env.timeout(Constants.HOUR)
