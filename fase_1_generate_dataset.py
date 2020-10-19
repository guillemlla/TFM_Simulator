import simpy
import random
from Sim_Modules import json_parser
from utils import CSVHelper
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import Constants

dateTime = datetime.now().strftime("%d-%m-%YT%H:%M:%S")
logging.basicConfig(filename='logs/log_fase_1_' + dateTime + '.log', level=logging.DEBUG)

RANDOM_SEED = 42
SIM_TIME = Constants.DAY * 7

random.seed(RANDOM_SEED)
env = simpy.Environment()

def count_time(env):
    i = 100
    while True:
        yield env.timeout(SIM_TIME/1000)
        logging.debug((env.now/SIM_TIME)*100)
        i += 100

env.process(count_time(env))

CSVHelper.create_csv_file()

user_groups, baseStations, scenario = json_parser.load_jsons(env, False, False)

env.run(until=SIM_TIME)

axis_x = []
for i in range(0, 72):
    axis_x.append(i)

fig, ax = plt.subplots()

for user_group in user_groups:

    print(axis_x.__len__())
    print(user_group.num_users_per_hour.__len__())
    print(user_group.num_users_per_hour)
    for user in user_group.users:
        print(user.print_result() + ' Final Pos: ' + str(user.coordinates))
        ax.add_artist(plt.scatter(user.coordinates_hist_x, user.coordinates_hist_y, color=user.color))

for baseStation in baseStations:
    print('{Base Station Id: ' + str(baseStation.id) + ' }')
    ax.add_artist(plt.scatter(baseStation.coordinates[0], baseStation.coordinates[1], s=200, color=[1, 1, 0]))
    ax.add_artist(plt.Circle((baseStation.coordinates[0], baseStation.coordinates[1]), baseStation.range, color=[1,1,0], fill=False))
    for slice in baseStation.slices.items():
        print(slice[1].print_result())
        print(slice[1].pipe.items.__len__())


fig.show()

CSVHelper.generate_predictor_dataset()
CSVHelper.generate_adapter_dataset()


