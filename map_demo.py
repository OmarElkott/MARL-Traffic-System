import matplotlib.pyplot as plt
from metadrive import (
    MultiAgentMetaDrive,
    MultiAgentTollgateEnv,
    MultiAgentBottleneckEnv,
    MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv,
    MultiAgentParkingLotEnv
)

env_classes = dict(
    roundabout=MultiAgentRoundaboutEnv,
    intersection=MultiAgentIntersectionEnv,
    tollgate=MultiAgentTollgateEnv,
    bottleneck=MultiAgentBottleneckEnv,
    parkinglot=MultiAgentParkingLotEnv,
    pgma=MultiAgentMetaDrive
)

fig, axs = plt.subplots(2, 3, figsize=(10, 6.5), dpi=200)
plt.tight_layout(pad=-3)

for i in range(2):
    for j in range(3):
        env = list(env_classes.values())[i*3+j]({"log_level":50})
        env.reset(seed=0)
        m = env.render(mode="topdown", 
                       # get the overview of the scene
                       film_size = (1000, 1000),
                       screen_size = (1000, 1000),
                       # set camer to map center
                       camera_position=env.current_map.get_center_point(), 
                       # auto determine the number of pixels for 1 meter 
                       scaling=None,
                       # do not pop window
                       window=False)
        ax = axs[i][j]
        ax.imshow(m, cmap="bone")
        ax.set_xticks([])
        ax.set_yticks([])
        env.close()
plt.show()