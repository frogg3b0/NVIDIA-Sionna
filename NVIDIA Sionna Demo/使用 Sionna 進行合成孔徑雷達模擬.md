```python
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver
from sionna.constants import SPEED_OF_LIGHT, PI
import sionna

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np 

from tqdm import tqdm

CPI_length=64
aperture_size = 100 
dist=500 

scene = load_scene(sionna.rt.scene.simple_reflector)
scene.frequency = 10e9

for object in scene.objects:

    scene.get(object).radio_material.scattering_coefficient = 0.5

# Configure the transmitter and receiver arrays
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

scene.rx_array = scene.tx_array

scene.add(Transmitter(name="tx", position=[0,0,0]))
scene.add(Receiver(name="rx", position=[0,0,0]))


x_pos = np.linspace(-aperture_size*0.5, aperture_size*0.5, CPI_length)
y_pos, z_pos = np.ones(CPI_length)*(dist / np.sqrt(2)), np.ones(CPI_length)*(dist / np.sqrt(2))
radar_trajectory = np.array([x_pos, y_pos, z_pos]).T 

mean_tau, range_ = [], []

with tqdm(range(CPI_length)) as pbar:

    for pulse in pbar:

        scene.transmitters["tx"].position = radar_trajectory[pulse]
        scene.receivers["rx"].position = radar_trajectory[pulse] 
        scene.transmitters["tx"].look_at([0,0,0])
        scene.receivers["rx"].look_at([0,0,0]) 

        paths = scene.compute_paths(los=False, 
                                    reflection=True, 
                                    scattering=True,
                                    scat_random_phases=False,
                                    scat_keep_prob=1.0,
                                    num_samples=1e7,
                                    max_depth=1)
        paths.normalize_delays=False
        ai, tau = paths.cir()
        
        mean_tau.append(tf.reduce_mean(tau).numpy())
        range_.append(np.linalg.norm(radar_trajectory[pulse]))

fig, ax = plt.subplots(figsize=(16,27), nrows=3)

ax[0].plot(np.array(range_))
ax[0].grid()
ax[0].set_title("theoretical quadratic range to reflector")
ax[0].set_xlabel("pulse index (azimuth dimension)")
ax[0].set_ylabel("range to reflector (meters)")

ax[1].plot(0.5*np.array(mean_tau)*SPEED_OF_LIGHT)
ax[1].set_title("average length of ray to reflector")
ax[1].grid()
ax[1].set_xlabel("pulse index (azimuth dimension)")
ax[1].set_ylabel("range to reflector (meters)")

# phase error caused by range error at X-band (10GHz frequency)
mean_tau, range_ = np.array(mean_tau), np.array(range_)
phase_diff = 2*PI*scene.frequency*(mean_tau - 2*range_*(1/SPEED_OF_LIGHT))

ax[2].plot(phase_diff)
ax[2].grid()
ax[2].set_title("phase difference (in radians) caused by the range between Sionna RT and theoretical range")
ax[2].set_xlabel("pulse index (azimuth dimension)")
ax[2].set_ylabel("phase difference (radians)")

plt.savefig("sionna_sanity_check.png", bbox_inches="tight")
plt.close()
```
