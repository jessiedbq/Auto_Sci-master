import numpy as np
import pandas as pd
from simulation import LotkaVolterra as LV
import os


lv_simulator = LV(true_param=(2.0, 1.0, 4.0, 1.0),
                  noise_variance=0.1**2)

states_obs, t_obs = lv_simulator.observe(initial_state=(5.0, 3.0),
                                         initial_time=0.0,
                                         final_time=2.0,
                                         t_delta_integration=0.01,
                                         t_delta_observation=0.1)
n_states, n_points = states_obs.shape

# print("Successfully generated %d observations"% (n_points))
#
#
# time_path = os.path.join(data_dir, "time.csv")
# pd.DataFrame(t_obs).to_csv(time_path)
# print('Saved timepoints to',time_path)
np.savetxt("time.csv", t_obs)
np.savetxt("observations.csv", states_obs)
