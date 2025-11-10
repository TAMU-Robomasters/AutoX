from filterpy.kalman import KalmanFilter
import numpy as np

# NOTE: this model is for 3D position estimation with dynamic velocity
#       Apparently, after talking with some researchers, the best/standard way
#       is to actually assume a constant velocity, and then assume everything else is noise
#       e.g. assume a lot of noise
#       not sure if this will work well for things that are rotating in circle 
#       but thats what is used commonly for other stuff

class KalmanFilterForHumans:
    def __init__(
        self,
        state_transition_matrix,
        measurement_matrix,
        process_noise,
        measurement_noise,
        initial_state,
    ):
        self.state_transition_matrix = state_transition_matrix
        self.measurement_matrix      = measurement_matrix
        self.process_noise           = process_noise
        self.measurement_noise       = measurement_noise
        self.initial_state           = initial_state
        self.prediction              = None
        self.kalman_filter           = None

    def reset(self):
        kalman_filter = KalmanFilter(
            dim_x=len(self.initial_state), dim_z=len(self.measurement_noise)
        )
        kalman_filter.x = self.initial_state
        kalman_filter.P = np.eye(len(self.initial_state))  # Initial state covariance matrix
        kalman_filter.F = self.state_transition_matrix
        kalman_filter.H = self.measurement_matrix
        kalman_filter.Q = self.process_noise
        kalman_filter.R = self.measurement_noise

        self.kalman_filter = kalman_filter
        self.kalman_filter.predict()

    def add_and_predict(self, sample):
        if self.kalman_filter == None:
            self.reset()
        self.kalman_filter.update(sample)
        self.kalman_filter.predict()
        self.prediction = self.kalman_filter.x
        return self.prediction


# transition matrix for 3D dynamic velocity model
time_step = 1.0  # time step (aka "dt"), this will need to change depending on the position and velocity units (e.g. meters/second, feet/second)
# IDK if this model has any kind of "weight" to it or if assumes massless movement
transition_matrix = np.array(
    [
        [1, 0, 0, time_step, 0        , 0        , 0.5 * time_step**2, 0                 , 0                  ],
        [0, 1, 0, 0        , time_step, 0        , 0                 , 0.5 * time_step**2, 0                  ],
        [0, 0, 1, 0        , 0        , time_step, 0                 , 0                 , 0.5 * time_step**2 ],
        [0, 0, 0, 1        , 0        , 0        , time_step         , 0                 , 0                  ],
        [0, 0, 0, 0        , 1        , 0        , 0                 , time_step         , 0                  ],
        [0, 0, 0, 0        , 0        , 1        , 0                 , 0                 , time_step          ],
        [0, 0, 0, 0        , 0        , 0        , 1                 , 0                 , 0                  ],
        [0, 0, 0, 0        , 0        , 0        , 0                 , 1                 , 0                  ],
        [0, 0, 0, 0        , 0        , 0        , 0                 , 0                 , 1                  ],
    ]
)

kalman_filter = KalmanFilterForHumans(
    state_transition_matrix=transition_matrix,
    initial_state=np.zeros(len(transition_matrix)), # Initial state estimate [x, y, z, vx, vy, vz, ax, ay, az], this will need to change
    # what are we measuring (just position, so first three)
    measurement_matrix=np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
        ]
    ),
    # in practice we are going to have a LOT of z (depth) noise, but less x,y noise. NOTE: we will have to do a transform from x,y pixels to x,y in real world meters/feet
    measurement_noise=np.diag([0.5, 0.5, 0.5]),
    # honestly I don't really understand the process noise
    process_noise=np.diag(
        [0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
    ),
)



# 
# 
# example/test code
# 
# 
if __name__ == "__main__":
    # Simulate noisy measurements
    true_position = np.zeros(3)
    true_velocity = np.zeros(3)
    measurements = []
    filtered_positions = [
        [0, 0, 0], # this is an inital value since the rest are predictions
    ]
    number_of_simulated_timesteps = 50
    for _ in range(number_of_simulated_timesteps):
        # lets say the object is changing velocity randomly
        true_velocity += np.random.normal(0, 0.1, size=3)
        # position after velocity change (duration of time_step)
        true_position += true_velocity 
        
        # add noise to measurements
        measured_position = true_position + np.random.normal(0, 0.5, size=3)
        measurements.append(measured_position)
        
        # where will it be
        prediction = kalman_filter.add_and_predict(measured_position)
        position = prediction[:3]
        filtered_positions.append(position)  # Estimated position
    
    def plot_results():
        import matplotlib.pyplot as plt
        global measurements, filtered_positions
        
        measurements = np.array(measurements)
        filtered_positions = np.array(filtered_positions)
        plt.plot(range(len(measurements)), measurements[:, 0], label="Measurements (x)")
        plt.plot(range(len(measurements)), measurements[:, 1], label="Measurements (y)")
        plt.plot(range(len(measurements)), measurements[:, 2], label="Measurements (z)")
        plt.plot(
            range(len(filtered_positions)),
            filtered_positions[:, 0],
            label="Filtered Position (x)",
        )
        plt.plot(
            range(len(filtered_positions)),
            filtered_positions[:, 1],
            label="Filtered Position (y)",
        )
        plt.plot(
            range(len(filtered_positions)),
            filtered_positions[:, 2],
            label="Filtered Position (z)",
        )
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.title("Kalman Filter for 3D Position Estimation with Dynamic Velocity")
        plt.legend()
        plt.show()
    
    plot_results()
