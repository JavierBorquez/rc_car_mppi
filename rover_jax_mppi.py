import numpy as np
import socket
import mat73
import jax
import jax.numpy as jnp
from jax import random, jit
import time
import gc


# ------------------     MATLAB STUFF  ----------------------------------
def get_matlab_variables(mat_file_path):
    variables = mat73.loadmat(mat_file_path)
    #double gets converted to np array by default
    value_func_data = variables['Vx']
    lx_data = variables['lx'] 
    tau2 = variables['tau2']

    #Deriv is cell which gets converted into list of lists
    deriv_x_data = np.array(variables['Deriv'][0])
    deriv_x_data = deriv_x_data.squeeze()
    deriv_y_data = np.array(variables['Deriv'][1])
    deriv_y_data = deriv_y_data.squeeze()
    deriv_th_data = np.array(variables['Deriv'][2])
    deriv_th_data = deriv_th_data.squeeze()
    
    #uOpt is also cell which gets converted into list of lists
    uOpt_vel = np.array(variables['uOpt'][0])
    uOpt_vel = uOpt_vel.squeeze()
    uOpt_angle = np.array(variables['uOpt'][1])
    uOpt_angle = uOpt_angle.squeeze() 

    #g is struct whic gets converted into dic
    #vs is cell which give a list
    x_coord=np.array(variables['g']['vs'][0])
    y_coord=np.array(variables['g']['vs'][1])
    th_coord=np.array(variables['g']['vs'][2])
    x_coord = x_coord.squeeze()
    y_coord = y_coord.squeeze()
    th_coord = th_coord.squeeze()

    matlab_var_dict = dict( value_func_data=value_func_data,
                            lx_data=lx_data,
                            deriv_x_data=deriv_x_data,
                            deriv_y_data=deriv_y_data,
                            deriv_th_data=deriv_th_data,
                            uOpt_vel=uOpt_vel,
                            uOpt_angle=uOpt_angle,
                            x_coord=x_coord,
                            y_coord=y_coord,
                            th_coord=th_coord,
                            tau2=tau2
                           )
    return matlab_var_dict


#---------------------- Load MATLAB ---------------------------------------------------------
#v3 added uopt lookup table
matlab_var_dict= get_matlab_variables('/home/javier/jax_work/mppi/rc_car_mppi/brt_rc_wh_fine_v3.mat')

data = matlab_var_dict['value_func_data']
data_lx = matlab_var_dict['lx_data']
uOpt_vel = matlab_var_dict['uOpt_vel']
uOpt_angle = matlab_var_dict['uOpt_angle']
coords = [matlab_var_dict['x_coord'], matlab_var_dict['y_coord'], matlab_var_dict['th_coord']]

data = jnp.array(data)
data_lx = jnp.array(data_lx)
uOpt_vel = jnp.array(uOpt_vel)
uOpt_angle = jnp.array(uOpt_angle)
coords = [jnp.array(coord) for coord in coords]


#---------------------- JAX MPPI ---------------------------------------------------------

# Experiment Constants
DT = 0.02
L = 0.235
V_MIN = 0.7
V_MAX = 1.4
DELTA_MIN = -0.436
DELTA_MAX = 0.436

# Example usage with jax.random
key = random.PRNGKey(0)

# Simulation parameters
HALLUCINATION_STEPS = 100
NUM_THREADS = 1000

# Safety filter parameters
# Safety filter parameters (big negative value to disable)
EXPERIMENT_THRESHOLD = 0.1
HALLUCINATIONS_THRESHOLD = 0.1

# MPPI cost function parameters
W_VEL = 1.0
W_CENTERING = 120.0
W_COLLISION = 50.0
W_IN_BRT = 0.0
TEMPERATURE = 0.005

# dict with all the parameters
experiment_params = {
    'DT': DT,
    'L': L,
    'V_MIN': V_MIN,
    'V_MAX': V_MAX,
    'DELTA_MIN': DELTA_MIN,
    'DELTA_MAX': DELTA_MAX,
    'HALLUCINATION_STEPS': HALLUCINATION_STEPS,
    'NUM_THREADS': NUM_THREADS,
    'EXPERIMENT_THRESHOLD': EXPERIMENT_THRESHOLD,
    'HALLUCINATIONS_THRESHOLD': HALLUCINATIONS_THRESHOLD,
    'W_VEL': W_VEL,
    'W_CENTERING': W_CENTERING,
    'W_COLLISION': W_COLLISION,
    'W_IN_BRT': W_IN_BRT,
    'TEMPERATURE': TEMPERATURE
}

# Data structures
state_now = []
state_history = []
control_now = (0.0, 0.0)
control_history = []
hallucination_history = []
hallucination_time_history = []
loop_time_history = []
lr_active = []

# timing structures
time_rollouts = []
time_sim_step = [] 

# Generate nominal control inputs
nominal_velocities = jnp.ones(HALLUCINATION_STEPS)
nominal_steering_angles = jnp.zeros(HALLUCINATION_STEPS)
nominal_controls = jnp.stack((nominal_velocities, nominal_steering_angles), axis=1)

@jit
def ackerman_dynamics(state, control, dt=DT, L=L):
    x, y, theta = state
    v, delta = control
    
    x_dot = v * jnp.cos(theta)
    y_dot = v * jnp.sin(theta)
    theta_dot = v * jnp.tan(delta) * (1 / L)
    
    new_x = x + x_dot * dt
    new_y = y + y_dot * dt
    new_theta = theta + theta_dot * dt    
    # Handle the angle wrap around
    new_theta = ((new_theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    
    return new_x, new_y, new_theta

@jit
def cost_function(states, controls, data_lx, data, v_target=V_MAX):
    x = states[0, :]
    y = states[1, :]
    th = states[2, :]
    x_idx = jnp.argmin(jnp.abs(coords[0] - x[:,jnp.newaxis]),axis=1)
    y_idx = jnp.argmin(jnp.abs(coords[1] - y[:,jnp.newaxis]),axis=1)
    th_idx= jnp.argmin(jnp.abs(coords[2] - th[:,jnp.newaxis]),axis=1)
    lx = data_lx[x_idx, y_idx]  # distance to nearest obstacle (max~0.45)
    vx = data[x_idx, y_idx, th_idx]  # value function (max~1.69)
    
    v = controls[:, 0]
    #delta = controls[:, 1]
    cost = jnp.sum((v - v_target) ** 2) * W_VEL  # magnitude 1.69
    cost += jnp.sum(0.45-lx) * W_CENTERING # magnitude 0.45 * W
    cost += jnp.sum(jnp.where(lx < 0.0, 1.0, 0.0)) * W_COLLISION  # magnitude 1 * W
    cost += jnp.sum(jnp.where(vx < 0.0, 1.0, 0.0)) * W_IN_BRT  # magnitude 1 * W
    return cost 

@jit
def simulate_ackerman(initial_state, disturbed_controls, data, data_lx, uOpt_vel, uOpt_angle, coords, dt=DT, L=L):
  
    def step(state, control):
        # Find index in coords closest to state to get value and optimal control
        x_idx = jnp.argmin(jnp.abs(coords[0] - state[0]))
        y_idx = jnp.argmin(jnp.abs(coords[1] - state[1]))
        th_idx = jnp.argmin(jnp.abs(coords[2] - state[2]))        
        value_now = data[x_idx, y_idx, th_idx]
        uOpt_vel_now = uOpt_vel[x_idx, y_idx, th_idx]
        uOpt_angle_now = uOpt_angle[x_idx, y_idx, th_idx]
        
        # Update control with optimal values
        updated_control = jnp.array([uOpt_vel_now, uOpt_angle_now]) * (value_now < HALLUCINATIONS_THRESHOLD) + control * (value_now >= HALLUCINATIONS_THRESHOLD)
        new_state = ackerman_dynamics(state, updated_control, dt, L)
        return new_state, (new_state, updated_control)
    
    # Use jax.lax.scan to iterate over the controls and accumulate the states and updated controls
    _, (states, updated_controls) = jax.lax.scan(step, initial_state, disturbed_controls)
    
    # Convert states and updated controls to JAX arrays
    states = jnp.array(states)
    updated_controls = jnp.array(updated_controls)
    
    # Compute the cost for the entire array of states and controls
    total_costs = cost_function(states, updated_controls, data_lx, data)
    
    return states, updated_controls, total_costs

# Vectorize the simulation function to run multiple trajectories in parallel
simulate_ackerman_parallel = jax.vmap(simulate_ackerman, in_axes=(None, 0, None, None, None, None, None))


#---------------------- UDP SERVER -----------------------------------
# Create a TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# Bind the socket to the server address
server_address = ('localhost', 8081)
sock.bind(server_address)
# Listen for incoming connections
sock.listen(1)
print("Server is listening on port 8081...")
# Accept a connection
connection, client_address = sock.accept()
print(f"Connection from {client_address}")

#---------------------- MAIN LOOP -----------------------------------

try:
    while True:
        # Receive the message in small chunks
        message = connection.recv(4096)
        if message:
            # Decode the message to a string
            received_str = message.decode().strip()

            # Split the string by commas
            x_str, y_str, th_str = received_str.split(',')

            # Convert the strings to floats
            x = float(x_str)
            y = float(y_str)
            th = float(th_str)
            print(f"Received x: {x}, y: {y}, th: {th}")
            
            ################################################################            
            state_now = (x, y, th)
            start_time_sim_step = time.time()
            # Generate random noise for multiple trajectories
            deltas_control = random.normal(key, shape=(NUM_THREADS, HALLUCINATION_STEPS, 2)) * jnp.array([0.2, 0.4])  # Adjust the scale of noise as needed

            # Combine controls and noise before passing to the simulation function, clip to valid range
            disturbed_controls = nominal_controls + deltas_control
            disturbed_controls = jnp.clip(disturbed_controls, jnp.array([V_MIN, DELTA_MIN]), jnp.array([V_MAX, DELTA_MAX]))

            # Perform the hallucination rollouts in parallel
            start_time_rollouts = time.time()
            states_parallel, updated_controls, costs_parallel = simulate_ackerman_parallel(state_now, disturbed_controls, data, data_lx, uOpt_vel, uOpt_angle, coords)
            end_time_rollouts = time.time()
            
            # Update nominal controls using the costs and noise
            weights = jnp.exp(-TEMPERATURE * (costs_parallel))
            weights = weights[:, jnp.newaxis, jnp.newaxis]  # Adjust shape for broadcasting
            deltas_control = updated_controls - nominal_controls # Consider the updated difference to the controls
            nominal_controls = nominal_controls + jnp.sum(weights * deltas_control, axis=0) / jnp.sum(weights)
            
            # Check value function and apply LR filter
            #find index in coords closest to state to get value and optimal control       
            x_idx = np.argmin(np.abs(coords[0] - x))
            y_idx = np.argmin(np.abs(coords[1] - y))
            th_idx = np.argmin(np.abs(coords[2] - th))                    
            value_now = data[x_idx, y_idx, th_idx]
            uOpt_vel_now = uOpt_vel[x_idx, y_idx, th_idx]
            uOpt_angle_now = uOpt_angle[x_idx, y_idx, th_idx]
            
            # BASIC LR TEST
            # control_now = np.array([1.0, 0.0])
            # control_now = np.array([uOpt_vel_now, uOpt_angle_now]) * (value_now < EXPERIMENT_THRESHOLD) + control_now * (value_now >= EXPERIMENT_THRESHOLD)
            
            if value_now < EXPERIMENT_THRESHOLD:
              control_now = jnp.array([uOpt_vel_now,uOpt_angle_now])
            else:
              control_now = nominal_controls[0]                

            # Move the control sequence one step forward and maintain the last control
            nominal_controls = jnp.roll(nominal_controls, -1, axis=0)
            nominal_controls = nominal_controls.at[-1].set(nominal_controls[-2])

            end_time_sim_step = time.time()
            # Print relevant information
            print(f"Value at this point: {value_now}")
            print(f"control: {control_now}")
            hallucination_time = (end_time_rollouts - start_time_rollouts)*1000
            loop_time = (end_time_sim_step - start_time_sim_step)*1000  
            print(f"Elapsed time for rollouts {hallucination_time :.1f} ms")
            print(f"Elapsed time for loop {loop_time :.1f} ms")
            
            # Store the state, control, and hallucinations for visualization
            state_history.append(state_now)
            control_history.append(control_now)
            hallucination_history.append(states_parallel)
            hallucination_time_history.append(hallucination_time)
            loop_time_history.append(loop_time)
            lr_active.append(value_now < EXPERIMENT_THRESHOLD)
                    
            #send value to client
            connection.sendall(str(control_now).encode())
                        
        # else:
        #     print("No message received")
        #     break
          
except KeyboardInterrupt:
    print("Stopping the UDP listener")    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f'mppi_data_{timestamp}.npz'
    np.savez(filename, state_history=state_history, control_history=control_history, hallucination_history=hallucination_history,
             hallucination_time_history=hallucination_time_history, loop_time_history=loop_time_history , lr_active=lr_active,
             experiment_params=experiment_params)
    print(f"Data saved to {filename}")         
finally:
    # Clean up the connection
    connection.close()
    jax.clear_backends()
    gc.collect()