import numpy as np
import socket
import mat73
import jax
import jax.numpy as jnp
from jax import random, jit
import time




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
matlab_var_dict= get_matlab_variables('/home/javier/jax_work/mppi/rc_car_mppi/brt_rc_wh_coarse_v3.mat')

data = matlab_var_dict['value_func_data']
data_lx = matlab_var_dict['lx_data']
uOpt_vel = matlab_var_dict['uOpt_vel']
uOpt_angle = matlab_var_dict['uOpt_angle']
coords = [matlab_var_dict['x_coord'], matlab_var_dict['y_coord'], matlab_var_dict['th_coord']]

data = jnp.array(data)
uOpt_vel = jnp.array(uOpt_vel)
uOpt_angle = jnp.array(uOpt_angle)
coords = [jnp.array(coord) for coord in coords]

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
            
            x_idx = np.argmin(np.abs(coords[0] - x))
            y_idx = np.argmin(np.abs(coords[1] - y))
            th_idx = np.argmin(np.abs(coords[2] - th))
                    
            value_now = data[x_idx, y_idx, th_idx]
            uOpt_vel_now = uOpt_vel[x_idx, y_idx, th_idx]
            uOpt_angle_now = uOpt_angle[x_idx, y_idx, th_idx]
            
            control = np.array([1.0, 0.0])
            control = np.array([uOpt_vel_now, uOpt_angle_now]) * (value_now < 0.1) + control * (value_now >= 0.1)
        
            print(f"Received x: {x}, y: {y}, th: {th}")
            print(f"Value at this point: {value_now}")
            print(f"control: {control}")
            
            #send value to client
            connection.sendall(str(control).encode())
            
        else:
            print("No message received")
            break
finally:
    # Clean up the connection
    connection.close()
    print("Socket closed")