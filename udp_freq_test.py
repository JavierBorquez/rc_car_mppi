import socket
import time

def main():
    # Define the address and port to listen on
    UDP_IP = "localhost"  # Listen on all available interfaces
    UDP_PORT = 8080

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"Listening on {UDP_IP}:{UDP_PORT}")

    # Initialize the message counter and start time
    message_count = 0
    start_time = time.time()

    try:
        while True:
            # Receive data from the socket
            data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
            message_count += 1

            # Calculate the elapsed time
            elapsed_time = time.time() - start_time

            # Calculate the message frequency
            if elapsed_time > 0:
                frequency = message_count / elapsed_time
                print(f"Message frequency: {frequency:.2f} messages per second")

    except KeyboardInterrupt:
        print("Stopping the UDP listener")

    finally:
        sock.close()

if __name__ == "__main__":
    main()