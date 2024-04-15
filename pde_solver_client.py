import zmq


def client(pde_code):
    # Establish connection to the server
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    # Send the PDE-solving code to the server
    socket.send_string(pde_code)

    # Receive the result from the server
    result = socket.recv_string()

    # Process the result (not needed in this case as the result is printed on the server side)
    print("Received result from server:", result)

if __name__ == "__main__":
    client()
