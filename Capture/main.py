

if __name__ == "__main__":
    interfaces = get_all_interfaces()

    for interface_name in interfaces:
        packet_q[interface_name] = deque()
        data_buffers[interface_name] = []
        data_locks[interface_name] = threading.Lock()

        # Start the packet capture thread for this interface
        threading.Thread(
            target=capture_packets,
            args=(interface_name, packet_q[interface_name]),
            daemon=True
        ).start()

        # Start the packet processing thread for this interface
        threading.Thread(
            target=process_packets,
            args=(packet_qs[interface_name], data_buffers[interface_name], data_locks[interface_name]),
            daemon=True
        ).start()

        # Start the model training thread for this interface
        threading.Thread(
            target=train_model,
            args=(interface_name, data_buffers[interface_name], data_locks[interface_name], trained_models, trained_scalers),
            daemon=True
        ).start()

        # Start the anomaly detection thread for this interface
        threading.Thread(
            target=detect_anomalies,
            args=(interface_name, packet_qs[interface_name], data_locks[interface_name], trained_models, trained_scalers),
            daemon=True
        ).start()

    # Keep the main thread alive
    while True:
        time.sleep(1)