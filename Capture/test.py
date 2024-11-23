import pandas as pd
import numpy as np
import joblib
from collections import deque
import time
import pyshark
import threading

# Load the model and pipeline components
pipeline = joblib.load('pipeline_components_with_features.joblib')

class Flow:
    def is_expired(self, timeout=60):
        return (time.time() - self.flow_end_time) > timeout

    def __init__(self, src_ip, src_port, dst_ip, dst_port, protocol):
        # Flow identifiers
        self.src_ip = src_ip
        self.src_port = src_port
        self.dst_ip = dst_ip
        self.dst_port = dst_port
        self.protocol = protocol

        # Packet tracking
        self.total_fwd_packets = 0
        self.total_bwd_packets = 0
        self.total_length_fwd_packets = 0
        self.total_length_bwd_packets = 0

        # Initialize timestamps
        self.flow_start_time = time.time()
        self.flow_end_time = self.flow_start_time
        
        # Initialize other required metrics
        self.fwd_packet_lengths = []
        self.bwd_packet_lengths = []
        self.fwd_iat = []
        self.bwd_iat = []
        self.last_fwd_packet_time = None
        self.last_bwd_packet_time = None
        self.fwd_header_length = 0
        self.bwd_header_length = 0
        self.init_win_bytes_forward = None
        self.init_win_bytes_backward = None
        self.act_data_pkt_fwd = 0

    def add_packet(self, packet, direction):
        try:
            current_time = float(packet.sniff_timestamp)
            packet_length = int(packet.length)
            
            if direction == 'forward':
                self.total_fwd_packets += 1
                self.total_length_fwd_packets += packet_length
                self.fwd_packet_lengths.append(packet_length)
                self.act_data_pkt_fwd += 1
                
                if self.last_fwd_packet_time is not None:
                    self.fwd_iat.append(current_time - self.last_fwd_packet_time)
                self.last_fwd_packet_time = current_time
            else:
                self.total_bwd_packets += 1
                self.total_length_bwd_packets += packet_length
                self.bwd_packet_lengths.append(packet_length)
                
                if self.last_bwd_packet_time is not None:
                    self.bwd_iat.append(current_time - self.last_bwd_packet_time)
                self.last_bwd_packet_time = current_time
            
            self.flow_end_time = current_time

        except Exception as e:
            print(f"Error processing packet: {e}")

    def compute_features(self):
        fwd_packet_lengths = np.array(self.fwd_packet_lengths or [0])
        bwd_packet_lengths = np.array(self.bwd_packet_lengths or [0])
        fwd_iat = np.array(self.fwd_iat or [0])
        
        features = {
            ' Destination Port': self.dst_port,
            ' Total Fwd Packets': self.total_fwd_packets,
            ' Total Backward Packets': self.total_bwd_packets,
            'Total Length of Fwd Packets': self.total_length_fwd_packets,
            ' Fwd Packet Length Max': np.max(fwd_packet_lengths),
            ' Fwd Packet Length Mean': np.mean(fwd_packet_lengths),
            ' Fwd Packet Length Std': np.std(fwd_packet_lengths),
            ' Bwd Packet Length Min': np.min(bwd_packet_lengths),
            ' Fwd IAT Mean': np.mean(fwd_iat),
            ' Fwd IAT Std': np.std(fwd_iat),
            ' Fwd IAT Max': np.max(fwd_iat),
            ' Fwd Header Length': self.fwd_header_length,
            ' Avg Fwd Segment Size': self.total_length_fwd_packets / self.total_fwd_packets if self.total_fwd_packets > 0 else 0,
            ' Fwd Header Length.1': self.fwd_header_length,
            'Subflow Fwd Packets': self.total_fwd_packets,
            ' Subflow Fwd Bytes': self.total_length_fwd_packets,
            ' Subflow Bwd Packets': self.total_bwd_packets,
            'Init_Win_bytes_forward': self.init_win_bytes_forward or 0,
            ' act_data_pkt_fwd': self.act_data_pkt_fwd
        }
        return features

def process_packets(packet_queue, flow_dict, pipeline):
    packet_count = 0
    flow_timeout = 15
    
    # Get the model directly - skip the feature selection pipeline
    model = pipeline['model']
    scaler = pipeline['scaler']
    expected_features = pipeline['selected_features']
    
    while True:
        try:
            if packet_queue:
                packet = packet_queue.popleft()
                packet_count += 1
                
                if not hasattr(packet, 'ip'):
                    continue

                # Extract packet info...
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                packet_length = int(packet.length)

                if hasattr(packet, 'tcp'):
                    protocol = 'TCP'
                    src_port = int(packet.tcp.srcport)
                    dst_port = int(packet.tcp.dstport)
                elif hasattr(packet, 'udp'):
                    protocol = 'UDP'
                    src_port = int(packet.udp.srcport)
                    dst_port = int(packet.udp.dstport)
                else:
                    continue

                print(f"\rPacket #{packet_count} | {protocol} | {src_ip}:{src_port} → {dst_ip}:{dst_port}", end='')

                # Process flow...
                forward_key = (src_ip, src_port, dst_ip, dst_port, protocol)
                backward_key = (dst_ip, dst_port, src_ip, src_port, protocol)

                if forward_key in flow_dict:
                    flow = flow_dict[forward_key]
                    direction = 'forward'
                elif backward_key in flow_dict:
                    flow = flow_dict[backward_key]
                    direction = 'backward'
                else:
                    flow = Flow(src_ip, src_port, dst_ip, dst_port, protocol)
                    flow_dict[forward_key] = flow
                    direction = 'forward'

                flow.add_packet(packet, direction)

                # Check for completed flows
                current_time = time.time()
                for flow_key, flow in list(flow_dict.items()):
                    if (current_time - flow.flow_start_time > flow_timeout):
                        try:
                            # Get features
                            features = flow.compute_features()
                            
                            # Create DataFrame with only the expected features
                            features_df = pd.DataFrame([{k: features.get(k, 0) for k in expected_features}])
                            
                            # Only scale and predict - skip feature selection
                            X = scaler.transform(features_df)
                            prediction = model.predict(X)

                            # Print results
                            src_ip, src_port, dst_ip, dst_port, proto = flow_key
                            status = "🚨 DDoS" if prediction[0] == 1 else "Normal"
                            
                            print(f"\n\nFlow Analysis:")
                            print(f"{'='*50}")
                            print(f"Source: {src_ip}:{src_port} → {dst_ip}:{dst_port}")
                            print(f"Protocol: {proto}")
                            print(f"Packets: Fwd={flow.total_fwd_packets}, Bwd={flow.total_bwd_packets}")
                            print(f"Bytes: Fwd={flow.total_length_fwd_packets}, Bwd={flow.total_length_bwd_packets}")
                            print(f"Duration: {current_time - flow.flow_start_time:.2f}s")
                            print(f"Status: {status}")
                            print(f"{'='*50}")
                            
                        except Exception as e:
                            print(f"\nError analyzing flow: {e}")
                        finally:
                            del flow_dict[flow_key]
            else:
                time.sleep(0.1)
        except Exception as e:
            print(f"\nError processing packet: {e}")
            
# Update the Flow class timeout
def is_expired(self, timeout=15):  # Reduced from 60 to 15 seconds
    return (time.time() - self.flow_end_time) > timeout
def capture_packets(interface_name, packet_queue, stop_event):
    try:
        capture = pyshark.LiveCapture(interface=interface_name)
        for packet in capture.sniff_continuously():
            if stop_event.is_set():
                break
            packet_queue.append(packet)
    except Exception as e:
        print(f"Error capturing packets: {e}")

# Test the system
def main():
    print("Starting DDoS Detection System")
    print("-" * 50)
    
    # Initialize components
    packet_queue = deque(maxlen=1000)
    flow_dict = {}
    stop_event = threading.Event()
    
    # Use a test interface (replace with your actual interface)
    interface_name = "en0"  # Change this to your network interface
    
    # Start the capture thread
    capture_thread = threading.Thread(
        target=capture_packets,
        args=(interface_name, packet_queue, stop_event),
        daemon=True
    )
    capture_thread.start()
    
    # Start the processing thread
    processing_thread = threading.Thread(
        target=process_packets,
        args=(packet_queue, flow_dict, pipeline),
        daemon=True
    )
    processing_thread.start()
    
    # Run for a specified duration
    try:
        print(f"Monitoring traffic on {interface_name}...")
        time.sleep(60)  # Monitor for 60 seconds
    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        stop_event.set()
        capture_thread.join(timeout=5)
        processing_thread.join(timeout=5)

if __name__ == "__main__":
    main()