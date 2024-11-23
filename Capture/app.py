import pyshark
import threading
import time
import numpy as np
import pandas as pd
from queue import Queue, Empty
import netifaces as net
import os
import joblib
from threading import Lock
FEATURE_NAMES = [
    ' Destination Port', 
    ' Flow Duration',
    ' Total Fwd Packets',
    ' Total Backward Packets',
    'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets',
    ' Fwd Packet Length Max',
    ' Fwd Packet Length Min',
    ' Fwd Packet Length Mean',
    ' Fwd Packet Length Std',
    'Bwd Packet Length Max',
    ' Bwd Packet Length Min',
    ' Bwd Packet Length Mean',
    ' Bwd Packet Length Std',
    'Flow Bytes/s',
    ' Flow Packets/s',
    ' Flow IAT Mean',
    ' Flow IAT Std',
    ' Flow IAT Max',
    ' Flow IAT Min',
    'Fwd IAT Total',
    ' Fwd IAT Mean',
    ' Fwd IAT Std',
    ' Fwd IAT Max',
    ' Fwd IAT Min',
    'Bwd IAT Total',
    ' Bwd IAT Mean',
    ' Bwd IAT Std',
    ' Bwd IAT Max',
    ' Bwd IAT Min',
    'Fwd PSH Flags',
    ' Bwd PSH Flags',
    ' Fwd URG Flags',
    ' Bwd URG Flags',
    ' Fwd Header Length',
    ' Bwd Header Length',
    'Fwd Packets/s',
    ' Bwd Packets/s',
    ' Min Packet Length',
    ' Max Packet Length',
    ' Packet Length Mean',
    ' Packet Length Std',
    ' Packet Length Variance',
    'FIN Flag Count',
    ' SYN Flag Count',
    ' RST Flag Count',
    ' PSH Flag Count',
    ' ACK Flag Count',
    ' URG Flag Count',
    ' CWE Flag Count',
    ' ECE Flag Count',
    ' Down/Up Ratio',
    ' Average Packet Size',
    ' Avg Fwd Segment Size',
    ' Avg Bwd Segment Size',
    ' Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk',
    ' Fwd Avg Packets/Bulk',
    ' Fwd Avg Bulk Rate',
    ' Bwd Avg Bytes/Bulk',
    ' Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets',
    ' Subflow Fwd Bytes',
    ' Subflow Bwd Packets',
    ' Subflow Bwd Bytes',
    'Init_Win_bytes_forward',
    ' Init_Win_bytes_backward',
    ' act_data_pkt_fwd',
    ' min_seg_size_forward',
    'Active Mean',
    ' Active Std',
    ' Active Max',
    ' Active Min',
    'Idle Mean',
    ' Idle Std',
    ' Idle Max',
    ' Idle Min'
]

# Add verification after loading the model
def verify_features():
    """Verify that we have all required features and they match exactly"""
    print(f"\nFeature Verification:")
    print(f"Total features in training data: {len(FEATURE_NAMES)}")
    print(f"Features in loaded model: {len(pipeline['selected_features'])}")
    
    # Check for missing features
    missing_features = set(FEATURE_NAMES) - set(pipeline['selected_features'])
    if missing_features:
        print("\nWARNING: Missing features in model:")
        for feature in missing_features:
            print(f"- {feature}")
    
    # Check for extra features
    extra_features = set(pipeline['selected_features']) - set(FEATURE_NAMES)
    if extra_features:
        print("\nWARNING: Extra features in model:")
        for feature in extra_features:
            print(f"- {feature}")
    
    # Print first few features for verification
    print("\nFirst 5 features:")
    for i, feature in enumerate(FEATURE_NAMES[:5]):
        print(f"{i+1}. '{feature}'")



# Add these constants at the top
PACKET_TIMEOUT = 60  # Flow expiration timeout in seconds
QUEUE_SIZE = 1000    # Maximum packet queue size
VERBOSE = False      # Enable/disable detailed logging

# Add a packet counter class
class PacketStats:
    def __init__(self):
        self.total_packets = 0
        self.ddos_flows = 0
        self.benign_flows = 0
        self.start_time = time.time()
        self.lock = threading.Lock()

    def update_stats(self, is_ddos):
        with self.lock:
            self.total_packets += 1
            if is_ddos:
                self.ddos_flows += 1
            else:
                self.benign_flows += 1

    def print_stats(self):
        with self.lock:
            elapsed_time = time.time() - self.start_time
            print(f"\nMonitoring Statistics:")
            print(f"Running time: {elapsed_time:.2f} seconds")
            print(f"Total packets processed: {self.total_packets}")
            print(f"DDoS flows detected: {self.ddos_flows}")
            print(f"Benign flows detected: {self.benign_flows}")


try:
    print("Loading model...")
    model_data = joblib.load('Anomaly_Model.joblib')
    pipeline = {
        'model': model_data['model'],
        'scaler': model_data['model'].named_steps['scaler'],
        'selector': model_data['model'].named_steps['feature_selection'],
        'variance_selector': model_data['model'].named_steps['variance_threshold'],
        'selected_features': model_data['feature_names']  
    }
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise



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

        # Packet lengths
        self.fwd_packet_lengths = []
        self.bwd_packet_lengths = []
        self.packet_lengths = []  # All packet lengths

        # Inter-arrival times
        self.fwd_iat = []
        self.bwd_iat = []
        self.flow_iat = []
        self.last_fwd_packet_time = None
        self.last_bwd_packet_time = None
        self.last_packet_time = None

        # Header lengths
        self.fwd_header_length = 0
        self.bwd_header_length = 0

        # Flags
        self.fin_flag_count = 0
        self.syn_flag_count = 0
        self.rst_flag_count = 0
        self.psh_flag_count = 0
        self.ack_flag_count = 0
        self.urg_flag_count = 0
        self.cwe_flag_count = 0
        self.ece_flag_count = 0

        # Window sizes
        self.init_win_bytes_forward = None
        self.init_win_bytes_backward = None
        self.act_data_pkt_fwd = 0
        self.min_seg_size_forward = None

        # Active and Idle times
        self.flow_start_time = time.time()
        self.flow_end_time = self.flow_start_time
        self.active_times = []
        self.idle_times = []
        self.last_active_time = None

        # Other features
        self.flow_packet_times = []

    def add_packet(self, packet, direction):
        try:
            if not hasattr(packet, 'sniff_timestamp'):
                if VERBOSE:
                    print(f"Packet missing sniff_timestamp: {packet}")
                return
            
            current_time = float(packet.sniff_timestamp)
            self.flow_end_time = current_time

            # Track packet times for flow IAT
            self.flow_packet_times.append(current_time)
            if len(self.flow_packet_times) > 1:
                iat = self.flow_packet_times[-1] - self.flow_packet_times[-2]
                self.flow_iat.append(iat)

            # Packet length - add error checking
            try:
                packet_length = int(packet.length)
            except (AttributeError, ValueError) as e:
                if VERBOSE:
                    print(f"Error getting packet length: {e}, packet: {packet}")
                return

            self.packet_lengths.append(packet_length)

            # Detailed error handling for IP addresses
            try:
                if hasattr(packet, 'ip'):
                    src_ip = packet.ip.src
                    dst_ip = packet.ip.dst
                elif hasattr(packet, 'ipv6'):
                    src_ip = packet.ipv6.src
                    dst_ip = packet.ipv6.dst
                else:
                    if VERBOSE:
                        print(f"Packet has no IP layer: {packet}")
                    return
            except AttributeError as e:
                if VERBOSE:
                    print(f"Error accessing IP addresses: {e}, packet: {packet}")
            return
            current_time = float(packet.sniff_timestamp)
            self.flow_end_time = current_time

            # Track packet times for flow IAT
            self.flow_packet_times.append(current_time)
            if len(self.flow_packet_times) > 1:
                iat = self.flow_packet_times[-1] - self.flow_packet_times[-2]
                self.flow_iat.append(iat)

            # Packet length
            packet_length = int(packet.length)
            self.packet_lengths.append(packet_length)

            # Header length calculation
            header_length = 14  # Ethernet header
            if hasattr(packet, 'ip'):
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
            elif hasattr(packet, 'ipv6'):
                src_ip = packet.ipv6.src
                dst_ip = packet.ipv6.dst
            else:
                return

            if hasattr(packet, 'tcp'):
                header_length += int(packet.tcp.hdr_len or 0)
                if hasattr(packet.tcp, 'flags'):
                    flags = int(packet.tcp.flags_hex, 16)
                    self.fin_flag_count += bool(flags & 0x01)
                    self.syn_flag_count += bool(flags & 0x02)
                    self.rst_flag_count += bool(flags & 0x04)
                    self.psh_flag_count += bool(flags & 0x08)
                    self.ack_flag_count += bool(flags & 0x10)
                    self.urg_flag_count += bool(flags & 0x20)
                    self.ece_flag_count += bool(flags & 0x40)
                    self.cwe_flag_count += bool(flags & 0x80)

                if direction == 'forward' and self.init_win_bytes_forward is None:
                    self.init_win_bytes_forward = int(packet.tcp.window_size or 0)
                elif direction == 'backward' and self.init_win_bytes_backward is None:
                    self.init_win_bytes_backward = int(packet.tcp.window_size or 0)

                if self.min_seg_size_forward is None:
                    self.min_seg_size_forward = int(packet.tcp.hdr_len or 0)

            elif hasattr(packet, 'udp'):
                header_length += 8

            if direction == 'forward':
                self.total_fwd_packets += 1
                self.total_length_fwd_packets += packet_length
                self.fwd_packet_lengths.append(packet_length)
                self.fwd_header_length += header_length
                self.act_data_pkt_fwd += 1

                if self.last_fwd_packet_time is not None:
                    iat = current_time - self.last_fwd_packet_time
                    self.fwd_iat.append(iat)
                self.last_fwd_packet_time = current_time
            else:
                self.total_bwd_packets += 1
                self.total_length_bwd_packets += packet_length
                self.bwd_packet_lengths.append(packet_length)
                self.bwd_header_length += header_length

                if self.last_bwd_packet_time is not None:
                    iat = current_time - self.last_bwd_packet_time
                    self.bwd_iat.append(iat)
                self.last_bwd_packet_time = current_time

        except Exception as e:
            if VERBOSE:
                print(f"Error processing packet: {str(e)}")
                print(f"Packet details: {packet}")
                import traceback
                print(traceback.format_exc())


    def compute_features(self):
        # Compute statistical features for packet lengths
        fwd_pl_array = np.array(self.fwd_packet_lengths)
        bwd_pl_array = np.array(self.bwd_packet_lengths)
        all_pl_array = np.array(self.packet_lengths)

        # Handle empty arrays
        if len(fwd_pl_array) == 0:
            fwd_pl_array = np.array([0])
        if len(bwd_pl_array) == 0:
            bwd_pl_array = np.array([0])
        if len(all_pl_array) == 0:
            all_pl_array = np.array([0])
        if len(self.fwd_iat) == 0:
            self.fwd_iat = [0]
        if len(self.bwd_iat) == 0:
            self.bwd_iat = [0]
        if len(self.flow_iat) == 0:
            self.flow_iat = [0]

        flow_duration = (self.flow_end_time - self.flow_start_time) * 1e6  # in microseconds

        # Compute features
        features = {
            ' Destination Port': self.dst_port,
            ' Flow Duration': flow_duration,
            ' Total Fwd Packets': self.total_fwd_packets,
            ' Total Backward Packets': self.total_bwd_packets,
            'Total Length of Fwd Packets': self.total_length_fwd_packets,
            ' Total Length of Bwd Packets': self.total_length_bwd_packets,
            ' Fwd Packet Length Max': np.max(fwd_pl_array),
            ' Fwd Packet Length Min': np.min(fwd_pl_array),
            ' Fwd Packet Length Mean': np.mean(fwd_pl_array),
            ' Fwd Packet Length Std': np.std(fwd_pl_array),
            'Bwd Packet Length Max': np.max(bwd_pl_array),
            ' Bwd Packet Length Min': np.min(bwd_pl_array),
            ' Bwd Packet Length Mean': np.mean(bwd_pl_array),
            ' Bwd Packet Length Std': np.std(bwd_pl_array),
            'Flow Bytes/s': ((self.total_length_fwd_packets + self.total_length_bwd_packets) / flow_duration) * 1e6 if flow_duration > 0 else 0,
            ' Flow Packets/s': ((self.total_fwd_packets + self.total_bwd_packets) / flow_duration) * 1e6 if flow_duration > 0 else 0,
            ' Flow IAT Mean': np.mean(self.flow_iat),
            ' Flow IAT Std': np.std(self.flow_iat),
            ' Flow IAT Max': np.max(self.flow_iat),
            ' Flow IAT Min': np.min(self.flow_iat),
            'Fwd IAT Total': sum(self.fwd_iat),
            ' Fwd IAT Mean': np.mean(self.fwd_iat),
            ' Fwd IAT Std': np.std(self.fwd_iat),
            ' Fwd IAT Max': np.max(self.fwd_iat),
            ' Fwd IAT Min': np.min(self.fwd_iat),
            'Bwd IAT Total': sum(self.bwd_iat),
            ' Bwd IAT Mean': np.mean(self.bwd_iat),
            ' Bwd IAT Std': np.std(self.bwd_iat),
            ' Bwd IAT Max': np.max(self.bwd_iat),
            ' Bwd IAT Min': np.min(self.bwd_iat),
            'Fwd PSH Flags': 0,  # Not computed
            ' Bwd PSH Flags': 0,  # Not computed
            ' Fwd URG Flags': 0,  # Not computed
            ' Bwd URG Flags': 0,  # Not computed
            ' Fwd Header Length': self.fwd_header_length,
            ' Bwd Header Length': self.bwd_header_length,
            'Fwd Packets/s': (self.total_fwd_packets / flow_duration) * 1e6 if flow_duration > 0 else 0,
            ' Bwd Packets/s': (self.total_bwd_packets / flow_duration) * 1e6 if flow_duration > 0 else 0,
            ' Min Packet Length': np.min(all_pl_array),
            ' Max Packet Length': np.max(all_pl_array),
            ' Packet Length Mean': np.mean(all_pl_array),
            ' Packet Length Std': np.std(all_pl_array),
            ' Packet Length Variance': np.var(all_pl_array),
            'FIN Flag Count': self.fin_flag_count,
            ' SYN Flag Count': self.syn_flag_count,
            ' RST Flag Count': self.rst_flag_count,
            ' PSH Flag Count': self.psh_flag_count,
            ' ACK Flag Count': self.ack_flag_count,
            ' URG Flag Count': self.urg_flag_count,
            ' CWE Flag Count': self.cwe_flag_count,
            ' ECE Flag Count': self.ece_flag_count,
            ' Down/Up Ratio': (self.total_fwd_packets / self.total_bwd_packets) if self.total_bwd_packets > 0 else 0,
            ' Average Packet Size': (np.mean(all_pl_array)) if len(all_pl_array) > 0 else 0,
            ' Avg Fwd Segment Size': (self.total_length_fwd_packets / self.total_fwd_packets) if self.total_fwd_packets > 0 else 0,
            ' Avg Bwd Segment Size': (self.total_length_bwd_packets / self.total_bwd_packets) if self.total_bwd_packets > 0 else 0,
            ' Fwd Header Length.1': self.fwd_header_length,
            'Fwd Avg Bytes/Bulk': 0,  # Not computed
            ' Fwd Avg Packets/Bulk': 0,  # Not computed
            ' Fwd Avg Bulk Rate': 0,  # Not computed
            ' Bwd Avg Bytes/Bulk': 0,  # Not computed
            ' Bwd Avg Packets/Bulk': 0,  # Not computed
            'Bwd Avg Bulk Rate': 0,  # Not computed
            'Subflow Fwd Packets': self.total_fwd_packets,
            ' Subflow Fwd Bytes': self.total_length_fwd_packets,
            ' Subflow Bwd Packets': self.total_bwd_packets,
            ' Subflow Bwd Bytes': self.total_length_bwd_packets,
            'Init_Win_bytes_forward': self.init_win_bytes_forward or 0,
            ' Init_Win_bytes_backward': self.init_win_bytes_backward or 0,
            ' act_data_pkt_fwd': self.act_data_pkt_fwd,
            ' min_seg_size_forward': self.min_seg_size_forward or 0,
            'Active Mean': 0,  # Not computed
            ' Active Std': 0,  # Not computed
            ' Active Max': 0,  # Not computed
            ' Active Min': 0,  # Not computed
            'Idle Mean': 0,  # Not computed
            ' Idle Std': 0,  # Not computed
            ' Idle Max': 0,  # Not computed
            ' Idle Min': 0,  # Not computed
        }

        for feature in FEATURE_NAMES:  # Replace `expected_features` with the full list of 78 features
            if feature not in features:
                features[feature] = 0

        return features
    
def get_all_interfaces():
    """
    Get all available network interfaces with their IP addresses.
    
    Returns:
        list: Available network interfaces with IP addresses
    """
    try:
        interfaces = net.interfaces()
        excluded_interfaces = ['lo', 'lo0', 'bridge', 'docker', 'vmnet']
        available_interfaces = []

        for iface in interfaces:
            if any(excluded in iface for excluded in excluded_interfaces):
                continue
            
            try:
                addrs = net.ifaddresses(iface)
                ip_info = addrs.get(net.AF_INET)
                if ip_info:
                    ip_addr = ip_info[0].get('addr', 'N/A')
                    available_interfaces.append((iface, ip_addr))
                else:
                    available_interfaces.append((iface, 'N/A'))
            except ValueError:
                continue
        
        return available_interfaces
    except Exception as e:
        print(f"Error getting network interfaces: {e}")
        return []

def capture_packets(interface_name, packet_queue, stop_event):
    try:
        capture = pyshark.LiveCapture(interface=interface_name)
        for packet in capture.sniff_continuously():
            if stop_event.is_set():
                break
            packet_queue.put(packet)
    except Exception as e:
        print(f"Error capturing packets: {e}")

def process_packets(packet_queue, flow_dict, pipeline):
    while True:
        try:
            # Use get() with timeout instead of popleft()
            try:
                packet = packet_queue.get(timeout=1)
            except Queue.Empty:  # Note: Changed from queue.Empty to Queue.Empty
                continue
                
            # Skip non-IP packets
            if not hasattr(packet, 'ip'):
                continue

            # Extract flow information
            src_ip = packet.ip.src
            dst_ip = packet.ip.dst

            # Get port information based on protocol
            if hasattr(packet, 'tcp'):
                src_port = int(packet.tcp.srcport)
                dst_port = int(packet.tcp.dstport)
                protocol = 'TCP'
            elif hasattr(packet, 'udp'):
                src_port = int(packet.udp.srcport)
                dst_port = int(packet.udp.dstport)
                protocol = 'UDP'
            else:
                continue

            # Create unique flow keys for both directions
            forward_key = (src_ip, src_port, dst_ip, dst_port, protocol)
            backward_key = (dst_ip, dst_port, src_ip, src_port, protocol)

            # Determine flow direction and get/create flow object
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

            # Add packet to flow
            flow.add_packet(packet, direction)

            # Check for expired flows
            for flow_key, flow in list(flow_dict.items()):
                if flow.is_expired(timeout=60):
                    try:
                        # Get features
                        features = flow.compute_features()
                        
                        # Create DataFrame with features
                        features_df = pd.DataFrame([features])
                        
                        # Ensure features are in the correct order
                        features_df = features_df[pipeline['selected_features']]
                        
                        # Make prediction
                        try:
                            X = features_df.copy()
                            X = pipeline['variance_selector'].transform(X)
                            X = pipeline['scaler'].transform(X)
                            X = pipeline['selector'].transform(X)
                            prediction = pipeline['model'].predict(X)
                            
                            # Log prediction
                            src_ip, src_port, dst_ip, dst_port, proto = flow_key
                            status = 'DDoS' if prediction[0] == 1 else 'Normal'
                            packets = flow.total_fwd_packets + flow.total_bwd_packets
                            print(f"[{time.strftime('%H:%M:%S')}] {src_ip}:{src_port} → {dst_ip}:{dst_port} | {proto} | Packets: {packets} | Status: {status}")
                            
                        except Exception as e:
                            print(f"Prediction error: {e}")
                            
                        finally:
                            del flow_dict[flow_key]
                            
                    except Exception as e:
                        print(f"Error processing flow: {e}")
                        del flow_dict[flow_key]

        except Exception as e:
            print(f"Processing error: {e}")
            continue


def process_packets(packet_queue, flow_dict, pipeline, stats):
    while True:
        try:
            try:
                packet = packet_queue.get(timeout=1)
            except Empty:
                continue
                
            if not hasattr(packet, 'ip'):
                if VERBOSE:
                    print(f"Skipping non-IP packet: {packet}")
                continue

            try:
                # Extract flow information with error checking
                if not hasattr(packet.ip, 'src') or not hasattr(packet.ip, 'dst'):
                    if VERBOSE:
                        print(f"Packet missing IP addresses: {packet}")
                    continue
                    
                src_ip = packet.ip.src
                dst_ip = packet.ip.dst
                
                # Get port information with better error handling
                if hasattr(packet, 'tcp'):
                    try:
                        src_port = int(packet.tcp.srcport)
                        dst_port = int(packet.tcp.dstport)
                        protocol = 'TCP'
                    except (AttributeError, ValueError) as e:
                        if VERBOSE:
                            print(f"Error getting TCP ports: {e}")
                        continue
                elif hasattr(packet, 'udp'):
                    try:
                        src_port = int(packet.udp.srcport)
                        dst_port = int(packet.udp.dstport)
                        protocol = 'UDP'
                    except (AttributeError, ValueError) as e:
                        if VERBOSE:
                            print(f"Error getting UDP ports: {e}")
                        continue
                else:
                    if VERBOSE:
                        print(f"Packet is neither TCP nor UDP: {packet}")
                    continue

                # Process flow and update statistics
                forward_key = (src_ip, src_port, dst_ip, dst_port, protocol)
                backward_key = (dst_ip, dst_port, src_ip, src_port, protocol)

                # Update stats first
                stats.update_stats(False)

                # Get or create flow
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

            except AttributeError as e:
                if VERBOSE:
                    print(f"Packet parsing error: {e}")
                    print(f"Packet details: {packet}")
                continue

        except Exception as e:
            if VERBOSE:
                print(f"Processing error: {e}")
                import traceback
                print(traceback.format_exc())
            continue

def select_interface(interfaces):
    """Select network interface for packet capture"""
    if len(interfaces) == 1:
        # If only one active interface, automatically select it
        interface_name = interfaces[0][0]
        print(f"Automatically selected interface: {interface_name} (IP: {interfaces[0][1]})")
        return interface_name
    
    # Display multiple active interfaces and let user select
    print("\nAvailable Network Interfaces:")
    for idx, (iface, ip_addr) in enumerate(interfaces):
        print(f"{idx}: {iface} (IP: {ip_addr})")
    
    while True:
        try:
            selected_idx = int(input("\nSelect interface index for capture: "))
            if 0 <= selected_idx < len(interfaces):
                return interfaces[selected_idx][0]
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def predict_flow(flow, pipeline):
    """Make prediction for a single flow"""
    features = flow.compute_features()
    features_df = pd.DataFrame([features])
    features_df = features_df[pipeline['selected_features']]
    
    X = features_df.copy()
    X = pipeline['variance_selector'].transform(X)
    X = pipeline['scaler'].transform(X)
    X = pipeline['selector'].transform(X)
    return pipeline['model'].predict(X)[0]

def start_capture_threads(interface_name, packet_queue, flow_dict, pipeline, stats, stop_event):
    """Start capture and processing threads"""
    capture_thread = threading.Thread(
        target=capture_packets,
        args=(interface_name, packet_queue, stop_event),
        daemon=True
    )
    
    processing_thread = threading.Thread(
        target=process_packets,
        args=(packet_queue, flow_dict, pipeline, stats),
        daemon=True
    )
    
    capture_thread.start()
    processing_thread.start()
    
    return [capture_thread, processing_thread]

def cleanup(stop_event, threads, stats):
    """Clean up threads and display final statistics"""
    stop_event.set()
    for thread in threads:
        thread.join(timeout=5)
    stats.print_stats()
    print("\nCapture stopped.")


def main():
    print("Network Traffic DDoS Monitor")
    
    # Verify features first thing in main
    verify_features()
    
    # Initialize statistics
    stats = PacketStats()
    
    # Initialize queue and flow tracking
    packet_queue = Queue(maxsize=QUEUE_SIZE)
    flow_dict = {}

    # Get interfaces and setup capture
    interfaces = [(iface, ip) for iface, ip in get_all_interfaces() if ip != 'N/A']
    if not interfaces:
        print("No active network interfaces found.")
        return

    interface_name = select_interface(interfaces)
    print(f"\nStarting capture on: {interface_name}")

    # Start capture
    stop_event = threading.Event()
    threads = start_capture_threads(interface_name, packet_queue, flow_dict, pipeline, stats, stop_event)
    
    try:
        while True:
            time.sleep(10)
            stats.print_stats()
            
    except KeyboardInterrupt:
        print("\nStopping capture...")
    finally:
        cleanup(stop_event, threads, stats)

    # Feature extraction and prediction
    print("\nProcessing captured network flows...")
    features_list = []
    predictions = []

    for flow_key, flow in flow_dict.items():
        try:
            # Get features
            features = flow.compute_features()
            features_list.append(features)

            # Create DataFrame with only the required features
            features_df = pd.DataFrame([features])
            feature_vector = pd.DataFrame(columns=pipeline['selected_features'])
            for feature in pipeline['selected_features']:
                feature_vector[feature] = features_df.get(feature, 0)

            # Apply the pipeline transformations
            X = pipeline['variance_selector'].transform(feature_vector)
            X = pipeline['scaler'].transform(X)
            X = pipeline['selector'].transform(X)

            # Make prediction
            prediction = pipeline['model'].predict(X)
            predictions.append(prediction[0])

            # Print prediction
            src_ip, src_port, dst_ip, dst_port, proto = flow_key
            print(f"Flow: {src_ip}:{src_port} -> {dst_ip}:{dst_port} ({proto})")
            print(f"Prediction: {'BENIGN' if prediction[0] == 0 else 'DDoS'}")
            print(f"Total packets: Forward={flow.total_fwd_packets}, Backward={flow.total_bwd_packets}")
            print("-" * 50)

        except Exception as e:
            print(f"Error processing flow: {str(e)}")

    # Save results if we have any
    if features_list:
        df = pd.DataFrame(features_list)
        df['Prediction'] = predictions
        output_file = 'network_traffic_predictions.csv'
        df.to_csv(output_file, index=False)
        print(f"\nFeatures and predictions saved to {output_file}")
        print(f"Total flows captured: {len(features_list)}")
    else:
        print("No network flows were captured.")
if __name__ == "__main__":
    main()