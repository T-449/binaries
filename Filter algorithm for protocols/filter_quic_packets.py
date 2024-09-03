import hashlib
import pyshark
from filter_tls_packets import classify_packets

key_share_groups = {
    23: 'secp256r1',
    24: 'secp384r1',
    25: 'secp521r1',
    29: 'x25519',
    30: 'x448',
    # Add more mappings as needed
}

packets = {}


# Function to initialize a key with None values if not present
def initialize_key(key):
    if key not in packets:
        packets[key] = (None, None)


# Function to update the first value
def update_value1(key, value1):
    initialize_key(key)
    _, value2 = packets[key]
    packets[key] = (value1, value2)


# Function to update the second value
def update_value2(key, value2):
    initialize_key(key)
    value1, _ = packets[key]
    packets[key] = (value1, value2)


# Function to retrieve values by key
def get_values(key):
    return packets.get(key, (None, None))


def simple_hash(elements):
    # Combine all elements in the list into a single string
    combined_elements = ''.join(str(element) for element in elements)

    # Use SHA-256 to hash the combined string
    hash_object = hashlib.sha256(combined_elements.encode())

    # Get the hexadecimal representation of the hash
    hash_hex = hash_object.hexdigest()

    return hash_hex

if __name__ == "__main__":

    # Path to the PCAP file
    pcap_file = ''
    #pcap_file = '/home/heresy/Downloads/quic.pcap'

    # Open the PCAP file

    capture = pyshark.FileCapture(pcap_file)
    # Iterate through the packets and print all fields in the QUIC layer

    '''
    for packet in capture:
        if 'QUIC' in packet:
            quic_layer = packet['QUIC']
            print(f"Packet Number: {packet.number}")
            for field in quic_layer.field_names:
                print(f"{field}: {getattr(quic_layer, field)}")
            print("-" * 50)
    '''

    for packet in capture:
        if not hasattr(packet, 'quic'):
            continue
        quic_layer = packet['quic']
        udp_layer = packet['udp']
        if hasattr(quic_layer, 'tls_handshake_type'):
            handshake_type = int(getattr(quic_layer, 'tls_handshake_type'))
            if 'ipv6' in packet:
                ipv6_layer = packet.ipv6
                ip_src = ipv6_layer.src
                ip_dst = ipv6_layer.dst
            elif 'ip' in packet:
                ipv4_layer = packet.ip
                ip_src = ipv4_layer.src
                ip_dst = ipv4_layer.dst
            if handshake_type == 2:
                if hasattr(quic_layer, 'tls_handshake_extensions_supported_version'):
                    supported_version = getattr(quic_layer, 'tls_handshake_extensions_supported_version')
                    if supported_version == '0x0304':
                        #print("TLS Layer Fields:")
                        #for field in tls_layer.field_names:
                        #    print(f"{field}: {getattr(tls_layer, field)}")
                        # have a check for algorithm type
                        #print(packet.length)
                        if hasattr(quic_layer, 'tls_handshake_extensions_key_share_group'):
                            key_share_group = quic_layer.tls_handshake_extensions_key_share_group
                            group_name = key_share_groups.get(int(key_share_group), 'Unknown')
                            print("Kex Algo: " + group_name)
                        #print("TLS 1.3")
                        elements = [ip_dst, ip_src, udp_layer.dstport, udp_layer.srcport]
                        key = simple_hash(elements)
                        update_value2(key, packet)
            elif handshake_type == 1:
                elements = [ip_src, ip_dst, udp_layer.srcport, udp_layer.dstport]
                key = simple_hash(elements)
                update_value1(key, packet)
                #for field in tls_layer.field_names:
                #    print(f"{field}: {getattr(tls_layer, field)}")
                #supported_version = getattr(tls_layer, 'handshake_extensions_supported_version')
        else:
            continue

    #complete_packets = {k: v for k, v in packets.items() if all(value is not None for value in v)}
    #print(complete_packets)

    for k, (v1, v2) in packets.items():
        if v2 is not None:
            quic_layer = v2['quic']
            for field in quic_layer.field_names:
                print(f"{field}: {getattr(quic_layer, field)}")
            classify_packets(v1, v2, 0)

    # Close the capture
    capture.close()

    #for key, (value1, value2) in complete_packets.items():
    #    classify_packets(value1, value2, 0)
