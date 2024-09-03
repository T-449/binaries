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
    pcap_file = ''

    # Open the PCAP file

    capture = pyshark.FileCapture(pcap_file)

    for packet in capture:
        if not hasattr(packet, 'openvpn'):
            continue
        if not hasattr(packet, 'tls'):
            continue
        tls_layer = packet['tls']
        if hasattr(packet, 'udp'):
            transport_layer = packet['udp']
        elif hasattr(packet, 'tcp'):
            transport_layer = packet['tcp']
        #for field in tls_layer.field_names:
        #    print(f"{field}: {getattr(tls_layer, field)}")
        if not hasattr(tls_layer, 'handshake_type'):
            continue
        handshake_type = int(packet.tls.handshake_type)
        if handshake_type == 2:
            if hasattr(tls_layer, 'handshake_extensions_supported_version'):
                supported_version = getattr(tls_layer, 'handshake_extensions_supported_version')
                if supported_version == '0x0304':
                    # print("TLS Layer Fields:")
                    # for field in tls_layer.field_names:
                    #    print(f"{field}: {getattr(tls_layer, field)}")
                    # have a check for algorithm type
                    # print(packet.length)
                    if hasattr(tls_layer, 'handshake_extensions_key_share_group'):
                        key_share_group = tls_layer.handshake_extensions_key_share_group
                        group_name = key_share_groups.get(int(key_share_group), 'Unknown')
                        print("Kex Algo: " + group_name)
                    # print("TLS 1.3")
                    elements = [packet.ip.dst, packet.ip.src, transport_layer.dstport, transport_layer.srcport,
                                tls_layer.handshake_session_id]
                    key = simple_hash(elements)
                    update_value2(key, packet)
        elif handshake_type == 1:
            if hasattr(tls_layer, 'handshake_extensions_supported_version'):
                supported_version = getattr(tls_layer, 'handshake_extensions_supported_version')
                if supported_version == '0x0304':
                    elements = [packet.ip.src, packet.ip.dst, transport_layer.srcport, transport_layer.dstport,
                                tls_layer.handshake_session_id]
                    key = simple_hash(elements)
                    update_value1(key, packet)
            # for field in transport_layer.field_names:
            #    print(f"{field}: {getattr(transport_layer, field)}")
            # for field in tls_layer.field_names:
            #    print(f"{field}: {getattr(tls_layer, field)}")
            # supported_version = getattr(tls_layer, 'handshake_extensions_supported_version')

        elif handshake_type == 12:
            if hasattr(tls_layer, 'handshake_server_named_curve'):
                # for field in tls_layer.field_names:
                #    print(f"{field}: {getattr(tls_layer, field)}")
                key_share_group = int(tls_layer.handshake_server_named_curve, 16)
                group_name = key_share_groups.get(key_share_group, 'Unknown')
                print("Kex Algo: " + group_name)
                elements = [packet.ip.src, packet.ip.dst, transport_layer.srcport, transport_layer.dstport, transport_layer.ack]
                key = simple_hash(elements)
                update_value2(key, packet)
        else:
            elements = [packet.ip.dst, packet.ip.src, transport_layer.dstport, transport_layer.srcport, transport_layer.seq]
            key = simple_hash(elements)
            update_value1(key, packet)
            # print('TLS 1.2')
            # haver a check for the algorithm

    #complete_packets = {k: v for k, v in packets.items() if all(value is not None for value in v)}
    #print(complete_packets)
    # Close the capture file
    capture.close()

    for key, (value1, value2) in packets.items():
        if value2 is not None:
            classify_packets(value1, value2, 1)
