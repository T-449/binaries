import pyshark
import hashlib

pq_algorithms = {
    # Post-Quantum Algorithms
    "Classic-McEliece-348864": (261120, 96),
    "Classic-McEliece-460896": (524160, 156),
    "Classic-McEliece-6688128": (1044992, 208),
    "Classic-McEliece-6960119": (1047319, 194),
    "Classic-McEliece-8192128": (1357824, 208),
    "BIKE-L1": (1541, 1573),
    "BIKE-L3": (3083, 3115),
    "BIKE-L5": (5122, 5154),
    "Kyber512": (800, 768),
    "Kyber768": (1184, 1088),
    "Kyber1024": (1568, 1568),
    "FrodoKEM-640": (9616, 9720),
    "FrodoKEM-976": (15632, 15744),
    "FrodoKEM-1344": (21520, 21632),
    "HQC-128": (2249, 4433),
    "HQC-192": (4522, 8978),
    "HQC-256": (7245, 14421),
    "SNTRUP761": (1158, 1039)
}

classical_algorithms = {
    # Classical Algorithms
    "RSA-2048": 256,
    "RSA-3072": 384,
    "RSA-4096": 512,
    "DSA-2048": 256,
    "DSA-3072": 384,
    "ECDSA-P256": 64,
    "ECDSA-P384": 96,
    "ECDSA-P521": 132,
    "ECDH-P256": 65,
    "ECDH-P384": 97,
    "ECDH-P521": 133,
    "X25519": 32,
    "X448": 56,
    "DH-2048": 256,
    "DH-3072": 384,
    "DH-4096": 512,
    "DH-8192": 1024
}

# Specify the path to the capture file
capture_file = ''

ip_address = "10.110.218.173"

# TLS handshake type values
TLS_HANDSHAKE_TYPE_CLIENT_HELLO = 1
TLS_HANDSHAKE_TYPE_SERVER_HELLO = 2
TLS_HANDSHAKE_TYPE_CLIENT_KEY_EXCHANGE = 16
TLS_HANDSHAKE_TYPE_SERVER_KEY_EXCHANGE = 12

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


def check_classical(packet_size):
    keys = ''
    #print(packet_size)
    for key, val in classical_algorithms.items():
        if val - 5 < packet_size < val + 5:
            keys = keys + ' ,' + key
        else:
            continue
    return keys


def check_quantum(packet_size, pk=1):
    #print(packet_size)
    keys = ''
    if pk:
        for key, (val1, val2) in pq_algorithms.items():
            if val1 - 5 < packet_size < val1 + 5:
                keys = keys + ' ,' + key
            else:
                continue
    else:
        for key, (val1, val2) in pq_algorithms.items():
            if val2 - 5 < packet_size < val2 + 5:
                keys = keys + ' ,' + key
            else:
                continue
    return keys


def check_hybrid(packet_size, pk=1):
    keys = ''

    if pk:
        for key_c, val in classical_algorithms.items():
            for key_q, (val1, val2) in pq_algorithms.items():
                if val1 - 5 < (packet_size - val) < val1 + 5:
                    #print("hello" + packet_size)
                    keys = keys + ',' + key_c + ' + ' + key_q
                    continue
                else:
                    continue
    else:
        for key_c, val in classical_algorithms.items():
            for key_q, (val1, val2) in pq_algorithms.items():
                if val2 - 5 < (packet_size - val) < val2 + 5:
                    keys = keys + ',' + key_c + ' + ' + key_q
                else:
                    continue
    return keys


def find_common_keys(*strings):
    # Split each string by ';' and convert to sets
    sets_of_elements = [set(s.split(';')) for s in strings]

    # Find the intersection of all sets
    common_elements = set.intersection(*sets_of_elements)

    return common_elements


def classify_packets(init_packet, reply_packet, tls=1):
    if tls:
        version_attribute = 'handshake_extensions_supported_version'
        key_length_attribute = 'handshake_extensions_key_share_key_exchange_length'
        if init_packet is not None:
            tls_layer_init = init_packet['tls']
        tls_layer_reply = reply_packet['tls']
    else:
        version_attribute = 'tls_handshake_extensions_supported_version'
        key_length_attribute = 'tls_handshake_extensions_key_share_key_exchange_length'
        if init_packet is not None:
            tls_layer_init = init_packet['quic']
        tls_layer_reply = reply_packet['quic']

    key_classical_init = ''
    key_classical_reply = ''
    key_quantum_init = ''
    key_quantum_reply = ''
    key_hybrid_init = ''
    key_hybrid_reply = ''

    '''
    print(init_packet.length)

    for field in tls_layer_init.field_names:
        print(f"{field}: {getattr(tls_layer_init, field)}")

    for field in tls_layer_reply.field_names:
        print(f"{field}: {getattr(tls_layer_reply, field)}")    
    print(reply_packet.length)    
    '''
    if init_packet is not None:
        if hasattr(tls_layer_init, version_attribute) and getattr(tls_layer_init, version_attribute) == '0x0304':
            if hasattr(tls_layer_init, key_length_attribute):
                key_classical_init = check_classical(
                    int(getattr(tls_layer_init, key_length_attribute)))
                key_quantum_init = check_quantum(
                    int(getattr(tls_layer_init, key_length_attribute)))
                key_hybrid_init = check_hybrid(
                    int(getattr(tls_layer_init, key_length_attribute)))
        else:
            if hasattr(tls_layer_init, 'handshake_client_point_len'):
                key_classical_init = check_classical(
                    int(getattr(tls_layer_init, 'handshake_client_point_len')))
                key_quantum_init = check_quantum(
                    int(getattr(tls_layer_init, 'handshake_client_point_len')))
                key_hybrid_init = check_hybrid(
                    int(getattr(tls_layer_init, 'handshake_client_point_len')))

    if hasattr(tls_layer_reply, version_attribute) and getattr(tls_layer_reply, version_attribute) == '0x0304':
        if hasattr(tls_layer_reply, key_length_attribute):
            key_classical_reply = check_classical(
                int(getattr(tls_layer_reply, key_length_attribute)))
            key_quantum_reply = check_quantum(
                int(getattr(tls_layer_reply, key_length_attribute)), 0)
            key_hybrid_reply = check_hybrid(
                int(getattr(tls_layer_reply, key_length_attribute)), 0)
    else:
        if hasattr(tls_layer_reply, 'handshake_server_point_len'):
            key_classical_reply = check_classical(
                int(getattr(tls_layer_reply, 'handshake_server_point_len')))
            key_quantum_reply = check_quantum(
                int(getattr(tls_layer_reply, 'handshake_server_point_len')), 0)
            key_hybrid_reply = check_hybrid(
                int(getattr(tls_layer_reply, 'handshake_server_point_len')), 0)

    priority = 0

    if init_packet is not None and key_classical_init != '' and key_classical_reply != '' and key_classical_init == key_classical_reply:
        print(key_classical_init,end="")
        priority = 1
    elif key_classical_reply != '':
        print(key_classical_reply,end="")
    elif init_packet is not None and key_classical_init != '':
        print(key_classical_init,end="")

    if priority == 0:
        if init_packet is not None and key_quantum_init != '' and key_quantum_reply != '' and key_quantum_init == key_quantum_reply:
            print(key_quantum_init,end="")
            priority = 1
        elif key_quantum_reply != '':
            print(key_quantum_reply,end="")
        elif init_packet is not None and key_quantum_init != '':
            print(key_quantum_init,end="")

    if priority == 0:
        if init_packet is not None and key_hybrid_init != '' and key_hybrid_reply != '' and key_hybrid_init == key_hybrid_reply:
            print(key_hybrid_init,end="")
        elif key_hybrid_reply != '':
            print(key_hybrid_reply,end="")
        elif init_packet is not None and key_hybrid_init != '':
            print(key_hybrid_init,end="")

    #print('------------------------')
    if not (key_classical_init == '' and key_classical_reply == '' and key_quantum_init == '' and key_quantum_reply == '' and key_hybrid_init == '' and key_hybrid_reply == ''):
        print("")


if __name__ == "__main__":
    capture = pyshark.FileCapture(
        capture_file,
        display_filter=(
            #f'ip.src == {ip_address} and ip.dst == {ip_address} and '
            f'(tls.handshake.type == {TLS_HANDSHAKE_TYPE_CLIENT_HELLO} or '
            f'tls.handshake.type == {TLS_HANDSHAKE_TYPE_SERVER_HELLO} or '
            f'tls.handshake.type == {TLS_HANDSHAKE_TYPE_CLIENT_KEY_EXCHANGE} or '
            f'tls.handshake.type == {TLS_HANDSHAKE_TYPE_SERVER_KEY_EXCHANGE})'
        )
    )

    # Iterate over each packet in the capture file
    for packet in capture:
        if not hasattr(packet, 'tls'):
            continue
        tls_layer = packet['tls']
        if hasattr(packet, 'tcp'):
            transport_layer = packet['tcp']
        elif hasattr(packet, 'udp'):
            transport_layer = packet['udp']
        tls_version = packet.tls.record_version  # Check TLS version3e
        handshake_type = int(packet.tls.handshake_type)
        if 'ipv6' in packet:
            ipv6_layer = packet.ipv6
            ip_src = ipv6_layer.src
            ip_dst = ipv6_layer.dst
        elif 'ip' in packet:
            ipv4_layer = packet.ip
            ip_src = ipv4_layer.src
            ip_dst = ipv4_layer.dst
        if handshake_type == 2:
            if hasattr(tls_layer, 'handshake_extensions_supported_version'):
                supported_version = getattr(tls_layer, 'handshake_extensions_supported_version')
                if supported_version == '0x0304':
                    #print("TLS Layer Fields:")
                    #for field in tls_layer.field_names:
                    #    print(f"{field}: {getattr(tls_layer, field)}")
                    # have a check for algorithm type
                    #print(packet.length)
                    if hasattr(tls_layer, 'handshake_extensions_key_share_group'):
                        key_share_group = tls_layer.handshake_extensions_key_share_group
                        group_name = key_share_groups.get(int(key_share_group), 'Unknown')
                        print("Kex Algo: " + group_name)
                    #print("TLS 1.3")
                    elements = [ip_dst, ip_src, transport_layer.dstport, transport_layer.srcport, tls_layer.handshake_session_id]
                    key = simple_hash(elements)
                    update_value2(key, packet)
        elif handshake_type == 1:
            #print(packet.length)
            #for field in tls_layer.field_names:
            #   print(f"{field}: {getattr(tls_layer, field)}")
            if hasattr(tls_layer, 'handshake_extensions_supported_version'):
                supported_version = getattr(tls_layer, 'handshake_extensions_supported_version')
                if supported_version == '0x0304':
                    elements = [ip_src, ip_dst, transport_layer.srcport, transport_layer.dstport, tls_layer.handshake_session_id]
                    key = simple_hash(elements)
                    update_value1(key, packet)
            #for field in transport_layer.field_names:
            #    print(f"{field}: {getattr(transport_layer, field)}")

            #supported_version = getattr(tls_layer, 'handshake_extensions_supported_version')

        elif handshake_type == 12:
            if hasattr(tls_layer, 'handshake_server_named_curve'):
                #for field in tls_layer.field_names:
                #    print(f"{field}: {getattr(tls_layer, field)}")
                key_share_group = int(tls_layer.handshake_server_named_curve, 16)
                group_name = key_share_groups.get(key_share_group, 'Unknown')
                print("Kex Algo: " + group_name)
                if hasattr(transport_layer, 'stream'):
                    elements = [ip_src, ip_dst, transport_layer.srcport, transport_layer.dstport, transport_layer.stream]
                else:
                    elements = [ip_src, ip_dst, transport_layer.srcport, transport_layer.dstport]
                key = simple_hash(elements)
                update_value2(key, packet)

        elif handshake_type == 16:
            if hasattr(transport_layer, 'stream'):
                elements = [ip_src, ip_dst, transport_layer.srcport, transport_layer.dstport, transport_layer.stream]
            else:
                elements = [ip_src, ip_dst, transport_layer.srcport, transport_layer.dstport]
            key = simple_hash(elements)
            update_value1(key, packet)
            #print('TLS 1.2')
            # haver a check for the algorithm

    #complete_packets = {k: v for k, v in packets.items() if all(value is not None for value in v)}
    #print(complete_packets)
    # Close the capture file
    capture.close()

    for k, (v1, v2) in packets.items():
        if v2 is not None:
            classify_packets(v1, v2)

    #for key, (value1, value2) in complete_packets.items():
    #    classify_packets(value1, value2, 1)
