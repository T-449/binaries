import hashlib

import pyshark

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


# Read the pcap file
capture = pyshark.FileCapture('')

ecdh_reply_packets = []
alt = 0
seen_init = 0

init_packet_size = [338, 594, 1106, 146, 218, 178, 114, 1274]
reply_packet_size = [814, 1070, 1514, 622, 694, 654, 590, 1514]
algo_name = ["diffie-hellman-group14-sha256", "diffie-hellman-group16-sha512",
             "diffie-hellman-group18-sha512/diffie-hellman-group-exchange-sha256",
             "ecdh-sha2-nistp256", "ecdh-sha2-nistp521", "ecdh-sha2-nistp384", "curve25519-sha256",
             "sntrup761x25519-sha512"]


def check_classical(packet_size):
    keys = ''
    #print(packet_size)
    for key, val in classical_algorithms.items():
        if val - 3 < packet_size < val + 3:
            keys = keys + ' /' + key
        else:
            continue
    return keys


def check_quantum(packet_size, pk=1):
    #print(packet_size)
    keys = ''
    if pk:
        for key, (val1, val2) in pq_algorithms.items():
            if val1 - 3 < packet_size < val1 + 3:
                keys = keys + ' /' + key
            else:
                continue
    else:
        for key, (val1, val2) in pq_algorithms.items():
            if val2 - 3 < packet_size < val2 + 3:
                keys = keys + ' /' + key
            else:
                continue
    return keys


def check_hybrid(packet_size, pk=1):
    keys = ''
    if pk:
        for key_c, val in classical_algorithms.items():
            for key_q, (val1, val2) in pq_algorithms.items():
                if val1 - 3 < (packet_size - val) < val1 + 3:
                    keys = keys + '/' + key_c + ' + ' + key_q
                    continue
                else:
                    continue
    else:
        for key_c, val in classical_algorithms.items():
            for key_q, (val1, val2) in pq_algorithms.items():
                if val2 - 3 < (packet_size - val) < val2 + 3:
                    keys = keys + '/' + key_c + ' + ' + key_q
                else:
                    continue
    return keys


def classify_packets(init_packet, reply_packet):
    if init_packet is not None:
        ssh_layer_init = init_packet['ssh']
    ssh_layer_reply = reply_packet['ssh']

    for field in ssh_layer_init.field_names:
        print(f"{field}: {getattr(ssh_layer_init, field)}")

    for field in ssh_layer_reply.field_names:
        print(f"{field}: {getattr(ssh_layer_reply, field)}")

    key_classical_init = None
    key_classical_reply = None
    key_quantum_init = None
    key_quantum_reply = None
    key_hybrid_init = None
    key_hybrid_reply = None

    if init_packet is not None:
        if hasattr(ssh_layer_init, 'ecdh_q_c_length'):
            key_classical_init = check_classical(int(getattr(ssh_layer_init, 'ecdh_q_c_length')))
            key_quantum_init = check_quantum(int(getattr(ssh_layer_init, 'ecdh_q_c_length')))
            key_hybrid_init = check_hybrid(int(getattr(ssh_layer_init, 'ecdh_q_c_length')))
        elif hasattr(ssh_layer_init, 'mpint_length'):
            key_classical_init = check_classical(int(getattr(ssh_layer_init, 'mpint_length')))
            key_quantum_init = check_quantum(int(getattr(ssh_layer_init, 'mpint_length')))
            key_hybrid_init = check_hybrid(int(getattr(ssh_layer_init, 'mpint_length')))

    if hasattr(ssh_layer_reply, 'ecdh_q_s_length'):
        key_classical_reply = check_classical(int(getattr(ssh_layer_reply, 'ecdh_q_s_length')))
        key_quantum_reply = (int(getattr(ssh_layer_reply, 'ecdh_q_s_length')), 0)
        key_hybrid_reply = check_hybrid(int(getattr(ssh_layer_reply, 'ecdh_q_s_length')), 0)
    elif hasattr(ssh_layer_reply, 'mpint_length'):
        key_classical_reply = check_classical(int(getattr(ssh_layer_reply, 'mpint_length')))
        key_quantum_reply = check_quantum(int(getattr(ssh_layer_reply, 'mpint_length')), 0)
        key_hybrid_reply = check_hybrid(int(getattr(ssh_layer_reply, 'mpint_length')), 0)

    if init_packet is not None:
        if key_classical_init != None and key_classical_reply != None and key_classical_init == key_classical_reply:
            print(key_classical_init)
        if key_quantum_init != None and key_quantum_reply != None and key_quantum_init == key_quantum_reply:
            print(key_quantum_init)
        if key_hybrid_init != None and key_hybrid_reply != None and key_hybrid_init == key_hybrid_reply:
            print(key_hybrid_init)
    else:
        if key_classical_reply is not None:
            print(key_classical_reply)
        if key_quantum_reply is not None:
            print(key_quantum_reply)
        if key_hybrid_reply is not None:
            print(key_hybrid_reply)


for packet in capture:
    if 'ipv6' in packet:
        ipv6_layer = packet.ipv6
        ip_src = ipv6_layer.src
        ip_dst = ipv6_layer.dst
    elif 'ip' in packet:
        ipv4_layer = packet.ip
        ip_src = ipv4_layer.src
        ip_dst = ipv4_layer.dst
    if 'SSH' in packet:
        try:
            tcp_layer = packet['tcp']
            # Get the message code field
            message_code_field = packet.ssh.get_field('message_code')

            # Check if the message code field is not None
            if message_code_field is not None:
                message_code = int(message_code_field)
                # Check for ECDH key exchange message codes
                if message_code == 30 and alt == 0 and seen_init == 0:  # SSH_MSG_KEX_ECDH_INIT
                    ecdh_init_packet = packet
                    seen_init = 1
                    elements = [ip_src, ip_dst, tcp_layer.srcport, tcp_layer.dstport, tcp_layer.stream]
                    key = simple_hash(elements)
                    update_value1(key, packet)
                    ssh_layer = packet['SSH']
                    #for field in ssh_layer.field_names:
                    #    print(f"{field}: {getattr(ssh_layer, field)}")
                elif message_code == 31 and alt == 0 and seen_init == 1:
                    ecdh_reply_packets.append(packet)
                    seen_init = 0
                    elements = [ip_dst, ip_src, tcp_layer.dstport, tcp_layer.srcport, tcp_layer.stream]
                    key = simple_hash(elements)
                    update_value2(key, packet)
                    ssh_layer = packet['SSH']
                    #for field in ssh_layer.field_names:
                    #   print(f"{field}: {getattr(ssh_layer, field)}") #ecdh_q_s_length: 133 for classical, mpint_length: 1071 quantum
                elif message_code == 32 and alt == 0 and seen_init == 0:  # ALT SSH_MSG_KEX_ECDH_INIT
                    ecdh_init_packet = packet
                    alt = 1
                    seen_init = 1
                    elements = [ip_src, ip_dst, tcp_layer.srcport, tcp_layer.dstport, tcp_layer.stream]
                    key = simple_hash(elements)
                    update_value1(key, packet)
                elif message_code == 33 and alt == 1 and seen_init == 1:
                    ecdh_reply_packets.append(packet)
                    alt = 0
                    seen_init = 0
                    elements = [ip_dst, ip_src, tcp_layer.dstport, tcp_layer.srcport, tcp_layer.stream]
                    key = simple_hash(elements)
                    update_value2(key, packet)
        except AttributeError:
            continue

#complete_packets = {k: v for k, v in packets.items() if all(value is not None for value in v)}
# print(complete_packets)
print(packets)
for key, (value1, value2) in packets.items():
    if value2 is not None:
        classify_packets(value1, value2)
    # classify_packet(value1, value2)

'''
# Print or process the found packets
print("ECDH Key Exchange Init Packets:")
for packet in ecdh_init_packets:
    ssh_layer_str = str(packet.ssh)
    # Check if 'method: ' is in the string
    if 'method' in ssh_layer_str:
        # Find the start index of 'method: '
        method_start = ssh_layer_str.find('method:') + len('method:')
        # Find the end of the line
        method_end = ssh_layer_str.find('\n', method_start)
        # Extract the algorithm name
        algo_name = ssh_layer_str[method_start:method_end].strip()
        print(algo_name)
    print(packet.length)

print("ECDH Key Exchange Reply Packets:")
for packet in ecdh_reply_packets:
    print(packet.length)
'''
