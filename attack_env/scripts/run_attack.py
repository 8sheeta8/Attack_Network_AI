import sys
from scapy.all import *

if len(sys.argv) != 2:
    print("Usage: python3 run_attack.py <pcap_file>")
    sys.exit(1)

pcap_path = sys.argv[1]
packets = rdpcap(pcap_path)

print(f"[INFO] Sending packets from: {pcap_path}")
sendp(packets, iface="eth0", verbose=True)
