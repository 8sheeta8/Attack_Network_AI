import sys
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP

def extract_features(pkt):
    proto = None
    if TCP in pkt:
        proto = 'TCP'
    elif UDP in pkt:
        proto = 'UDP'
    else:
        proto = 'OTHER'

    return {
        "src_ip": pkt[IP].src if IP in pkt else "",
        "dst_ip": pkt[IP].dst if IP in pkt else "",
        "proto": proto,
        "pkt_len": len(pkt)
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 flow_converter.py <pcap_file>")
        sys.exit(1)

    pcap_path = sys.argv[1]
    packets = rdpcap(pcap_path)

    flows = [extract_features(pkt) for pkt in packets if IP in pkt]
    df = pd.DataFrame(flows)
    df.to_csv("flow.csv", index=False)
    print("[INFO] flow.csv created.")
