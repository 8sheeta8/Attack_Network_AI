import os
import subprocess

PCAP_DIR = "pcaps"

def get_latest_pcap():
    pcaps = [f for f in os.listdir(PCAP_DIR) if f.endswith(".pcap")]
    if not pcaps:
        print("[ERROR] No .pcap file found in pcaps/")
        exit(1)
    pcaps.sort(key=lambda f: os.path.getmtime(os.path.join(PCAP_DIR, f)))
    return os.path.join(PCAP_DIR, pcaps[-1])

if __name__ == "__main__":
    pcap_path = get_latest_pcap()

    print(f"[PIPELINE] Running for: {pcap_path}")

    subprocess.run(["bash", "scripts/reset_env.sh"], check=True)
    subprocess.run(["python3", "scripts/run_attack.py", pcap_path], check=True)
    subprocess.run(["python3", "model/flow_converter.py", pcap_path], check=True)
    subprocess.run(["python3", "scripts/run_model.py", pcap_path], check=True)

    print("[PIPELINE] Done.")
