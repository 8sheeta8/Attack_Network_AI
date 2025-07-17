from scapy.all import *

# ▶ 일부분 잘라서 사용 (처음 64 바이트 정도 예시)
hex_str = "47371642440242ff4921430e37262b38316b7c4e2d5c186f3f7d333056053a49283eff1a522946504a37"
payload = bytes.fromhex(hex_str)

# ▶ TCP 페이로드로 구성 (Dst IP/Port는 예시용, 수정 가능)
pkt = IP(dst="192.168.0.99") / TCP(sport=12345, dport=80, flags="PA") / Raw(load=payload)

# ▶ pcap 파일로 저장
wrpcap("/content/drive/MyDrive/doyean/partial_attack_payload.pcap", [pkt])
