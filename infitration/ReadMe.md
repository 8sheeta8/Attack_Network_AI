# infitration에 대한 npy 및 pacp 제작 코드
make_pacp.py는 코랩에서 사용하는 pacp제작 코드입니다. 
선수 코드로는 ``!pip install scapy``를 입력해야하므로, 해당 작업 수행 후 실행해주시길 바랍니다.

더불어 서버 환경에 맞는 ip 및 port 수정이 필요하므로 이를 확인하여 실행해주시길 바랍니다.

### npy의 정보
npy 파일 내부에는 raw형태의 데이터가 기록되어 있습니다.
그 정보는 payload에 대한 내용이므로 raw단위(60자 내외)로 읽은 뒤, 그 값을 생성한 것이기에 그에 맞춰 생성하면 됩니다.
