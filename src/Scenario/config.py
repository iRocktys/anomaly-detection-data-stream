from pathlib import Path

DATA_DIR = Path(
    r"E:\CICDDoS2019\01-12\CSV"
)

OUTPUT_DIR = Path(
    r"C:\GitHub\anomaly-detection-data-stream\Estudos\window\output"
)

WINDOW = "1s"

ATTACK_RATIO_THRESHOLD = 0.50

DNS_SAMPLE_ROWS = 1000
DNS_SAMPLE_START_ROW = 0

ATTACK_INSERT_DURATION = "5min"

BENIGN_SOURCE_FILES = [
    "DrDoS_DNS.csv",
    "DrDoS_LDAP.csv",
    "DrDoS_MSSQL.csv",
    "DrDoS_NetBIOS.csv",
    "DrDoS_NTP.csv",
    "DrDoS_SNMP.csv",
    "DrDoS_SSDP.csv",
    "DrDoS_UDP.csv",
    "Syn.csv",
    "TFTP.csv",
    "UDPLag.csv",
]

ATTACK_FILES = {
    "DNS": "DrDoS_DNS.csv",
    "LDAP": "DrDoS_LDAP.csv",
    "SYN": "Syn.csv",
}

SCENARIOS = {
    "Consistency": [
        "DNS",
        "DNS",
        "DNS",
    ],
    "Generalization": [
        "DNS",
        "LDAP",
        "DNS",
    ],
    "Adaptation": [
        "DNS",
        "SYN",
        "DNS",
    ],
    "Recurrence": [
        "DNS",
        "SYN",
        "LDAP",
        "DNS",
        "SYN",
        "LDAP",
    ],
}