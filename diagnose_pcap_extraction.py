"""
diagnose_pcap_extraction.py

Inspects the flow features our extractor produced for a given pcap
file, to sanity-check whether the numbers look like what we'd expect
for an attack capture (many packets per flow, unusual port patterns,
etc.) or genuinely look like normal short connections.

Usage: python diagnose_pcap_extraction.py path/to/file.pcap
"""

import sys
import pandas as pd

sys.path.insert(0, ".")
from app.utils.flow_extractor import extract_features_from_pcap

if len(sys.argv) < 2:
    print("Usage: python diagnose_pcap_extraction.py path/to/file.pcap")
    sys.exit(1)

pcap_path = sys.argv[1]
df = extract_features_from_pcap(pcap_path)

print(f"Total flows extracted: {len(df)}")
print("\n--- Key feature statistics ---")
print(df[[
    "Total Fwd Packets", "Total Length of Fwd Packets",
    "Flow Duration", "Destination Port", "Flow Bytes/s", "Flow Packets/s",
]].describe())

print("\n--- Destination port distribution (top 10) ---")
print(df["Destination Port"].value_counts().head(10))

print("\n--- Sample of 5 flows (raw values) ---")
print(df[[
    "Src_IP", "Dst_IP", "Destination Port", "Total Fwd Packets",
    "Total Length of Fwd Packets", "Flow Duration", "Flow Bytes/s",
]].head(10).to_string())