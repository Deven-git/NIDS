"""
app/utils/pcap_converter.py

Converts an uploaded .pcap/.pcapng file into flow-level features
matching (as closely as possible) the CICIDS2017 feature naming
convention our model was trained on, using the `cicflowmeter`
package (which replicates the original CICFlowMeter tool's
flow-reconstruction methodology).

IMPORTANT: cicflowmeter's output column naming may vary slightly by
version. This mapping is best-effort based on the package's typical
snake_case convention. After first real use, verify actual output
columns (printed to console) against this mapping and adjust as
needed - this is expected, not a sign something is broken.
"""

import subprocess
import os
import pandas as pd

# Best-effort mapping: cicflowmeter's typical column name -> our
# training feature name (from CICIDS2017 / feature_list.json)
COLUMN_MAPPING = {
    "dst_port": "Destination Port",
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Total Fwd Packets",
    "totlen_fwd_pkts": "Total Length of Fwd Packets",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "flow_iat_std": "Flow IAT Std",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Total",
    "bwd_iat_min": "Bwd IAT Min",
    "fwd_header_len": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "bwd_pkts_s": "Bwd Packets/s",
    "pkt_len_min": "Min Packet Length",
    "pkt_len_max": "Max Packet Length",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_var": "Packet Length Variance",
    "init_fwd_win_byts": "Init_Win_bytes_forward",
    "init_bwd_win_byts": "Init_Win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
}


def convert_pcap_to_csv(pcap_path, output_csv_path):
    """
    Runs cicflowmeter on the given pcap file, producing a CSV of
    flow-level features at output_csv_path.
    """
    result = subprocess.run(
        ["cicflowmeter", "-f", pcap_path, "-c", output_csv_path],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"cicflowmeter failed (exit code {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    if not os.path.exists(output_csv_path):
        raise RuntimeError(
            "cicflowmeter reported success but no output CSV was created. "
            f"stdout: {result.stdout}"
        )
    return output_csv_path


def map_columns_to_training_features(df, required_features):
    """
    Renames cicflowmeter's output columns to match our training
    feature names, using COLUMN_MAPPING. Returns:
      - the renamed dataframe
      - list of required features that WERE successfully matched
      - list of required features that were NOT found (missing)
    """
    # Normalize existing columns: lowercase, strip whitespace
    df.columns = [c.strip() for c in df.columns]

    # Build a normalized lookup of this file's actual columns
    normalized_to_actual = {c.strip().lower(): c for c in df.columns}

    rename_map = {}
    for source_col, target_col in COLUMN_MAPPING.items():
        if source_col in normalized_to_actual:
            rename_map[normalized_to_actual[source_col]] = target_col

    df = df.rename(columns=rename_map)

    matched = [f for f in required_features if f in df.columns]
    missing = [f for f in required_features if f not in df.columns]

    print(f"PCAP conversion: matched {len(matched)}/{len(required_features)} required features")
    if missing:
        print(f"Missing/unmatched features: {missing}")
        print(f"Actual columns in converted file: {list(df.columns)}")

    return df, matched, missing