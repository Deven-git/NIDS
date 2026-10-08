"""
app/utils/flow_extractor.py

Driver-free .pcap/.pcapng -> flow-feature extractor.

Uses scapy's rdpcap() to parse the pcap file directly (pure Python
parsing of the file format - no Npcap/WinPcap/tcpdump required, since
we are reading an already-captured file, not sniffing live traffic).

Groups packets into flows (5-tuple: src_ip, dst_ip, src_port, dst_port,
protocol) and computes the same 30 features our model was trained on,
directly - no column-name guessing/mapping needed since we control the
computation ourselves.

LIMITATION (documented, not hidden): this is a simplified, from-scratch
reimplementation of flow feature computation, not a byte-for-byte replica
of the original CICFlowMeter Java tool's exact internal logic (e.g. its
precise flow-timeout/subflow rules). It computes the same named
statistics using standard, explainable definitions, which is sufficient
for demonstrating the system's real-time capability, but exact numeric
values may differ slightly from what CICFlowMeter itself would compute
for the same traffic.
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP

FLOW_TIMEOUT = 120  # seconds of inactivity before a flow is considered ended


def _get_flow_key(pkt):
    """
    Returns:
        canonical_key: used only for grouping both directions
        actual_flow_key: preserves the first observed direction
    """

    if IP not in pkt:
        return None

    src_ip = pkt[IP].src
    dst_ip = pkt[IP].dst
    proto = pkt[IP].proto

    if TCP in pkt:
        src_port = int(pkt[TCP].sport)
        dst_port = int(pkt[TCP].dport)

    elif UDP in pkt:
        src_port = int(pkt[UDP].sport)
        dst_port = int(pkt[UDP].dport)

    else:
        return None

    actual_flow_key = (
        src_ip,
        dst_ip,
        src_port,
        dst_port,
        proto,
    )

    # Direction-independent grouping key
    endpoints = (
        (src_ip, src_port),
        (dst_ip, dst_port),
    )

    canonical_key = tuple(
        item
        for endpoint in sorted(endpoints)
        for item in endpoint
    ) + (proto,)

    return canonical_key, actual_flow_key


def _extract_flows(packets):
    """
    Groups packets bidirectionally but preserves the first observed
    packet direction for feature computation.
    """

    grouped = defaultdict(list)
    flow_directions = {}

    for pkt in packets:

        result = _get_flow_key(pkt)

        if result is None:
            continue

        canonical_key, actual_flow_key = result

        # Preserve the first observed direction of the flow.
        if canonical_key not in flow_directions:
            flow_directions[canonical_key] = actual_flow_key

        flow_key = flow_directions[canonical_key]

        pkt_src_ip = pkt[IP].src
        pkt_dst_ip = pkt[IP].dst

        if TCP in pkt:
            pkt_src_port = int(pkt[TCP].sport)
            pkt_dst_port = int(pkt[TCP].dport)

        elif UDP in pkt:
            pkt_src_port = int(pkt[UDP].sport)
            pkt_dst_port = int(pkt[UDP].dport)

        else:
            continue

        is_forward = (
            pkt_src_ip == flow_key[0]
            and pkt_dst_ip == flow_key[1]
            and pkt_src_port == flow_key[2]
            and pkt_dst_port == flow_key[3]
        )

        flags = ""

        if TCP in pkt:
            flags = str(pkt[TCP].flags)

        grouped[canonical_key].append({
            "time": float(pkt.time),
            "length": len(pkt),
            "forward": is_forward,
            "header_len": (
                pkt[IP].ihl * 4
                if IP in pkt
                else 0
            ),
            "win": (
                pkt[TCP].window
                if TCP in pkt
                else 0
            ),
            "flags": flags,
        })

    # Convert back to:
    # actual first-direction flow key -> packet list
    return {
        flow_directions[canonical_key]: packets
        for canonical_key, packets
        in grouped.items()
    }


def _safe_stats(values):
    """Returns (mean, std, max, min) safely, handling empty lists."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std()), float(arr.max()), float(arr.min())


def _compute_iat(timestamps):
    """Inter-arrival times between consecutive timestamps."""
    if len(timestamps) < 2:
        return []
    sorted_ts = sorted(timestamps)
    return [sorted_ts[i + 1] - sorted_ts[i] for i in range(len(sorted_ts) - 1)]


def compute_flow_features(flow_key, packets):
    """Computes the 30 required features for ONE flow's packet list."""
    src_ip, dst_ip, src_port, dst_port, proto = flow_key

    fwd_pkts = [p for p in packets if p["forward"]]
    bwd_pkts = [p for p in packets if not p["forward"]]

    all_times = [p["time"] for p in packets]
    fwd_times = [p["time"] for p in fwd_pkts]

    duration = (max(all_times) - min(all_times)) if len(all_times) > 1 else 0.0
    duration_us = duration * 1_000_000  # CICIDS2017 uses microseconds

    fwd_lengths = [p["length"] for p in fwd_pkts]
    bwd_lengths = [p["length"] for p in bwd_pkts]
    all_lengths = [p["length"] for p in packets]

    total_fwd_bytes = sum(fwd_lengths)
    fwd_mean, fwd_std, fwd_max, fwd_min = _safe_stats(fwd_lengths)
    bwd_mean, bwd_std, bwd_max, bwd_min = _safe_stats(bwd_lengths)
    pkt_mean, pkt_std, pkt_max, pkt_min = _safe_stats(all_lengths)
    pkt_var = pkt_std ** 2

    flow_iat = _compute_iat(all_times)
    fwd_iat = _compute_iat(fwd_times)
    bwd_iat = _compute_iat([p["time"] for p in bwd_pkts])

    fiat_mean, fiat_std, fiat_max, fiat_min = _safe_stats(flow_iat)
    fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _safe_stats(fwd_iat)
    bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _safe_stats(bwd_iat)

    flow_bytes_s = (sum(all_lengths) / duration) if duration > 0 else 0.0
    flow_pkts_s = (len(packets) / duration) if duration > 0 else 0.0
    bwd_pkts_s = (len(bwd_pkts) / duration) if duration > 0 else 0.0

    fwd_header_len = sum(p["header_len"] for p in fwd_pkts)
    bwd_header_len = sum(p["header_len"] for p in bwd_pkts)

    init_win_fwd = fwd_pkts[0]["win"] if fwd_pkts else 0
    init_win_bwd = bwd_pkts[0]["win"] if bwd_pkts else 0

    psh_count = sum(1 for p in packets if "P" in p["flags"])

    act_data_pkt_fwd = sum(1 for p in fwd_pkts if p["length"] > p["header_len"])
    min_seg_size_fwd = min([p["header_len"] for p in fwd_pkts], default=0)

    active_times = flow_iat if flow_iat else [0.0]
    active_max = max(active_times) if active_times else 0.0

    return {
        "Destination Port": dst_port,
        "Fwd Packet Length Mean": fwd_mean,
        "Total Length of Fwd Packets": total_fwd_bytes,
        "Fwd Packet Length Max": fwd_max,
        "Flow Duration": duration_us,
        "Flow IAT Max": fiat_max * 1_000_000,
        "Bwd Header Length": bwd_header_len,
        "Init_Win_bytes_backward": init_win_bwd,
        "Packet Length Mean": pkt_mean,
        "Packet Length Variance": pkt_var,
        "Bwd Packets/s": bwd_pkts_s,
        "Init_Win_bytes_forward": init_win_fwd,
        "Total Fwd Packets": len(fwd_pkts),
        "Bwd Packet Length Min": bwd_min,
        "Bwd Packet Length Max": bwd_max,
        "Max Packet Length": pkt_max,
        "Fwd IAT Std": fwd_iat_std * 1_000_000,
        "act_data_pkt_fwd": act_data_pkt_fwd,
        "Fwd Header Length": fwd_header_len,
        "Fwd IAT Mean": fwd_iat_mean * 1_000_000,
        "min_seg_size_forward": min_seg_size_fwd,
        "Flow IAT Std": fiat_std * 1_000_000,
        "Flow IAT Mean": fiat_mean * 1_000_000,
        "Flow Packets/s": flow_pkts_s,
        "PSH Flag Count": psh_count,
        "Min Packet Length": pkt_min,
        "Fwd Packet Length Min": fwd_min,
        "Flow Bytes/s": flow_bytes_s,
        "Fwd IAT Min": fwd_iat_min * 1_000_000,
        "Active Max": active_max * 1_000_000,
        # Extra context columns (not model features, useful for display)
        "Src_IP": src_ip,
        "Dst_IP": dst_ip,
        "Src_Port": src_port,
        "Protocol": proto,
    }


def extract_features_from_pcap(pcap_path):
    """
    Main entry point: reads a pcap file and returns a DataFrame with
    one row per flow, containing all required model features plus
    IP/port context columns.
    """
    packets = rdpcap(pcap_path)
    flows = _extract_flows(packets)

    rows = []
    for flow_key, pkt_list in flows.items():
        if len(pkt_list) < 1:
            continue
        rows.append(compute_flow_features(flow_key, pkt_list))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)