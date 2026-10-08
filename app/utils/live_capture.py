"""
app/utils/live_capture.py

Live packet capture between two specified IP addresses, using tshark
(Wireshark's CLI tool) rather than scapy directly - tshark manages its
own interaction with the Npcap driver via dumpcap.exe, which tends to
be more reliable on Windows than scapy's own driver detection (see
project notes on the scapy/Npcap issues encountered during development).

Requires: Wireshark installed (tshark.exe on PATH), and the capturing
process run with Administrator privileges (required for live packet
capture on Windows).
"""

import subprocess
import re


def list_interfaces():
    """
    Returns a list of (index, description) tuples for available
    capture interfaces, parsed from `tshark -D`.
    """
    result = subprocess.run(
        ["tshark", "-D"], capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not list interfaces: {result.stderr}")

    interfaces = []
    for line in result.stdout.strip().split("\n"):
        # Format: "1. \Device\NPF_{GUID} (Description)"
        match = re.match(r"(\d+)\.\s+(.+)", line)
        if match:
            idx, desc = match.groups()
            interfaces.append((idx, desc.strip()))
    return interfaces


def capture_between_ips(interface_index, ip1, ip2, duration, output_pcap_path):
    """
    Captures traffic between ip1 and ip2 on the given interface for
    `duration` seconds, saving to output_pcap_path.
    """
    capture_filter = f"host {ip1} and host {ip2}"

    cmd = [
        "tshark",
        "-i", str(interface_index),
        "-f", capture_filter,
        "-a", f"duration:{duration}",
        "-w", output_pcap_path,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=duration + 20
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"tshark capture failed (exit code {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}\n\n"
            f"Common causes: not running as Administrator, invalid interface "
            f"index, or no traffic matched the filter within the capture window."
        )

    return output_pcap_path