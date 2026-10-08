"""
app/utils/rate_heuristic.py

Behavioral detection layer alongside the per-flow ML classifier.

Detects:
    1. Port Scan
    2. Brute Force
    3. DoS
    4. DDoS

The CatBoost prediction remains the baseline. Behavioral rules override
the ML prediction when a strong cross-flow pattern is detected.
"""

import pandas as pd


# ============================================================
# DoS / DDoS
# ============================================================

DEFAULT_FLOW_COUNT_THRESHOLD = 30
DEFAULT_UNIQUE_SRC_THRESHOLD = 5


# ============================================================
# Port Scan
# ============================================================

DEFAULT_PORT_SCAN_UNIQUE_PORT_THRESHOLD = 5
DEFAULT_PORT_SCAN_MIN_FLOW_THRESHOLD = 5


# ============================================================
# Brute Force
# ============================================================

# FTP / SSH / SMTP / POP3 / IMAP are common authentication services.
BRUTE_FORCE_PORTS = {
    21,     # FTP
    22,     # SSH
    25,     # SMTP
    110,    # POP3
    143,    # IMAP
}

# Number of flows from the same source to the same service
# before we consider it suspicious.
DEFAULT_BRUTE_FORCE_FLOW_THRESHOLD = 5


def _is_tcp_series(series):
    """Return True for TCP rows."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    text = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return numeric.eq(6) | text.eq("TCP")


def _find_port_scan_pairs(
    df,
    unique_port_threshold=DEFAULT_PORT_SCAN_UNIQUE_PORT_THRESHOLD,
    min_flow_threshold=DEFAULT_PORT_SCAN_MIN_FLOW_THRESHOLD,
):
    """
    Detect vertical TCP port scans.

    Same source + same destination + many destination ports.
    """

    required = {
        "Src_IP",
        "Dst_IP",
        "Protocol",
        "Destination Port",
    }

    if not required.issubset(df.columns):
        return pd.DataFrame()

    work = df.copy()

    work = work[
        _is_tcp_series(
            work["Protocol"]
        )
    ].copy()

    if work.empty:
        return pd.DataFrame()

    work["Scan_Dst_Port"] = pd.to_numeric(
        work["Destination Port"],
        errors="coerce",
    )

    work = work.dropna(
        subset=["Scan_Dst_Port"]
    )

    if work.empty:
        return pd.DataFrame()

    stats = (
        work.groupby(
            ["Src_IP", "Dst_IP"],
            dropna=False,
        )
        .agg(
            flow_count=("Scan_Dst_Port", "size"),
            unique_dst_ports=("Scan_Dst_Port", "nunique"),
        )
    )

    flagged = stats[
        (stats["flow_count"] >= min_flow_threshold)
        &
        (
            stats["unique_dst_ports"]
            >= unique_port_threshold
        )
    ]

    return flagged


def _find_brute_force_pairs(
    df,
    flow_threshold=DEFAULT_BRUTE_FORCE_FLOW_THRESHOLD,
):
    """
    Detect repeated authentication-service connections.

    Same source + same destination + same authentication service
    + multiple TCP flows.

    This is a behavioral rule and is intended for the controlled
    validation environment.
    """

    required = {
        "Src_IP",
        "Dst_IP",
        "Protocol",
        "Destination Port",
    }

    if not required.issubset(df.columns):
        return pd.DataFrame()

    work = df.copy()

    # TCP only
    work = work[
        _is_tcp_series(
            work["Protocol"]
        )
    ].copy()

    if work.empty:
        return pd.DataFrame()

    # Convert destination port to numeric
    work["BF_Dst_Port"] = pd.to_numeric(
        work["Destination Port"],
        errors="coerce",
    )

    work = work[
        work["BF_Dst_Port"].isin(
            BRUTE_FORCE_PORTS
        )
    ].copy()

    if work.empty:
        return pd.DataFrame()

    stats = (
        work.groupby(
            [
                "Src_IP",
                "Dst_IP",
                "BF_Dst_Port",
            ],
            dropna=False,
        )
        .agg(
            authentication_flows=(
                "BF_Dst_Port",
                "size",
            ),
        )
    )

    flagged = stats[
        stats["authentication_flows"]
        >= flow_threshold
    ]

    return flagged


def apply_rate_heuristic(
    df,
    flow_count_threshold=DEFAULT_FLOW_COUNT_THRESHOLD,
    unique_src_threshold=DEFAULT_UNIQUE_SRC_THRESHOLD,
    port_scan_unique_port_threshold=DEFAULT_PORT_SCAN_UNIQUE_PORT_THRESHOLD,
    port_scan_min_flow_threshold=DEFAULT_PORT_SCAN_MIN_FLOW_THRESHOLD,
    brute_force_flow_threshold=DEFAULT_BRUTE_FORCE_FLOW_THRESHOLD,
):
    """
    Combine ML prediction with behavioral detection.

    Priority:

        1. Port Scan
        2. Brute Force
        3. DDoS
        4. DoS
        5. Original ML prediction
    """

    df = df.copy()

    if "Predicted_Label" not in df.columns:
        raise ValueError(
            "apply_rate_heuristic requires "
            "'Predicted_Label'."
        )

    df["Final_Label"] = (
        df["Predicted_Label"]
    )

    df["Heuristic_Detection"] = "ML"

    required_context = {
        "Src_IP",
        "Dst_IP",
        "Protocol",
        "Destination Port",
    }

    if not required_context.issubset(
        df.columns
    ):
        return df, False

    # ========================================================
    # 1. PORT SCAN
    # ========================================================

    port_scan_pairs = _find_port_scan_pairs(
        df,
        unique_port_threshold=(
            port_scan_unique_port_threshold
        ),
        min_flow_threshold=(
            port_scan_min_flow_threshold
        ),
    )

    if not port_scan_pairs.empty:

        scan_pairs = set(
            port_scan_pairs.index.tolist()
        )

        scan_mask = pd.Series(
            [
                (
                    row["Src_IP"],
                    row["Dst_IP"],
                ) in scan_pairs
                for _, row in df.iterrows()
            ],
            index=df.index,
        )

        df.loc[
            scan_mask,
            "Final_Label"
        ] = "Port Scan"

        df.loc[
            scan_mask,
            "Heuristic_Detection"
        ] = "Port Scan (Behavioral)"

    else:

        scan_mask = pd.Series(
            False,
            index=df.index,
        )

    # ========================================================
    # 2. BRUTE FORCE
    # ========================================================

    brute_force_pairs = (
        _find_brute_force_pairs(
            df,
            flow_threshold=(
                brute_force_flow_threshold
            ),
        )
    )

    if not brute_force_pairs.empty:

        brute_force_keys = set(
            brute_force_pairs.index.tolist()
        )

        brute_force_mask = pd.Series(
            [
                (
                    row["Src_IP"],
                    row["Dst_IP"],
                    pd.to_numeric(
                        row["Destination Port"],
                        errors="coerce",
                    ),
                ) in brute_force_keys
                for _, row in df.iterrows()
            ],
            index=df.index,
        )

        # Never overwrite a Port Scan classification.
        brute_force_mask &= ~scan_mask

        df.loc[
            brute_force_mask,
            "Final_Label"
        ] = "Brute Force"

        df.loc[
            brute_force_mask,
            "Heuristic_Detection"
        ] = "Brute Force (Behavioral)"

    else:

        brute_force_mask = pd.Series(
            False,
            index=df.index,
        )

    # ========================================================
    # 3. DoS / DDoS
    # ========================================================

    dest_stats = (
        df.groupby("Dst_IP")
        .agg(
            flow_count=(
                "Dst_IP",
                "size",
            ),
            unique_src_count=(
                "Src_IP",
                "nunique",
            ),
        )
    )

    flagged_destinations = (
        dest_stats[
            dest_stats["flow_count"]
            >= flow_count_threshold
        ]
    )

    rate_destination_mask = (
        df["Dst_IP"].isin(
            flagged_destinations.index
        )
        &
        ~scan_mask
        &
        ~brute_force_mask
    )

    if not flagged_destinations.empty:

        source_counts = (
            flagged_destinations[
                "unique_src_count"
            ]
            .to_dict()
        )

        unique_src_for_row = (
            df["Dst_IP"]
            .map(source_counts)
            .fillna(0)
        )

        ddos_mask = (
            rate_destination_mask
            &
            unique_src_for_row.ge(
                unique_src_threshold
            )
        )

        dos_mask = (
            rate_destination_mask
            &
            ~ddos_mask
        )

        df.loc[
            ddos_mask,
            "Final_Label"
        ] = "DDoS (Rate-Based)"

        df.loc[
            ddos_mask,
            "Heuristic_Detection"
        ] = "DDoS (Rate-Based)"

        df.loc[
            dos_mask,
            "Final_Label"
        ] = "DoS (Rate-Based)"

        df.loc[
            dos_mask,
            "Heuristic_Detection"
        ] = "DoS (Rate-Based)"

    triggered = bool(
        scan_mask.any()
        or brute_force_mask.any()
        or rate_destination_mask.any()
    )

    return df, triggered