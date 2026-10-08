"""
app/routes/report.py
Generates a downloadable PDF summary report of the most recent
prediction run, using ReportLab. Uses Final_Label (ML prediction +
rate-based heuristic combined) where available.
"""

import os
import io
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Blueprint, session, current_app, send_file, redirect, url_for, flash
from flask_login import login_required

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

report_bp = Blueprint("report", __name__)


def build_pie_chart_image(class_counts):
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = list(class_counts.keys())
    values = list(class_counts.values())
    ax.pie(values, labels=labels, autopct="%1.1f%%", textprops={"fontsize": 8})
    ax.set_title("Attack Distribution")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


@report_bp.route("/download-report")
@login_required
def download_report():
    predictions_path = session.get("predictions_path")
    if not predictions_path or not os.path.exists(predictions_path):
        flash("No predictions available yet. Please upload and predict on a file first.", "warning")
        return redirect(url_for("upload.upload"))

    df = pd.read_csv(predictions_path, low_memory=False)
    label_col = "Final_Label" if "Final_Label" in df.columns else "Predicted_Label"

    total_rows = len(df)
    class_counts = df[label_col].value_counts().to_dict()
    benign_count = class_counts.get("Benign", 0)
    attack_count = total_rows - benign_count
    avg_confidence = round(float(df["Confidence"].mean()) * 100, 2)

    original_name = session.get("uploaded_original_name", "uploaded_file.csv")

    comparison_csv = os.path.join(current_app.config["REPORTS_DIR"], "model_comparison.csv")
    comparison_df = pd.read_csv(comparison_csv)
    best_row = comparison_df.sort_values("f1_macro", ascending=False).iloc[0]

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=18)
    elements = []

    elements.append(Paragraph("Network Intrusion Detection - Prediction Report", title_style))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]
    ))
    elements.append(Paragraph(f"Source file: {original_name}", styles["Normal"]))
    elements.append(Spacer(1, 0.5 * cm))

    summary_data = [
        ["Metric", "Value"],
        ["Total Traffic Analyzed", f"{total_rows:,}"],
        ["Benign Traffic", f"{benign_count:,}"],
        ["Attack Traffic", f"{attack_count:,}"],
        ["Attack Percentage", f"{round((attack_count / total_rows) * 100, 2)}%"],
        ["Average Prediction Confidence", f"{avg_confidence}%"],
        ["Model Used", "CatBoost (production model) + rate-based heuristic layer"],
    ]
    summary_table = Table(summary_data, colWidths=[8 * cm, 8 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f0f0f0")]),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.7 * cm))

    elements.append(Paragraph("Class Breakdown", styles["Heading2"]))
    class_data = [["Class", "Count"]] + [[k, f"{v:,}"] for k, v in class_counts.items()]
    class_table = Table(class_data, colWidths=[8 * cm, 8 * cm])
    class_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f0f0f0")]),
    ]))
    elements.append(class_table)
    elements.append(Spacer(1, 0.7 * cm))

    pie_buf = build_pie_chart_image(class_counts)
    elements.append(Image(pie_buf, width=10 * cm, height=10 * cm))
    elements.append(Spacer(1, 0.7 * cm))

    elements.append(Paragraph("Production Model Validated Performance", styles["Heading2"]))
    model_stats = [
        ["Metric", "Value"],
        ["Model", best_row["model_name"]],
        ["Test Accuracy", f"{best_row['accuracy'] * 100:.2f}%"],
        ["F1-Score (macro)", f"{best_row['f1_macro'] * 100:.2f}%"],
        ["ROC-AUC (macro)", f"{best_row['roc_auc_macro'] * 100:.2f}%"],
    ]
    model_table = Table(model_stats, colWidths=[8 * cm, 8 * cm])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    elements.append(model_table)

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"NIDS_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
    )