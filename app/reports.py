"""PDF research reports generated from the selected asset's actual available data."""
from __future__ import annotations

from io import BytesIO
from math import ceil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether,
)

NAVY = colors.HexColor("#101827")
GOLD = colors.HexColor("#E7B85C")
INK = colors.HexColor("#233044")
MUTED = colors.HexColor("#64748B")
PALE = colors.HexColor("#F1F5F9")
GREEN = colors.HexColor("#16845B")
RED = colors.HexColor("#C2413A")


def _fmt_money(value, currency="USD"):
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{currency} {float(value):,.4f}" if abs(float(value)) < 1 else f"{currency} {float(value):,.2f}"


def _fmt_pct(value):
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value) * 100:,.2f}%"


def _fmt_num(value, digits=2):
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _build_price_chart(df: pd.DataFrame):
    """Build a compact vector chart; downsample dense histories for legibility."""
    close = df["close"].astype(float).reset_index(drop=True)
    dates = pd.to_datetime(df["date"]).reset_index(drop=True)
    n = len(close)
    step = max(1, ceil(n / 90))
    sampled = list(range(0, n, step))
    if sampled[-1] != n - 1:
        sampled.append(n - 1)
    values = [float(close.iloc[i]) for i in sampled]
    labels = [dates.iloc[i].strftime("%b %y") for i in sampled]

    drawing = Drawing(480, 185)
    chart = HorizontalLineChart()
    chart.x = 38
    chart.y = 28
    chart.width = 425
    chart.height = 130
    chart.data = [values]
    label_step = max(1, len(labels) // 8)
    chart.categoryAxis.categoryNames = [label if i % label_step == 0 or i == len(labels) - 1 else "" for i, label in enumerate(labels)]
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 6
    chart.categoryAxis.labels.angle = 35
    chart.categoryAxis.labels.boxAnchor = "ne"
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.labelTextFormat = "%.2f"
    chart.valueAxis.forceZero = False
    low, high = min(values), max(values)
    pad = (high - low) * 0.08 if high > low else max(abs(high) * 0.02, 1)
    chart.valueAxis.valueMin = low - pad
    chart.valueAxis.valueMax = high + pad
    chart.valueAxis.valueStep = max((high - low + 2 * pad) / 4, 0.000001)
    chart.lines[0].strokeColor = GOLD
    chart.lines[0].strokeWidth = 2
    chart.categoryAxis.strokeColor = colors.HexColor("#CBD5E1")
    chart.valueAxis.strokeColor = colors.HexColor("#CBD5E1")
    chart.categoryAxis.gridStrokeColor = colors.HexColor("#E2E8F0")
    chart.valueAxis.gridStrokeColor = colors.HexColor("#E2E8F0")
    drawing.add(chart)
    return drawing


def _page_decor(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 13 * mm, width, 13 * mm, stroke=0, fill=1)
    canvas.setFillColor(GOLD)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(18 * mm, height - 8.5 * mm, "CRYPTOSENSE AI  /  INVESTOR INTELLIGENCE")
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(18 * mm, 10 * mm, "Research and educational use only · Not investment advice")
    canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def generate_report_pdf(df: pd.DataFrame, quality: dict, meta: dict, symbol: str, period_label: str) -> bytes:
    """Create a PDF report with data provenance, historical metrics, chart and recent rows."""
    if df is None or df.empty:
        raise ValueError("No observations are available for this report period.")

    d = df.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(d["close"], errors="coerce").dropna()
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    volume = pd.to_numeric(d.get("volume", pd.Series(dtype=float)), errors="coerce").dropna()
    latest = float(close.iloc[-1])
    first = float(close.iloc[0])
    period_return = latest / first - 1 if first else np.nan
    peak = close.cummax()
    max_drawdown = float((close / peak - 1).min()) if len(close) else np.nan
    annual_vol = float(returns.std(ddof=1) * np.sqrt(365)) if len(returns) > 1 else np.nan
    daily_mean = float(returns.mean()) if len(returns) else np.nan
    daily_std = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    best = float(returns.max()) if len(returns) else np.nan
    worst = float(returns.min()) if len(returns) else np.nan
    up_day_share = float((returns > 0).mean()) if len(returns) else np.nan
    currency = str(meta.get("currency", "USD"))
    start_date = pd.to_datetime(d.date.iloc[0]).strftime("%d %b %Y")
    end_date = pd.to_datetime(d.date.iloc[-1]).strftime("%d %b %Y")
    generated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=23,
                              leading=27, textColor=NAVY, alignment=TA_LEFT, spaceAfter=3))
    styles.add(ParagraphStyle(name="SubTitleCS", parent=styles["Normal"], fontSize=9, leading=13,
                              textColor=MUTED, spaceAfter=9))
    styles.add(ParagraphStyle(name="SectionCS", parent=styles["Heading2"], fontName="Helvetica-Bold",
                              fontSize=12, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyCS", parent=styles["BodyText"], fontSize=8.5, leading=12,
                              textColor=INK, spaceAfter=5))
    styles.add(ParagraphStyle(name="TinyCS", parent=styles["BodyText"], fontSize=7, leading=9,
                              textColor=MUTED, spaceAfter=3))
    styles.add(ParagraphStyle(name="CalloutCS", parent=styles["BodyText"], fontSize=8, leading=11,
                              textColor=INK, backColor=PALE, borderColor=colors.HexColor("#CBD5E1"),
                              borderWidth=0.5, borderPadding=7, spaceBefore=5, spaceAfter=7))

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
                            topMargin=21 * mm, bottomMargin=21 * mm,
                            title=f"CryptoSense AI {symbol} {period_label} report",
                            author="CryptoSense AI")
    story = [
        Paragraph(f"{symbol} Research Report", styles["ReportTitle"]),
        Paragraph(f"{period_label} · {start_date} to {end_date} · Generated {generated}", styles["SubTitleCS"]),
        Paragraph(
            f"This report analyses <b>{len(d):,} daily observations</b> for {symbol}. "
            f"It uses the historical data currently available to CryptoSense AI for this asset and period.",
            styles["BodyCS"]),
    ]

    provenance = [
        ["ASSET", symbol, "QUOTE CURRENCY", currency],
        ["DATA SOURCE", str(meta.get("source", "Unknown")), "OBSERVATIONS", f"{len(d):,}"],
        ["ACTUAL COVERAGE", f"{start_date} – {end_date}", "RETRIEVED", str(meta.get("retrieved_at", "Unknown"))],
    ]
    prov = Table(provenance, colWidths=[27 * mm, 63 * mm, 28 * mm, 56 * mm])
    prov.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE), ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (2, 0), (2, -1), MUTED), ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7), ("FONTSIZE", (1, 0), (1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.35, colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [prov, Paragraph("Historical performance snapshot", styles["SectionCS"])]

    metrics = [
        ("Latest close", _fmt_money(latest, currency)),
        ("Period return", _fmt_pct(period_return)),
        ("Period high", _fmt_money(float(close.max()), currency)),
        ("Period low", _fmt_money(float(close.min()), currency)),
        ("Annualized volatility", _fmt_pct(annual_vol)),
        ("Maximum drawdown", _fmt_pct(max_drawdown)),
        ("Average daily return", _fmt_pct(daily_mean)),
        ("Daily return standard deviation", _fmt_pct(daily_std)),
        ("Best daily return", _fmt_pct(best)),
        ("Worst daily return", _fmt_pct(worst)),
        ("Positive days", _fmt_pct(up_day_share)),
        ("Average daily volume", _fmt_num(float(volume.mean()), 0) if len(volume) else "n/a"),
    ]
    metric_rows = [["METRIC", "VALUE", "METRIC", "VALUE"]]
    for i in range(0, len(metrics), 2):
        a, b = metrics[i], metrics[i + 1]
        metric_rows.append([a[0], a[1], b[0], b[1]])
    mt = Table(metric_rows, colWidths=[43 * mm, 43 * mm, 43 * mm, 45 * mm], repeatRows=1)
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"), ("TEXTCOLOR", (0, 1), (0, -1), MUTED),
        ("TEXTCOLOR", (2, 1), (2, -1), MUTED), ("TEXTCOLOR", (1, 1), (1, -1), INK),
        ("TEXTCOLOR", (3, 1), (3, -1), INK), ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DCE3EA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [mt, Paragraph("Price history", styles["SectionCS"]), _build_price_chart(d)]
    story.append(Paragraph(
        "The chart plots daily closing prices. It is a historical view, not a forward price forecast. "
        "Annualized volatility uses the sample standard deviation of daily simple returns multiplied by √365.",
        styles["TinyCS"]))

    story.append(Paragraph("Data quality and interpretation", styles["SectionCS"]))
    quality_lines = [
        f"Original rows in selected window: {quality.get('original_rows', len(d)):,}.",
        f"Duplicate dates removed: {quality.get('duplicates_removed', 0):,}; invalid or missing closes removed: {quality.get('invalid_close_removed', 0):,}.",
        f"Missing calendar days: {quality.get('missing_day_count', 0):,}; OHLC inconsistencies flagged: {quality.get('ohlc_violations_flagged', 0):,}.",
        f"OHLC fields available: {'Yes' if quality.get('has_ohlc') else 'No / incomplete'}. Missing values were not filled with fabricated prices.",
    ]
    story.append(Paragraph("<br/>".join(quality_lines), styles["BodyCS"]))
    if meta.get("warning"):
        story.append(Paragraph(f"<b>Data availability notice:</b> {meta['warning']}", styles["CalloutCS"]))
    if len(returns) < 20:
        story.append(Paragraph(
            "This period contains fewer than 20 daily returns. Volatility and return-distribution statistics "
            "may be unstable or unavailable; consider a longer reporting period.", styles["CalloutCS"]))

    story.append(PageBreak())
    story.append(Paragraph("Recent daily observations", styles["SectionCS"]))
    recent = d.tail(10).iloc[::-1]
    rows = [["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
    for _, row in recent.iterrows():
        rows.append([
            pd.to_datetime(row.date).strftime("%d %b %Y"),
            _fmt_num(row.get("open"), 4), _fmt_num(row.get("high"), 4),
            _fmt_num(row.get("low"), 4), _fmt_num(row.get("close"), 4),
            _fmt_num(row.get("volume"), 0),
        ])
    recent_table = Table(rows, colWidths=[31 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm, 31 * mm], repeatRows=1)
    recent_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#DCE3EA")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(recent_table)

    story.append(Paragraph("Methodology and important limitations", styles["SectionCS"]))
    story.append(Paragraph(
        "This report describes the supplied historical daily OHLCV data for the selected asset. "
        "Period return is calculated from the first and last closing prices in the report window. "
        "Maximum drawdown is the largest peak-to-subsequent-trough decline in the closing-price series. "
        "Volatility is annualized using 365 trading days because cryptocurrency markets operate every day.", styles["BodyCS"]))
    story.append(Paragraph(
        "The report does not include circulating supply, market capitalization, tokenomics, news sentiment, "
        "on-chain activity, exchange order books, or a verified fundamental description of the asset because "
        "those data are not supplied by the current OHLCV data pipeline. A shorter actual coverage than the "
        "selected period means the provider or uploaded file did not contain the full requested history.", styles["BodyCS"]))
    story.append(Paragraph(
        "Educational and research use only. This document is not investment, trading, tax, or legal advice. "
        "Historical performance does not predict future results. Cryptoassets are volatile and may lose substantial "
        "or all of their value. Data providers can revise, delay, limit, or discontinue their feeds.", styles["CalloutCS"]))

    doc.build(story, onFirstPage=_page_decor, onLaterPages=_page_decor)
    return buf.getvalue()
