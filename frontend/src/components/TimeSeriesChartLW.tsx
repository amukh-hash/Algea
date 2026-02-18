"use client";

import { useEffect, useMemo, useRef } from "react";
import { createChart, IChartApi, ISeriesApi, LineSeries, ColorType, CrosshairMode, Time, LineData } from "lightweight-charts";

export type LWPoint = { time: number; value: number };
export type SeriesSpec = { key: string; name?: string; data: LWPoint[]; color?: string; visible?: boolean };

const palette = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)", "var(--series-5)"];

export function TimeSeriesChartLW({
  title,
  series,
  height = 220,
  mode = "full",
  alignment = "absolute",
  normalize = "raw",
  annotations,
  onCrosshairMove,
  externalCrosshairTime,
}: {
  title: string;
  series: SeriesSpec[];
  height?: number;
  mode?: "compact" | "full";
  timeRange?: { start?: number; end?: number };
  alignment?: "absolute" | "relative";
  normalize?: "raw" | "index100";
  annotations?: { time: number; label: string; kind: "info" | "warn" | "error" }[];
  onCrosshairMove?: (time: number | null) => void;
  externalCrosshairTime?: number | null;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRefs = useRef<ISeriesApi<"Line">[]>([]);

  const normalized = useMemo(() => series.map((s) => {
    const start = s.data[0]?.value ?? 1;
    const data = s.data.map((p) => ({ time: alignment === "relative" ? p.time - (s.data[0]?.time ?? 0) : p.time, value: normalize === "index100" ? (p.value / start) * 100 : p.value }));
    return { ...s, data };
  }), [series, alignment, normalize]);

  useEffect(() => {
    if (!ref.current) return;
    const chart = createChart(ref.current, {
      width: ref.current.clientWidth,
      height,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "var(--color-text-secondary)" },
      crosshair: { mode: CrosshairMode.Magnet },
      grid: { vertLines: { visible: mode !== "compact", color: "rgba(148,163,184,0.08)" }, horzLines: { color: "rgba(148,163,184,0.08)" } },
      timeScale: { timeVisible: true },
    });
    chartRef.current = chart;
    const resize = new ResizeObserver(([entry]) => chart.applyOptions({ width: entry.contentRect.width }));
    resize.observe(ref.current);

    chart.subscribeCrosshairMove((param) => onCrosshairMove?.(param.time ? Number(param.time) : null));
    return () => { resize.disconnect(); chart.remove(); };
  }, [height, mode, onCrosshairMove]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    seriesRefs.current.forEach((s) => chart.removeSeries(s));
    seriesRefs.current = [];
    normalized.forEach((s, i) => {
      const ls = chart.addSeries(LineSeries, { color: s.color ?? palette[i % palette.length], lineWidth: 2, title: s.name ?? s.key });
      ls.setData(s.data as unknown as LineData[]);
      seriesRefs.current.push(ls);
    });
    annotations?.forEach((ann) => {
      const marker = chart.addSeries(LineSeries, { color: ann.kind === "error" ? "var(--color-error)" : ann.kind === "warn" ? "var(--color-warning)" : "var(--color-info)", lineWidth: 1 });
      marker.setData([{ time: ann.time as Time, value: normalized[0]?.data[0]?.value ?? 0 }, { time: ann.time as Time, value: normalized[0]?.data.at(-1)?.value ?? 0 }] as unknown as LineData[]);
    });
    chart.timeScale().fitContent();
  }, [normalized, annotations]);

  useEffect(() => {
    if (!externalCrosshairTime || !chartRef.current) return;
    chartRef.current.setCrosshairPosition(0, externalCrosshairTime as Time, seriesRefs.current[0]);
  }, [externalCrosshairTime]);

  return (
    <section className="rounded-md border border-border bg-surface-1 p-3" aria-label={`${title} chart`}>
      <div className="mb-2 text-xs text-secondary">{title}</div>
      <div ref={ref} style={{ height }} />
      <div className="sr-only" aria-live="polite">Latest values {normalized.map((s) => `${s.name ?? s.key}: ${s.data.at(-1)?.value ?? "none"}`).join(", ")}</div>
    </section>
  );
}
