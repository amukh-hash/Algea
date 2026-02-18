"use client";

import { memo, useEffect, useMemo, useRef } from "react";
import { ColorType, createChart, LineSeries, Time } from "lightweight-charts";

export type LWPoint = { time: number; value: number };
export type SeriesSpec = { key: string; name?: string; data: LWPoint[]; color?: string; visible?: boolean };

type Props = {
  title: string;
  series: SeriesSpec[];
  height?: number;
  mode?: "compact" | "full";
  alignment?: "absolute" | "relative";
  normalize?: "raw" | "index100";
  annotations?: { time: number; label: string; kind: "info" | "warn" | "error" }[];
  onCrosshairMove?: (time: number | null) => void;
  externalCrosshairTime?: number | null;
};

function normalizeData(data: LWPoint[], normalize: "raw" | "index100") {
  if (normalize === "raw" || data.length === 0) return data;
  const base = data[0].value || 1;
  return data.map((p) => ({ ...p, value: (p.value / base) * 100 }));
}

function alignData(data: LWPoint[], mode: "absolute" | "relative") {
  if (mode === "absolute" || data.length === 0) return data;
  const t0 = data[0].time;
  return data.map((p) => ({ ...p, time: p.time - t0 }));
}

function TimeSeriesChartLWInner({ title, series, height = 260, mode = "full", alignment = "absolute", normalize = "raw", onCrosshairMove }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  const processed = useMemo(() => series.map((s, idx) => ({ ...s, color: s.color ?? `var(--colors-series${(idx % 6) + 1})`, data: alignData(normalizeData(s.data, normalize), alignment) })), [series, alignment, normalize]);

  useEffect(() => {
    if (!ref.current) return;
    const chart = createChart(ref.current, {
      width: ref.current.clientWidth,
      height,
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#acb6ca" },
      grid: { vertLines: { color: "rgba(172,182,202,.08)" }, horzLines: { color: "rgba(172,182,202,.08)" } },
      rightPriceScale: { visible: true },
      timeScale: { timeVisible: mode === "full" },
    });

    const lineSeries = processed.filter((s) => s.visible !== false).map((s) => {
      const l = chart.addSeries(LineSeries, { color: s.color, lineWidth: 2, title: s.name ?? s.key });
      l.setData(s.data.map((p) => ({ time: p.time as Time, value: p.value })));
      return l;
    });

    chart.subscribeCrosshairMove((param) => onCrosshairMove?.(typeof param.time === "number" ? param.time : null));
    chart.timeScale().fitContent();

    const ro = new ResizeObserver(([entry]) => chart.applyOptions({ width: entry.contentRect.width }));
    ro.observe(ref.current);

    return () => {
      ro.disconnect();
      lineSeries.forEach((s) => chart.removeSeries(s));
      chart.remove();
    };
  }, [processed, height, mode, onCrosshairMove]);

  const last = processed.map((s) => `${s.name ?? s.key}: ${s.data.at(-1)?.value?.toFixed(4) ?? "—"}`).join(" | ");

  return (
    <section className="rounded-lg border border-border bg-surface-1 p-3" aria-label={`Chart ${title}`}>
      <div className="mb-2 flex items-center justify-between"><h3 className="text-sm font-medium">{title}</h3><div className="flex flex-wrap gap-2 text-xs text-secondary">{processed.map((s) => <span key={s.key}>{s.name ?? s.key}</span>)}</div></div>
      <p className="sr-only">{last}</p>
      <div ref={ref} style={{ height }} />
    </section>
  );
}

export const TimeSeriesChartLW = memo(TimeSeriesChartLWInner);
