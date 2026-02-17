"use client";

import { useEffect, useRef, useCallback, memo } from "react";
import {
    createChart,
    IChartApi,
    ISeriesApi,
    LineSeries,
    HistogramSeries,
    ColorType,
    CrosshairMode,
    LineData,
    HistogramData,
    Time,
} from "lightweight-charts";

export type LWPoint = { time: number; value: number };

export type SeriesSpec = {
    key: string;
    name?: string;
    type?: "line" | "histogram";
    data: LWPoint[];
    color?: string;
};

// Auto-generate pleasant colors for series
const PALETTE = [
    "#38bdf8", // sky-400
    "#22c55e", // green-500
    "#f97316", // orange-500
    "#a78bfa", // violet-400
    "#f43f5e", // rose-500
    "#facc15", // yellow-400
    "#06b6d4", // cyan-500
    "#ec4899", // pink-500
    "#14b8a6", // teal-500
    "#8b5cf6", // violet-500
];

function getColor(idx: number, explicit?: string): string {
    return explicit ?? PALETTE[idx % PALETTE.length];
}

interface Props {
    title: string;
    series: SeriesSpec[];
    height?: number;
    rightPriceScale?: boolean;
}

function TimeSeriesChartLWInner({ title, series, height = 220, rightPriceScale = true }: Props) {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const seriesRefs = useRef<ISeriesApi<any>[]>([]);

    // Initialize chart
    useEffect(() => {
        if (!containerRef.current) return;

        const chart = createChart(containerRef.current, {
            width: containerRef.current.clientWidth,
            height,
            layout: {
                background: { type: ColorType.Solid, color: "transparent" },
                textColor: "#94a3b8",
                fontSize: 11,
            },
            grid: {
                vertLines: { color: "rgba(148, 163, 184, 0.06)" },
                horzLines: { color: "rgba(148, 163, 184, 0.06)" },
            },
            crosshair: {
                mode: CrosshairMode.Magnet,
            },
            rightPriceScale: {
                visible: rightPriceScale,
                borderColor: "rgba(148, 163, 184, 0.15)",
            },
            leftPriceScale: {
                visible: false,
            },
            timeScale: {
                borderColor: "rgba(148, 163, 184, 0.15)",
                timeVisible: true,
            },
            handleScale: true,
            handleScroll: true,
        });

        chartRef.current = chart;

        // Handle resize
        const resizeObserver = new ResizeObserver((entries) => {
            const { width } = entries[0].contentRect;
            chart.applyOptions({ width });
        });
        resizeObserver.observe(containerRef.current);

        return () => {
            resizeObserver.disconnect();
            chart.remove();
            chartRef.current = null;
            seriesRefs.current = [];
        };
    }, [height, rightPriceScale]);

    // Update series data
    useEffect(() => {
        const chart = chartRef.current;
        if (!chart) return;

        // Remove old series
        seriesRefs.current.forEach((s) => {
            try {
                chart.removeSeries(s);
            } catch {
                // already removed
            }
        });
        seriesRefs.current = [];

        // Add new series
        series.forEach((spec, idx) => {
            const color = getColor(idx, spec.color);
            const sorted = [...spec.data].sort((a, b) => a.time - b.time);

            if (spec.type === "histogram") {
                const histSeries = chart.addSeries(HistogramSeries, {
                    color,
                    priceFormat: { type: "volume" },
                    title: spec.name ?? spec.key,
                });
                histSeries.setData(
                    sorted.map((p) => ({ time: p.time as Time, value: p.value, color })) as HistogramData[]
                );
                seriesRefs.current.push(histSeries);
            } else {
                const lineSeries = chart.addSeries(LineSeries, {
                    color,
                    lineWidth: 2,
                    title: spec.name ?? spec.key,
                    crosshairMarkerRadius: 4,
                    priceFormat: { type: "price", precision: 4, minMove: 0.0001 },
                });
                lineSeries.setData(
                    sorted.map((p) => ({ time: p.time as Time, value: p.value })) as LineData[]
                );
                seriesRefs.current.push(lineSeries);
            }
        });

        chart.timeScale().fitContent();
    }, [series]);

    return (
        <div className="rounded border border-slate-800 bg-slate-900/50 p-3">
            <div className="mb-2 flex items-center justify-between">
                <span className="text-xs font-medium text-slate-400">{title}</span>
                <div className="flex gap-2">
                    {series.map((s, i) => (
                        <span key={s.key} className="flex items-center gap-1 text-[10px] text-slate-500">
                            <span
                                className="inline-block h-2 w-2 rounded-full"
                                style={{ backgroundColor: getColor(i, s.color) }}
                            />
                            {s.name ?? s.key}
                        </span>
                    ))}
                </div>
            </div>
            <div ref={containerRef} style={{ height }} />
        </div>
    );
}

export const TimeSeriesChartLW = memo(TimeSeriesChartLWInner);
