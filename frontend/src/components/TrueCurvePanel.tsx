"use client";

import { useMemo, useState } from "react";
import { Brush, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CurvePoint } from "@/lib/orch";

const METRICS: Record<string, { key: keyof CurvePoint; color: string }> = {
  unscaled: { key: "cum_net_unscaled", color: "#38bdf8" },
  volscaled: { key: "cum_net_volscaled", color: "#22c55e" },
  drawdown: { key: "drawdown", color: "#ef4444" },
  vol: { key: "rolling_vol", color: "#f59e0b" },
  sharpe: { key: "rolling_sharpe", color: "#a855f7" },
  turnover: { key: "turnover", color: "#f97316" },
  cost: { key: "cost", color: "#14b8a6" },
};

export function TrueCurvePanel({ title, series }: { title: string; series: CurvePoint[] }) {
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const [compare, setCompare] = useState(false);

  const rows = useMemo(() => {
    if (!compare || !series.length) return series.map((p) => ({ ...p, label: p.t }));
    const baseU = series[0].cum_net_unscaled || 1;
    const baseV = series[0].cum_net_volscaled || 1;
    return series.map((p) => ({
      ...p,
      label: p.t,
      cum_net_unscaled: (p.cum_net_unscaled / baseU) - 1,
      cum_net_volscaled: (p.cum_net_volscaled / baseV) - 1,
    }));
  }, [series, compare]);

  return (
    <div className="rounded border border-border bg-surface-1 p-3">
      <div className="mb-2 flex items-center justify-between"><div className="text-sm font-semibold">{title}</div><button className="rounded border border-border px-2 py-1 text-xs" onClick={() => setCompare((v) => !v)}>{compare ? "Disable" : "Enable"} compare sleeves</button></div>
      <div className="mb-2 flex flex-wrap gap-2 text-xs">
        {Object.keys(METRICS).map((k) => <button key={k} onClick={() => setHidden((s) => ({ ...s, [k]: !s[k] }))} className="rounded border border-border px-2 py-1">{hidden[k] ? "☐" : "☑"} {k}</button>)}
      </div>
      <div className="h-64">
        <ResponsiveContainer>
          <LineChart data={rows}>
            <XAxis dataKey="label" minTickGap={24} />
            <YAxis domain={["auto", "auto"]} />
            <Tooltip />
            <Legend />
            {Object.entries(METRICS).map(([name, cfg]) => hidden[name] ? null : <Line key={name} type="monotone" dot={false} dataKey={cfg.key as string} stroke={cfg.color} strokeWidth={2} name={name} />)}
            <Brush dataKey="label" height={20} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
