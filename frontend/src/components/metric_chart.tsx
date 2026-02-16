"use client";

import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { MetricPoint } from "@/lib/types";

export function MetricChart({ data, color = "#38bdf8" }: { data: MetricPoint[]; color?: string }) {
  return (
    <div className="h-36 w-full">
      <ResponsiveContainer>
        <LineChart data={data}>
          <XAxis dataKey="ts" hide />
          <YAxis hide domain={["auto", "auto"]} />
          <Tooltip />
          <Line type="monotone" dot={false} dataKey="value" stroke={color} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
