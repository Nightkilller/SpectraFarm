import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  ComposedChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Area,
} from "recharts";
import { cn } from "@/lib/utils";
import { buildSeries, T, type FarmPayload, type Lang } from "@/lib/agrin-data";

export function Analytics({ farm, lang }: { farm: FarmPayload; lang: Lang }) {
  const t = T[lang];
  const [days, setDays] = useState(90);
  const data = useMemo(() => buildSeries(farm, days), [farm, days]);
  const ranges = [
    { d: 30, label: t.d30 },
    { d: 60, label: t.d60 },
    { d: 90, label: t.d90 },
  ];
  const features = farm.prediction.features.map((f) => ({
    name: lang === "hi" ? f.name_hi : f.name,
    value: Number((f.importance * 100).toFixed(1)),
  }));

  return (
    <div className="grid gap-4 xl:grid-cols-3">
      <section className="panel p-5 xl:col-span-2">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="font-display text-lg font-semibold">{t.analytics}</h2>
            <p className="text-xs text-muted-foreground">{t.analyticsSub}</p>
          </div>
          <div className="flex rounded-xl border border-border bg-secondary/60 p-0.5">
            {ranges.map((r) => (
              <button
                key={r.d}
                onClick={() => setDays(r.d)}
                className={cn(
                  "rounded-[10px] px-3 py-1.5 text-xs font-medium transition-colors",
                  days === r.d
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {r.label}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-5 h-[320px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data} margin={{ top: 8, right: 8, left: -14, bottom: 0 }}>
              <defs>
                <linearGradient id="ndviFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--color-ndvi)" stopOpacity={0.28} />
                  <stop offset="100%" stopColor="var(--color-ndvi)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 6" vertical={false} />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                tickLine={false}
                axisLine={false}
                minTickGap={28}
              />
              <YAxis
                yAxisId="ndvi"
                domain={[0, 1]}
                tick={{ fontSize: 11, fill: "var(--color-ndvi)" }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                yAxisId="sar"
                orientation="right"
                domain={[-22, -4]}
                tick={{ fontSize: 11, fill: "var(--color-sar)" }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{
                  background: "var(--color-popover)",
                  border: "1px solid var(--color-border)",
                  borderRadius: 12,
                  fontSize: 12,
                  color: "var(--color-popover-foreground)",
                }}
              />
              <Area
                yAxisId="ndvi"
                type="monotone"
                dataKey="ndvi"
                stroke="none"
                fill="url(#ndviFill)"
                isAnimationActive={false}
              />
              <Line
                yAxisId="ndvi"
                type="monotone"
                dataKey="ndvi"
                name="NDVI"
                stroke="var(--color-ndvi)"
                strokeWidth={2.4}
                dot={false}
              />
              <Line
                yAxisId="sar"
                type="monotone"
                dataKey="sar_vv"
                name="SAR VV (dB)"
                stroke="var(--color-sar)"
                strokeWidth={2}
                strokeDasharray="5 4"
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex flex-wrap gap-4 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-2">
            <span className="h-0.5 w-5 rounded bg-primary" /> Optical NDVI (Sentinel-2)
          </span>
          <span className="inline-flex items-center gap-2">
            <span className="h-0.5 w-5 rounded bg-sar" /> SAR VV backscatter (Sentinel-1)
          </span>
        </div>
      </section>

      <section className="panel p-5">
        <h2 className="font-display text-lg font-semibold">{t.features}</h2>
        <p className="text-xs text-muted-foreground">
          {lang === "hi" ? "वर्गीकरण में योगदान (%)" : "Contribution to classification (%)"}
        </p>
        <div className="mt-5 h-[320px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={features} layout="vertical" margin={{ left: 8, right: 16 }}>
              <CartesianGrid stroke="var(--color-border)" strokeDasharray="3 6" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }} axisLine={false} tickLine={false} />
              <YAxis
                type="category"
                dataKey="name"
                width={130}
                tick={{ fontSize: 10, fill: "var(--color-muted-foreground)" }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: "var(--color-secondary)" }}
                contentStyle={{
                  background: "var(--color-popover)",
                  border: "1px solid var(--color-border)",
                  borderRadius: 12,
                  fontSize: 12,
                  color: "var(--color-popover-foreground)",
                }}
              />
              <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={16}>
                {features.map((_, i) => (
                  <Cell key={i} fill={i % 2 === 0 ? "var(--color-ndvi)" : "var(--color-sar)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}
