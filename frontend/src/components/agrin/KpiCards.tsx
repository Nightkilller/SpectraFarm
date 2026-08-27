import { ArrowDownRight, ArrowUpRight, CloudRain, Leaf, Radar, ShieldAlert, Sprout } from "lucide-react";
import { cn } from "@/lib/utils";
import { T, type FarmPayload, type Lang, type StressLevel } from "@/lib/agrin-data";

const STRESS_STYLE: Record<StressLevel, string> = {
  healthy: "bg-primary/12 text-primary ring-primary/25",
  mild: "bg-warning/15 text-warning ring-warning/30",
  severe: "bg-danger/12 text-danger ring-danger/25",
};

function Shell({
  icon,
  title,
  accent,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  accent: string;
  children: React.ReactNode;
}) {
  return (
    <div className="panel group relative overflow-hidden p-5 transition-all duration-300 hover:-translate-y-1 hover:shadow-glow">
      <div
        className={cn("absolute inset-x-0 top-0 h-px opacity-70", accent)}
        style={{ background: "currentColor" }}
      />
      <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
        <span className={cn("grid size-7 place-items-center rounded-lg bg-secondary", accent)}>{icon}</span>
        {title}
      </div>
      <div className="mt-4">{children}</div>
    </div>
  );
}

export function KpiCards({ farm, lang }: { farm: FarmPayload; lang: Lang }) {
  const t = T[lang];
  const m = farm.metrics;
  const up = m.ndvi_trend_pct >= 0;
  const stressLabel = t[m.stress_level as "healthy" | "mild" | "severe"];

  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      <Shell icon={<Leaf className="size-4" />} title={t.ndviTitle} accent="text-primary">
        <div className="flex items-end gap-3">
          <span className="font-display text-4xl font-semibold tabular-nums">{m.current_ndvi.toFixed(2)}</span>
          <span
            className={cn(
              "mb-1 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold",
              up ? "bg-primary/12 text-primary" : "bg-danger/12 text-danger",
            )}
          >
            {up ? <ArrowUpRight className="size-3.5" /> : <ArrowDownRight className="size-3.5" />}
            {Math.abs(m.ndvi_trend_pct)}%
          </span>
        </div>
        <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
          <div
            className="h-full rounded-full bg-primary transition-all duration-700"
            style={{ width: `${Math.min(100, (m.current_ndvi / 0.85) * 100)}%` }}
          />
        </div>
        <p className="mt-2 text-[11px] text-muted-foreground">
          {t.dayTrend} · {t.optimal}
        </p>
      </Shell>

      <Shell icon={<Radar className="size-4" />} title={t.sarTitle} accent="text-sar">
        <div className="flex items-end gap-3">
          <span className="font-display text-4xl font-semibold tabular-nums">{m.sar_vv_db}</span>
          <span className="mb-1.5 text-sm text-muted-foreground">dB VV</span>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <span className="rounded-full bg-sar/12 px-2.5 py-1 text-xs font-semibold text-sar">
            {m.moisture_category}
          </span>
          <span className="inline-flex items-center gap-1 rounded-full bg-secondary px-2.5 py-1 text-[11px] text-muted-foreground">
            <CloudRain className="size-3" />
            {t.allWeather}
          </span>
        </div>
        <p className="mt-2 font-mono text-[11px] text-muted-foreground">
          VH {m.sar_vh_db} dB · ratio {m.sar_ratio}
        </p>
      </Shell>

      <Shell icon={<Sprout className="size-4" />} title={t.cropTitle} accent="text-primary">
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-2 rounded-xl bg-primary/12 px-3 py-2 font-display text-lg font-semibold text-primary ring-1 ring-primary/25">
            {farm.farm.crop_emoji} {lang === "hi" ? farm.farm.crop_hi : farm.prediction.predicted_crop}
          </span>
        </div>
        <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
          <div
            className="h-full rounded-full bg-primary transition-all duration-700"
            style={{ width: `${farm.prediction.confidence * 100}%` }}
          />
        </div>
        <p className="mt-2 text-[11px] text-muted-foreground">
          {(farm.prediction.confidence * 100).toFixed(1)}% {t.confidence} · 2nd:{" "}
          {farm.prediction.secondary_crop} {(farm.prediction.secondary_confidence * 100).toFixed(1)}%
        </p>
      </Shell>

      <Shell icon={<ShieldAlert className="size-4" />} title={t.stressTitle} accent="text-warning">
        <span
          className={cn(
            "inline-flex items-center gap-2 rounded-full px-3.5 py-2 font-display text-base font-semibold ring-1",
            STRESS_STYLE[m.stress_level],
          )}
        >
          <span className="size-2 rounded-full bg-current live-dot" />
          {stressLabel}
        </span>
        <p className="mt-4 text-xs text-muted-foreground">
          {lang === "hi" ? "विसंगति पहचान" : "Anomaly detection"}:{" "}
          <span className="font-medium text-foreground">
            {m.ndvi_trend_pct < -10
              ? lang === "hi"
                ? "10 दिनों में तीव्र गिरावट"
                : "Sharp 10-day NDVI drop"
              : m.ndvi_trend_pct < 0
                ? lang === "hi"
                  ? "धीमी गिरावट"
                  : "Slow decline detected"
                : lang === "hi"
                  ? "कोई विसंगति नहीं"
                  : "No anomaly detected"}
          </span>
        </p>
        <p className="mt-1 text-[11px] text-muted-foreground">
          NDWI {m.current_ndwi} · {m.health_trend}
        </p>
      </Shell>
    </div>
  );
}
