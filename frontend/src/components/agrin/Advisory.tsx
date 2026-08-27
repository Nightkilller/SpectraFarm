import { Droplets, Info, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";
import { T, type FarmPayload, type Lang, type StressLevel } from "@/lib/agrin-data";

const URGENCY: Record<StressLevel, string> = {
  healthy: "border-primary/25 bg-primary/8",
  mild: "border-warning/30 bg-warning/10",
  severe: "border-danger/25 bg-danger/8",
};

const URGENCY_DOT: Record<StressLevel, string> = {
  healthy: "bg-primary",
  mild: "bg-warning",
  severe: "bg-danger",
};

export function Advisory({
  farm,
  lang,
  onLangToggle,
}: {
  farm: FarmPayload;
  lang: Lang;
  onLangToggle: () => void;
}) {
  const t = T[lang];
  const a = farm.advisory;

  return (
    <section className="panel grid-noise p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="grid size-10 place-items-center rounded-xl bg-primary/12 text-primary ring-1 ring-primary/25">
            <Sparkles className="size-5" />
          </span>
          <div>
            <h2 className="font-display text-lg font-semibold">{t.advisory}</h2>
            <p className="text-xs text-muted-foreground">{t.advisorySub}</p>
          </div>
        </div>
        <button
          onClick={onLangToggle}
          className="rounded-xl border border-border bg-surface px-3 py-2 text-xs font-medium transition-colors hover:bg-secondary"
        >
          {lang === "en" ? "हिंदी में पढ़ें" : "Read in English"}
        </button>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-4">
          <div className="rounded-xl border border-border bg-surface-2/60 p-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{t.summary}</p>
            <p className="mt-2 text-sm leading-relaxed">{a.summary[lang]}</p>
          </div>

          <div>
            <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">{t.actions}</p>
            <div className="mt-2 grid gap-3 sm:grid-cols-2">
              {a.action_items.map((item, i) => (
                <div
                  key={i}
                  className={cn(
                    "rounded-xl border p-4 transition-transform duration-300 hover:-translate-y-0.5",
                    URGENCY[item.urgency],
                  )}
                >
                  <div className="flex items-center gap-2">
                    <span className="grid size-6 place-items-center rounded-lg bg-surface text-xs font-bold">
                      {i + 1}
                    </span>
                    <span className={cn("size-1.5 rounded-full", URGENCY_DOT[item.urgency])} />
                    <p className="text-sm font-semibold">{item.title[lang]}</p>
                  </div>
                  <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{item.detail[lang]}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-xl border border-sar/25 bg-sar/8 p-4">
            <p className="inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-sar">
              <Droplets className="size-3.5" />
              {t.irrigation}
            </p>
            <p className="mt-2 text-sm leading-relaxed">{a.irrigation_advice[lang]}</p>
            <dl className="mt-4 grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-lg bg-surface p-2">
                <dt className="text-muted-foreground">VV</dt>
                <dd className="font-mono font-semibold">{farm.metrics.sar_vv_db} dB</dd>
              </div>
              <div className="rounded-lg bg-surface p-2">
                <dt className="text-muted-foreground">VH</dt>
                <dd className="font-mono font-semibold">{farm.metrics.sar_vh_db} dB</dd>
              </div>
              <div className="rounded-lg bg-surface p-2">
                <dt className="text-muted-foreground">NDWI</dt>
                <dd className="font-mono font-semibold">{farm.metrics.current_ndwi}</dd>
              </div>
              <div className="rounded-lg bg-surface p-2">
                <dt className="text-muted-foreground">72h</dt>
                <dd className="font-semibold">{farm.metrics.moisture_category}</dd>
              </div>
            </dl>
          </div>

          <p className="flex gap-2 rounded-xl bg-secondary/60 p-3 text-[11px] leading-relaxed text-muted-foreground">
            <Info className="mt-0.5 size-3.5 shrink-0" />
            {t.provenance}
          </p>
        </div>
      </div>
    </section>
  );
}
