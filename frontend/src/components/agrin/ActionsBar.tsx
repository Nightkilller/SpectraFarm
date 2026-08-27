import { useState } from "react";
import { Download, RefreshCw, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { T, type FarmPayload, type Lang } from "@/lib/agrin-data";

export function ActionsBar({ farm, lang }: { farm: FarmPayload; lang: Lang }) {
  const t = T[lang];
  const [spinning, setSpinning] = useState(false);

  const refresh = () => {
    setSpinning(true);
    toast.loading("Pulling latest Sentinel-1/2 scene…", { id: "sat" });
    setTimeout(() => {
      setSpinning(false);
      toast.success(`Imagery refreshed for ${farm.farm.name}`, { id: "sat" });
    }, 1600);
  };

  const download = () => {
    const report = {
      generated_at: new Date().toISOString(),
      ...farm,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${farm.farm.farm_id}-health-report.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Farm health report exported");
  };

  const share = () => {
    const text = `SpectraFarm advisory — ${farm.farm.name}\nNDVI ${farm.metrics.current_ndvi} | SAR ${farm.metrics.sar_vv_db} dB\n${farm.advisory.summary[lang]}\n${farm.advisory.irrigation_advice[lang]}`;
    window.open(`https://wa.me/?text=${encodeURIComponent(text)}`, "_blank", "noopener");
  };

  return (
    <div className="panel flex flex-wrap items-center gap-3 p-4">
      <Button onClick={download} className="rounded-xl">
        <Download className="size-4" />
        {t.report}
      </Button>
      <Button variant="outline" onClick={refresh} className="rounded-xl">
        <RefreshCw className={cn("size-4", spinning && "animate-spin")} />
        {t.refresh}
      </Button>
      <Button variant="outline" onClick={share} className="rounded-xl">
        <Share2 className="size-4" />
        {t.share}
      </Button>
      <p className="ml-auto font-mono text-[11px] text-muted-foreground">
        Sentinel-1 GRD · Sentinel-2 L2A · AOI {farm.farm.area_ha} ha
      </p>
    </div>
  );
}
