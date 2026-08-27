import { useEffect, useMemo, useState } from "react";
import { Toaster } from "@/components/ui/sonner";
import { TopNav } from "@/components/agrin/TopNav";
import { MapSection } from "@/components/agrin/MapSection";
import { KpiCards } from "@/components/agrin/KpiCards";
import { Analytics } from "@/components/agrin/Analytics";
import { Advisory } from "@/components/agrin/Advisory";
import { AskAgriN } from "@/components/agrin/AskAgriN";
import { ActionsBar } from "@/components/agrin/ActionsBar";
import { FARMS, type Lang } from "@/lib/agrin-data";

export default function App() {
  const [farmId, setFarmId] = useState(FARMS[0]!.farm.farm_id);
  const [lang, setLang] = useState<Lang>("en");
  const [live, setLive] = useState(true);
  const [dark, setDark] = useState(false);
  const [customOpen, setCustomOpen] = useState(false);

  const farm = useMemo(
    () => FARMS.find((f) => f.farm.farm_id === farmId) ?? FARMS[0]!,
    [farmId],
  );
  const [coords, setCoords] = useState({
    lat: farm.farm.latitude,
    lng: farm.farm.longitude,
  });

  useEffect(() => {
    setCoords({ lat: farm.farm.latitude, lng: farm.farm.longitude });
  }, [farm]);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  const alerts = FARMS.filter(
    (f) => f.metrics.stress_level !== "healthy",
  ).length;

  return (
    <div className="min-h-screen bg-background">
      <TopNav
        farm={farm}
        onFarmChange={setFarmId}
        lang={lang}
        onLangChange={setLang}
        live={live}
        onLiveChange={setLive}
        dark={dark}
        onDarkChange={setDark}
        onAddField={() => setCustomOpen(true)}
        alerts={alerts}
      />

      <main className="mx-auto max-w-[1600px] space-y-4 px-4 py-6 lg:px-8">
        <MapSection
          farm={farm}
          lang={lang}
          coords={coords}
          onCoords={setCoords}
          customOpen={customOpen}
          onCustomOpenChange={setCustomOpen}
        />
        <KpiCards farm={farm} lang={lang} />
        <Analytics farm={farm} lang={lang} />
        <Advisory
          farm={farm}
          lang={lang}
          onLangToggle={() => setLang(lang === "en" ? "hi" : "en")}
        />
        <ActionsBar farm={farm} lang={lang} />
        <footer className="pb-24 pt-2 text-center text-[11px] text-muted-foreground">
          SpectraFarm · AgriN — {live ? "Live Earth Engine feed" : "Demo mode"}{" "}
          · Copernicus Sentinel data
        </footer>
      </main>

      <AskAgriN farm={farm} lang={lang} />
      <Toaster position="top-center" richColors />
    </div>
  );
}
