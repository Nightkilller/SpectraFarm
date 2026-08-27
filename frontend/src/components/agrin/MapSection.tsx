import { lazy, Suspense, useEffect, useState } from "react";
import { Crosshair, LocateFixed, Layers, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { T, type FarmPayload, type Lang } from "@/lib/agrin-data";
import type { MapLayer } from "./LeafletMap";
import { toast } from "sonner";

const LeafletMap = lazy(() => import("./LeafletMap"));

const LAYERS: { id: MapLayer; en: string; hi: string; tone: string }[] = [
  { id: "satellite", en: "Satellite Base", hi: "सैटेलाइट बेस", tone: "text-foreground" },
  { id: "ndvi", en: "NDVI Greenness", hi: "एनडीवीआई हरियाली", tone: "text-primary" },
  { id: "sar", en: "SAR Moisture", hi: "एसएआर नमी", tone: "text-sar" },
  { id: "stress", en: "Stress Risk", hi: "तनाव जोखिम", tone: "text-danger" },
];

const RADII = [50, 100, 250, 500];

export function MapSection({
  farm,
  lang,
  coords,
  onCoords,
  customOpen,
  onCustomOpenChange,
}: {
  farm: FarmPayload;
  lang: Lang;
  coords: { lat: number; lng: number };
  onCoords: (c: { lat: number; lng: number }) => void;
  customOpen: boolean;
  onCustomOpenChange: (v: boolean) => void;
}) {
  const t = T[lang];
  const [layer, setLayer] = useState<MapLayer>("satellite");
  const [radius, setRadius] = useState(250);
  const [mounted, setMounted] = useState(false);
  const [form, setForm] = useState({ lat: "", lng: "", name: "", crop: "", sown: "" });

  useEffect(() => setMounted(true), []);

  const locate = () => {
    if (!navigator.geolocation) {
      toast.error("Geolocation unavailable on this device");
      return;
    }
    toast.loading("Acquiring GPS fix…", { id: "geo" });
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        onCoords({ lat: pos.coords.latitude, lng: pos.coords.longitude });
        toast.success("Map centred on your position", { id: "geo" });
      },
      () => toast.error("Location permission denied", { id: "geo" }),
    );
  };

  return (
    <section className="panel relative overflow-hidden">
      <div className="relative h-[440px] w-full lg:h-[520px]">
        {mounted ? (
          <Suspense fallback={<div className="size-full animate-pulse bg-surface-2" />}>
            <LeafletMap
              lat={coords.lat}
              lng={coords.lng}
              radius={radius}
              layer={layer}
              farmName={lang === "hi" ? farm.farm.name_hi : farm.farm.name}
              cropName={lang === "hi" ? farm.farm.crop_hi : farm.farm.crop}
              onCoordsChange={(c) => {
                onCoords(c);
                toast.info(`Target moved to ${c.lat.toFixed(4)}°N, ${c.lng.toFixed(4)}°E`);
              }}
            />
          </Suspense>
        ) : (
          <div className="size-full animate-pulse bg-surface-2" />
        )}

        {/* Layer tabs */}
        <div className="absolute left-4 top-4 z-[401] flex flex-wrap gap-1 rounded-xl border border-border bg-surface/85 p-1 backdrop-blur-md">
          <span className="hidden items-center gap-1.5 px-2 text-[11px] font-medium text-muted-foreground sm:inline-flex">
            <Layers className="size-3.5" />
          </span>
          {LAYERS.map((l) => (
            <button
              key={l.id}
              onClick={() => setLayer(l.id)}
              className={cn(
                "rounded-lg px-3 py-1.5 text-xs font-medium transition-all",
                layer === l.id
                  ? "bg-secondary shadow-sm " + l.tone
                  : "text-muted-foreground hover:bg-secondary/60",
              )}
            >
              {lang === "hi" ? l.hi : l.en}
            </button>
          ))}
        </div>

        {/* AOI + locate */}
        <div className="absolute bottom-4 left-4 z-[401] flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 rounded-xl border border-border bg-surface/85 p-1 backdrop-blur-md">
            <span className="px-2 text-[11px] font-medium text-muted-foreground">{t.aoi}</span>
            {RADII.map((r) => (
              <button
                key={r}
                onClick={() => setRadius(r)}
                className={cn(
                  "rounded-lg px-2.5 py-1.5 text-xs font-medium transition-all",
                  radius === r ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-secondary",
                )}
              >
                {r}m
              </button>
            ))}
          </div>
          <Button onClick={locate} className="rounded-xl" size="sm">
            <LocateFixed className="size-4" />
            {t.locate}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="rounded-xl bg-surface/85 backdrop-blur-md"
            onClick={() => onCustomOpenChange(true)}
          >
            <Plus className="size-4" />
            {lang === "hi" ? "निर्देशांक" : "Coordinates"}
          </Button>
        </div>

        {/* Readout */}
        <div className="absolute right-4 top-4 z-[401] rounded-xl border border-border bg-surface/85 px-4 py-3 text-right backdrop-blur-md">
          <p className="font-display text-sm font-semibold">
            {farm.farm.crop_emoji} {lang === "hi" ? farm.farm.crop_hi : farm.farm.crop}
          </p>
          <p className="font-mono text-[11px] text-muted-foreground">
            {coords.lat.toFixed(4)}°N, {coords.lng.toFixed(4)}°E
          </p>
          <p className="text-[11px] text-muted-foreground">
            {farm.farm.area_ha} ha · {farm.farm.region}
          </p>
        </div>
      </div>

      <Dialog open={customOpen} onOpenChange={onCustomOpenChange}>
        <DialogContent className="rounded-2xl sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{lang === "hi" ? "नया खेत जोड़ें" : "Add custom field"}</DialogTitle>
            <DialogDescription>
              {lang === "hi"
                ? "निर्देशांक दर्ज करें, हम उस पर उपग्रह विश्लेषण चलाएँगे।"
                : "Enter coordinates and we will run the satellite stack over that AOI."}
            </DialogDescription>
          </DialogHeader>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label>Latitude</Label>
              <Input
                placeholder="23.2045"
                value={form.lat}
                onChange={(e) => setForm({ ...form, lat: e.target.value })}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Longitude</Label>
              <Input
                placeholder="77.0825"
                value={form.lng}
                onChange={(e) => setForm({ ...form, lng: e.target.value })}
              />
            </div>
            <div className="col-span-2 space-y-1.5">
              <Label>Field name</Label>
              <Input
                placeholder="North block"
                value={form.name}
                onChange={(e) => setForm({ ...form, name: e.target.value })}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Crop type</Label>
              <Input
                placeholder="Wheat"
                value={form.crop}
                onChange={(e) => setForm({ ...form, crop: e.target.value })}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Sowing date</Label>
              <Input type="date" value={form.sown} onChange={(e) => setForm({ ...form, sown: e.target.value })} />
            </div>
          </div>
          <DialogFooter>
            <Button
              className="w-full rounded-xl"
              onClick={() => {
                const lat = Number(form.lat);
                const lng = Number(form.lng);
                if (Number.isNaN(lat) || Number.isNaN(lng) || !form.lat || !form.lng) {
                  toast.error("Enter valid latitude and longitude");
                  return;
                }
                onCoords({ lat, lng });
                onCustomOpenChange(false);
                toast.success(`AOI queued for ${form.name || "custom field"}`);
              }}
            >
              {lang === "hi" ? "क्षेत्र विश्लेषण करें" : "Analyse AOI"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  );
}
