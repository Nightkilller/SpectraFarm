import { Bell, Globe, Moon, Radio, Satellite, Sun, ChevronDown, MapPin } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { cn } from "@/lib/utils";
import { FARMS, T, type FarmPayload, type Lang } from "@/lib/agrin-data";

interface Props {
  farm: FarmPayload;
  onFarmChange: (id: string) => void;
  lang: Lang;
  onLangChange: (l: Lang) => void;
  live: boolean;
  onLiveChange: (v: boolean) => void;
  dark: boolean;
  onDarkChange: (v: boolean) => void;
  onAddField: () => void;
  alerts: number;
}

export function TopNav({
  farm,
  onFarmChange,
  lang,
  onLangChange,
  live,
  onLiveChange,
  dark,
  onDarkChange,
  onAddField,
  alerts,
}: Props) {
  const t = T[lang];
  return (
    <header className="sticky top-0 z-50 glass">
      <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-3 px-4 py-3 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="grid size-10 place-items-center rounded-xl bg-primary/12 text-primary ring-1 ring-primary/25">
            <Satellite className="size-5" />
          </div>
          <div className="leading-tight">
            <h1 className="font-display text-lg font-semibold">SpectraFarm</h1>
            <p className="text-[11px] text-muted-foreground">AgriN Intelligence</p>
          </div>
          <span className="ml-2 hidden items-center gap-2 rounded-full bg-primary/10 px-3 py-1.5 text-[11px] font-medium text-primary xl:inline-flex">
            <span className="size-1.5 rounded-full bg-primary live-dot" />
            {t.liveFeed}
          </span>
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" className="rounded-xl">
                <MapPin className="size-4 text-primary" />
                <span className="max-w-[190px] truncate">
                  {lang === "hi" ? farm.farm.name_hi : farm.farm.name}
                </span>
                <ChevronDown className="size-4 opacity-60" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-72">
              <DropdownMenuLabel>Fields</DropdownMenuLabel>
              {FARMS.map((f) => (
                <DropdownMenuItem key={f.farm.farm_id} onSelect={() => onFarmChange(f.farm.farm_id)}>
                  <span className="mr-1">{f.farm.crop_emoji}</span>
                  {lang === "hi" ? f.farm.name_hi : f.farm.name}
                </DropdownMenuItem>
              ))}
              <DropdownMenuSeparator />
              <DropdownMenuItem onSelect={onAddField} className="text-primary">
                {t.addField}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <div className="flex rounded-xl border border-border bg-secondary/60 p-0.5">
            {(["en", "hi"] as Lang[]).map((l) => (
              <button
                key={l}
                onClick={() => onLangChange(l)}
                className={cn(
                  "rounded-[10px] px-3 py-1.5 text-xs font-medium transition-colors",
                  lang === l
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {l === "en" ? "English" : "हिंदी"}
              </button>
            ))}
          </div>

          <button
            onClick={() => onLiveChange(!live)}
            className={cn(
              "inline-flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-medium transition-colors",
              live
                ? "border-primary/30 bg-primary/10 text-primary"
                : "border-warning/35 bg-warning/12 text-warning",
            )}
          >
            <Radio className={cn("size-3.5", live && "live-dot rounded-full")} />
            {live ? t.liveMode : t.demoMode}
          </button>

          <Button
            variant="ghost"
            size="icon"
            className="rounded-xl"
            onClick={() => onDarkChange(!dark)}
            aria-label="Toggle theme"
          >
            {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </Button>

          <Button variant="ghost" size="icon" className="relative rounded-xl" aria-label="Alerts">
            <Bell className="size-4" />
            {alerts > 0 && (
              <span className="absolute right-1 top-1 grid size-4 place-items-center rounded-full bg-danger text-[9px] font-bold text-danger-foreground">
                {alerts}
              </span>
            )}
          </Button>

          <Avatar className="size-9 ring-1 ring-border">
            <AvatarFallback className="bg-primary/12 text-xs font-semibold text-primary">RS</AvatarFallback>
          </Avatar>
          <Globe className="hidden size-0" />
        </div>
      </div>
    </header>
  );
}
