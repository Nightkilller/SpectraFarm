import { useState } from "react";
import { MessageCircle, Send, Sparkles } from "lucide-react";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { answerFor, SUGGESTIONS, T, type FarmPayload, type Lang } from "@/lib/agrin-data";

interface Msg {
  role: "user" | "ai";
  text: string;
}

export function AskAgriN({ farm, lang }: { farm: FarmPayload; lang: Lang }) {
  const t = T[lang];
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [msgs, setMsgs] = useState<Msg[]>([
    {
      role: "ai",
      text:
        lang === "hi"
          ? "नमस्ते! मैं AgriN हूँ। मैं आपके खेत की सेंटिनल-1 और सेंटिनल-2 रीडिंग देख सकता हूँ। पूछिए।"
          : "Hi! I'm AgriN. I can read your field's live Sentinel-1 radar and Sentinel-2 optical signals. Ask me anything.",
    },
  ]);

  const send = (text: string) => {
    const q = text.trim();
    if (!q) return;
    setMsgs((m) => [...m, { role: "user", text: q }]);
    setInput("");
    setThinking(true);
    setTimeout(() => {
      setMsgs((m) => [...m, { role: "ai", text: answerFor(q, farm, lang) }]);
      setThinking(false);
    }, 750);
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button
          size="lg"
          className="fixed bottom-6 right-6 z-50 rounded-full px-5 shadow-glow transition-transform hover:scale-105"
        >
          <MessageCircle className="size-5" />
          {t.ask}
        </Button>
      </SheetTrigger>
      <SheetContent className="flex w-full flex-col gap-0 p-0 sm:max-w-md">
        <SheetHeader className="border-b border-border p-5">
          <SheetTitle className="flex items-center gap-2 font-display">
            <span className="grid size-8 place-items-center rounded-lg bg-primary/12 text-primary">
              <Sparkles className="size-4" />
            </span>
            {t.ask}
          </SheetTitle>
          <SheetDescription>{t.askSub}</SheetDescription>
        </SheetHeader>

        <ScrollArea className="flex-1 p-5">
          <div className="space-y-4">
            {msgs.map((m, i) => (
              <div key={i} className={cn("flex", m.role === "user" ? "justify-end" : "justify-start")}>
                <div
                  className={cn(
                    "max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
                    m.role === "user"
                      ? "rounded-br-sm bg-primary text-primary-foreground"
                      : "rounded-bl-sm bg-secondary text-secondary-foreground",
                  )}
                >
                  {m.text}
                </div>
              </div>
            ))}
            {thinking && (
              <div className="flex gap-1.5 rounded-2xl bg-secondary px-4 py-3 w-fit">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="size-1.5 animate-bounce rounded-full bg-muted-foreground"
                    style={{ animationDelay: `${i * 120}ms` }}
                  />
                ))}
              </div>
            )}
          </div>
        </ScrollArea>

        <div className="border-t border-border p-4">
          <div className="mb-3 flex flex-wrap gap-2">
            {SUGGESTIONS[lang].map((s) => (
              <button
                key={s}
                onClick={() => send(s)}
                className="rounded-full border border-border bg-secondary/60 px-3 py-1.5 text-[11px] text-muted-foreground transition-colors hover:border-primary/40 hover:text-primary"
              >
                {s}
              </button>
            ))}
          </div>
          <form
            className="flex gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
          >
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={t.placeholder}
              className="rounded-xl"
            />
            <Button type="submit" size="icon" className="rounded-xl">
              <Send className="size-4" />
            </Button>
          </form>
        </div>
      </SheetContent>
    </Sheet>
  );
}
