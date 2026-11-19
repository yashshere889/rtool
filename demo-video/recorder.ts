/** Records a Playwright page as JPEG frames through Chrome's screencast, on a clock that only runs while recording.
 * Waits for the demo pipeline are cut out (pause/resume); encode.swift turns the frames into an MP4.
 * Captions, title cards and the cursor are overlays drawn on top of the real app; they never change it. */
import type { CDPSession, Locator, Page } from "@playwright/test";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const OVERLAY = `
(() => {
  if (window.__demo) return;
  const css = document.createElement("style");
  css.textContent = \`
    #demo-cap { position: fixed; left: 50%; bottom: 28px; transform: translateX(-50%); max-width: 1100px; z-index: 2147483646;
      background: rgba(17,24,39,.92); color: #fff; font: 500 21px/1.4 system-ui, sans-serif; padding: 12px 22px;
      border-radius: 12px; box-shadow: 0 6px 24px rgba(0,0,0,.25); text-align: center; pointer-events: none; transition: opacity .3s }
    #demo-cap:empty { opacity: 0 }
    #demo-tag { position: fixed; bottom: 10px; left: 12px; z-index: 2147483646; background: rgba(180,83,9,.92); color: #fff;
      font: 600 12px system-ui, sans-serif; padding: 3px 9px; border-radius: 999px; pointer-events: none }
    #demo-card { position: fixed; inset: 0; z-index: 2147483647; background: #0f172a; color: #fff; display: none;
      flex-direction: column; align-items: center; justify-content: center; gap: 18px; font-family: system-ui, sans-serif; text-align: center }
    #demo-card h1 { font-size: 46px; font-weight: 700; margin: 0 }
    #demo-card p { font-size: 22px; opacity: .85; margin: 0; max-width: 900px; line-height: 1.45 }
    #demo-cursor { position: fixed; z-index: 2147483647; width: 22px; height: 22px; margin: -11px 0 0 -11px; border-radius: 50%;
      background: rgba(239,68,68,.35); border: 2px solid rgb(239,68,68); pointer-events: none; left: -50px; top: -50px;
      transition: left .6s ease, top .6s ease, transform .15s }
    #demo-cursor.press { transform: scale(.6) }\`;
  document.head.appendChild(css);
  const add = (id) => { const el = document.createElement("div"); el.id = id; document.body.appendChild(el); return el; };
  const cap = add("demo-cap"), tag = add("demo-tag"), card = add("demo-card"), cursor = add("demo-cursor");
  tag.textContent = window.__demoTag || "Demo · simulated data and model";
  const saved = JSON.parse(sessionStorage.getItem("__demo") || "{}");
  cap.textContent = saved.cap || "";
  if (saved.x !== undefined) { cursor.style.transition = "none"; cursor.style.left = saved.x + "px"; cursor.style.top = saved.y + "px";
    requestAnimationFrame(() => (cursor.style.transition = "")); }
  const save = (k, v) => { const s = JSON.parse(sessionStorage.getItem("__demo") || "{}"); s[k] = v; sessionStorage.setItem("__demo", JSON.stringify(s)); };
  window.__demo = {
    caption: (t) => { cap.textContent = t; save("cap", t); },
    card: (h, p) => { card.innerHTML = h ? "<h1></h1><p></p>" : ""; if (h) { card.querySelector("h1").textContent = h;
      card.querySelector("p").textContent = p || ""; } card.style.display = h ? "flex" : "none"; },
    move: (x, y) => { cursor.style.left = x + "px"; cursor.style.top = y + "px"; save("x", x); save("y", y); },
    press: (on) => cursor.classList.toggle("press", on),
  };
})();`;

export class Recorder {
  private cdp!: CDPSession;
  private clock = 0; // ms of finished video so far, up to `since`
  private since: number | null = null; // wall time recording (re)started, null while paused
  private lastFrame: Buffer | null = null;
  private frames = 0;

  constructor(private page: Page, private dir: string, private tag?: string) {}

  private now(): number {
    return this.clock + (this.since === null ? 0 : Date.now() - this.since);
  }

  async start(): Promise<void> {
    rmSync(this.dir, { recursive: true, force: true });
    mkdirSync(this.dir, { recursive: true });
    // On every later page load too (the overlay keeps its caption and cursor in sessionStorage).
    if (this.tag) await this.page.addInitScript(`window.__demoTag = ${JSON.stringify(this.tag)};`);
    if (this.tag) await this.page.evaluate(`window.__demoTag = ${JSON.stringify(this.tag)};`);
    await this.page.addInitScript(`addEventListener("DOMContentLoaded", () => ${OVERLAY.trim().replace(/;$/, "")});`);
    await this.page.evaluate(OVERLAY);
    this.cdp = await this.page.context().newCDPSession(this.page);
    this.cdp.on("Page.screencastFrame", async (f) => {
      await this.cdp.send("Page.screencastFrameAck", { sessionId: f.sessionId }).catch(() => {});
      this.lastFrame = Buffer.from(f.data, "base64");
      if (this.since !== null) this.write(this.lastFrame);
    });
    const vp = this.page.viewportSize()!;
    await this.cdp.send("Page.startScreencast", { format: "jpeg", quality: 92, maxWidth: vp.width, maxHeight: vp.height });
    this.since = Date.now();
  }

  private write(frame: Buffer): void {
    writeFileSync(join(this.dir, `${String(this.now()).padStart(8, "0")}.jpg`), frame);
    this.frames++;
  }

  /** Stop the clock: nothing until resume() reaches the video. */
  pause(): void {
    if (this.since === null) return;
    this.clock = this.now();
    this.since = null;
  }

  /** Start the clock again, beginning with what the page shows now. */
  async resume(): Promise<void> {
    if (this.since !== null) return;
    this.since = Date.now();
    await this.page.waitForTimeout(150); // let a fresh frame arrive; the last one bridges until then
    if (this.lastFrame) this.write(this.lastFrame);
  }

  /** Run `wait` off the record (a pipeline step, a page load). */
  async offRecord<T>(wait: () => Promise<T>): Promise<T> {
    this.pause();
    try {
      return await wait();
    } finally {
      await this.resume();
    }
  }

  async hold(ms: number): Promise<void> {
    await this.page.waitForTimeout(ms);
  }

  readonly captions: { at: number; text: string }[] = []; // the video time of each caption, for a chapter list

  async caption(text: string, ms = 0): Promise<void> {
    if (text) this.captions.push({ at: this.now(), text });
    await this.page.evaluate((t) => (window as any).__demo?.caption(t), text);
    if (ms) await this.hold(ms);
  }

  async card(title: string, text = "", ms = 3500): Promise<void> {
    await this.page.evaluate(([h, p]) => (window as any).__demo?.card(h, p), [title, text]);
    await this.hold(ms);
    await this.page.evaluate(() => (window as any).__demo?.card("", ""));
  }

  /** Glide the cursor to the element, then click it for real. */
  async click(target: Locator, options: { hold?: number } = {}): Promise<void> {
    await this.point(target);
    await this.page.evaluate(() => (window as any).__demo?.press(true));
    await this.hold(150);
    await this.page.evaluate(() => (window as any).__demo?.press(false));
    await target.click();
    await this.hold(options.hold ?? 700);
  }

  async point(target: Locator): Promise<void> {
    await target.scrollIntoViewIfNeeded();
    const box = (await target.boundingBox())!;
    await this.page.evaluate(([x, y]) => (window as any).__demo?.move(x, y), [box.x + box.width / 2, box.y + box.height / 2]);
    await this.hold(750);
  }

  async type(target: Locator, text: string): Promise<void> {
    await this.click(target, { hold: 200 });
    await target.pressSequentially(text, { delay: 35 });
    await this.hold(400);
  }

  async scroll(dy: number, ms = 900): Promise<void> {
    await this.page.mouse.move(850, 450); // over the content column, which is what scrolls
    for (let i = 0; i < 10; i++) {
      await this.page.mouse.wheel(0, dy / 10);
      await this.page.waitForTimeout(30);
    }
    await this.hold(ms);
  }

  /** Smoothly back to the top of whatever scrolls (the window or an inner panel). */
  async toTop(ms = 900): Promise<void> {
    await this.page.evaluate(() => {
      for (const el of [document.scrollingElement, ...document.querySelectorAll("*")])
        if (el && el.scrollTop > 0) el.scrollTo({ top: 0, behavior: "smooth" });
    });
    await this.hold(ms);
  }

  async stop(): Promise<number> {
    const end = this.now();
    this.pause();
    await this.cdp.send("Page.stopScreencast");
    writeFileSync(join(this.dir, "END"), String(end));
    writeFileSync(join(this.dir, "captions.json"), JSON.stringify(this.captions, null, 1));
    return this.frames;
  }
}
