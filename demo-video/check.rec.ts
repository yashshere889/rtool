/** Plays a finished MP4 in Chrome and saves what viewers see at given seconds:  VIDEO=out/x.mp4 AT=1,5 npx playwright test check.rec.ts */
import { test } from "@playwright/test";
import { resolve } from "node:path";

test("stills from the finished video", async ({ page }) => {
  const video = resolve(process.env.VIDEO!);
  await page.goto(`file://${video}`);
  await page.locator("video").evaluate((el: HTMLVideoElement) => el.removeAttribute("controls")); // the bar hides the bottom strip
  const v = page.locator("video");
  const info = await v.evaluate(async (el: HTMLVideoElement) => {
    if (el.readyState < 1) await new Promise((r) => el.addEventListener("loadedmetadata", r, { once: true }));
    return { duration: el.duration, w: el.videoWidth, h: el.videoHeight, error: el.error?.message ?? null };
  });
  console.log(JSON.stringify(info));
  for (const s of (process.env.AT ?? "1").split(",")) {
    await v.evaluate(async (el: HTMLVideoElement, t) => {
      el.pause();
      el.currentTime = t;
      await new Promise((r) => el.addEventListener("seeked", r, { once: true }));
    }, Number(s));
    await page.waitForTimeout(300);
    const quality = await v.evaluate((el: HTMLVideoElement) => el.getVideoPlaybackQuality?.().totalVideoFrames ?? -1);
    await v.screenshot({ path: `${process.env.OUT ?? "out/check"}/at-${s}.png` });
    console.log(s, "frames decoded so far", quality);
  }
});
