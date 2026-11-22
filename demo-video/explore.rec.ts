/** Dumps every stage page of the staged demo runs (accessibility snapshot + screenshot), to plan the tour. */
import { expect, test } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";

const OUT = process.env.OUT ?? "out/explore";
test("explore", async ({ page }) => {
  mkdirSync(OUT, { recursive: true });
  await page.goto(`/#token=${readFileSync(process.env.DEMO_TOKEN!, "utf8").trim()}`);
  await expect(page.getByRole("heading", { name: "Research projects" })).toBeVisible();
  for (const run of (process.env.RUNS ?? "1-literature,2-hypotheses,3-plan,4-gpu-runs,5-final-report,6-done").split(",")) {
    await page.goto(`/runs/${process.env.PROJECT ?? "sleep-demo"}/${run}`);
    const nav = page.getByRole("navigation", { name: "Run" });
    await expect(nav).toBeVisible();
    const names = await nav.getByRole("link").allInnerTexts();
    for (const [i, name] of names.entries()) {
      if (name.includes("All projects")) continue;
      await nav.getByRole("link").nth(i).click({ timeout: 10_000 });
      await page.waitForTimeout(1500);
      const slug = `${run}-${i}-${name.split("\n").pop()!.replace(/\W+/g, "_")}`;
      writeFileSync(`${OUT}/${slug}.txt`, `${page.url()}\n${await page.title()}\n${await page.locator("main").ariaSnapshot()}`);
      await page.screenshot({ path: `${OUT}/${slug}.png`, fullPage: true });
    }
  }
});
