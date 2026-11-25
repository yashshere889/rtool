import AxeBuilder from "@axe-core/playwright";
import { expect, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

export const LONG = { timeout: 120_000 }; // the demo pipeline runs in the server between checkpoints

export async function signIn(page: Page): Promise<void> {
  const token = readFileSync(".e2e/token", "utf8").trim();
  await page.goto(`/#token=${token}`);
  await expect(page.getByRole("heading", { name: "Research projects" })).toBeVisible();
}

/** No serious or critical WCAG 2.1 A/AA problems on the page as it is now. */
export async function accessible(page: Page, name: string): Promise<void> {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .exclude("[data-sonner-toaster]") // notifications are checked mid-animation (half transparent); their colours pass
    .analyze();
  const problems = results.violations
    .filter((v) => v.impact === "serious" || v.impact === "critical")
    .map((v) => `${v.id} (${v.help}): ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
  expect(problems, `accessibility problems on ${name}`).toEqual([]);
}

export async function openStage(page: Page, title: RegExp): Promise<void> {
  await page.getByRole("navigation", { name: "Run" }).getByRole("link", { name: title }).click();
}
