import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

const LONG = { timeout: 120_000 }; // the demo pipeline runs in the server between checkpoints

async function signIn(page: Page): Promise<void> {
  const token = readFileSync(".e2e/token", "utf8").trim();
  await page.goto(`/#token=${token}`);
  await expect(page.getByRole("heading", { name: "Research projects" })).toBeVisible();
}

/** No serious or critical WCAG 2.1 A/AA problems on the page as it is now. */
async function accessible(page: Page, name: string): Promise<void> {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .exclude("[data-sonner-toaster]") // notifications are checked mid-animation (half transparent); their colours pass
    .analyze();
  const problems = results.violations
    .filter((v) => v.impact === "serious" || v.impact === "critical")
    .map((v) => `${v.id} (${v.help}): ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
  expect(problems, `accessibility problems on ${name}`).toEqual([]);
}

async function openStage(page: Page, title: RegExp): Promise<void> {
  await page.getByRole("navigation", { name: "Run" }).getByRole("link", { name: title }).click();
}

test("a question goes from the literature to an approved report, every checkpoint decided in the browser", async ({ page }) => {
  await signIn(page);
  await accessible(page, "projects");

  await page.getByRole("link", { name: "New research" }).first().click();
  await page.getByLabel("Research question or topic").fill("Does sound played during deep sleep improve memory?");
  await accessible(page, "new research");
  await page.getByRole("button", { name: "Start research" }).click();

  // literature
  await expect(page.getByRole("button", { name: "Approve & continue" })).toBeVisible(LONG);
  await accessible(page, "overview");
  await openStage(page, /Literature/);
  await expect(page.getByRole("heading", { name: "Literature", level: 1 })).toBeVisible();
  await page.getByRole("button", { name: /Summary/ }).first().click();
  await expect(page.getByText("AI-written · not checked against quotes").first()).toBeVisible();
  await accessible(page, "literature");
  await page.getByRole("link", { name: "Open full paper" }).first().click();
  await expect(page.getByRole("button", { name: /Text with findings marked/ })).toBeVisible();
  await accessible(page, "paper reader");
  await page.getByRole("link", { name: "All papers" }).click();
  await page.getByRole("button", { name: "Approve & continue" }).click();

  // hypotheses: edit the second one, then choose it
  await openStage(page, /Hypotheses/);
  await expect(page.getByRole("button", { name: /Edit & choose/ }).nth(1)).toBeVisible(LONG);
  await accessible(page, "hypotheses");
  await page.getByRole("button", { name: /Edit & choose/ }).nth(1).click();
  const statement = page.getByRole("textbox", { name: /^Hypothesis/ });
  await statement.fill("If the detection threshold adapts to each older adult, their slow-oscillation response matches the young");
  await page.getByRole("button", { name: "Choose my version" }).click();
  await expect(page.getByRole("region", { name: "Your choice" })).toBeVisible(LONG);

  // plan: more seeds, then approve
  await openStage(page, /Experiment plan/);
  await expect(page.getByRole("button", { name: "Approve plan & start downloads" })).toBeVisible(LONG);
  await accessible(page, "plan");
  await page.getByRole("button", { name: /Edit the plan/ }).click();
  await page.getByRole("spinbutton", { name: /seeds\) per condition/ }).fill("5");
  const apply = page.getByRole("button", { name: "Apply my edits" });
  await expect(apply).toBeEnabled();
  await accessible(page, "plan editor");
  await apply.click();
  await expect(page.getByText(/Version 2 · edited by/)).toBeVisible(LONG);
  await page.getByRole("button", { name: "Approve plan & start downloads" }).click();

  // code: approve GPU time
  await openStage(page, /Code & runs/);
  await expect(page.getByRole("button", { name: "Approve GPU runs" })).toBeVisible(LONG);
  await expect(page.getByRole("textbox", { name: "experiment.py" })).toBeVisible();
  await accessible(page, "code");
  await page.getByRole("button", { name: "Approve GPU runs" }).click();

  // report and review
  await openStage(page, /Review/);
  await expect(page.getByRole("button", { name: "Approve & finish" })).toBeVisible(LONG);
  await expect(page.getByRole("region", { name: "Verdict" })).toBeVisible();
  await accessible(page, "review");
  await page.getByRole("tab", { name: "Final report" }).click();
  await expect(page.getByRole("article")).toBeVisible();
  await accessible(page, "final report");
  await openStage(page, /Report/);
  await expect(page.getByRole("article")).toBeVisible();
  await accessible(page, "report");
  await openStage(page, /Review/);
  await page.getByRole("button", { name: "Approve & finish" }).click();

  await openStage(page, /Overview/);
  await expect(page.getByText("Finished").first()).toBeVisible(LONG);

  // the same pages in the dark theme
  await page.emulateMedia({ colorScheme: "dark" });
  await page.reload();
  await expect(page.locator("html")).toHaveClass(/dark/);
  for (const [stage, ready] of [
    [/Overview/, page.getByText("Finished").first()],
    [/Literature/, page.getByRole("heading", { name: "Literature", level: 1 })],
    [/Hypotheses/, page.getByRole("region", { name: "Your choice" })],
    [/Experiment plan/, page.getByRole("heading", { name: "Experiment plan", level: 1 })],
    [/Code & runs/, page.getByRole("tab", { name: "Results" })],
    [/Review/, page.getByRole("region", { name: "Verdict" })],
  ] as const) {
    await openStage(page, stage);
    await expect(ready).toBeVisible();
    await accessible(page, `${stage.source} (dark)`);
  }
});

test("keyboard users can skip straight to the content", async ({ page }) => {
  await signIn(page);
  await page.keyboard.press("Tab");
  const skip = page.getByRole("link", { name: "Skip to content" });
  await expect(skip).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#main$/);
});
