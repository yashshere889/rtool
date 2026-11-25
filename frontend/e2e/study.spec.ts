import { expect, test } from "@playwright/test";
import { readFileSync, writeFileSync } from "node:fs";

import { accessible, LONG, openStage, signIn } from "./helpers";

/** The template's rows with a number in the metric cell (the treated condition scores higher). */
function filled(template: string): string {
  const [header, ...rows] = template.trim().split(/\r?\n/); // the template uses CSV's own CRLF
  return [header, ...rows.map((row, i) => `${row}${(3 + (i % 5) + (row.includes("phase_locked") ? 3 : 0)).toFixed(1)}`)].join("\n") + "\n";
}

test("a study goes from the question to an approved report: protocol edited, data uploaded, every checkpoint decided", async ({ page }, info) => {
  await signIn(page);
  await page.getByRole("link", { name: "New research" }).first().click();
  await page.getByLabel("Research question or topic").fill("Does sound played during deep sleep improve memory?");
  await page.getByRole("button", { name: /More options/ }).click();
  await page.getByLabel("Field").selectOption("study");
  await expect(page.getByText(/A study you carry out, then analyse/)).toBeVisible();
  await accessible(page, "new research (study)");
  await page.getByRole("button", { name: "Start research" }).click();

  await expect(page.getByRole("button", { name: "Approve & continue" })).toBeVisible(LONG);
  await openStage(page, /Literature/);
  await page.getByRole("button", { name: "Approve & continue" }).click();

  await openStage(page, /Hypotheses/);
  await expect(page.getByRole("button", { name: "Choose this one" }).first()).toBeVisible(LONG);
  await page.getByRole("button", { name: "Choose this one" }).first().click();
  await page.getByRole("button", { name: "Choose this hypothesis" }).click();
  await expect(page.getByRole("region", { name: "Your choice" })).toBeVisible(LONG);

  // protocol: a smaller effect needs more units; then approve
  await openStage(page, /Study protocol/);
  await expect(page.getByRole("button", { name: "Approve protocol" })).toBeVisible(LONG);
  await expect(page.getByText(/26 participants per condition/)).toBeVisible();
  await accessible(page, "protocol");
  await page.getByRole("button", { name: /Edit the protocol/ }).click();
  await page.getByRole("spinbutton", { name: /Smallest effect/ }).fill("0.5");
  await expect(page.getByText(/was 26/)).toBeVisible();
  await accessible(page, "protocol editor");
  await page.getByRole("button", { name: "Apply my edits" }).click();
  await expect(page.getByText(/Version 2 · edited by/)).toBeVisible(LONG);
  await expect(page.getByText(/64 participants per condition/)).toBeVisible();

  // the design behind it: change what is expected, and the protocol is worked out from the design again
  await page.getByRole("button", { name: /Edit the design/ }).click();
  await expect(page.getByRole("heading", { name: "Study design", level: 1 })).toBeVisible();
  await accessible(page, "study design");
  await page.getByRole("button", { name: /Edit the plan/ }).click();
  await page.getByLabel("Expected results").fill("Phase-locked clicks give the largest recall gain.");
  await expect(page.getByText("No problems found.")).toBeVisible();
  await accessible(page, "study design editor");
  await page.getByRole("button", { name: "Apply my edits" }).click();
  await expect(page.getByText(/Version 2 · edited by/)).toBeVisible(LONG);
  await page.getByRole("button", { name: /Back to the protocol/ }).click();
  await expect(page.getByText(/64 participants per condition/)).toBeVisible();
  await page.getByRole("button", { name: "Approve protocol" }).click();

  // data: a file the protocol rejects, then one that fits
  await openStage(page, /Study data/);
  await expect(page.getByRole("button", { name: "Upload and check" })).toBeVisible(LONG);
  await expect(page.getByRole("button", { name: "Analyse the data" })).toHaveCount(0);
  await accessible(page, "study data");
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: /Template with the units/ }).click();
  const template = readFileSync(await (await download).path(), "utf8");
  expect(template.trim().split(/\r?\n/)).toHaveLength(129); // a header and 2 × 64 units

  const bad = info.outputPath("bad.csv");
  writeFileSync(bad, "unit,condition\nU001,sham\n");
  await page.getByLabel("data.csv", { exact: true }).setInputFiles(bad);
  await page.getByRole("button", { name: "Upload and check" }).click();
  await expect(page.getByRole("alert")).toContainText("does not match the protocol");
  await accessible(page, "study data (refused upload)");
  const good = info.outputPath("data.csv");
  writeFileSync(good, filled(template));
  await page.getByLabel("data.csv", { exact: true }).setInputFiles(good);
  await page.getByRole("button", { name: "Upload and check" }).click();
  await expect(page.getByText(/Received 128 rows/)).toBeVisible();
  await accessible(page, "study data (received)");
  await page.getByRole("button", { name: "Analyse the data" }).click();

  // analysis, report, review
  await openStage(page, /Analysis/);
  await expect(page.getByText("Primary outcome").first()).toBeVisible(LONG);
  await accessible(page, "analysis");
  await openStage(page, /Review/);
  await expect(page.getByRole("button", { name: "Approve & finish" })).toBeVisible(LONG);
  await expect(page.getByRole("region", { name: "Verdict" })).toBeVisible();
  await accessible(page, "review (study)");
  await openStage(page, /Report/);
  await expect(page.getByRole("article")).toBeVisible();
  await expect(page.getByText(/Not measured\./)).toBeVisible(); // a field nobody has measured claims no level
  await accessible(page, "report (study)");
  await openStage(page, /Review/);
  await page.getByRole("button", { name: "Approve & finish" }).click();
  await openStage(page, /Overview/);
  await expect(page.getByText("Finished").first()).toBeVisible(LONG);
  await accessible(page, "overview (study, finished)");

  await page.emulateMedia({ colorScheme: "dark" });
  await page.reload();
  await expect(page.locator("html")).toHaveClass(/dark/);
  for (const [stage, ready] of [
    [/Study protocol/, page.getByRole("heading", { name: "Study protocol", level: 1 })],
    [/Study data/, page.getByText(/Received 128 rows|rows read/).first()],
    [/Analysis/, page.getByText("Primary outcome").first()],
    [/Review/, page.getByRole("region", { name: "Verdict" })],
  ] as const) {
    await openStage(page, stage);
    await expect(ready).toBeVisible();
    await accessible(page, `${stage.source} (study, dark)`);
  }
});
