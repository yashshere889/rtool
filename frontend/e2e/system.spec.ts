import { expect, test } from "@playwright/test";

import { accessible, LONG, signIn } from "./helpers";

test("the settings tabs and the run-level dialogs are accessible, and say what they cannot do", async ({ page }) => {
  await signIn(page);
  await page.getByRole("link", { name: "Settings" }).click();
  for (const [tab, ready] of [
    ["Fields", page.getByText("Any field").first()],
    ["Pipelines & agents", page.getByText("Study pipeline")],
    ["Validity", page.getByText(/No field has been measured on this server yet/)],
    ["Checks", page.getByText(/Nothing has been checked yet/)],
  ] as const) {
    await page.getByRole("tab", { name: tab }).click();
    await expect(ready).toBeVisible();
    await accessible(page, `settings: ${tab}`);
  }
  await page.getByRole("button", { name: "Run the checks" }).click();
  await expect(page.getByText(/0\.1\.0 on Python/)).toBeVisible();
  await accessible(page, "settings: checks run");

  // a run that stops at its first checkpoint is enough for the dialogs
  await page.getByRole("link", { name: "New research" }).first().click();
  await page.getByLabel("Research question or topic").fill("Is sound during deep sleep good for memory?");
  await page.getByRole("button", { name: /More options/ }).click();
  await page.getByLabel("Field").selectOption("medicine");
  await page.getByRole("button", { name: "Start research" }).click();
  await expect(page.getByRole("button", { name: "Approve & continue" })).toBeVisible(LONG);
  await expect(page.getByRole("heading", { name: "This run as a whole" })).toBeVisible();
  await accessible(page, "overview with the run tools");

  await page.getByRole("button", { name: "Archive this run" }).click();
  await expect(page.getByText(/set paths.archive_root/)).toBeVisible(); // a demo server has no archive folder
  await accessible(page, "archive dialog");
  await page.keyboard.press("Escape");

  await page.getByRole("button", { name: "Redo a stage in a copy" }).click();
  await page.getByLabel("Stage to redo").selectOption({ label: "Review" });
  await accessible(page, "replay dialog");
  await page.getByRole("button", { name: "Make the copy" }).click();
  await expect(page.getByRole("heading", { name: "This run as a whole" })).toBeVisible(LONG); // the copy's Overview
  await expect(page).toHaveURL(/-synthesis-again/);
});
