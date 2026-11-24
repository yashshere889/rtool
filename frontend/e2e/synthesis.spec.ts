import { expect, test } from "@playwright/test";

import { accessible, LONG, openStage, signIn } from "./helpers";

test("a literature review: every sentence traced to a quote, checkpoints decided in the browser", async ({ page }) => {
  await signIn(page);
  await page.getByRole("link", { name: "New research" }).first().click();
  await page.getByLabel("Research question or topic").fill("What does the evidence say about sound during deep sleep and memory?");
  await page.getByRole("button", { name: /More options/ }).click();
  await page.getByLabel("Field").selectOption("medicine");
  await expect(page.getByText(/A literature review only, with no experiment/)).toBeVisible();
  await page.getByRole("button", { name: "Start research" }).click();

  await expect(page.getByRole("button", { name: "Approve & continue" })).toBeVisible(LONG);
  await openStage(page, /Literature/);
  await page.getByRole("button", { name: "Approve & continue" }).click();

  await openStage(page, /Review/);
  await expect(page.getByRole("heading", { name: "Literature review", level: 1 })).toBeVisible(LONG);
  await expect(page.getByRole("button", { name: "Approve & finish" })).toBeVisible(LONG);
  await expect(page.getByText(/sentences? dropped/)).toBeVisible();
  await expect(page.getByText("Stimulation also cures insomnia.").first()).toBeVisible(); // left out, with its reason
  await accessible(page, "synthesis");
  await page.getByText(/^Rests on C\d+/).first().click();
  await expect(page.getByRole("link", { name: "Show in the paper" }).first()).toBeVisible();
  await accessible(page, "synthesis (a sentence's findings open)");
  await page.getByRole("link", { name: "Show in the paper" }).first().click();
  await expect(page.getByRole("button", { name: /Text with findings marked/ })).toBeVisible();
  await accessible(page, "paper reader (from the review)");
  await page.goBack();

  // rewrite a sentence: refused if it invents a number, kept when it rests on its findings
  await page.getByRole("button", { name: /Edit the review/ }).click();
  const sentence = page.getByRole("textbox", { name: /^Sentence 1 of / }).first();
  const original = await sentence.inputValue();
  await sentence.fill(`${original.replace(/\.$/, "")}, in 97 % of people.`);
  await expect(page.getByText(/states numbers its findings do not/)).toBeVisible();
  await accessible(page, "review editor (a refused sentence)");
  await sentence.fill(`${original.replace(/\.$/, "")}, as the study reports.`);
  await expect(page.getByText("Every sentence rests on verified findings.")).toBeVisible();
  await page.getByRole("button", { name: "Apply my edits" }).click();
  await expect(page.getByText(/Edited by/)).toBeVisible(LONG);
  await expect(page.getByText(/as the study reports/).first()).toBeVisible();
  await accessible(page, "synthesis (edited)");
  await page.getByRole("button", { name: "Approve & finish" }).click();

  await openStage(page, /Overview/);
  await expect(page.getByText("Finished").first()).toBeVisible(LONG);
  await accessible(page, "overview (synthesis, finished)");

  await page.emulateMedia({ colorScheme: "dark" });
  await page.reload();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await openStage(page, /Review/);
  await expect(page.getByRole("heading", { name: "Literature review", level: 1 })).toBeVisible();
  await accessible(page, "synthesis (dark)");
});
