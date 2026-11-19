/** The demo video: one research question taken from the question to an approved report in the web UI, every
 * checkpoint decided by clicking the real controls. Waits for the (simulated) pipeline are cut from the video.
 *
 *   DEMO_TOKEN=<token file> npx playwright test tour.rec.ts      then      swift encode.swift out/tour-frames out/rtool-demo.mp4
 */
import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { Recorder } from "./recorder";

const WAIT = { timeout: 300_000 };
const QUESTION = "Does playing quiet clicks in time with deep-sleep brain waves improve overnight memory?";

test("tour", async ({ page }) => {
  await page.goto(`/#token=${readFileSync(process.env.DEMO_TOKEN!, "utf8").trim()}`);
  await expect(page.getByRole("heading", { name: "Research projects" })).toBeVisible();
  const rec = new Recorder(page, "out/tour-frames");
  await rec.start();
  const r = rec;
  const main = page.locator("main");
  const stage = (name: RegExp) => page.getByRole("navigation", { name: "Run" }).getByRole("link", { name });
  const tab = async () => (await page.title()).split(" · ")[0]; // the part that changes; the rest is the run's title
  const until = (what: Parameters<Page["waitForTimeout"]>[0] | (() => Promise<unknown>)) =>
    r.offRecord(async () => (typeof what === "function" ? what() : page.waitForTimeout(what)));

  // ---- opening
  await r.card(
    "Research Tool (rtool)",
    "An AI research assistant that runs a study end to end: literature, hypotheses, experiment plan, code and GPU runs, report and review. You make every important decision.",
    7000,
  );
  await r.card(
    "About this recording",
    "Demo mode: the papers, AI model and cluster are simulated so the whole flow fits in a few minutes. On Barkla the same pages drive real searches, the Nemotron model and real GPU jobs. Waiting time is cut out.",
    8000,
  );

  // ---- projects and a new question
  await r.caption("All research runs in one place: where each one is, and which ones are waiting for you.", 4500);
  await r.caption(`The browser tab counts them too: “${await tab()}”`, 4000);
  await r.click(page.getByRole("link", { name: "New research" }).first());
  await r.caption("Start a study by asking a question in plain words.", 1500);
  await r.type(page.getByLabel("Research question or topic"), QUESTION);
  await r.caption("Optional settings: field, which steps need your approval, GPU budget and constraints.", 0);
  await r.scroll(500, 3500);
  await r.toTop(800);
  await r.click(page.getByRole("button", { name: "Start research" }), { hold: 1200 });

  // ---- live progress
  await expect(main.getByText(/Working: Literature|Literature/).first()).toBeVisible(WAIT);
  await r.caption("Live progress: each step of the literature agent is ticked off as it finishes (in demo mode they take seconds).", 6000);
  await r.caption("The activity feed explains each event in plain words, newest first.", 0);
  await r.scroll(700, 3500);
  await r.toTop(600);
  await until(() => expect(page.getByRole("button", { name: "Approve & continue" })).toBeVisible(WAIT));
  await r.caption(`Checkpoint: the run stops for your decision. Tab: “${await tab()}”`, 4500);

  // ---- literature checkpoint
  await r.click(stage(/^Literature/), { hold: 1500 });
  await r.point(page.getByRole("note", { name: "Warnings" }));
  await r.caption("“Check before you decide”: problems the tool found itself, such as too few papers.", 4500);
  await r.caption("Counts at a glance, then how the AI understood the topic, so you can correct it early.", 0);
  await r.scroll(420, 4500);
  await r.point(page.getByRole("region", { name: "Quality checks" }));
  await r.caption("Automatic quality checks on the paper set.", 3500);
  const first = page.getByRole("article").first();
  await r.caption("Every paper: why it is relevant, how it was found, and its summary.", 0);
  await r.point(first.getByRole("heading"));
  await r.hold(2500);
  await r.click(first.getByRole("button", { name: /Key findings/ }), { hold: 1200 });
  await r.caption("Key findings are extracted by the AI and each one is checked against a quote from the paper.", 4500);

  // keep / drop marks first, so they can be seen surviving a visit to the reader
  await r.caption("You can mark papers to keep or drop.", 0);
  const articles = page.getByRole("article");
  await r.click(articles.nth(0).getByRole("button", { name: "Keep" }), { hold: 500 });
  for (const i of [9, 8, 7, 6, 5]) await r.click(articles.nth(i).getByRole("button", { name: "Drop" }), { hold: 300 });
  await r.click(first.getByRole("link", { name: "Open full paper" }), { hold: 1500 });
  await expect(page.getByRole("button", { name: /Text with findings marked/ })).toBeVisible();
  await r.caption("The paper reader: the full text with every extracted finding marked where it comes from.", 5000);
  await r.scroll(400, 2500);
  await r.click(page.getByRole("link", { name: "All papers" }), { hold: 1200 });
  await r.point(articles.nth(0).getByRole("button", { name: "Keep" }));
  await r.caption("Back on the list, your marks are still there.", 3500);
  for (const [name, text] of [
    ["Map", "A map of the field: sub-topics, methods and how the papers connect."],
    ["Gaps & open problems", "Gaps and open problems the papers point to: where a new study can add something."],
  ] as const) {
    await r.click(page.getByRole("tab", { name }), { hold: 600 });
    await r.caption(text, 4000);
  }
  await r.click(page.getByRole("tab", { name: "Papers" }), { hold: 600 });

  // send the marks with a suggested request
  await r.toTop(900);
  await r.point(page.getByRole("button", { name: "Approve & continue" }));
  await r.caption("Unsent marks block “Approve”, so a choice is never lost by accident.", 4500);
  await r.click(page.getByRole("button", { name: /Send to the literature agent/ }), { hold: 900 });
  await r.caption("The request lists your choices. Suggestions come from what the tool found; one click adds one.", 4000);
  const dialog = page.getByRole("dialog");
  await r.click(dialog.getByRole("listitem").getByRole("button").first(), { hold: 1500 });
  await r.click(dialog.getByRole("button", { name: "Send", exact: true }), { hold: 1500 });
  await r.caption("The literature agent runs again with your choices and comment.", 3000);
  await until(async () => {
    await expect(page.getByRole("region", { name: "What changed" })).toBeVisible(WAIT);
    await expect(page.getByRole("button", { name: "Approve & continue" })).toBeEnabled(WAIT);
  });
  await r.point(page.getByRole("region", { name: "What changed" }));
  await r.caption("Each new round says what it acted on: your comment and the papers you kept or dropped.", 5000);
  const more = page.getByRole("region", { name: "What changed" }).getByRole("button", { name: /and \d+ more/ });
  if (await more.count()) {
    await r.click(more.first(), { hold: 800 });
    await r.caption("Long lists are shortened; one click shows them all.", 3000);
  }
  await r.click(page.getByRole("button", { name: "Approve & continue" }), { hold: 1200 });

  // ---- hypotheses
  await r.click(stage(/^Hypotheses/), { hold: 600 });
  await until(() => expect(page.getByRole("button", { name: /Edit & choose/ }).first()).toBeVisible(WAIT));
  await r.caption("Ranked hypotheses, each tied to the papers that support it.", 4500);
  await r.point(page.getByRole("link", { name: "Show in paper" }).first());
  await r.caption("“Show in paper” jumps to the quote behind a claim.", 3500);
  await r.click(page.getByRole("tab", { name: "Compare" }), { hold: 800 });
  await r.caption("Compare the candidates side by side on the same criteria.", 4000);
  await r.click(page.getByRole("tab", { name: "Side by side" }), { hold: 500 });
  await r.caption("Not convinced? Ask for new ones, saying what should change.", 0);
  await r.click(page.getByRole("button", { name: "Ask for new hypotheses" }), { hold: 600 });
  await r.type(page.getByRole("dialog").getByRole("textbox"), "Include one hypothesis about older adults.");
  await r.click(page.getByRole("dialog").getByRole("button", { name: "Send", exact: true }), { hold: 1200 });
  await until(() => expect(page.getByRole("region", { name: "What changed" })).toBeVisible(WAIT));
  await until(() => expect(page.getByRole("button", { name: /Edit & choose/ }).first()).toBeVisible(WAIT));
  await r.point(page.getByRole("region", { name: "What changed" }));
  await r.caption(
    (await page.getByRole("region", { name: "What changed" }).innerText()).includes("same set")
      ? "Each redo says what it acted on. (The demo model returned the same set; a real redo keeps the old set one click away.)"
      : "The new set says why it was redone, and the previous set stays one click away.",
    5500,
  );
  const previous = page.getByRole("button", { name: /See the previous set/ });
  if (await previous.count()) {
    await r.click(previous.first(), { hold: 3500 });
    await r.click(previous.first(), { hold: 600 });
  }
  await r.click(page.getByRole("button", { name: /Edit & choose/ }).first(), { hold: 800 });
  await r.caption("Choose one as it is, or edit it first in your own words.", 2000);
  const statement = page.getByRole("dialog").getByRole("textbox", { name: "Hypothesis", exact: true });
  await r.click(statement, { hold: 200 });
  await page.keyboard.press("End");
  await statement.pressSequentially(", in adults of any age", { delay: 35 });
  await r.hold(1200);
  await r.click(page.getByRole("button", { name: "Choose my version" }), { hold: 800 });
  await until(() => expect(page.getByRole("region", { name: "Your choice" })).toBeVisible(WAIT));

  // ---- plan
  await r.click(stage(/^Experiment plan/), { hold: 600 });
  await until(() => expect(page.getByRole("button", { name: "Approve plan & start downloads" })).toBeVisible(WAIT));
  await r.caption("The experiment plan: variables, conditions, measurements and statistics, datasets and compute.", 0);
  await r.scroll(600, 3000);
  await r.scroll(600, 3000);
  await r.toTop(600);
  await r.click(page.getByRole("button", { name: /Edit the plan/ }), { hold: 800 });
  await r.caption("Change anything in a form; every edit is checked before it is applied.", 0);
  const seeds = page.getByRole("spinbutton", { name: /seeds\) per condition/ });
  await r.point(seeds);
  await seeds.fill("5");
  await r.hold(1200);
  await r.click(page.getByRole("button", { name: "Apply my edits" }), { hold: 600 });
  await until(() => expect(page.getByText(/Version 2 · edited by/)).toBeVisible(WAIT));
  await r.caption("Your edit becomes version 2; earlier versions are kept.", 3500);
  await r.click(page.getByRole("button", { name: "Approve plan & start downloads" }), { hold: 1000 });

  // ---- datasets, code and GPU approval
  await r.click(stage(/^Code & runs/), { hold: 600 });
  await until(() => expect(page.getByRole("button", { name: "Approve GPU runs" })).toBeVisible(WAIT));
  await r.click(stage(/^Datasets & models/), { hold: 1200 });
  await r.caption("Datasets and models named in the plan are downloaded and checked first.", 4000);
  await r.click(stage(/^Code & runs/), { hold: 800 });
  await r.caption("The coding agent wrote the experiment and tested it on a CPU. Nothing uses a GPU until you approve.", 5000);
  await r.point(page.getByRole("textbox", { name: "experiment.py" }));
  await r.caption("You can read and edit the code here before it runs.", 3500);
  await r.click(page.getByRole("tab", { name: /Tests & repairs/ }), { hold: 600 });
  await r.caption("Tests and any automatic repairs are listed.", 3000);
  await r.click(page.getByRole("tab", { name: "Code" }), { hold: 400 });
  await r.click(page.getByRole("button", { name: "Approve GPU runs" }), { hold: 1200 });
  await r.caption("Approved: the runs are submitted to the cluster.", 0);
  await r.hold(4000);
  await until(() => expect(page.getByRole("tab", { name: "Results" })).toBeVisible(WAIT));
  await until(() => expect(stage(/^Code & runs/)).toContainText(/Done/, WAIT));
  await r.click(page.getByRole("tab", { name: "Results" }), { hold: 800 });
  await r.caption("Results per condition, with the statistics the plan asked for.", 0);
  await r.scroll(500, 4500);
  await r.toTop(400);

  // ---- report and review
  await r.click(stage(/^Review/), { hold: 600 });
  await until(() => expect(page.getByRole("button", { name: "Approve & finish" })).toBeVisible(WAIT));
  await r.click(stage(/^Report/), { hold: 1200 });
  await r.caption("The report is written from the results; every number is filled in from the data, not typed by the AI.", 0);
  await r.scroll(700, 3500);
  await r.point(page.getByRole("link", { name: /et al\., 20/ }).first());
  await r.caption("Citations link back to the papers from the literature step.", 3500);
  await r.toTop(500);
  await r.point(page.getByRole("link", { name: "Everything (.zip)" }));
  await r.caption("Download it as Markdown, BibTeX, PDF, or everything in one zip.", 3500);
  await r.click(stage(/^Review/), { hold: 1200 });
  await r.caption("A separate reviewer scores the report and runs automated checks on it.", 0);
  await r.scroll(500, 4500);
  await r.toTop(400);
  await r.click(page.getByRole("button", { name: "Approve & finish" }), { hold: 1200 });

  // ---- finished
  await r.click(stage(/^Overview/), { hold: 600 });
  await until(() => expect(page.getByRole("region", { name: "What this study found" })).toBeVisible(WAIT));
  await r.point(page.getByRole("region", { name: "What this study found" }));
  await r.caption("When it finishes, the Overview sums up the study: the hypothesis, the verdict and the main result.", 5500);
  await r.caption("Below: the full activity history and how much AI and GPU time the study used.", 0);
  await r.scroll(800, 3500);
  await r.toTop(500);
  await r.caption("", 0);
  const theme = page.getByRole("button", { name: /^Theme:/ });
  await r.click(theme, { hold: 300 }); // system -> light
  await r.click(theme, { hold: 300 }); // light -> dark
  await r.caption("There is a dark theme too.", 3500);
  await r.click(theme, { hold: 600 }); // back to the system setting
  await r.caption("", 0);

  await r.card(
    "Research Tool (rtool)",
    "Runs on the University of Liverpool's Barkla cluster with an open model (Nemotron Nano). Every step can be checked, edited or sent back.",
    7000,
  );
  console.log("frames", await r.stop());
});
