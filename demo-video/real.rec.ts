/** The real-run supplement: the same web UI on Barkla, showing real runs (real papers, the Nemotron model on the
 * cluster's GPUs, real Slurm jobs). Read only: it opens pages, tabs and the paper reader and never decides anything.
 *
 *   DEMO_PORT=8780 DEMO_TOKEN=<Barkla UI token file> npx playwright test real.rec.ts
 *   swift encode.swift out/real-frames rtool-real-run.mp4
 */
import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { Recorder } from "./recorder";

const LIVE = "/runs/ui-real-test/20261116-013530-1fc7"; // waiting at the hypothesis checkpoint
const DONE = "/runs/verify-audit-fixes/verify-20261112"; // finished end to end
const STUCK = "/runs/verify-audit-fixes/verify-20261112-code-budget3"; // stopped with an error

test("real", async ({ page }) => {
  await page.goto(`/#token=${readFileSync(process.env.DEMO_TOKEN!, "utf8").trim()}`);
  await expect(page.getByRole("heading", { name: "Research projects" })).toBeVisible();
  await page.goto(LIVE); // not the projects list: it shows every project on the account
  const main = page.locator("main");
  const activity = page.getByRole("list", { name: "Activity" });
  await expect(activity).toBeVisible();
  const r = new Recorder(page, "out/real-frames", "Real run · Barkla GPUs · Nemotron Nano");
  await r.start();
  const stage = (name: RegExp) => page.getByRole("navigation", { name: "Run" }).getByRole("link", { name });
  const go = async (path: string, ready: () => Promise<unknown>) =>
    r.offRecord(async () => {
      await page.goto(path);
      await ready();
      await page.waitForTimeout(800);
    });
  const text = async (pattern: RegExp) => (await main.getByText(pattern).first().innerText()).trim();

  await r.card(
    "rtool on Barkla: real runs",
    "The same pages as the demo, on real runs: real literature searches, the Nemotron Nano model on the University's GPUs, real cluster jobs. Nothing is simulated; only waiting time is cut, and nothing is decided in this recording.",
    9000,
  );

  // ---- a live run, waiting for its researcher
  await r.point(page.getByRole("region", { name: "Right now" }));
  await r.caption(`A live run: “${(await page.getByRole("heading", { level: 1 }).innerText()).trim()}” It is waiting for its researcher to choose a hypothesis.`, 6000);
  const queued = activity.getByText(/^Queued a GPU worker/).first();
  await r.point(queued);
  await r.caption(`Each step asks the cluster for a GPU and shows Slurm's own start estimate: “${(await queued.innerText()).trim()}”`, 6000);
  const ready = activity.getByText(/^AI model ready on/).first();
  await r.point(ready);
  await r.caption(`The model is started on whatever GPU the cluster gives: “${(await ready.innerText()).trim()}”`, 5000);
  const interrupted = activity.getByText(/was interrupted and will continue/).last(); // the first one: a preemption
  await r.point(interrupted);
  await r.caption("When a job is stopped part-way (the cluster can take a GPU back), the step resumes where it stopped, on another node.", 5000);
  const found = activity.getByText(/^Search found \d+ candidate papers/).first();
  await r.point(found);
  await r.caption(`Real searches: “${(await found.innerText()).trim()}”, then screening, citation following and reading.`, 5000);
  await r.toTop(600);

  // ---- literature after a second round
  await r.click(stage(/^Literature/), { hold: 1200 });
  const changed = page.getByRole("region", { name: "What changed" });
  await r.point(changed);
  await r.caption("The researcher sent the first paper set back with keep and drop marks; the second round says what it acted on.", 6000);
  const full = (await main.getByText("papers read in full", { exact: true }).locator("xpath=preceding-sibling::*[1]").innerText()).trim();
  await r.caption(`${full} papers read in full; every extracted finding is checked against a quote from the paper.`, 0);
  await r.scroll(450, 4500);
  const first = page.getByRole("article").first();
  await r.point(first.getByRole("heading"));
  await r.caption("Real papers, each with why it is relevant and a summary.", 3500);
  await r.click(first.getByRole("button", { name: /Key findings/ }), { hold: 1200 });
  await r.caption("Its key findings, as the model extracted them.", 4000);
  await r.click(first.getByRole("link", { name: "Open full paper" }), { hold: 1500 });
  await r.click(page.getByRole("button", { name: /Text with findings marked/ }), { hold: 1500 });
  await r.caption("The reader shows the paper's own text with each finding marked where it comes from.", 5000);
  await r.scroll(500, 3000);
  await r.click(page.getByRole("link", { name: "All papers" }), { hold: 1000 });
  await r.click(page.getByRole("tab", { name: "Map" }), { hold: 800 });
  await r.scroll(600, 300);
  await r.caption("The map of the field, built from the papers this run read.", 4500);
  await r.click(page.getByRole("tab", { name: "Papers" }), { hold: 400 });
  await r.toTop(500);

  // ---- hypotheses: shown, not chosen
  await r.click(stage(/^Hypotheses/), { hold: 1200 });
  await r.caption(`${await text(/offered from \d+ drafted/)}: each one is checked against the papers it cites and scored.`, 5000);
  await r.point(page.getByRole("link", { name: "Show in paper" }).first());
  await r.caption("Each cited claim shows the quote it rests on.", 3500);
  await r.point(page.getByRole("button", { name: "Ask for new hypotheses" }));
  await r.caption("These are the model's drafts, unedited (two even share a title). The researcher chooses one, edits it, or asks for new ones.", 6500);
  await r.caption("", 0);

  // ---- a run that finished
  await go(DONE, () => expect(page.getByRole("region", { name: "What this study found" })).toBeVisible());
  await r.point(page.getByRole("region", { name: "What this study found" }));
  await r.caption(`A run that went all the way: “${(await page.getByRole("heading", { level: 1 }).innerText()).trim()}”`, 5000);
  await r.caption(`The verdict is honest: ${await text(/^(Supported|Refuted|Inconclusive)$/)}, judged against the success criterion set before the experiment.`, 6000);
  await r.click(stage(/^Code & runs/), { hold: 1000 });
  await r.click(page.getByRole("tab", { name: /^Runs/ }), { hold: 1000 });
  await r.caption("The experiment ran on the cluster's GPUs: each cell is one run's result, with its cluster log a click away.", 5000);
  await r.click(page.getByRole("tab", { name: "Results" }), { hold: 1000 });
  await r.caption("Results with 95% confidence intervals, computed from the run outputs.", 0);
  await r.scroll(450, 5000);
  await r.toTop(400);
  await r.click(stage(/^Report/), { hold: 1500 });
  await r.caption("The report the Writer drafted from those results; the researcher approved it after review.", 0);
  await r.scroll(700, 4000);
  await r.toTop(400);
  await r.caption("", 0);

  // ---- a run that needs help
  await go(`${STUCK}/code`, () => expect(page.getByRole("region", { name: "Needs attention" })).toBeVisible());
  await r.point(page.getByRole("region", { name: "Needs attention" }));
  await r.caption("When a step cannot fix itself, it stops, says why, and waits for the researcher.", 6000);
  await r.point(page.getByRole("link", { name: "See the tests and repairs" }));
  await r.caption("It shows the repairs it already tried on its own.", 4000);
  await r.point(page.getByRole("button", { name: "Continue from where it stopped" }));
  await r.caption("The researcher can edit the code, add guidance, then continue or start the step again.", 5500);
  await r.caption("", 0);

  await r.card(
    "rtool on Barkla",
    "Open model, University GPUs, and every step can be checked, edited or sent back by the researcher.",
    6000,
  );
  console.log("frames", await r.stop());
});
