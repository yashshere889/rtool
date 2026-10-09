// Same rule as the server (web/api/runs.py project_name): the group a new run goes into when none is given.
const STOPWORDS = new Set(
  "a an and the of on in for to with during does do is are how".split(" "),
);

export function projectName(topic: string): string {
  const words = (topic.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((w) => !STOPWORDS.has(w));
  const name = words.slice(0, 4).join("-").slice(0, 60).replace(/^-+|-+$/g, "");
  return name || "research";
}

export function lines(text: string): string[] {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}
