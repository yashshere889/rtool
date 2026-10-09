import { describe, expect, it } from "vitest";

import { lines, projectName } from "./project";

describe("new research helpers", () => {
  it("makes the same project names as the server", () => {
    expect(projectName("Does sound during deep sleep help older adults remember?")).toBe("sound-deep-sleep-help");
    expect(projectName("?!")).toBe("research");
  });

  it("splits one-per-line fields", () => {
    expect(lines(" first \n\n second\n")).toEqual(["first", "second"]);
  });
});
