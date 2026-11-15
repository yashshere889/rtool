import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Round } from "@/api/types";

import { RoundBanner } from "./RoundBanner";

const round = (extra: Partial<Round>): Round => ({
  previous_version: 1,
  previous_at: new Date().toISOString(),
  feedback: [],
  inputs_changed: [],
  unchanged: false,
  ...extra,
});
const you = { by: "you" as const, from_stage: null, text: "", kept: [], dropped: [] };

describe("RoundBanner", () => {
  it("quotes the researcher's comment and lists the papers they dropped", () => {
    render(
      <RoundBanner round={round({ feedback: [{ ...you, text: "Focus on older adults.", dropped: ["Paper A"] }] })} what="set of papers">
        <p>1 new paper</p>
      </RoundBanner>,
    );
    expect(screen.getByText("You asked: “Focus on older adults.”")).toBeInTheDocument();
    expect(screen.getByText("You dropped “Paper A”")).toBeInTheDocument();
    expect(screen.getByText("1 new paper")).toBeInTheDocument();
  });

  it("says when a request came back with the same result, without a comparison", () => {
    render(
      <RoundBanner round={round({ unchanged: true, feedback: [{ ...you, text: "More on ageing" }] })} what="set of hypotheses">
        <p>See the previous set</p>
      </RoundBanner>,
    );
    expect(screen.getByText("The stage ran again on this request and came back with the same set of hypotheses.")).toBeInTheDocument();
    expect(screen.getByText("You asked: “More on ageing”")).toBeInTheDocument();
    expect(screen.queryByText("See the previous set")).not.toBeInTheDocument();
  });

  it("says when there was no comment, and when earlier results changed instead", () => {
    const { unmount } = render(<RoundBanner round={round({ feedback: [you] })} what="set of hypotheses" />);
    expect(screen.getByText("You asked for changes without a comment")).toBeInTheDocument();
    unmount();
    render(<RoundBanner round={round({ inputs_changed: ["corpus"] })} what="set of hypotheses" />);
    expect(screen.getByText(/redone because earlier results changed \(corpus\)/i)).toBeInTheDocument();
  });
});
