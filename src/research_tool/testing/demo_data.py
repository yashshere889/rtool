"""Content of the demo research run: a small, fictional literature on closed-loop auditory stimulation.

Everything here is invented for demonstrating and testing the tool (papers,
authors, DOIs under the 10.5555 example prefix, results). Each paper has full
text in sections, and the facts the Literature agent extracts from it, with
quotes copied word for word from that text so they pass quote verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TOPIC = "Closed-loop auditory stimulation during slow-wave sleep and memory consolidation"
QUESTIONS = ["Does phase-locked stimulation improve overnight recall more than untimed stimulation?"]
SEARCH_QUERIES = [
    "closed-loop auditory stimulation slow oscillation memory",
    "phase-locked acoustic stimulation sleep recall",
    "slow oscillation spindle coupling memory consolidation",
    "auditory stimulation sleep older adults",
]


@dataclass(frozen=True)
class Finding:
    claim: str
    quote: str
    method: str = ""
    dataset: str = ""
    metric: str = ""
    value: str = ""


@dataclass(frozen=True)
class DemoPaper:
    key: str  # used for URLs and file markers in the demo only
    title: str
    authors: tuple[str, ...]
    year: int
    venue: str
    abstract: str
    sections: tuple[tuple[str, str], ...]  # (heading, text)
    relevance: int  # what the fake model answers when screening (0-10)
    reason: str
    problem: str = ""
    approach: str = ""
    methods: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    results: tuple[Finding, ...] = ()
    limitations: tuple[tuple[str, str], ...] = ()  # (statement, quote)
    future_work: tuple[tuple[str, str], ...] = ()
    doi: str | None = None
    arxiv_id: str | None = None
    access: str = "pdf"  # pdf | html (arXiv HTML) | none (paywalled: needs a manual upload)
    citations: int = 0
    found_by_snowballing: bool = False  # only reachable as a reference of the first paper
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """The paper as Markdown, as the PDF converter would return it (with the usual closing sections)."""
        parts = [f"# {self.title}", "", ", ".join(self.authors), "", "## Abstract", self.abstract, ""]
        for heading, body in [*self.sections, *self.closing_sections()]:
            parts += [f"## {heading}", body, ""]
        cited = [BY_KEY[k] for k in REFERENCES.get(self.key, ())]
        parts += ["## References"]
        parts += [f"[{n}] {', '.join(c.authors)}. {c.title}. {c.venue}, {c.year}." for n, c in enumerate(cited, 1)]
        parts += [f"[{len(cited) + 1}] Further references are listed in the supplementary material."]
        return "\n".join(parts)

    def closing_sections(self) -> list[tuple[str, str]]:
        simulated = "network model" in self.methods or "meta-analysis" in self.methods
        ethics = (
            "This study used simulations and published summary data only, so no ethical approval was required. "
            "All simulation settings are listed in the supplementary material so that others can repeat them."
            if simulated
            else "All participants gave written informed consent, and the study was approved by the local research "
            "ethics committee. Participants were screened for sleep disorders, hearing problems and medication that "
            "affects sleep before taking part, and they could withdraw at any time without giving a reason."
        )
        return [
            ("Ethics statement", ethics),
            (
                "Data availability",
                "Anonymised data and analysis code are available from the authors on reasonable request. The "
                "processing pipeline, including all parameters, is described in the supplementary material, and "
                "summary statistics for every figure are provided as tables.",
            ),
            (
                "Author contributions",
                f"{self.authors[0]} designed the study and analysed the data. All authors interpreted the results, "
                "wrote the manuscript and approved the final version.",
            ),
        ]

    def extraction(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "approach": self.approach,
            "methods": list(self.methods),
            "datasets": list(self.datasets),
            "metrics": list(self.metrics),
            "key_results": [f.__dict__ for f in self.results],
            "limitations": [{"statement": s, "quote": q} for s, q in self.limitations],
            "future_work": [{"statement": s, "quote": q} for s, q in self.future_work],
        }

    def digest(self) -> dict[str, Any]:
        """What the fake model answers for the paper's digest (built from the same facts)."""
        methods = next((body for heading, body in self.sections if "method" in heading.lower()), "")
        first = methods.split(". ")[0].rstrip(".") + "." if methods else "not stated"
        finding = self.results[0].claim if self.results else ""
        return {
            "summary": " ".join(part for part in (self.problem, self.approach, finding) if part),
            "design": self.approach or "not stated",
            "sample": ", ".join(self.datasets) or "not stated",
            "procedure": first,
            "gaps": [s for s, _ in self.limitations][:2] + [s for s, _ in self.future_work][:1],
        }


CLAS = "closed-loop auditory stimulation"
OPEN = "open-loop auditory stimulation"
COUPLING = "phase-coupling analysis"
SPINDLES = "spindle detection"

PAPERS: tuple[DemoPaper, ...] = (
    DemoPaper(
        key="morgan2019",
        title="Phase-locked auditory stimulation enhances slow oscillations and overnight recall",
        authors=("Alice Morgan", "Lei Chen", "Rahim Iqbal"),
        year=2019,
        venue="Sleep & Memory Letters",
        doi="10.5555/demo.2019.001",
        citations=214,
        abstract=(
            "Slow oscillations during deep sleep are thought to support the consolidation of declarative memory. "
            "We delivered brief pink-noise clicks phase-locked to the up-state of ongoing slow oscillations in "
            "healthy young adults. Stimulation increased slow-oscillation amplitude and improved overnight "
            "word-pair recall compared with a sham night."
        ),
        sections=(
            (
                "Introduction",
                "Sleep after learning benefits the retention of new declarative memories. Slow oscillations, the "
                "dominant rhythm of deep non-REM sleep, group thalamocortical spindles and hippocampal ripples into "
                "windows of heightened plasticity. Whether these oscillations can be strengthened from outside the "
                "brain, and whether doing so improves memory, remained an open question.",
            ),
            (
                "Methods",
                "Twenty-four healthy adults aged between 19 and 31 years took part in a within-subject design with "
                "a stimulation night and a sham night in counterbalanced order. Participants learned 120 word pairs "
                "in the evening and were tested before sleep and the next morning. A real-time algorithm detected "
                "the negative peak of each slow oscillation in a frontal EEG channel and delivered two clicks timed "
                "to the following up-state. During the sham night the algorithm ran but no sound was played.",
            ),
            (
                "Results",
                "Stimulation increased the amplitude of the slow oscillations that followed the clicks by 24 percent "
                "relative to sham. Overnight retention of word pairs was higher after the stimulation night than "
                "after the sham night, with a mean gain of 2.3 additional word pairs. The memory benefit correlated "
                "with the increase in slow-oscillation amplitude across participants.",
            ),
            (
                "Discussion",
                "Timing the clicks to the up-state appears to be essential, because sounds that arrive during the "
                "down-state tend to disrupt the oscillation. Our results show that a non-invasive and inexpensive "
                "intervention can strengthen a sleep rhythm that is causally involved in memory.",
            ),
            (
                "Limitations",
                "The sample consisted only of young, healthy university students, so the effect in older or "
                "clinical populations is unknown. We tested memory only on the morning after a single night of "
                "stimulation. Future studies should test whether repeated nights of stimulation produce lasting "
                "memory benefits.",
            ),
        ),
        relevance=10,
        reason="Directly tests closed-loop stimulation of slow oscillations and its effect on recall.",
        problem="Whether strengthening slow oscillations with sound during sleep improves declarative memory.",
        approach="Within-subject comparison of a night of up-state phase-locked clicks against a sham night.",
        methods=(CLAS, "real-time slow-wave detection"),
        datasets=("in-lab young adult cohort",),
        metrics=("word-pair recall", "slow-oscillation amplitude"),
        results=(
            Finding(
                "Phase-locked clicks enlarged the following slow oscillations.",
                "Stimulation increased the amplitude of the slow oscillations that followed the clicks by 24 "
                "percent relative to sham.",
                CLAS,
                "in-lab young adult cohort",
                "slow-oscillation amplitude",
                "24 percent",
            ),
            Finding(
                "Overnight word-pair retention improved after stimulation.",
                "Overnight retention of word pairs was higher after the stimulation night than after the sham "
                "night, with a mean gain of 2.3 additional word pairs.",
                CLAS,
                "in-lab young adult cohort",
                "word-pair recall",
                "2.3 word pairs",
            ),
            Finding(
                "The memory gain tracked the amplitude increase.",
                "The memory benefit correlated with the increase in slow-oscillation amplitude across participants.",
                CLAS,
                "in-lab young adult cohort",
                "word-pair recall",
            ),
        ),
        limitations=(
            (
                "Only young healthy students were tested.",
                "The sample consisted only of young, healthy university students, so the effect in older or "
                "clinical populations is unknown.",
            ),
            (
                "Memory was tested after a single night only.",
                "We tested memory only on the morning after a single night of stimulation.",
            ),
        ),
        future_work=(
            (
                "Test repeated nights of stimulation.",
                "Future studies should test whether repeated nights of stimulation produce lasting memory benefits.",
            ),
        ),
    ),
    DemoPaper(
        key="novak2020",
        title="Open-loop versus closed-loop acoustic stimulation in young adults: a randomised crossover study",
        authors=("Sara Novak", "Thomas Hale"),
        year=2020,
        venue="Journal of Sleep Methods",
        doi="10.5555/demo.2020.014",
        citations=131,
        abstract=(
            "Acoustic stimulation during slow-wave sleep can be delivered at fixed intervals or timed to the "
            "phase of the slow oscillation. In a randomised crossover study we compared open-loop stimulation, "
            "closed-loop stimulation and sham in the same participants. Only closed-loop stimulation improved "
            "overnight recall, while open-loop stimulation fragmented sleep in some participants."
        ),
        sections=(
            (
                "Introduction",
                "Most stimulation studies use expensive real-time systems that track the phase of the slow "
                "oscillation. Open-loop protocols, which play sounds at fixed intervals, are much simpler to deploy. "
                "It is not known how much of the memory benefit depends on precise phase targeting.",
            ),
            (
                "Methods",
                "Thirty adults completed three nights in randomised order: closed-loop stimulation, open-loop "
                "stimulation at a fixed interval of one second, and sham. Recall of a word-pair list and sleep "
                "architecture were measured on each night. Sleep stages were scored by two raters blind to the "
                "condition.",
            ),
            (
                "Results",
                "Closed-loop stimulation improved overnight recall relative to sham, whereas open-loop stimulation "
                "did not differ from sham. Open-loop stimulation increased the number of arousals during slow-wave "
                "sleep. The difference between closed-loop and open-loop stimulation in recall was moderate in size.",
            ),
            (
                "Discussion",
                "The benefit of stimulation depends on hitting the right phase of the oscillation. Simple timers "
                "are not a substitute for real-time phase detection when the goal is memory enhancement.",
            ),
            (
                "Limitations",
                "Each condition was tested on a single night, and practice effects across the three sessions may "
                "have reduced the differences between conditions. Future work should compare stimulation protocols "
                "in larger samples and across several nights.",
            ),
        ),
        relevance=9,
        reason="Directly compares phase-locked and untimed stimulation, the central question of the topic.",
        problem="Whether the memory benefit of acoustic stimulation requires phase-locked timing.",
        approach="Randomised crossover of closed-loop stimulation, open-loop stimulation and sham nights.",
        methods=(CLAS, OPEN),
        datasets=("in-lab young adult cohort",),
        metrics=("word-pair recall", "arousal count"),
        results=(
            Finding(
                "Only phase-locked stimulation improved recall.",
                "Closed-loop stimulation improved overnight recall relative to sham, whereas open-loop stimulation "
                "did not differ from sham.",
                CLAS,
                "in-lab young adult cohort",
                "word-pair recall",
            ),
            Finding(
                "Untimed stimulation disturbed deep sleep.",
                "Open-loop stimulation increased the number of arousals during slow-wave sleep.",
                OPEN,
                "in-lab young adult cohort",
                "arousal count",
            ),
        ),
        limitations=(
            (
                "One night per condition; practice effects possible.",
                "Each condition was tested on a single night, and practice effects across the three sessions may "
                "have reduced the differences between conditions.",
            ),
        ),
        future_work=(
            (
                "Larger, multi-night comparisons of protocols.",
                "Future work should compare stimulation protocols in larger samples and across several nights.",
            ),
        ),
    ),
    DemoPaper(
        key="osei2021",
        title="Slow oscillation-spindle coupling predicts memory retention across the lifespan",
        authors=("Kwame Osei", "Marie Laurent", "Jisoo Park"),
        year=2021,
        venue="arXiv preprint",
        arxiv_id="2103.04512",
        access="html",
        citations=96,
        abstract=(
            "Memory consolidation during sleep may depend on the precise timing of spindles relative to slow "
            "oscillations. Using overnight recordings from a public sleep database, we quantified the coupling "
            "between slow oscillations and spindles in adults of different ages. Tighter coupling predicted better "
            "overnight retention, and coupling declined with age."
        ),
        sections=(
            (
                "Introduction",
                "Spindles that occur near the peak of the slow oscillation are thought to be the most effective for "
                "memory. Ageing changes both oscillations, but the consequences for their coordination are unclear.",
            ),
            (
                "Methods",
                "We analysed polysomnography from 180 adults in the MASS database together with a word-pair task. "
                "Spindles were detected automatically and their phase relative to the slow oscillation was "
                "estimated with a Hilbert transform. Coupling strength was measured as the consistency of the "
                "spindle phase across events.",
            ),
            (
                "Results",
                "Participants whose spindles clustered near the slow-oscillation peak retained more word pairs "
                "overnight. Coupling strength declined steadily with age and explained part of the age-related loss "
                "of overnight retention. Spindle density alone was a weaker predictor than coupling strength.",
            ),
            (
                "Discussion",
                "The coordination between oscillations may matter more for memory than the amount of either "
                "oscillation. Interventions that restore coupling could therefore help older adults.",
            ),
            (
                "Limitations",
                "The analysis is correlational, so it cannot show that coupling causes better memory. Future "
                "studies should manipulate coupling directly, for example with phase-targeted stimulation.",
            ),
        ),
        relevance=9,
        reason="Links slow oscillation-spindle coupling to memory, the mechanism stimulation is meant to target.",
        problem="Whether the timing of spindles relative to slow oscillations predicts overnight memory retention.",
        approach="Correlational analysis of coupling strength in public polysomnography with a word-pair task.",
        methods=(COUPLING, SPINDLES),
        datasets=("MASS",),
        metrics=("coupling strength", "word-pair recall", "spindle density"),
        results=(
            Finding(
                "Spindles near the slow-oscillation peak predicted better retention.",
                "Participants whose spindles clustered near the slow-oscillation peak retained more word pairs "
                "overnight.",
                COUPLING,
                "MASS",
                "word-pair recall",
            ),
            Finding(
                "Coupling weakens with age.",
                "Coupling strength declined steadily with age and explained part of the age-related loss of "
                "overnight retention.",
                COUPLING,
                "MASS",
                "coupling strength",
            ),
            Finding(
                "Spindle density predicts memory less well than coupling.",
                "Spindle density alone was a weaker predictor than coupling strength.",
                SPINDLES,
                "MASS",
                "spindle density",
            ),
        ),
        limitations=(
            (
                "Correlational design.",
                "The analysis is correlational, so it cannot show that coupling causes better memory.",
            ),
        ),
        future_work=(
            (
                "Manipulate coupling with phase-targeted stimulation.",
                "Future studies should manipulate coupling directly, for example with phase-targeted stimulation.",
            ),
        ),
    ),
    DemoPaper(
        key="brandt2022",
        title="Ageing weakens the response to auditory slow-wave stimulation",
        authors=("Hanna Brandt", "Eduardo Silva"),
        year=2022,
        venue="Neurobiology of Ageing Reports",
        doi="10.5555/demo.2022.031",
        citations=58,
        abstract=(
            "Closed-loop auditory stimulation enhances slow oscillations in young adults, but older adults have "
            "smaller and fewer slow oscillations. We applied the same stimulation protocol to young and older "
            "adults. Older adults showed a smaller increase in slow-oscillation amplitude and no reliable memory "
            "benefit."
        ),
        sections=(
            (
                "Introduction",
                "Slow-wave sleep declines with age, and so does the overnight retention of new memories. If "
                "stimulation could restore slow oscillations in older adults, it might offer a simple way to support "
                "memory in later life.",
            ),
            (
                "Methods",
                "Twenty young and twenty older adults received closed-loop stimulation and sham on separate nights. "
                "Clicks were timed to the slow-oscillation up-state with the same detection threshold in both age "
                "groups. Memory was assessed with a word-pair task.",
            ),
            (
                "Results",
                "Older adults showed a smaller increase in slow-oscillation amplitude after the clicks than young "
                "adults. Young adults recalled more word pairs after stimulation, whereas older adults showed no "
                "reliable memory benefit.",
            ),
            (
                "Discussion",
                "A fixed detection threshold may miss many of the smaller slow waves of older adults. Adapting the "
                "threshold to each participant could make stimulation more effective in this group.",
            ),
            (
                "Limitations",
                "We used a fixed detection threshold that was tuned on young adults. Future studies should adapt the "
                "stimulation parameters to the individual sleep of each older participant.",
            ),
        ),
        relevance=9,
        reason="Shows how age changes the response to closed-loop stimulation, a key moderator.",
        problem="Whether closed-loop auditory stimulation helps older adults as it helps young adults.",
        approach="Stimulation versus sham nights in young and older adults with the same protocol.",
        methods=(CLAS,),
        datasets=("in-lab ageing cohort",),
        metrics=("slow-oscillation amplitude", "word-pair recall"),
        results=(
            Finding(
                "The amplitude response is smaller in older adults.",
                "Older adults showed a smaller increase in slow-oscillation amplitude after the clicks than young "
                "adults.",
                CLAS,
                "in-lab ageing cohort",
                "slow-oscillation amplitude",
            ),
            Finding(
                "No reliable memory benefit in older adults.",
                "Young adults recalled more word pairs after stimulation, whereas older adults showed no reliable "
                "memory benefit.",
                CLAS,
                "in-lab ageing cohort",
                "word-pair recall",
            ),
        ),
        limitations=(
            (
                "Detection threshold tuned on young adults.",
                "We used a fixed detection threshold that was tuned on young adults.",
            ),
        ),
        future_work=(
            (
                "Individualise stimulation parameters for older adults.",
                "Future studies should adapt the stimulation parameters to the individual sleep of each older "
                "participant.",
            ),
        ),
    ),
    DemoPaper(
        key="rossi2022",
        title="Automated real-time slow-wave detection on consumer EEG headbands",
        authors=("Davide Rossi", "Priya Gupta"),
        year=2022,
        venue="arXiv preprint",
        arxiv_id="2206.11873",
        citations=44,
        abstract=(
            "Closed-loop stimulation requires detecting slow waves as they happen. We evaluated a lightweight "
            "real-time detector on recordings from consumer EEG headbands and compared it with offline expert "
            "scoring. The detector found most slow waves with a short delay, making home stimulation feasible."
        ),
        sections=(
            (
                "Introduction",
                "Laboratory stimulation studies use research-grade amplifiers, which limits the number of nights "
                "that can be recorded. Consumer headbands could enable stimulation at home over many nights.",
            ),
            (
                "Methods",
                "We recorded 300 nights from 40 participants with a dry-electrode headband. The detector used a "
                "band-pass filter and an adaptive amplitude threshold, and its detections were compared with expert "
                "annotations.",
            ),
            (
                "Results",
                "The detector identified 87 percent of the slow waves marked by experts. The median delay between "
                "the true negative peak and the detection was 40 milliseconds.",
            ),
            (
                "Discussion",
                "Real-time detection on consumer hardware is accurate enough for phase-targeted stimulation at home.",
            ),
            (
                "Limitations",
                "Detection accuracy dropped in participants with frequent movement artefacts. We did not test "
                "whether stimulation delivered by the headband improves memory.",
            ),
        ),
        relevance=7,
        reason="Provides the detection method that phase-locked stimulation depends on; useful background.",
        problem="Whether slow waves can be detected in real time on inexpensive consumer EEG hardware.",
        approach="Comparison of a lightweight online detector with expert scoring on headband recordings.",
        methods=("real-time slow-wave detection",),
        datasets=("headband home recordings",),
        metrics=("detection accuracy", "detection delay"),
        results=(
            Finding(
                "Most expert-marked slow waves were detected online.",
                "The detector identified 87 percent of the slow waves marked by experts.",
                "real-time slow-wave detection",
                "headband home recordings",
                "detection accuracy",
                "87 percent",
            ),
            Finding(
                "Detections arrive quickly enough for phase targeting.",
                "The median delay between the true negative peak and the detection was 40 milliseconds.",
                "real-time slow-wave detection",
                "headband home recordings",
                "detection delay",
                "40 milliseconds",
            ),
        ),
        limitations=(
            (
                "Accuracy falls with movement artefacts.",
                "Detection accuracy dropped in participants with frequent movement artefacts.",
            ),
            (
                "Memory effects of home stimulation untested.",
                "We did not test whether stimulation delivered by the headband improves memory.",
            ),
        ),
    ),
    DemoPaper(
        key="adeyemi2023",
        title="Dose-response of acoustic stimulation: one, two or five clicks per slow wave",
        authors=("Folake Adeyemi", "Claire Moreau"),
        year=2023,
        venue="Journal of Sleep Methods",
        doi="10.5555/demo.2023.007",
        citations=27,
        abstract=(
            "Stimulation protocols differ in how many clicks are played after each detected slow wave. We compared "
            "one, two and five clicks per detection in a within-subject design. Two clicks produced the largest "
            "memory benefit, while five clicks reduced the benefit and increased arousals."
        ),
        sections=(
            (
                "Introduction",
                "The number of sounds played per slow wave is an important but rarely tested parameter. Too few "
                "clicks may not entrain the oscillation, while too many may wake the sleeper.",
            ),
            (
                "Methods",
                "Eighteen adults completed four nights: one, two or five clicks per detected slow wave, and sham. "
                "Clicks were spaced to coincide with successive up-states. Word-pair recall and arousals were "
                "measured.",
            ),
            (
                "Results",
                "Two clicks per slow wave produced the largest improvement in overnight recall. Five clicks per slow "
                "wave reduced the memory benefit and increased the number of arousals.",
            ),
            (
                "Discussion",
                "Short bursts of stimulation seem to strike the best balance between entraining slow oscillations "
                "and preserving sleep.",
            ),
            (
                "Limitations",
                "The sample size was small and each dose was tested on one night only.",
            ),
        ),
        relevance=8,
        reason="Tests a stimulation parameter that any closed-loop experiment has to choose.",
        problem="How many clicks per detected slow wave give the best memory benefit.",
        approach="Within-subject comparison of one, two and five clicks per detection against sham.",
        methods=(CLAS,),
        datasets=("in-lab young adult cohort",),
        metrics=("word-pair recall", "arousal count"),
        results=(
            Finding(
                "Two clicks gave the largest recall improvement.",
                "Two clicks per slow wave produced the largest improvement in overnight recall.",
                CLAS,
                "in-lab young adult cohort",
                "word-pair recall",
            ),
            Finding(
                "Five clicks hurt sleep and memory.",
                "Five clicks per slow wave reduced the memory benefit and increased the number of arousals.",
                CLAS,
                "in-lab young adult cohort",
                "arousal count",
            ),
        ),
        limitations=(
            (
                "Small sample, one night per dose.",
                "The sample size was small and each dose was tested on one night only.",
            ),
        ),
    ),
    DemoPaper(
        key="ivanova2021",
        title="Targeted memory reactivation during slow-wave sleep: a meta-analysis",
        authors=("Nadia Ivanova", "Gabriel Torres", "Bora Kim"),
        year=2021,
        venue="Memory & Cognition Reviews",
        doi="10.5555/demo.2021.052",
        citations=188,
        abstract=(
            "Targeted memory reactivation presents learning-related cues during sleep to strengthen specific "
            "memories. We pooled studies that delivered cues during slow-wave sleep. Cueing improved later memory "
            "for cued items with a small to moderate effect, and effects were larger when cues were delivered "
            "during slow-oscillation up-states."
        ),
        sections=(
            (
                "Introduction",
                "Cueing memories during sleep is a second way of influencing consolidation, alongside the "
                "enhancement of slow oscillations themselves.",
            ),
            (
                "Methods",
                "We searched for studies of targeted memory reactivation during slow-wave sleep and combined their "
                "effect sizes with a random-effects model.",
            ),
            (
                "Results",
                "Cueing during slow-wave sleep improved later memory for cued items with a small to moderate effect. "
                "Effects were larger when cues were timed to slow-oscillation up-states.",
            ),
            (
                "Discussion",
                "Timing matters for both cueing and stimulation, which suggests a shared mechanism in the up-state.",
            ),
            (
                "Limitations",
                "Most included studies had small samples and there was evidence of publication bias.",
            ),
        ),
        relevance=7,
        reason="Related intervention during slow-wave sleep; shows that up-state timing matters.",
        problem="How reliable the memory benefit of cueing during slow-wave sleep is across studies.",
        approach="Random-effects meta-analysis of targeted memory reactivation studies.",
        methods=("targeted memory reactivation", "meta-analysis"),
        datasets=("published TMR studies",),
        metrics=("effect size",),
        results=(
            Finding(
                "Cueing during deep sleep helps memory.",
                "Cueing during slow-wave sleep improved later memory for cued items with a small to moderate effect.",
                "targeted memory reactivation",
                "published TMR studies",
                "effect size",
            ),
            Finding(
                "Up-state timing strengthens cueing effects.",
                "Effects were larger when cues were timed to slow-oscillation up-states.",
                "targeted memory reactivation",
                "published TMR studies",
                "effect size",
            ),
        ),
        limitations=(
            (
                "Small samples and publication bias.",
                "Most included studies had small samples and there was evidence of publication bias.",
            ),
        ),
    ),
    DemoPaper(
        key="haddad2024",
        title="Closed-loop stimulation in mild cognitive impairment: a pilot study",
        authors=("Rania Haddad", "Yuki Sato"),
        year=2024,
        venue="Clinical Sleep Neuroscience",
        doi="10.5555/demo.2024.003",
        access="none",
        citations=12,
        abstract=(
            "People with mild cognitive impairment have reduced slow-wave sleep. In a pilot study, closed-loop "
            "auditory stimulation was well tolerated by people with mild cognitive impairment and increased "
            "slow-oscillation activity. Memory effects were not reliable in this small sample."
        ),
        sections=(),
        relevance=8,
        reason="Extends stimulation to a clinical population; important for generalisation.",
        problem="Whether closed-loop stimulation is feasible in mild cognitive impairment.",
        approach="Pilot study of stimulation and sham nights in people with mild cognitive impairment.",
        methods=(CLAS,),
        datasets=("MCI pilot cohort",),
        metrics=("slow-oscillation activity", "tolerability"),
        results=(
            Finding(
                "Stimulation was tolerated and increased slow-oscillation activity.",
                "In a pilot study, closed-loop auditory stimulation was well tolerated by people with mild cognitive "
                "impairment and increased slow-oscillation activity.",
                CLAS,
                "MCI pilot cohort",
                "slow-oscillation activity",
            ),
        ),
        limitations=(
            ("Memory effects unreliable in a small sample.", "Memory effects were not reliable in this small sample."),
        ),
    ),
    DemoPaper(
        key="petrov2023",
        title="Simulating sleep-dependent consolidation with a thalamocortical network model",
        authors=("Viktor Petrov", "Anna Lindqvist"),
        year=2023,
        venue="arXiv preprint",
        arxiv_id="2309.01456",
        citations=19,
        abstract=(
            "We built a thalamocortical network model that produces slow oscillations and spindles and stores "
            "associations in plastic synapses. Simulated stimulation timed to the up-state strengthened stored "
            "associations, while stimulation in the down-state weakened them."
        ),
        sections=(
            (
                "Introduction",
                "Computational models can test hypotheses about stimulation timing that are expensive to test in "
                "people.",
            ),
            (
                "Methods",
                "The model contained cortical and thalamic populations with spike-timing-dependent plasticity. We "
                "simulated auditory input as a brief excitatory drive delivered at different phases of the slow "
                "oscillation.",
            ),
            (
                "Results",
                "Simulated stimulation timed to the up-state strengthened the stored associations. Stimulation "
                "delivered in the down-state weakened the stored associations and desynchronised the network.",
            ),
            (
                "Discussion",
                "The model reproduces the phase dependence seen in human studies and predicts that mistimed "
                "stimulation is harmful.",
            ),
            (
                "Limitations",
                "The model is simplified and its parameters were not fitted to individual recordings. Future work "
                "should fit the model to human sleep recordings before using it to choose stimulation parameters.",
            ),
        ),
        relevance=6,
        reason="Modelling work that explains why phase matters; useful background for the design.",
        problem="Why the phase of stimulation determines its effect on memory.",
        approach="Simulation of stimulation at different slow-oscillation phases in a plastic network model.",
        methods=("network model",),
        datasets=("simulated recordings",),
        metrics=("association strength",),
        results=(
            Finding(
                "Up-state input strengthens memories in the model.",
                "Simulated stimulation timed to the up-state strengthened the stored associations.",
                "network model",
                "simulated recordings",
                "association strength",
            ),
            Finding(
                "Down-state input weakens memories in the model.",
                "Stimulation delivered in the down-state weakened the stored associations and desynchronised the "
                "network.",
                "network model",
                "simulated recordings",
                "association strength",
            ),
        ),
        limitations=(
            (
                "Simplified, not fitted to recordings.",
                "The model is simplified and its parameters were not fitted to individual recordings.",
            ),
        ),
        future_work=(
            (
                "Fit the model to human recordings.",
                "Future work should fit the model to human sleep recordings before using it to choose stimulation "
                "parameters.",
            ),
        ),
    ),
    DemoPaper(
        key="mensah2018",
        title="Spindle density and declarative memory: a replication in 120 adults",
        authors=("Ophelia Mensah", "Wei Zhou"),
        year=2018,
        venue="Sleep & Memory Letters",
        doi="10.5555/demo.2018.021",
        citations=302,
        found_by_snowballing=True,
        abstract=(
            "Earlier small studies reported that sleep spindle density predicts overnight memory retention. We "
            "replicated this association in 120 adults. Spindle density predicted retention, but the association "
            "was weaker than in the original reports."
        ),
        sections=(
            (
                "Introduction",
                "Replications of sleep and memory findings in large samples are rare.",
            ),
            (
                "Methods",
                "We recorded one night of polysomnography in 120 adults after they learned a word-pair list, and "
                "detected spindles automatically.",
            ),
            (
                "Results",
                "Spindle density predicted overnight retention, but the association was weaker than in the original "
                "reports.",
            ),
            (
                "Discussion",
                "Effects reported in small samples are likely to be overestimated. Our estimate is smaller but more "
                "precise, and it is consistent with a modest role of spindles in the overnight retention of word "
                "pairs. Future work should pre-register analyses and record several nights per participant.",
            ),
            (
                "Limitations",
                "We recorded only one night per participant, which adds noise to the spindle estimates.",
            ),
        ),
        relevance=8,
        reason="Large replication of the spindle-memory link; strong background evidence.",
        problem="Whether the spindle density-memory association replicates in a large sample.",
        approach="Polysomnography and a word-pair task in a sample of adults.",
        methods=(SPINDLES,),
        datasets=("in-lab replication cohort",),
        metrics=("spindle density", "word-pair recall"),
        results=(
            Finding(
                "Spindle density predicts retention, but weakly.",
                "Spindle density predicted overnight retention, but the association was weaker than in the original "
                "reports.",
                SPINDLES,
                "in-lab replication cohort",
                "spindle density",
            ),
        ),
        limitations=(
            (
                "A single night per participant.",
                "We recorded only one night per participant, which adds noise to the spindle estimates.",
            ),
        ),
    ),
    # off-topic search results, excluded at screening
    DemoPaper(
        key="lund2022",
        title="Graph neural surrogates for ocean heat transport",
        authors=("Erik Lund",),
        year=2022,
        venue="Climate Informatics",
        doi="10.5555/demo.2022.090",
        citations=33,
        abstract="We train graph neural networks to emulate an ocean circulation model.",
        sections=(),
        relevance=1,
        reason="Different field (climate modelling).",
    ),
    DemoPaper(
        key="mbeki2021",
        title="Crop yield forecasting from satellite time series",
        authors=("Thabo Mbeki",),
        year=2021,
        venue="Remote Sensing Letters",
        doi="10.5555/demo.2021.077",
        citations=58,
        abstract="We forecast crop yields from multispectral satellite images with recurrent networks.",
        sections=(),
        relevance=0,
        reason="Unrelated to sleep or memory.",
    ),
)

BY_TITLE = {p.title: p for p in PAPERS}
BY_KEY = {p.key: p for p in PAPERS}

# who cites whom inside the demo corpus (reference lists, and snowballing in the demo)
REFERENCES: dict[str, tuple[str, ...]] = {
    "morgan2019": ("mensah2018",),
    "novak2020": ("morgan2019", "mensah2018"),
    "osei2021": ("mensah2018", "morgan2019"),
    "brandt2022": ("morgan2019", "osei2021"),
    "adeyemi2023": ("morgan2019", "novak2020"),
    "haddad2024": ("brandt2022", "morgan2019"),
    "petrov2023": ("morgan2019", "osei2021"),
    "rossi2022": ("morgan2019",),
}

# ------------------------------------------------------------------ hypotheses
# Drafted one per opportunity, in this order; evidence is taken from the opportunity the tool offers.
HYPOTHESES: tuple[dict[str, Any], ...] = (
    {
        "title": "Phase-locked clicks beat untimed clicks for recall",
        "statement": (
            "If clicks are delivered in phase with the slow-oscillation up-state, then overnight word-pair recall "
            "improves more than with clicks delivered at fixed intervals"
        ),
        "null_hypothesis": "Overnight recall does not differ between phase-locked and fixed-interval clicks",
        "rationale": (
            "Studies agree that stimulation timed to the up-state enlarges slow oscillations, while untimed sounds "
            "fragment sleep; a direct comparison in one simulated cohort isolates the effect of timing."
        ),
        "independent_variables": ["stimulation timing (phase-locked, fixed-interval, sham)"],
        "dependent_variables": ["overnight recall gain", "slow-oscillation amplitude"],
        "controls": ["same number of clicks per night", "same sound intensity"],
        "datasets": ["Sleep-EDF"],
        "methods": [CLAS, OPEN],
        "success_criteria": (
            "Phase-locked stimulation yields a significantly larger recall gain than fixed-interval stimulation "
            "(paired test, alpha 0.05); no difference refutes the hypothesis"
        ),
        "risks": ["simulated recall is only a proxy for human memory"],
        "compute": "small_gpu",
        "scores": {"novelty": 4, "feasibility": 5, "testability": 5, "significance": 4, "clarity": 5},
        "comment": "A clean comparison with an obvious baseline; the main weakness is that recall is simulated.",
    },
    {
        "title": "Adaptive thresholds rescue the stimulation effect in ageing",
        "statement": (
            "If the slow-wave detection threshold is adapted to each older participant, then stimulation increases "
            "slow-oscillation amplitude in older adults as much as a fixed threshold does in young adults"
        ),
        "null_hypothesis": "An adaptive threshold does not increase the amplitude response in older adults",
        "rationale": (
            "Older adults respond weakly to protocols tuned on young adults, and the authors suggest individual "
            "parameters; an adaptive detector can be evaluated on recordings of older adults."
        ),
        "independent_variables": ["detection threshold (fixed, adaptive)", "age group"],
        "dependent_variables": ["slow-oscillation amplitude response"],
        "controls": ["same stimulation protocol"],
        "datasets": ["MASS"],
        "methods": ["real-time slow-wave detection", CLAS],
        "success_criteria": "The amplitude response of older adults with an adaptive threshold is within 10% of young",
        "risks": ["amplitude response is simulated from recordings, not measured during stimulation"],
        "compute": "small_gpu",
        "scores": {"novelty": 5, "feasibility": 3, "testability": 4, "significance": 5, "clarity": 4},
        "comment": "Addresses the ageing gap directly, but the amplitude response can only be approximated offline.",
    },
    {
        "title": "Coupling, not spindle density, carries the memory benefit",
        "statement": (
            "If stimulation improves recall, then the improvement is explained by stronger slow oscillation-spindle "
            "coupling rather than by higher spindle density"
        ),
        "null_hypothesis": "Coupling strength does not explain more of the recall gain than spindle density",
        "rationale": (
            "Coupling predicts retention better than spindle density, and phase-targeted stimulation is proposed as "
            "a way to manipulate coupling directly."
        ),
        "independent_variables": ["stimulation (on, sham)"],
        "dependent_variables": ["coupling strength", "spindle density", "recall gain"],
        "controls": ["sleep duration"],
        "datasets": ["MASS"],
        "methods": [COUPLING, SPINDLES],
        "success_criteria": "Coupling mediates a significant share of the recall gain while spindle density does not",
        "risks": ["mediation analysis needs many participants"],
        "compute": "cpu_only",
        "scores": {"novelty": 4, "feasibility": 4, "testability": 3, "significance": 4, "clarity": 3},
        "comment": "Mechanistically interesting; a mediation analysis needs more recordings than are likely available.",
    },
    {
        "title": "Two clicks per slow wave are optimal in home recordings",
        "statement": (
            "If stimulation is delivered at home with a consumer headband, then two clicks per detected slow wave "
            "improve recall more than one or five clicks"
        ),
        "null_hypothesis": "Recall does not depend on the number of clicks per slow wave at home",
        "rationale": "Laboratory dose-response data favour two clicks, and headband detection is accurate enough.",
        "independent_variables": ["clicks per slow wave (one, two, five)"],
        "dependent_variables": ["recall gain", "arousal count"],
        "controls": ["same headband"],
        "datasets": ["Sleep-EDF"],
        "methods": ["real-time slow-wave detection", CLAS],
        "success_criteria": "Two clicks give the highest recall gain with no increase in arousals",
        "risks": ["home recordings are noisy"],
        "compute": "small_gpu",
        "scores": {"novelty": 3, "feasibility": 4, "testability": 4, "significance": 3, "clarity": 4},
        "comment": "Practical, but laboratory dose-response work already covers most of it.",
    },
)

# ------------------------------------------------------------------ plan and code
PLAN_CORE: dict[str, Any] = {
    "objective": (
        "Test whether phase-locked clicks improve simulated overnight recall more than fixed-interval clicks"
    ),
    "design": (
        "A within-subject simulation on public sleep recordings: for each recording, slow waves are detected, "
        "stimulation is simulated as phase-locked clicks, fixed-interval clicks or sham, and a consolidation model "
        "converts the resulting slow-oscillation activity into a predicted recall gain. Seeds vary the detection "
        "noise and the sampled recordings."
    ),
    "independent_variables": [
        {"name": "stimulation", "description": "timing of the clicks", "values": ["sham", "phase-locked", "fixed"]}
    ],
    "dependent_variables": [
        {"name": "recall gain", "description": "predicted overnight change in recalled word pairs"},
        {"name": "slow-oscillation amplitude", "description": "mean amplitude after stimulation, in microvolts"},
    ],
    "controls": ["same recordings in every condition", "same number of clicks per night"],
    "conditions": [
        {"name": "Sham", "description": "detection runs, no clicks", "is_baseline": True},
        {"name": "Phase locked", "description": "two clicks timed to each detected up-state"},
        {"name": "Fixed interval", "description": "two clicks per second regardless of phase"},
    ],
    "ablations": [],
    "metrics": [
        {
            "name": "Recall gain",
            "definition": "predicted overnight gain in recalled word pairs",
            "higher_is_better": True,
        },
        {
            "name": "SO amplitude",
            "definition": "mean slow-oscillation amplitude after clicks",
            "higher_is_better": True,
        },
    ],
    "statistical_tests": [
        {"test": "paired t-test", "comparison": "phase_locked vs sham and fixed_interval vs sham on recall_gain"},
    ],
    "n_seeds": 4,
    "procedure": [
        "Load the Sleep-EDF recordings and keep nights with at least 30 minutes of N3 sleep",
        "Detect slow waves with the real-time detector and estimate their phase",
        "Simulate the clicks of each condition and the resulting slow-oscillation amplitude",
        "Convert slow-oscillation activity into a predicted recall gain with the consolidation model",
        "Write the metrics of each run to the results file",
    ],
    "expected_results": "Phase-locked clicks give the largest recall gain; fixed-interval clicks are close to sham",
    "risks": ["the consolidation model is a simplification", "Sleep-EDF has few older participants"],
}

RESOURCES: list[dict[str, Any]] = [
    {
        "kind": "dataset",
        "name": "Sleep-EDF",
        "source": "huggingface",
        "identifier": "demo-org/sleep-edf",
        "purpose": "overnight EEG recordings with sleep stages",
    },
    {
        "kind": "code",
        "name": "numpy",
        "source": "pip",
        "identifier": "numpy",
        "purpose": "signal processing",
    },
]

COMPUTE = {"gpu_memory_gb": 4, "gpus_per_run": 1, "hours_per_run": 0.1, "justification": "light signal processing"}

EXPERIMENT = '''"""Closed-loop auditory stimulation, simulated on sleep recordings (demo experiment)."""
# requirements:
import csv
import random
from pathlib import Path

EFFECT = {"sham": 0.0, "phase_locked": 2.1, "fixed_interval": 0.4}
AMPLITUDE = {"sham": 0.0, "phase_locked": 9.5, "fixed_interval": 1.5}


def deep_sleep_minutes(folder):
    """Minutes of N3 sleep in each recorded night."""
    with open(Path(folder) / "recordings.csv", newline="") as handle:
        return [float(row["n3_minutes"]) for row in csv.DictReader(handle)]


def run(condition, seed, resources, smoke, workdir):
    """One simulated night per seed and recording: returns the metrics the plan asks for."""
    minutes = deep_sleep_minutes(resources["Sleep-EDF"])
    rng = random.Random(seed * 7919 + len(condition))
    nights = 3 if smoke else 40
    gains, amplitudes = [], []
    for night in range(nights):
        deep = minutes[night % len(minutes)] / 90  # clicks only reach deep sleep
        baseline = rng.gauss(1.0, 0.6)
        # conditions added to the plan by hand get a middling effect
        gains.append(baseline + deep * EFFECT.get(condition, 1.2) + rng.gauss(0.0, 0.5))
        amplitudes.append(70.0 + deep * AMPLITUDE.get(condition, 5.0) + rng.gauss(0.0, 3.0))
    return {"recall_gain": sum(gains) / nights, "so_amplitude": sum(amplitudes) / nights}
'''

# ------------------------------------------------------------------ report
SECTION_TEXT = {  # {cite:<demo key>} becomes [@bib_key] when the Writer offers that paper; Results are generated
    "Introduction": (
        "Sleep after learning helps people keep new memories, and slow oscillations during deep sleep are a "
        "central part of this process{cite:morgan2019}. Sounds played in phase with these oscillations can "
        "enlarge them, and several studies report better recall the next morning{cite:novak2020}.\n\n"
        "It is less clear how much of this benefit depends on precise timing. Simpler protocols that play sounds "
        "at fixed intervals are easier to deploy at home, but they may disturb sleep{cite:novak2020}. This study "
        "tests whether phase-locked clicks improve recall more than fixed-interval clicks."
    ),
    "Related work": (
        "Laboratory studies agree that stimulation timed to the up-state enlarges the following slow "
        "oscillations{cite:morgan2019}. Direct comparisons with untimed stimulation found a memory benefit only "
        "for the phase-locked version{cite:novak2020}.\n\n"
        "Work on the mechanism points to the coupling between slow oscillations and spindles, which predicts "
        "overnight retention better than spindle density alone{cite:osei2021}. Models of the thalamocortical "
        "network reproduce the phase dependence and predict that mistimed input is harmful{cite:petrov2023}.\n\n"
        "The benefit appears to shrink with age, possibly because detection thresholds tuned on young adults miss "
        "the smaller slow waves of older adults{cite:brandt2022}."
    ),
    "Discussion": (
        "The results support the hypothesis that the timing of the clicks matters: phase-locked stimulation "
        "improved simulated recall, while untimed stimulation did little. This agrees with laboratory "
        "comparisons of the two protocols{cite:novak2020} and with modelling work on phase "
        "dependence{cite:petrov2023}.\n\n"
        "Limitations. The recall gain comes from a consolidation model, not from people, and the recordings "
        "include few older adults, so the size of the effect in older or clinical groups remains unknown."
    ),
    "Conclusion": (
        "Phase-locked clicks are the better choice for stimulation aimed at memory. The next step is to test the "
        "protocol in people, including older adults."
    ),
}
