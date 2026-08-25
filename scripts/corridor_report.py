"""The corridor questions, answered on a road, WITH THE COVERAGE OF EVERY ANSWER BESIDE IT.

    .venv/bin/python scripts/corridor_report.py
    .venv/bin/python scripts/corridor_report.py --road "Broad Street" --model-notes

WHAT THIS REPLACES, and why the shape of the output is the whole point. Every corridor number in
the session recorded in docs/network-model.md came from a scratch script on raw OSM, and three of
them were wrong. Not wrong by a rounding error - wrong because a count taken over the part of a
street that happens to be surveyed was printed as a fact about the street. A stall count over a
corridor that is 43% traced is not a corridor figure; it is a figure about 43% of a corridor, and
the difference is the difference between "Broad St has room for 179 cars" and the truth.

So this module cannot emit a bare number. A `Figure` carries a `Coverage`, `Coverage` has no
default and no None, and `figure.line()` prints the two together. That is a structural guarantee
rather than a habit: tests/test_corridor.py asserts it over every figure of every road, so a new
question added here without a denominator fails the build instead of quietly reading as a total.

WHAT A DENOMINATOR IS, per figure, because they are not all the same question:

  * a WIDTH is only a measurement where BOTH kerbs are traced, so its denominator is
    Corridor.both_traced_ft against the road's length;
  * an OPENING is a fact about one kerb, so its denominator is traced kerb against both kerb
    lines - twice the length;
  * anything counted out of OSM is bounded by what was FETCHED, so its denominator is
    network.osm_window_spans - "nothing fetched" and "nothing mapped" arrive identically and only
    the first is fixable;
  * a STALL COUNT gets two figures rather than one, because there are two honest answers: what the
    law leaves parkable by LENGTH, and how much of that length has a measured width to check it
    against. Printing only the first is the mistake this file exists to stop; printing only the
    second understates a street somebody could go and trace.

Nothing here renders and nothing here writes a golden - see docs/network-renderer-plan.md, stream
C. It is a reading of src/geometry/network/, which is the object all of this now hangs off.
"""
import argparse
import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.geometry.corridor_paint import centred_on_its_kerbs
from src.geometry.network import (Corridor, CorridorFacts, KERB_FROM_TRACING, corridor_facts,
                                 corridors_from_models, marked_parking_capacity, osm_window_spans)

# Every site whose junction sits on one of the corridors this project reads. nj31_wdelaware is
# left out because it is in Pennington and on no borough corridor - corridor_render.py excludes
# it for the same reason. princeton_eprospect belongs here or Princeton Ave reports over ONE of
# its two modelled junctions and reads as a shorter street than it is.
DEFAULT_SITES = ("broad_st_greenwood", "ebroad_princeton", "columbia_princeton",
                 "princeton_eprospect", "wbroad_louellen")

# A gap in the tracing shorter than this is a side street's mouth or a driveway - real, and already
# accounted for by the coverage fractions. Longer than this is a stretch of street nobody has
# traced, which is a different statement and worth naming on its own line. One block is ~400 ft
# here, so this catches "most of a block" without listing every kerb ramp.
NOTABLE_GAP_FT = 100.0

# Two crossing ways this close together are one intersection: OSM splits a street at its junction,
# so N and S Greenwood Ave arrive as two ways meeting Broad St at the same station. Counting ways
# would report 9 crossings on Broad St where a person walking it meets 8.
SAME_JUNCTION_FT = 40.0


@dataclass(frozen=True)
class Coverage:
    """What a figure was actually measured over, and out of what.

    NO DEFAULT AND NO None, deliberately. A Coverage that could be omitted would be omitted, and
    the figure would then read as a whole-corridor total - which is exactly how the three wrong
    answers in docs/network-model.md were produced.
    """
    measured_ft: float
    total_ft: float
    basis: str

    @property
    def fraction(self) -> float:
        return self.measured_ft / self.total_ft if self.total_ft else 0.0

    def __str__(self) -> str:
        return (f"{self.measured_ft:,.0f} of {self.total_ft:,.0f} ft "
                f"({self.fraction * 100:.0f}%) {self.basis}")


@dataclass(frozen=True)
class Figure:
    """One corridor answer, its coverage, and why the reader might want to know."""
    label: str
    value: str
    coverage: Coverage
    note: str = ""

    def line(self) -> str:
        head = f"  {self.label:<30}{self.value:<34}[{self.coverage}]"
        return f"{head}\n      {self.note}" if self.note else head


@dataclass(frozen=True)
class CorridorReport:
    road: str
    sites: tuple[str, ...]
    figures: tuple[Figure, ...]
    notes: tuple[str, ...] = ()

    def text(self) -> str:
        head = (f"{self.road} - {len(self.sites)} modelled junction(s): "
                f"{', '.join(self.sites)}")
        lines = [head, "-" * len(head), *(figure.line() for figure in self.figures)]
        if self.notes:
            lines.extend(["", *(f"  NOTE: {note}" for note in self.notes)])
        return "\n".join(lines)


def _spans_ft(spans) -> float:
    return sum(hi - lo for lo, hi in spans)


def corridor_report(corridor: Corridor, facts: CorridorFacts, models) -> CorridorReport:
    """Every corridor question about one road, each with the coverage of its own answer."""
    length = corridor.length_ft
    traced = Coverage(corridor.both_traced_ft, length, "with both kerbs traced")
    kerb = Coverage(corridor.traced_ft("left") + corridor.traced_ft("right"), 2 * length,
                    "of kerb traced (both sides)")
    fetched = Coverage(_spans_ft(osm_window_spans(corridor, models)), length,
                       "inside the OSM fetch window")

    figures = [
        Figure("length", f"{length:,.0f} ft", traced,
               "The centreline is continuous: each modelled junction's own fitted centre, bridged "
               "along the NJDOT alignment its legs were cut from. The coverage is the kerb."),
        _narrowest_figure(corridor, traced),
        _openings_figure(facts, kerb),
        _crossings_figure(facts, fetched),
        _pedestrian_figure(facts, fetched),
        *_parking_figures(corridor, facts),
    ]
    return CorridorReport(road=corridor.name, sites=corridor.sites,
                          figures=tuple(figures), notes=_notes(corridor, models))


def _narrowest_figure(corridor: Corridor, traced: Coverage) -> Figure:
    narrowest = corridor.narrowest_width_ft()
    if narrowest is None:
        return Figure("narrowest traced width", "no measurable cross-section", traced,
                      "Neither kerb is traced anywhere the other is, so this road has no width "
                      "this project can measure. Trace one side and the figure appears.")
    width, station = narrowest
    return Figure("narrowest traced width", f"{width:.1f} ft at station {station:,.0f}", traced,
                  "Kerb to kerb, sampled only where both kerbs are traced - the nominal width is "
                  "a summary and is routinely the wrong number for asking whether there is room.")


def _openings_figure(facts: CorridorFacts, kerb: Coverage) -> Figure:
    surveyed = sum(1 for _side, opening in facts.openings if opening.is_surveyed_width)
    assumed = len(facts.openings) - surveyed
    return Figure("driveway openings", f"{len(facts.openings)} "
                  f"({surveyed} surveyed, {assumed} assumed)", kerb,
                  "Places a vehicle crosses one of the two kerbs: a dropped kerb tagged "
                  "wheelchair=no, or a mapped driveway reaching a kerb. An assumed-width opening "
                  "is a driveway with no dropped kerb tagged at its mouth - tagging it would "
                  "replace a 10 ft guess with a measurement.")


def _crossings_figure(facts: CorridorFacts, fetched: Coverage) -> Figure:
    stations = sorted(cross.station_ft for cross in facts.crossings)
    junctions = sum(1 for i, station in enumerate(stations)
                    if not i or station - stations[i - 1] > SAME_JUNCTION_FT)
    named = sorted({cross.name for cross in facts.crossings if cross.name})
    return Figure("streets crossing",
                  f"{junctions} location{'' if junctions == 1 else 's'} "
                  f"({len(facts.crossings)} OSM way{'' if len(facts.crossings) == 1 else 's'})",
                  fetched,
                  "R.S. 39:4-138(e) applies at every one of them, not only at the junction a "
                  "drawing is centred on. Fewer locations than ways wherever OSM splits a street "
                  "at its junction, so both halves arrive as two ways at one place. "
                  + (f"Named: {', '.join(named)}." if named else ""))


def _pedestrian_figure(facts: CorridorFacts, fetched: Coverage) -> Figure:
    marked = sum(1 for _station, markings in facts.marked_crossings if markings)
    return Figure("pedestrian crossings", f"{len(facts.marked_crossings)} "
                  f"({marked} with markings tagged)", fetched,
                  "Surveyed crossings on this road, counted here and drawn as traced by "
                  "src/geometry/surveyed.py. A crossing with no markings tag is not a marked "
                  "crossing and must not be drawn as one.")


def _parking_figures(corridor: Corridor, facts: CorridorFacts) -> list[Figure]:
    """The stall count, twice: what the law leaves parkable, and how much of it has a width.

    TWO FIGURES AND NOT ONE, because there are two honest answers and reporting either alone
    misleads in a different direction. The length-only count is what R.S. 39:4-138 and OSM's
    recorded restrictions leave over; the width-tested count is the part of that where both kerbs
    are traced, so an 8 ft parking lane can be checked against the road that is actually there.
    """
    parkable, by_length, tested_ft, by_width = 0.0, 0, 0.0, 0
    for side in ("left", "right"):
        parkable += _spans_ft(facts.by_side("parkable", side))
        stalls, _ft = marked_parking_capacity(corridor, facts, side)
        by_length += stalls
        stalls, measured = marked_parking_capacity(corridor, facts, side,
                                                   within=corridor.both_traced_spans())
        by_width += stalls
        tested_ft += measured
    return [
        Figure("marked parking (length only)", f"{by_length} stalls",
               Coverage(parkable, 2 * corridor.length_ft,
                        "of kerb left parkable by R.S. 39:4-138 and OSM"),
               "Every 22 ft of legally parkable kerb, counted whether or not the street's width "
               "there was ever measured. An upper bound, not a proposal."),
        Figure("marked parking (width tested)", f"{by_width} stalls",
               Coverage(tested_ft, 2 * corridor.length_ft,
                        "of kerb both parkable and with both kerbs traced"),
               f"The same count restricted to kerb where the road's width is a measurement - "
               f"{tested_ft:,.0f} of the {parkable:,.0f} parkable ft. The gap between the two "
               f"figures is the tracing that would settle it. Denominator is the whole of both "
               f"kerbs, as above, so the two figures are read against the same length."),
    ]


def _notes(corridor: Corridor, models) -> tuple[str, ...]:
    """The qualitative record: which route carries which stretch, and where the tracing stops."""
    notes = []
    if len(corridor.sri_spans) > 1:
        spans = "; ".join(f"SRI {sri} over stations {lo:,.0f}-{hi:,.0f}"
                          for lo, hi, sri in corridor.sri_spans)
        notes.append(f"this road carries more than one NJDOT route - {spans}. A report that "
                     f"named one of them would be wrong about the rest.")
    unmeasurable = corridor.unmeasurable_gaps_ft()
    for lo, hi in corridor.untraced_gaps_ft(NOTABLE_GAP_FT):
        where = ", ".join(f"{run.side} kerb traced to station {run.end_ft:,.0f}"
                          for run in corridor.kerb_runs
                          if run.source == KERB_FROM_TRACING and abs(run.end_ft - lo) < 5.0)
        blind = sum(min(hi, b) - max(lo, a) for a, b in unmeasurable if b > lo and a < hi)
        notes.append(
            f"no traced kerb on one or both sides over stations {lo:,.0f}-{hi:,.0f} "
            f"({hi - lo:,.0f} ft)" + (f" - {where}" if where else "") + ". Every width figure "
            "above excludes it. " + (
                f"{blind:,.0f} ft of it has no kerb line at all, so width_at_ft returns None there "
                f"rather than interpolating across it." if blind > 1.0 else
                "A modelled junction's own kerb line covers it, so a width is still reported - the "
                "per-leg model's answer, on the per-leg model's terms, not a survey."))
    if corridor.seams:
        worst = max(gap for _station, gap in corridor.seams)
        notes.append(f"beyond the modelled junctions the centreline is NJDOT's alignment, eased "
                     f"onto each junction's fitted centre at {len(corridor.seams)} seam(s) - "
                     f"largest lateral correction {worst:.1f} ft. An SRI line is a "
                     f"linear-referencing reference, not a surveyed carriageway centre.")
    window = osm_window_spans(corridor, models)
    if _spans_ft(window) < corridor.length_ft - 1.0:
        notes.append(f"{corridor.length_ft - _spans_ft(window):,.0f} ft of this road lies outside "
                     f"every OSM fetch window, so nothing counted from OSM covers it. Add a site "
                     f"on it, or widen CORRIDOR_KERB_RADIUS_M.")
    return tuple(notes)


def build_reports(sites=DEFAULT_SITES, road: str | None = None,
                  model_notes: bool = False) -> list[CorridorReport]:
    """Load the sites, resolve their shared roads, and answer the questions on each."""
    from src.geometry.intersection import load_intersection_model

    models = {}
    for site in sites:
        with contextlib.nullcontext() if model_notes else contextlib.redirect_stdout(io.StringIO()):
            models[site] = load_intersection_model(site=site)
    reports = []
    for raw_corridor in corridors_from_models(models):
        if road and road.lower() not in raw_corridor.name.lower():
            continue
        # Same recentring corridor_render.py applies before painting - CorridorFacts is offset-
        # dependent (kerb-matching in _road_spans_on), so a report built off the raw NJDOT
        # alignment is a second, divergent way to resolve the same facts.
        corridor = centred_on_its_kerbs(raw_corridor)
        reports.append(corridor_report(corridor, corridor_facts(corridor, models), models))
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", action="append", dest="sites", metavar="SITE",
                        help="a site to include; repeatable. Default: all four.")
    parser.add_argument("--road", help="only roads whose name contains this, e.g. \"Broad\".")
    parser.add_argument("--model-notes", action="store_true",
                        help="show each junction model's own load-time provenance notes.")
    args = parser.parse_args(argv)

    reports = build_reports(tuple(args.sites or DEFAULT_SITES), args.road, args.model_notes)
    if not reports:
        print("No road matched. Roads are named after the street their legs share.")
        return 1
    for report in sorted(reports, key=lambda r: -len(r.sites)):
        print(report.text())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
