"""How many jobs this machine runs at once - in one place, because it had three.

`build_all.py --jobs` defaulted to 4, `export_all_scenarios.py` to one worker per site, and
`phase4_render_3d.blender_job_limit` derived its own number from RAM. Three answers to one
question about one machine, and the two that were guesses were the two that were wrong.

THE MACHINE: 36 GB, shared with a Blender that costs ~11 GB a copy, an editor, and usually
more than one agent session. **MAX_BUILD_JOBS is a house rule, not a derivation.** Measured
peak RSS is 0.21 GB for an export worker and 0.27 GB for a pytest worker, so on paper a dozen
of either fit - but the measurement is of one worker in isolation and the machine is not, and
the operator was watching it hit OOM while these numbers said there was room. So: one at a
time unless you pass a number, and the arithmetic below is a ceiling, never a target.

So there are two limits and they answer different questions:

  * job_limit(peak_gb) - what fits in RAM. Use it for anything whose footprint is measured in
    gigabytes; it is what saved four OOM-killed Blenders (see BLENDER_PEAK_RAM_GB).
  * MAX_BUILD_JOBS - how many the operator of this machine wants running at once, regardless
    of what fits. Any knob that fans out over sites or scenarios defaults to this.
"""
import os

# At most this many build jobs at once, whatever the arithmetic says fits. See the module
# docstring: a house rule about a 36 GB box that is doing other things, not a derivation from
# a measured footprint. One means the fan-out knobs are off by default and a caller has to ask.
MAX_BUILD_JOBS = 1

# Left for the OS, the editor, and whatever else is open. Without it a "safe" job count still
# pushes the machine into swap, which is slower than running the jobs one at a time.
RAM_HEADROOM_GB = 8


def total_ram_gb() -> float | None:
    """Physical RAM, or None when the platform will not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        return None


def job_limit(peak_gb: float, requested: int | None = None) -> int:
    """How many jobs of `peak_gb` each this machine can hold, capped by MAX_BUILD_JOBS.

    An explicit `requested` wins over the RAM arithmetic - the operator may know something
    this does not - but never over 1, and it is still the caller's job to mean it.
    """
    if requested:
        return max(1, requested)
    total = total_ram_gb()
    if total is None:
        return 1  # can't tell: the safe answer is one at a time
    return max(1, min(MAX_BUILD_JOBS, int((total - RAM_HEADROOM_GB) // peak_gb)))
