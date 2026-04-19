"""Cross-match FITS tables using the STILTS ``tmatch2`` CLI.

STILTS (Starlink Tables Infrastructure Library Tool Set) ships a fast,
memory-bounded sky cross-matcher that is much faster than a pure Python
astropy implementation once either table has more than a few million
rows. This module is a thin Python wrapper that builds the Java command
and exposes the most useful ``tmatch2`` options.

Installation
------------
1. Install a Java runtime (Java 8 or newer; OpenJDK works). Verify::

       java -version

2. Download the standalone STILTS jar from the Starlink project page
   `<https://www.starlink.ac.uk/stilts/>`_. The direct download link for
   the current release is
   `<https://www.starlink.ac.uk/stilts/stilts.jar>`_::

       wget https://www.starlink.ac.uk/stilts/stilts.jar -O /opt/stilts/stilts.jar

3. Tell this module where the jar lives, by either:

   * passing ``stilts_path="/opt/stilts/stilts.jar"`` to
     :func:`stilts_crossmatch`, **or**
   * exporting the ``STILTS_JAR`` environment variable once per shell::

       export STILTS_JAR=/opt/stilts/stilts.jar

Example:
    >>> from lib.crossmatch import stilts_crossmatch
    >>> stilts_crossmatch(
    ...     "catalog_a.fits",
    ...     "catalog_b.fits",
    ...     "matched.fits",
    ...     match_radius=1.0,
    ... )
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional, Union


def stilts_crossmatch(
    input1: Union[str, Path],
    input2: Union[str, Path],
    output_file: Union[str, Path],
    stilts_path: Optional[Union[str, Path]] = None,
    match_radius: float = 1.0,
    values1: str = "ra dec",
    values2: str = "ra dec",
    join: str = "1and2",
    find: str = "best",
    heap_size: str = "4G",
    tmp_dir: Optional[Union[str, Path]] = None,
    runner: str = "parallel",
    progress: str = "log",
    verbose: bool = True,
) -> bool:
    """Sky-match two FITS tables via STILTS ``tmatch2``.

    Builds and runs a Java command of the form::

        java -Xmx{heap_size} [-Djava.io.tmpdir={tmp_dir}] \\
             -jar {stilts_path} tmatch2 matcher=sky params={radius} ...

    The matcher is fixed to ``sky`` (great-circle separation); ``params``
    is the radius in arcseconds.

    Args:
        input1: Path to the first input FITS file.
        input2: Path to the second input FITS file.
        output_file: Path for the matched output FITS file. Parent
            directories are created automatically.
        stilts_path: Path to ``stilts.jar``. If ``None`` (default), the
            function falls back to the ``STILTS_JAR`` environment
            variable; if that is also unset a :class:`RuntimeError` is
            raised. See the module docstring for install instructions.
        match_radius: Match radius in arcseconds.
        values1: Space-separated RA/Dec column names in ``input1`` (STILTS
            ``values1``), e.g. ``"ra dec"`` or ``"raStack decStack"``.
        values2: Space-separated RA/Dec column names in ``input2``.
        join: STILTS ``join`` option. One of:

            * ``"1and2"`` — inner join, keep only matched pairs (default).
            * ``"1or2"`` — outer join, keep every row from both tables.
            * ``"all1"`` / ``"all2"`` — left / right join.
            * ``"1not2"`` / ``"2not1"`` — anti-join (unmatched rows only).
            * ``"1xor2"`` — symmetric difference of the anti-joins.
        find: STILTS ``find`` option. One of:

            * ``"best"`` — single mutually-best match per pair (default).
            * ``"best1"`` — best match per ``input1`` row.
            * ``"best2"`` — best match per ``input2`` row.
            * ``"all"`` — every pair within the radius.
        heap_size: Java ``-Xmx`` heap size, e.g. ``"4G"`` or ``"16G"``.
            Scale up for large inputs; ``tmatch2`` can be memory-hungry
            even with disk-backed matchers.
        tmp_dir: Scratch directory passed as ``-Djava.io.tmpdir``. ``None``
            uses the JVM default (typically ``/tmp``). Created if missing.
        runner: STILTS ``runner`` option. ``"parallel"`` uses all cores
            (default), ``"parallelN"`` uses N threads, ``"sequential"``
            runs single-threaded.
        progress: STILTS ``progress`` option: ``"log"``, ``"none"``, or
            ``"time"``.
        verbose: If ``True``, print the resolved command plus a final
            success/failure summary.

    Returns:
        ``True`` if STILTS exited with status 0, ``False`` otherwise.

    Raises:
        RuntimeError: ``stilts_path`` was not supplied and the
            ``STILTS_JAR`` environment variable is unset.
        FileNotFoundError: The STILTS jar or either input file does not
            exist on disk.

    Example:
        >>> stilts_crossmatch(
        ...     "ps1.fits",
        ...     "legacy.fits",
        ...     "matched.fits",
        ...     match_radius=1.0,
        ...     values1="raStack decStack",
        ...     values2="ra dec",
        ...     join="all1",
        ...     heap_size="16G",
        ... )
    """
    if stilts_path is None:
        stilts_path = os.environ.get("STILTS_JAR")
        if not stilts_path:
            raise RuntimeError(
                "stilts_path was not provided and the STILTS_JAR environment "
                "variable is not set. Pass stilts_path=..., or export "
                "STILTS_JAR=/path/to/stilts.jar. See module docstring for "
                "install instructions."
            )

    input1 = str(input1)
    input2 = str(input2)
    output_file = str(output_file)
    stilts_path = str(stilts_path)

    if not os.path.exists(stilts_path):
        raise FileNotFoundError(f"STILTS jar not found: {stilts_path}")
    if not os.path.exists(input1):
        raise FileNotFoundError(f"Input file 1 not found: {input1}")
    if not os.path.exists(input2):
        raise FileNotFoundError(f"Input file 2 not found: {input2}")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    java_opts = ["java", f"-Xmx{heap_size}"]
    if tmp_dir is not None:
        tmp_dir = str(tmp_dir)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir, exist_ok=True)
        java_opts.append(f"-Djava.io.tmpdir={tmp_dir}")
    java_opts.extend(["-jar", stilts_path])

    cmd = java_opts + [
        "tmatch2",
        f"runner={runner}",
        f"in1={input1}",
        f"in2={input2}",
        f"out={output_file}",
        "matcher=sky",
        f"params={match_radius}",
        f"values1={values1}",
        f"values2={values2}",
        f"join={join}",
        f"find={find}",
        f"progress={progress}",
    ]

    if verbose:
        print("Starting STILTS tmatch2 ...")
        print(f"  input1 : {input1}")
        print(f"  input2 : {input2}")
        print(f"  output : {output_file}")
        print(f"  radius : {match_radius} arcsec")
        print(f"  join={join}, find={find}, runner={runner}")
        print(f"  command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        if verbose:
            print(f"\nMatch complete. Output written to: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nSTILTS tmatch2 failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(
            "\nJava runtime not found. Install Java (>= 8) and make sure "
            "`java` is on PATH."
        )
        return False
