"""Command line entry point:  python -m bib <command> ...

  fingerprint   write a hardware/software fingerprint            (bib/fingerprint.py)
  workloads     regenerate request files from public sources      (bib/workloads.py)
  run           execute a plan: servers x workloads x concurrency (bib/runner.py)
  summarize     raw results -> runs.csv / points.csv / points.json (bib/summarize.py)
  compare       evaluate pre-registered comparisons               (bib/compare.py)
  regress       regression report between two result sets         (bib/regress.py)
  dashboard     static HTML dashboard                              (bib/dashboard.py)
  report        summarize + compare + dashboard in one go
"""
import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    sys.argv = [f"bib {cmd}"] + rest
    if cmd == "fingerprint":
        from .fingerprint import __name__ as _  # noqa
        import runpy
        runpy.run_module("bib.fingerprint", run_name="__main__")
    elif cmd == "workloads":
        from .workloads import main as m
        m()
    elif cmd == "run":
        from .runner import main as m
        m()
    elif cmd == "summarize":
        from .summarize import summarize
        for r in rest:
            summarize(r)
    elif cmd == "compare":
        from .compare import main as m
        m()
    elif cmd == "regress":
        from .regress import main as m
        m(rest)
    elif cmd == "dashboard":
        import runpy
        runpy.run_module("bib.dashboard", run_name="__main__")
    elif cmd == "report":
        import glob
        import os
        from .summarize import summarize
        from .compare import main as cmp_main
        from .dashboard import build
        res = rest[0] if rest else "results"
        for d in sorted(glob.glob(os.path.join(res, "*", "plan.yaml"))):
            summarize(os.path.dirname(d))
        sys.argv = ["bib compare", "--results", res, "--out", os.path.join(res, "comparisons")]
        cmp_main()
        build(res, "dashboard/index.html")
    else:
        print(f"unknown command {cmd}\n{__doc__}")
        sys.exit(2)


if __name__ == "__main__":
    main()
