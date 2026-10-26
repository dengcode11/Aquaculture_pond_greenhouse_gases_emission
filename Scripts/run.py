from pathlib import Path
import argparse
import json
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "Scripts"


def main():
    parser = argparse.ArgumentParser(description="List key scripts or run one with the required inputs.")
    parser.add_argument("script", nargs="?", default="list", help="Use list, a two-digit number, or a script stem.")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--external-config", type=Path)
    parser.add_argument("--rscript", default="Rscript")
    arguments = parser.parse_args()
    scripts = sorted(path for path in SCRIPTS.iterdir() if path.is_file() and path.name[:2].isdigit())
    if arguments.script == "list":
        for script in scripts:
            number = int(script.name[:2])
            inputs = "supplied figure data" if number in (17, 18) else "separate upstream inputs required"
            print(f"{script.name}: {inputs}")
        return
    matches = [script for script in scripts if arguments.script in (script.stem, script.name[:2])]
    if len(matches) != 1:
        parser.error("Select a script shown by: python Scripts/run.py list")
    script = matches[0]
    number = int(script.name[:2])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(SCRIPTS / "_shared"), str(SCRIPTS), environment.get("PYTHONPATH", "")])
    environment["PYTHONIOENCODING"] = "utf-8"
    if number == 17:
        command = [arguments.rscript, str(script), str(ROOT / "Source_Data/Fig_2/flux_plot_values.csv"), str(ROOT / "outputs")]
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        return
    if number == 18:
        subprocess.run([sys.executable, str(script)], cwd=ROOT, env=environment, check=True)
        return
    if arguments.workspace is None:
        parser.error("This script requires --workspace with upstream inputs not included in the figure-data package.")
    workspace = arguments.workspace.resolve()
    if not workspace.is_dir():
        parser.error(f"Workspace does not exist: {workspace}")
    if number == 8:
        subprocess.run([sys.executable, str(script), "--root", str(workspace)], cwd=ROOT, env=environment, check=True)
        return
    working_directory = workspace / ("literature" if number <= 3 else "meta")
    if not working_directory.is_dir():
        parser.error(f"Missing input directory: {working_directory}. Use working copies of the upstream inputs.")
    environment["AQUACULTURE_WORKSPACE"] = str(workspace)
    environment["AQUACULTURE_R_RUNTIME"] = str(SCRIPTS / "_shared/aquaculture_paths.R")
    if arguments.external_config:
        configuration = arguments.external_config.resolve()
        if not configuration.is_file():
            parser.error(f"External configuration does not exist: {configuration}")
        mappings = json.loads(configuration.read_text(encoding="utf-8-sig"))
        environment["AQUACULTURE_EXTERNAL_CONFIG"] = str(configuration)
        if "download" in mappings:
            download = Path(mappings["download"])
            environment["AQUACULTURE_EXTERNAL_DOWNLOAD"] = str(download if download.is_absolute() else configuration.parent / download)
    command = [arguments.rscript if script.suffix == ".R" else sys.executable, str(script)]
    print("Running with separately supplied inputs; outputs may be written within the selected working-copy workspace.", flush=True)
    subprocess.run(command, cwd=working_directory, env=environment, check=True)


if __name__ == "__main__":
    main()
