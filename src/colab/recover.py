"""Recover the task-owned Colab export, checking every transferred byte."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
COMMAND = ["python3", str(Path(__file__).with_name("safe.py")), "--auth", "oauth2",
           "--config", "/home/ubuntu/.config/colab-cli/seqcredit-session.json",
           "download", "-s", "seqcredit-frozen-t4"]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    """Download to temporary storage and publish only verified artifact files."""
    with tempfile.TemporaryDirectory(prefix="seqcredit-recovery-") as temporary:
        staging = Path(temporary)
        manifest_path = staging / "manifest.json"
        subprocess.run([*COMMAND, "/content/seqcredit-export/manifest.json", str(manifest_path)], check=True)
        manifest = json.loads(manifest_path.read_text())
        names = [item["name"] for item in manifest["parts"]]
        if len(names) != len(set(names)):
            raise ValueError("Repeated transfer part names")

        def download(item: dict) -> Path:
            name = item["name"]
            if Path(name).name != name:
                raise ValueError("Invalid transfer part path")
            part = staging / name
            command = subprocess.run([*COMMAND, "/content/seqcredit-export/" + name, str(part)],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if command.returncode:
                raise RuntimeError(command.stdout)
            if part.stat().st_size != item["bytes"] or digest(part) != item["sha256"]:
                raise ValueError("Transfer part failed verification: " + name)
            return part

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(download, item) for item in manifest["parts"]]
            for number, future in enumerate(as_completed(futures), 1):
                future.result()
                print(f"Verified artifact part {number}/{len(futures)}", flush=True)
        archive = staging / "artifacts.tar"
        with archive.open("wb") as output:
            for item in manifest["parts"]:
                part = staging / item["name"]
                output.write(part.read_bytes())
                part.unlink()
        if archive.stat().st_size != manifest["bytes"] or digest(archive) != manifest["sha256"]:
            raise ValueError("Combined artifact archive failed verification")
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            for member in members:
                target = (ROOT / member.name).resolve()
                if not target.is_relative_to(ART.resolve()) or member.issym() or member.islnk():
                    raise ValueError("Invalid artifact archive member: " + member.name)
            for name in ("full_comparison.log", "evaluation.log"):
                source = ART / name
                target = ART / "20261024T224729Z-0f3990f5" / (name.removesuffix(".log") + ".cpu.log")
                if source.exists():
                    if target.exists():
                        raise FileExistsError("CPU log already preserved: " + str(target))
                    source.rename(target)
            # Remote source helpers retain their historical names in the export;
            # do not restore superseded scripts to artifacts/.
            extracted = [member for member in members if not (
                len(Path(member.name).parts) == 2 and
                Path(member.name).name.startswith("colab_") and
                Path(member.name).suffix in {".py", ".sh"})]
            bundle.extractall(ROOT, members=extracted, filter="data")
        (ART / "colab_export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        record = {"status": "transport_verified", "recovered_at": datetime.now(timezone.utc).isoformat(),
                  "archive_sha256": manifest["sha256"], "archive_bytes": manifest["bytes"],
                  "parts_verified": len(manifest["parts"]),
                  "scope": "artifacts only; no dataset, frozen configuration or training source replaced",
                  "historical_helper_scripts_restored": False}
        (ART / "colab_recovery.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
