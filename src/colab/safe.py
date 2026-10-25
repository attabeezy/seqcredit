"""Stream Colab CLI output while redacting proxy credentials from errors."""

import re
import subprocess
import sys

process = subprocess.Popen(["colab", *sys.argv[1:]], stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    line = re.sub(r"(colab-runtime-proxy-token=)[^&\s]+", r"\1<redacted>", line)
    line = re.sub(r"((?:access_token|refresh_token|id_token)=)[^&\s]+", r"\1<redacted>", line)
    sys.stdout.write(line)
    sys.stdout.flush()
sys.exit(process.wait())
