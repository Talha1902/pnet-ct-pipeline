"""Run a command on the HPC server: python server.py "command"

Reads SERVER_HOST, SERVER_PORT, SERVER_USERNAME, SERVER_PASSWORD from .env
next to this script.
"""
import sys
from pathlib import Path

import paramiko


def load_env(path):
    env = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def main():
    if len(sys.argv) < 2:
        print('usage: python server.py "command"', file=sys.stderr)
        sys.exit(2)
    command = " ".join(sys.argv[1:])
    env = load_env(Path(__file__).with_name(".env"))

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=env["SERVER_HOST"],
        port=int(env.get("SERVER_PORT", 22)),
        username=env["SERVER_USERNAME"],
        password=env["SERVER_PASSWORD"],
        look_for_keys=False,
        allow_agent=False,
        timeout=30,
    )
    try:
        _, stdout, stderr = client.exec_command(command)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
    finally:
        client.close()

    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.write(out)
    sys.stderr.write(err)
    sys.exit(code)


if __name__ == "__main__":
    main()
