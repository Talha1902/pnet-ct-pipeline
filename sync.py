"""Copy new files from ~/Thesis_Folder on the HPC into this folder.

    python sync.py                  # whole Thesis_Folder
    python sync.py Results Models   # only these subfolders
    python sync.py --dry-run        # list what would be copied

Files already present here are skipped. If a file exists in both places with a
different size it is reported, not overwritten. With --update it is replaced only
when the server copy is newer than the local one.
"""
import argparse
import posixpath
import sys
from pathlib import Path

import paramiko

from server import load_env

REMOTE_ROOT = "Thesis_Folder"
LOCAL_ROOT = Path(__file__).resolve().parent
SKIP_NAMES = {".env", "server.py", "sync.py", "__pycache__"}


def connect():
    env = load_env(LOCAL_ROOT / ".env")
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
    return client


def list_remote(client, subdirs):
    """Return {relative posix path: (size, mtime)} for files under the remote root."""
    targets = " ".join(f"'{d}'" for d in subdirs) if subdirs else "."
    cmd = f"cd ~/{REMOTE_ROOT} && find {targets} -type f -printf '%s\\t%T@\\t%p\\n'"
    _, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode("utf-8", errors="replace")
    if stdout.channel.recv_exit_status() != 0:
        sys.exit("remote listing failed: " + stderr.read().decode("utf-8", errors="replace"))
    files = {}
    for line in out.splitlines():
        size, mtime, path = line.split("\t", 2)
        path = posixpath.normpath(path)
        if SKIP_NAMES.intersection(path.split("/")) or posixpath.basename(path).startswith("~$"):
            continue
        files[path] = (int(size), float(mtime))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("subdirs", nargs="*", help="subfolders of Thesis_Folder to sync (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="only list what would be copied")
    parser.add_argument("--update", action="store_true",
                        help="also replace local files whose size differs, if the server copy is newer")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    client = connect()
    try:
        remote = list_remote(client, args.subdirs)
        missing, newer, local_newer = [], [], []
        for path, (size, mtime) in sorted(remote.items()):
            local = LOCAL_ROOT / Path(*path.split("/"))
            if not local.exists():
                missing.append((path, size))
            elif local.stat().st_size != size:
                # a local file edited after the server copy (e.g. a review sheet) is never replaced
                (newer if mtime > local.stat().st_mtime else local_newer).append((path, size))

        to_copy = missing + (newer if args.update else [])
        total = sum(size for _, size in to_copy)
        print(f"{len(remote)} remote files, {len(missing)} missing here, "
              f"{len(newer) + len(local_newer)} with a different size")
        for path, _ in newer:
            print(f"  server copy newer{' (will replace)' if args.update else ' (kept local copy)'}: {path}")
        for path, _ in local_newer:
            print(f"  local copy newer (kept local copy): {path}")
        print(f"to copy: {len(to_copy)} files, {total / 1e9:.2f} GB")
        if args.dry_run or not to_copy:
            return

        sftp = client.open_sftp()
        done = 0
        for i, (path, size) in enumerate(to_copy, 1):
            local = LOCAL_ROOT / Path(*path.split("/"))
            local.parent.mkdir(parents=True, exist_ok=True)
            part = local.with_name(local.name + ".part")
            sftp.get(f"{REMOTE_ROOT}/{path}", str(part))
            try:
                part.replace(local)
            except PermissionError:
                part.unlink()
                print(f"  skipped (file is open in another program): {path}")
                continue
            done += size
            print(f"[{i}/{len(to_copy)}] {done / 1e9:.2f}/{total / 1e9:.2f} GB  {path}", flush=True)
        sftp.close()
    finally:
        client.close()
    print("sync finished")


if __name__ == "__main__":
    main()
