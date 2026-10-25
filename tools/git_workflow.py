"""Fixed Git operations that preserve existing branch history."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


REPO = Path("/home/yunli/code/qwen/codex")
GIT = "/usr/bin/git"


class Rejected(RuntimeError):
    pass


def git(*args, capture=False, check=True):
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment["LC_ALL"] = "C"
    return subprocess.run(
        [GIT, "-C", str(REPO), *args],
        env=environment,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        check=check,
    )


def output(*args):
    return git(*args, capture=True).stdout.strip()


def branch_name(name, feature=False):
    if name.startswith("-") or ("/" in name and not name.startswith("codex/")):
        raise Rejected("Only codex/ branches and unprefixed branches are permitted.")
    if feature and not name.startswith("codex/"):
        raise Rejected("This operation requires a codex/ branch.")
    git("check-ref-format", "refs/heads/" + name, capture=True)
    return name


def current_branch(feature=False):
    return branch_name(output("symbolic-ref", "--quiet", "--short", "HEAD"), feature)


def ensure_idle():
    root = Path(output("rev-parse", "--show-toplevel")).resolve()
    if root != REPO.resolve():
        raise Rejected("Unexpected repository root.")
    for marker in (
        "MERGE_HEAD", "REVERT_HEAD", "CHERRY_PICK_HEAD",
        "rebase-merge", "rebase-apply", "sequencer",
    ):
        path = Path(output("rev-parse", "--git-path", marker))
        if not path.is_absolute():
            path = REPO / path
        if path.exists():
            raise Rejected("Finish the existing Git operation before starting another.")


def is_ancestor(old, new):
    result = git("merge-base", "--is-ancestor", old, new, capture=True, check=False)
    if result.returncode not in (0, 1):
        raise Rejected("Could not verify existing history.")
    return result.returncode == 0


def fetch_main():
    git("fetch", "--no-tags", "--refmap=", "origin", "refs/heads/main")
    new = output("rev-parse", "FETCH_HEAD^{commit}")
    ref = "refs/remotes/origin/main"
    old_result = git("rev-parse", "--verify", "--quiet", ref, capture=True, check=False)
    old = old_result.stdout.strip()
    if old_result.returncode not in (0, 1):
        raise Rejected("Could not inspect the tracking reference.")
    if old and not is_ancestor(old, new):
        raise Rejected("Remote main moved backwards; tracking history was preserved.")
    git("update-ref", ref, new, old or "0" * 40)


def push_branch(name):
    branch_name(name)
    ref = "refs/heads/" + name
    local = output("rev-parse", "--verify", ref + "^{commit}")
    remote_rows = output("ls-remote", "--heads", "origin", ref).splitlines()
    if remote_rows:
        if len(remote_rows) != 1 or remote_rows[0].split()[1] != ref:
            raise Rejected("Unexpected remote reference.")
        remote = remote_rows[0].split()[0]
        present = git("cat-file", "-e", remote + "^{commit}", capture=True, check=False)
        if present.returncode:
            git("fetch", "--no-tags", "--refmap=", "origin", ref)
        if not is_ancestor(remote, local):
            raise Rejected("Push would replace existing history; no push was performed.")
    git("push", "--porcelain", "origin", ref + ":" + ref)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("start", "switch", "push", "merge"):
        commands.add_parser(action).add_argument("branch")
    commands.add_parser("commit").add_argument("-m", "--message", required=True)
    commands.add_parser("revert").add_argument("commit")
    commands.add_parser("fetch")
    args = parser.parse_args(argv)
    try:
        ensure_idle()
        if args.action == "start":
            name = branch_name(args.branch, feature=True)
            git("switch", "--create", name, "main")
        elif args.action == "switch":
            git("switch", "--no-guess", branch_name(args.branch))
        elif args.action == "commit":
            current_branch(feature=True)
            git("commit", "--message=" + args.message)
        elif args.action == "push":
            push_branch(args.branch)
        elif args.action == "fetch":
            fetch_main()
        elif args.action == "merge":
            if current_branch() != "main":
                raise Rejected("Merge must start on main.")
            name = branch_name(args.branch, feature=True)
            git("merge", "--no-ff", "--message=Merge " + name, "refs/heads/" + name)
            push_branch("main")
        elif args.action == "revert":
            current_branch(feature=True)
            if not re.fullmatch(r"[0-9a-fA-F]{7,40}", args.commit):
                raise Rejected("Revert requires a commit hash.")
            commit = output("rev-parse", "--verify", args.commit + "^{commit}")
            git("revert", "--no-edit", commit)
    except (Rejected, subprocess.CalledProcessError) as error:
        print("Git operation rejected:", error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
