"""Exercise history protection using temporary repositories and a local bare remote."""

import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("workflow", Path(__file__).with_name("git_workflow.py"))
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class GitWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="qwen-git-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.remote = self.root / "remote.git"
        self.repo.mkdir()
        self.run_git(self.remote, "init", "--bare", "--initial-branch=main", str(self.remote))
        self.run_git(self.repo, "init", "--initial-branch=main")
        self.run_git(self.repo, "config", "user.name", "codex")
        self.run_git(self.repo, "config", "user.email", "codex@local.invalid")
        self.run_git(self.repo, "remote", "add", "origin", str(self.remote))
        (self.repo / "file.txt").write_text("base\n")
        self.run_git(self.repo, "add", "file.txt")
        self.run_git(self.repo, "commit", "-m", "Initial commit")
        self.base = self.tip()
        self.run_git(self.repo, "push", "origin", "main")
        original = workflow.REPO
        self.addCleanup(setattr, workflow, "REPO", original)
        workflow.REPO = self.repo

    def run_git(self, root, *args):
        if not root.exists():
            root.mkdir()
        return subprocess.check_output(
            ["/usr/bin/git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE
        ).strip()

    def tip(self, ref="HEAD", root=None):
        return self.run_git(root or self.repo, "rev-parse", ref)

    def act(self, *args):
        return workflow.main(list(args))

    def change(self):
        (self.repo / "file.txt").write_text("new\n")
        self.run_git(self.repo, "add", "file.txt")
        self.assertEqual(self.act("commit", "-m", "Add a change"), 0)
        return self.tip()

    def test_normal_start_commit_push_merge_and_revert(self):
        self.assertEqual(self.act("start", "codex/example"), 0)
        changed = self.change()
        self.assertEqual(self.tip("HEAD^"), self.base)
        self.assertEqual(self.act("push", "codex/example"), 0)
        self.assertEqual(self.act("switch", "main"), 0)
        self.assertEqual(self.act("merge", "codex/example"), 0)
        self.assertEqual(self.tip("main", self.remote), self.tip("main"))
        self.assertTrue(workflow.is_ancestor(changed, self.tip()))
        self.assertEqual(self.act("start", "codex/revert"), 0)
        before = self.tip()
        self.assertEqual(self.act("revert", changed), 0)
        self.assertEqual(self.tip("HEAD^"), before)

    def test_rejects_replacing_remote_history(self):
        self.assertEqual(self.act("start", "codex/example"), 0)
        changed = self.change()
        self.assertEqual(self.act("push", "codex/example"), 0)
        self.run_git(self.repo, "reset", "--hard", self.base)
        self.assertEqual(self.act("push", "codex/example"), 1)
        self.assertEqual(self.tip("codex/example", self.remote), changed)

    def test_rejects_destructive_arguments(self):
        before = self.tip()
        cases = (
            ("commit", "-m", "Example", "--amend"),
            ("push", "main", "--force"),
            ("push", "--delete", "main"),
            ("reset", "--hard", "HEAD~1"),
            ("rebase", "main"),
            ("branch", "-D", "main"),
            ("start", "codex/example", "--force-create", "main"),
        )
        for args in cases:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.act(*args)
        self.assertEqual(self.tip(), before)

    def test_rejects_main_commit_foreign_prefix_and_existing_branch(self):
        self.assertEqual(self.act("commit", "-m", "Forbidden"), 1)
        for action in ("start", "switch", "push", "merge"):
            self.assertEqual(self.act(action, "other/example"), 1)
        self.assertEqual(self.act("start", "codex/example"), 0)
        before = self.tip()
        self.assertEqual(self.act("start", "codex/example"), 1)
        self.assertEqual(self.tip(), before)

    def test_fetch_advances_but_refuses_rewinding_tracking_history(self):
        self.assertEqual(self.act("fetch"), 0)
        writer = self.root / "writer"
        self.run_git(self.root, "clone", str(self.remote), str(writer))
        self.run_git(writer, "config", "user.name", "codex")
        self.run_git(writer, "config", "user.email", "codex@local.invalid")
        (writer / "file.txt").write_text("remote change\n")
        self.run_git(writer, "commit", "-am", "Advance remote main")
        changed = self.tip(root=writer)
        self.run_git(writer, "push", "origin", "main")
        self.assertEqual(self.act("fetch"), 0)
        self.assertEqual(self.tip("origin/main"), changed)
        self.assertEqual(self.tip("main"), self.base)
        self.run_git(writer, "reset", "--hard", self.base)
        self.run_git(writer, "push", "--force", "origin", "main")
        self.assertEqual(self.act("fetch"), 1)
        self.assertEqual(self.tip("origin/main"), changed)

    def test_fetch_creates_a_missing_tracking_reference(self):
        self.run_git(self.repo, "update-ref", "-d", "refs/remotes/origin/main")
        self.assertEqual(self.act("fetch"), 0)
        self.assertEqual(self.tip("origin/main"), self.base)


if __name__ == "__main__":
    unittest.main()
