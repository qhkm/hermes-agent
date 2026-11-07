"""Real descriptor I/O and trusted policy validation; no terminal mocks."""
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from agent.jentera_lane_policy import LanePolicy, POLICY, POLICY_HASH, apply_policy

class LanePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'outputs'
        self.root.mkdir()
        self.value = dict(POLICY, hash=POLICY_HASH, read_roots=[str(self.root)], write_roots=[str(self.root)])
        self.policy = LanePolicy.parse(self.value)

    def call(self, name, **args):
        return json.loads(self.policy.invoke(name, args))

    def test_local_files_use_existing_schema_and_preserve_order(self):
        self.assertTrue(self.call('write_file', path='draft.txt', content='first')['success'])
        self.assertTrue(self.call('write_file', path='draft.txt', content='second')['success'])
        self.assertEqual(self.call('read_file', path='draft.txt')['content'], 'second')
        self.assertEqual((self.root / 'draft.txt').read_text(), 'second')

    def test_parent_absolute_hidden_symlink_hardlink_and_device_escape(self):
        outside = Path(self.temp.name) / 'private.txt'
        outside.write_text('SYNTHETIC_PRIVATE_VALUE')
        (self.root / 'link').symlink_to(outside)
        os.link(outside, self.root / 'hardlink')
        (self.root / 'directory-link').symlink_to(outside.parent, target_is_directory=True)
        for name in ['../private.txt', str(outside), '.secret', 'link', 'hardlink', 'directory-link/private.txt', '/dev/zero']:
            with self.subTest(name=name):
                self.assertIn('error', self.call('read_file', path=name))
                self.assertIn('error', self.call('write_file', path=name, content='overwrite'))
        self.assertEqual(outside.read_text(), 'SYNTHETIC_PRIVATE_VALUE')

    @unittest.skipUnless(sys.platform == 'linux', 'requires Linux O_PATH and Unix credentials')
    def test_traverse_only_roots_with_unprivileged_real_io(self):
        # A root runner must drop privileges: chmod alone cannot reproduce
        # directory read denial while CAP_DAC_OVERRIDE is effective.
        with tempfile.TemporaryDirectory(prefix='lane-traverse-', dir='/tmp') as temp:
            base = Path(temp)
            ancestor = base / 'aisar'
            root = ancestor / 'outputs'
            nested = root / 'nested'
            nested.mkdir(parents=True)
            outside = base / 'private.txt'
            outside.write_text('private')
            (root / 'link').symlink_to(outside)
            os.link(outside, root / 'hardlink')
            (root / 'directory-link').symlink_to(base, target_is_directory=True)
            uid = 65534 if os.geteuid() == 0 else os.geteuid()
            gid = 65534 if os.geteuid() == 0 else os.getegid()
            if os.geteuid() == 0:
                os.chown(root, uid, gid)
                os.chown(nested, uid, gid)
            base.chmod(0o755)
            # Owner execute-only also reproduces the denial on non-root CI.
            ancestor.chmod(0o111)
            nested.chmod(0o311)
            read_fd, write_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                try:
                    if os.geteuid() == 0:
                        os.setgroups([])
                        os.setgid(gid)
                        os.setuid(uid)
                    self.assertNotEqual(os.geteuid(), 0)
                    with self.assertRaises(PermissionError):
                        os.open(ancestor, os.O_RDONLY | os.O_DIRECTORY)
                    value = dict(POLICY, hash=POLICY_HASH,
                                 read_roots=[str(root)], write_roots=[str(root)])
                    policy = LanePolicy.parse(value)
                    written = json.loads(policy.invoke('write_file', {'path': 'nested/result.txt', 'content': 'ok'}))
                    self.assertTrue(written.get('success'), written)
                    self.assertEqual(json.loads(policy.invoke('read_file', {'path': 'nested/result.txt'})), {'content': 'ok'})
                    for path in ['link', 'hardlink', 'directory-link/private.txt', '../private.txt', str(outside)]:
                        for tool in ['read_file', 'write_file']:
                            self.assertIn('error', json.loads(policy.invoke(tool, {'path': path, **({'content': 'bad'} if tool == 'write_file' else {})})))
                    with self.assertRaises(OSError):
                        LanePolicy.parse(dict(value, read_roots=[str(root / 'directory-link')], write_roots=[str(root / 'directory-link')]))
                except BaseException as error:
                    os.write(write_fd, repr(error).encode())
                    os._exit(1)
                os._exit(0)
            os.close(write_fd)
            try:
                error = os.read(read_fd, 65536).decode()
                _, status = os.waitpid(pid, 0)
                self.assertEqual(os.waitstatus_to_exitcode(status), 0, error)
                self.assertEqual(outside.read_text(), 'private')
            finally:
                os.close(read_fd)
                ancestor.chmod(0o755)
                nested.chmod(0o755)

    def test_every_other_tool_is_refused_without_dispatch(self):
        for name in ['terminal','execute_code','browser_navigate','memory','session_search','tool_call','delegate_task','ask_specialist','vision_analyze','web_extract','search_files','patch','skill_manage']:
            self.assertIn('error', self.call(name, path=str(self.root / 'out'), url='http://127.0.0.1:9223', command='false'))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_policy_identity_and_roots_fail_closed(self):
        for patch in [dict(version=2), dict(id='other'), dict(hash='0'*64), dict(tools=['terminal']), dict(read_roots=[]), dict(write_roots=['relative'])]:
            with self.assertRaises((ValueError,OSError)):
                LanePolicy.parse(dict(self.value, **patch))
        linked = Path(self.temp.name) / 'root-link'
        linked.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(OSError):
            LanePolicy.parse(dict(self.value, read_roots=[str(linked)], write_roots=[str(linked)]))

    def test_schema_filter_is_allowlist_and_disables_provider_writes(self):
        from types import SimpleNamespace
        agent = SimpleNamespace(tools=[{'type':'function','function':{'name':n}} for n in ['terminal','read_file','write_file','memory','web_search']], _memory_manager=object())
        apply_policy(agent, self.policy)
        self.assertEqual(agent.valid_tool_names, {'read_file','write_file','web_search'})
        self.assertIsNone(agent._memory_manager)
        self.assertFalse(agent._memory_enabled)

    def test_real_agent_invocation_refuses_and_uses_confined_io(self):
        from run_agent import AIAgent
        agent = AIAgent.__new__(AIAgent)
        agent.jentera_lane_policy = self.policy
        result = json.loads(agent._invoke_tool('write_file', {'path': 'real.txt', 'content': 'actual dispatcher'}, 'session'))
        self.assertTrue(result['success'])
        self.assertIn('error', json.loads(agent._invoke_tool('terminal', {'command': 'false'}, 'session')))
        self.assertIn('error', json.loads(agent._invoke_tool('read_file', {'path': '/etc/passwd'}, 'session')))
        self.assertEqual((self.root / 'real.txt').read_text(), 'actual dispatcher')

    def test_remote_search_errors_are_value_free(self):
        from unittest.mock import MagicMock, patch
        process = MagicMock()
        process.communicate.return_value = (json.dumps({'ok':False,'error':'CANARY_CREDENTIAL /private/profile/path'}), None)
        process.returncode = 0
        process.poll.return_value = 0
        with patch('agent.jentera_lane_policy.subprocess.Popen', return_value=process):
            result = self.call('web_search', query='public facts')
        self.assertIn('error', result)
        self.assertNotIn('CANARY', json.dumps(result))
        self.assertNotIn('/private', json.dumps(result))

    def test_search_worker_has_fixed_program_minimal_environment_and_json_only_input(self):
        from unittest.mock import MagicMock, patch
        process=MagicMock(); process.returncode=0; process.poll.return_value=0
        process.communicate.return_value=(json.dumps({'ok':True,'results':[{'title':'Public hit','url':'https://example.com','description':'snippet'}]}),None)
        query='$(cat /private/profile) ; echo not-a-command'
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':'CANARY','API_SERVER_KEY':'PRIVATE'}), patch('agent.jentera_lane_policy.subprocess.Popen',return_value=process) as spawn:
            result=self.call('web_search',query=query,limit=3)
        self.assertTrue(result['success'])
        command=spawn.call_args.args[0]
        self.assertTrue(command[1].endswith('/plugins/web/ddgs/_search_worker.py'))
        self.assertNotIn(query,command)
        env=spawn.call_args.kwargs['env']
        self.assertNotIn('OPENROUTER_API_KEY',env); self.assertNotIn('API_SERVER_KEY',env)
        self.assertEqual(env['HOME'],'/nonexistent')
        sent=json.loads(process.communicate.call_args.args[0])
        self.assertEqual(sent,{'query':query,'safe_limit':3})
        process.wait.assert_called_once()

    def test_search_timeout_kills_and_reaps_before_tool_returns(self):
        import subprocess
        from unittest.mock import MagicMock, patch
        process=MagicMock(); process.pid=43210; process.poll.return_value=None
        process.communicate.side_effect=[subprocess.TimeoutExpired('fixed-worker',30),('',None)]
        with patch('agent.jentera_lane_policy.subprocess.Popen',return_value=process), patch('agent.jentera_lane_policy.os.killpg') as kill:
            result=self.call('web_search',query='public facts')
        self.assertIn('error',result); kill.assert_called_once(); process.wait.assert_called_once()

if __name__ == '__main__':
    unittest.main()
