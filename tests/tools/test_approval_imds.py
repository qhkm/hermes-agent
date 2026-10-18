"""Cloud metadata endpoint (IMDS) detection and the Jentera ``security.imds`` switch.

Default (``approve``) flags a metadata fetch for approval, as upstream does
(NousResearch/hermes-agent 24692ee790). ``deny`` refuses it outright with a plain
error and never emits an approval request.
"""

import pytest

from tools import approval as mod

IMDS_KEY = "cloud metadata endpoint access (instance credentials)"
AWS_IP = ".".join(["169", "254", "169", "254"])
ALI_IP = ".".join(["100", "100", "100", "200"])

IMDS_COMMANDS = (
    f"curl http://{AWS_IP}/latest/meta-data/iam/security-credentials/",
    'curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
    f"wget http://{AWS_IP}/latest/api/token",
    f'curl -H "Metadata: true" "http://{AWS_IP}/metadata/identity/oauth2/token?api-version=2018-02-01"',
    "curl http://[fd00:ec2::254]/latest/meta-data/",
    f"curl http://{ALI_IP}/latest/meta-data/ram/security-credentials/",
)


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("HERMES_YOLO_MODE", "HERMES_GATEWAY_SESSION", "HERMES_CRON_SESSION",
                "HERMES_EXEC_ASK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(mod, "_get_approval_config", lambda: {"mode": "manual", "deny": []})
    monkeypatch.setattr(mod, "_command_matches_permanent_allowlist", lambda c: False)


@pytest.fixture
def imds_mode(monkeypatch):
    def set_mode(value):
        monkeypatch.setattr(mod, "_imds_deny_enabled", lambda: value == "deny")
    return set_mode


class TestDetection:
    def test_metadata_credential_fetches_flagged(self):
        for cmd in IMDS_COMMANDS:
            is_dangerous, key, _ = mod.detect_dangerous_command(cmd)
            assert is_dangerous is True, cmd
            assert key == IMDS_KEY, cmd

    def test_other_link_local_and_ordinary_urls_not_flagged(self):
        for cmd in (
            "curl http://169.254.1.1/status",
            "ping 169.254.100.100",
            "curl https://example.com/api/169.254.169.2540",
            "curl https://metadata.google.internal.example.com/",
            "curl https://example.com/docs",
        ):
            is_dangerous, key, _ = mod.detect_dangerous_command(cmd)
            assert not (is_dangerous and key == IMDS_KEY), cmd


    @pytest.mark.parametrize("cmd", [
        f"curl http://{AWS_IP}.nip.io/latest/meta-data/",
        f"curl http://{AWS_IP}./latest/meta-data/",
        f"curl http://{AWS_IP}.:80/",
        "curl http://169.254.170.2/v2/credentials/abc",       # ECS task role
        "curl http://169.254.169.253/",                       # AWS DNS
        "curl http://2852039166/latest/meta-data/",           # decimal
        "curl http://0xa9fea9fe/latest/meta-data/",           # hex
        "curl 'http://[::ffff:a9fe:a9fe]/latest/meta-data/'",  # IPv4-mapped IPv6
        "curl http://[64:ff9b::a9fe:a9fe]/",                  # NAT64
        "curl -H 'Metadata-Flavor: Google' http://metadata/computeMetadata/v1/",
        "wget http://user@metadata:80/computeMetadata/v1/",
        "curl metadata/computeMetadata/v1/instance/",
        "curl -sH 'Metadata-Flavor: Google' metadata.google.internal/computeMetadata/v1/",
    ])
    def test_variant_spellings_flagged(self, cmd):
        is_dangerous, key, _ = mod.detect_dangerous_command(cmd)
        assert is_dangerous is True and key == IMDS_KEY, cmd

    @pytest.mark.parametrize("cmd", [
        "curl http://169.254.170.20/",
        "curl http://169.254.169.2540/",
        "curl https://example.com/2852039166x",
        "echo 12852039166",
        "cat metadata.json",
        "curl -o metadata.json https://example.com/a",
        "ls /srv/metadata/",
        "curl https://example.com/metadata",
        "curl https://metadata.example.com/",
        "echo metadata is a word",
        "git log --format=%H -- metadata",
        "curl http://[2001:db8::a9fe:a9fe]/",
    ])
    def test_ordinary_text_not_flagged(self, cmd):
        is_dangerous, key, _ = mod.detect_dangerous_command(cmd)
        assert not (is_dangerous and key == IMDS_KEY), cmd


class TestConfigSwitch:
    def test_default_is_approve(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["security"]["imds"] == "approve"

    def test_reads_deny_from_config(self, monkeypatch):
        import hermes_cli.config as cfg
        monkeypatch.setattr(cfg, "load_config", lambda: {"security": {"imds": "deny"}})
        assert mod._imds_deny_enabled() is True
        monkeypatch.setattr(cfg, "load_config", lambda: {"security": {"imds": "approve"}})
        assert mod._imds_deny_enabled() is False
        monkeypatch.setattr(cfg, "load_config", lambda: {"security": {}})
        assert mod._imds_deny_enabled() is False

    def test_unreadable_config_falls_back_to_approve(self, monkeypatch):
        import hermes_cli.config as cfg

        def boom():
            raise RuntimeError("no config")

        monkeypatch.setattr(cfg, "load_config", boom)
        assert mod._imds_deny_enabled() is False


class TestDenyMode:
    @pytest.mark.parametrize("guard", ["all", "dangerous"])
    def test_blocks_without_approval_request(self, imds_mode, clean_env, guard):
        imds_mode("deny")
        asked = []

        def callback(*args, **kwargs):
            asked.append(args)
            return "once"

        fn = mod.check_all_command_guards if guard == "all" else mod.check_dangerous_command
        for cmd in IMDS_COMMANDS:
            result = fn(cmd, "local", approval_callback=callback)
            assert result["approved"] is False, cmd
            assert result.get("imds_deny") is True
            assert "blocked: cloud metadata endpoint" in result["message"]
        assert asked == []

    def test_blocks_under_yolo(self, imds_mode, clean_env, monkeypatch):
        imds_mode("deny")
        monkeypatch.setattr(mod, "_YOLO_MODE_FROZEN", True)
        result = mod.check_all_command_guards(IMDS_COMMANDS[0], "local")
        assert result["approved"] is False
        assert result.get("imds_deny") is True

    def test_blocks_in_gateway_without_queueing_approval(self, imds_mode, clean_env, monkeypatch):
        imds_mode("deny")
        monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
        monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
        monkeypatch.setattr(mod, "_get_approval_config", lambda: {"mode": "manual", "deny": []})
        before = dict(mod._pending)
        result = mod.check_all_command_guards(IMDS_COMMANDS[0], "local")
        assert result["approved"] is False
        assert result.get("imds_deny") is True
        assert mod._pending == before

    def test_execute_code_script_naming_endpoint_is_blocked(self, imds_mode, clean_env):
        imds_mode("deny")
        code = f"import urllib.request\nurllib.request.urlopen('http://{AWS_IP}/latest/meta-data/')"
        result = mod.check_execute_code_guard(code, "local")
        assert result["approved"] is False
        assert result.get("imds_deny") is True

    def test_ordinary_external_url_unaffected(self, imds_mode, clean_env):
        imds_mode("deny")
        result = mod.check_all_command_guards("curl https://example.com/docs", "local")
        assert result["approved"] is True
        result = mod.check_execute_code_guard("print('hi')", "local")
        assert result["approved"] is True


class TestApproveMode:
    def test_default_still_asks(self, imds_mode, clean_env):
        imds_mode("approve")
        asked = []

        def callback(command, description, **kwargs):
            asked.append(description)
            return "deny"

        result = mod.check_all_command_guards(IMDS_COMMANDS[0], "local", approval_callback=callback)
        assert asked and "cloud metadata endpoint" in asked[0]
        assert result["approved"] is False
        assert not result.get("imds_deny")

    def test_ordinary_external_url_unaffected(self, imds_mode, clean_env):
        imds_mode("approve")
        asked = []
        result = mod.check_all_command_guards(
            "curl https://example.com/docs", "local",
            approval_callback=lambda *a, **k: asked.append(a) or "once")
        assert result["approved"] is True
        assert asked == []
