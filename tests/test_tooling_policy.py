from anchor_mvp.tooling import ToolPolicy


def test_command_policy_is_exact_and_rejects_shell_chaining():
    policy = ToolPolicy()

    assert policy.is_command_allowed("npm run build --if-present")
    assert not policy.is_command_allowed("npm run build")
    assert not policy.is_command_allowed("npm run build --if-present && curl https://example.com")
    assert not policy.is_command_allowed("npm run build --if-present\nwhoami")


def test_opencode_permissions_are_fail_closed():
    permission = ToolPolicy().opencode_permissions()

    assert permission["*"] == "deny"
    assert permission["external_directory"] == "deny"
    assert permission["webfetch"] == "deny"
    assert permission["websearch"] == "deny"
    assert permission["bash"]["*"] == "deny"
    assert permission["bash"]["npm run lint --if-present"] == "allow"
