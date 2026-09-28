"""
Tests for the privacy-related configuration.

Acceptance:
  - telemetry switches are set before `import gradio`
  - the theme uses system fonts (no Google Fonts)
  - the Chatbot call uses the 'messages' format
  - no share button, no public API documentation
"""

import unittest


class TestEnvironmentVariables(unittest.TestCase):
    """Check that the telemetry env variables are set in app.py."""

    def test_app_py_sets_analytics_disabled(self):
        """app.py sets GRADIO_ANALYTICS_ENABLED=False before `import gradio`."""
        import re
        src = open("app.py").read()
        env_pos = src.find('os.environ.setdefault("GRADIO_ANALYTICS_ENABLED"')
        # `import gradio` as a real statement (start of a line), not in the docstring
        m = re.search(r'^import gradio\b', src, re.MULTILINE)
        self.assertIsNotNone(m, "import gradio not found")
        gradio_import_pos = m.start()
        self.assertGreater(env_pos, 0,
                           "GRADIO_ANALYTICS_ENABLED must be set")
        self.assertGreater(gradio_import_pos, env_pos,
                           "telemetry switches must come BEFORE `import gradio`")

    def test_all_required_telemetry_switches(self):
        """All three telemetry switches are set."""
        src = open("app.py").read()
        for var in ["GRADIO_ANALYTICS_ENABLED",
                    "HF_HUB_DISABLE_TELEMETRY",
                    "DO_NOT_TRACK"]:
            self.assertIn(var, src,
                          f"telemetry variable {var} missing in app.py")

    def test_flagging_disabled(self):
        src = open("app.py").read()
        self.assertIn("GRADIO_FLAGGING_MODE", src)
        self.assertIn('"never"', src)

    def test_temp_dir_set(self):
        """GRADIO_TEMP_DIR is directed to a dedicated directory."""
        src = open("app.py").read()
        self.assertIn("GRADIO_TEMP_DIR", src)


class TestLaunchParameters(unittest.TestCase):
    """Check that all privacy-relevant launch() parameters are set."""

    def setUp(self):
        self.src = open("app.py").read()

    def test_share_false(self):
        self.assertIn("share=False", self.src)

    def test_pwa_false(self):
        self.assertIn("pwa=False", self.src)

    def test_ssr_mode_false(self):
        self.assertIn("ssr_mode=False", self.src)

    def test_max_file_size_set(self):
        self.assertIn("max_file_size", self.src)

    def test_blocked_paths_set(self):
        self.assertIn("blocked_paths", self.src)
        # at least these three protected directories
        for path in ["/etc", "/root", "/home"]:
            self.assertIn(path, self.src,
                          f"blocked_paths should contain {path}")

    def test_show_error_false(self):
        self.assertIn("show_error=False", self.src)

    def test_quiet_true(self):
        self.assertIn("quiet=True", self.src)

    def test_footer_links_empty(self):
        # footer_links=[] or without an API docs entry
        self.assertIn("footer_links=[]", self.src)


class TestTheme(unittest.TestCase):
    """Check the theme set-up."""

    def test_app_theme_uses_system_fonts(self):
        """APP_THEME has system fonts only, no GoogleFonts."""
        src = open("src/ui/gradio_app.py").read()
        # APP_THEME definition
        self.assertIn("APP_THEME = gr.themes.Default(", src)
        # no GoogleFont
        self.assertNotIn("GoogleFont", src,
                         "GoogleFont must not be used")
        # System-Font-Strings
        self.assertIn("ui-sans-serif", src)
        self.assertIn("ui-monospace", src)

    def test_theme_passed_to_launch_not_blocks(self):
        """In Gradio 6: theme goes to launch(), not to gr.Blocks()."""
        app_src = open("app.py").read()
        gradio_src = open("src/ui/gradio_app.py").read()
        # It must be in app.launch(... theme=APP_THEME ...)
        self.assertIn("theme=APP_THEME", app_src)
        # It must NOT be in gr.Blocks(...) (at least not as a
        # positional/keyword argument)
        # We check that `with gr.Blocks(` has no `theme=` parameter
        import re
        for m in re.finditer(r"with gr\.Blocks\(([^)]*)\)", gradio_src):
            block_args = m.group(1)
            self.assertNotIn(
                "theme=", block_args,
                f"gr.Blocks() must not have theme= any more in Gradio 6:\n{block_args}",
            )


class TestChatbotConfig(unittest.TestCase):
    """Check the Chatbot call."""

    def setUp(self):
        self.src = open("src/ui/gradio_app.py").read()

    def test_messages_type(self):
        """Gradio 6: the type parameter was removed; "messages" is the only
        mode.

        The code must set the parameter neither as type="messages" nor as
        type="tuples" — both fail at run time with `unexpected keyword
        argument 'type'`.
        """
        self.assertNotIn('type="messages"', self.src)
        self.assertNotIn('type="tuples"', self.src)

    def test_no_removed_params(self):
        """No removed Gradio 5 parameters on the Chatbot."""
        for old_param in [
            "bubble_full_width",
            "resizeable",            # typo form, removed in 6.x
            "show_copy_button",
            "show_copy_all_button",
            "show_share_button",
        ]:
            self.assertNotIn(
                old_param, self.src,
                f"{old_param} was removed in Gradio 6",
            )

    def test_no_share_button(self):
        """The buttons list contains no 'share'."""
        # Heuristic: buttons=["copy"] OK, buttons=["...share..."] NOT
        import re
        for m in re.finditer(r'buttons=\[([^\]]*)\]', self.src):
            arr = m.group(1)
            self.assertNotIn("share", arr,
                             f"buttons=[...] contains 'share': {arr}")


class TestRequirementsPin(unittest.TestCase):
    """Check the version pinning."""

    def test_gradio_pinned_to_6_11_or_higher(self):
        """gradio lower bound >= 6.11 and below the next major release.

        Gradio 6.0-6.10 contain UI freezes and dropdown slow-downs that
        the interface is sensitive to.
        """
        import re
        req = open("requirements.txt").read()
        m = re.search(r"^gradio>=(\d+)\.(\d+)[^,\n]*,<(\d+)\s*$", req, re.M)
        self.assertIsNotNone(m, "gradio must be pinned as gradio>=X.Y,<Z")
        major, minor, upper = (int(x) for x in m.groups())
        self.assertGreaterEqual((major, minor), (6, 11))
        self.assertEqual(upper, 7, "no jump to the next major release")


if __name__ == "__main__":
    unittest.main()
