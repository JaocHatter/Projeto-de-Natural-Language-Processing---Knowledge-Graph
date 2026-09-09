"""Streamlit integration tests; skipped when viewer dependencies are absent."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipUnless(AppTest is not None and (ROOT / "data/interim/gazetteer.db").exists(),
                     "Requires Streamlit and the local gazetteer")
class AppTests(unittest.TestCase):
    def test_case_filters_exports_and_new_text(self):
        at = AppTest.from_file(str(ROOT / "src/app.py"), default_timeout=30).run()
        self.assertFalse(at.exception)
        self.assertEqual([tab.label for tab in at.tabs], ["Annotated text", "Knowledge Graph", "Tables"])
        self.assertEqual(len(at.get("download_button")), 5)
        self.assertEqual(next(w for w in at.selectbox if w.label == "Measurements").value, "On node selection")
        next(w for w in at.checkbox if w.label == "Group by IS_A category").set_value(True).run()
        self.assertFalse(at.exception)
        self.assertTrue(next(w for w in at.selectbox if w.label == "Layout").disabled)
        for mode in ("Hide all", "On node selection"):
            next(w for w in at.selectbox if w.label == "Measurements").set_value(mode).run()
            self.assertFalse(at.exception)
        case_select = next(w for w in at.selectbox if w.label.startswith("Case ("))
        self.assertEqual(len(case_select.options), 50)
        case_select.set_value("PMC6083636_P1").run()
        self.assertFalse(at.exception)
        self.assertTrue(any("PMC6083636_03" in item.value for item in at.json))
        next(w for w in at.selectbox if w.label.startswith("Case (")).set_value("PMC4630775_01").run()
        self.assertFalse(at.exception)
        self.assertTrue(any("Age: **0**" in item.value for item in at.markdown))
        next(w for w in at.radio if w.label == "Representation").set_value("All occurrences").run()
        next(w for w in at.multiselect if w.label == "Entity types").set_value([]).run()
        self.assertFalse(at.exception)
        next(w for w in at.radio if w.label == "Source").set_value("New case").run()
        at.text_area[0].set_value('lipase 850 U/L. <script>alert("text")</script>').run()
        self.assertFalse(at.exception)
        self.assertTrue(any("&lt;script&gt;" in item.value for item in at.markdown))
        at.text_area[0].set_value("xyzzy").run()
        self.assertFalse(at.exception)


if __name__ == "__main__":
    unittest.main()
