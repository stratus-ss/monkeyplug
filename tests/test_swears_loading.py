#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle
import tempfile
import pytest
from monkeyplug.monkeyplug import Plugger, scrubword


class MockPlugger:
    """Minimal mock Plugger for testing swears loading without audio file requirements"""

    def __init__(self, swearsFileSpec, debug=False):
        self.swearsFileSpec = swearsFileSpec
        self.swearsMap = {}
        self.debug = debug

        # Import the methods we need to test
        self._load_swears_file = Plugger._load_swears_file.__get__(self)
        self._load_swears_from_json = Plugger._load_swears_from_json.__get__(self)
        self._load_swears_from_text = Plugger._load_swears_from_text.__get__(self)


class MockPluggerNoPresetAttr:
    """Like MockPlugger, but deliberately does NOT set self.swearsMap in
    __init__, so it relies on Plugger.swearsMap's class-level {} default
    the way real Plugger/WhisperPlugger instances do before _load_swears_file
    runs. Used to catch the class-attribute-vs-instance-attribute pickling
    regression (see test_swears_map_survives_pickling_for_multiprocessing).
    """

    def __init__(self, swearsFileSpec, debug=False):
        self.swearsFileSpec = swearsFileSpec
        self.debug = debug
        self._load_swears_file = Plugger._load_swears_file.__get__(self)
        self._load_swears_from_json = Plugger._load_swears_from_json.__get__(self)
        self._load_swears_from_text = Plugger._load_swears_from_text.__get__(self)


class TestSwearsLoading:
    """Test suite for swears list loading (JSON and text formats)"""

    def test_load_json_swears_from_file(self):
        """Test loading swears from JSON format file"""
        # Create a temporary JSON swears file
        test_swears = ["damn", "hell", "crap", "badword"]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_swears, f)
            temp_json_file = f.name

        try:
            # Create mock plugger to test swears loading
            plugger = MockPlugger(temp_json_file)
            plugger._load_swears_file()

            # Verify swears were loaded correctly
            assert len(plugger.swearsMap) == len(test_swears)

            # Verify each word is in the map
            for word in test_swears:
                assert scrubword(word) in plugger.swearsMap
                assert plugger.swearsMap[scrubword(word)] == "*****"

        finally:
            if os.path.exists(temp_json_file):
                os.unlink(temp_json_file)

    def test_load_text_swears_from_file(self):
        """Test loading swears from legacy text format (backward compatibility)"""
        # Create a temporary text swears file
        test_swears = [
            "damn|dang",
            "hell",
            "crap|crud",
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('\n'.join(test_swears))
            temp_text_file = f.name

        try:
            plugger = MockPlugger(temp_text_file)
            plugger._load_swears_file()

            # Verify swears were loaded correctly
            assert len(plugger.swearsMap) == 3

            # Verify words are mapped correctly
            assert scrubword("damn") in plugger.swearsMap
            assert plugger.swearsMap[scrubword("damn")] == "dang"

            assert scrubword("hell") in plugger.swearsMap
            assert plugger.swearsMap[scrubword("hell")] == "*****"

            assert scrubword("crap") in plugger.swearsMap
            assert plugger.swearsMap[scrubword("crap")] == "crud"

        finally:
            if os.path.exists(temp_text_file):
                os.unlink(temp_text_file)

    def test_json_format_auto_detection_by_extension(self):
        """Test that JSON format is auto-detected by .json extension"""
        test_swears = ["test1", "test2", "test3"]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_swears, f)
            temp_json_file = f.name

        try:
            plugger = MockPlugger(temp_json_file)
            plugger._load_swears_file()

            assert len(plugger.swearsMap) == 3

        finally:
            if os.path.exists(temp_json_file):
                os.unlink(temp_json_file)

    def test_json_and_text_produce_same_results(self):
        """Test that the same words in JSON and text format produce identical swearsMap"""
        # Define the same set of words
        test_words = ["damn", "hell", "crap", "badword"]

        # Create JSON version
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_words, f)
            temp_json_file = f.name

        # Create text version (simple format without replacements)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('\n'.join(test_words))
            temp_text_file = f.name

        try:
            # Load JSON format
            plugger_json = MockPlugger(temp_json_file)
            plugger_json._load_swears_file()

            # Load text format
            plugger_text = MockPlugger(temp_text_file)
            plugger_text._load_swears_file()

            # Both should have the same number of entries
            assert len(plugger_json.swearsMap) == len(plugger_text.swearsMap)
            assert len(plugger_json.swearsMap) == len(test_words)

            # Both should have identical keys (normalized words)
            assert set(plugger_json.swearsMap.keys()) == set(plugger_text.swearsMap.keys())

            # Verify each word is present and normalized the same way
            for word in test_words:
                normalized = scrubword(word)
                assert normalized in plugger_json.swearsMap
                assert normalized in plugger_text.swearsMap
                # Both should map to the default replacement
                assert plugger_json.swearsMap[normalized] == "*****"
                assert plugger_text.swearsMap[normalized] == "*****"

        finally:
            if os.path.exists(temp_json_file):
                os.unlink(temp_json_file)
            if os.path.exists(temp_text_file):
                os.unlink(temp_text_file)


class TestSwearsMapPickling:
    """Regression tests for the swearsMap class-attribute pickling bug.

    swearsMap is declared as a mutable class attribute default ({}) on
    Plugger. If _load_swears_file() mutates it in place (self.swearsMap[x] = y)
    without first assigning self.swearsMap = {} as an instance attribute, the
    populated dict only exists on the class, not in the instance's __dict__.
    When AudioChunker pickles a WhisperPlugger for ProcessPoolExecutor workers
    (parallel chunk encoding), only instance __dict__ survives pickling --
    the class attribute does not travel with it. Workers then see an empty
    swearsMap and silently scrub nothing, even though the archived transcript
    JSON correctly flagged words with scrub=true.
    """

    def test_swears_map_becomes_instance_attribute(self):
        """After _load_swears_file(), swearsMap must be in the instance's
        __dict__, not just inherited from the class."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(["damn", "hell"], f)
            temp_json_file = f.name

        try:
            plugger = MockPluggerNoPresetAttr(temp_json_file)
            assert "swearsMap" not in vars(plugger), (
                "test setup invalid: swearsMap should not be an instance "
                "attribute before _load_swears_file() runs"
            )

            plugger._load_swears_file()

            assert "swearsMap" in vars(plugger), (
                "swearsMap must become an instance attribute in "
                "_load_swears_file(), otherwise it is silently dropped when "
                "pickled for ProcessPoolExecutor workers"
            )
            assert scrubword("damn") in plugger.swearsMap
            assert scrubword("hell") in plugger.swearsMap
        finally:
            if os.path.exists(temp_json_file):
                os.unlink(temp_json_file)

    def test_swears_map_survives_pickling_for_multiprocessing(self):
        """Simulates what ProcessPoolExecutor does: pickle the object, then
        unpickle it in a fresh "worker" and confirm swearsMap is populated
        there too (not silently reset to the empty class-level default)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(["damn", "hell", "crap"], f)
            temp_json_file = f.name

        try:
            plugger = MockPluggerNoPresetAttr(temp_json_file)
            plugger._load_swears_file()
            assert len(plugger.swearsMap) == 3

            # MockPluggerNoPresetAttr binds test helper methods as instance
            # attributes (via __get__), which aren't picklable the way real
            # Plugger instance methods (looked up through the class) are.
            # Only the data attributes matter for this regression -- that's
            # exactly what ProcessPoolExecutor round-trips for a real
            # WhisperPlugger instance's __dict__.
            data_only_state = {
                k: v for k, v in plugger.__dict__.items() if not callable(v)
            }
            unpickled_state = pickle.loads(pickle.dumps(data_only_state))

            assert "swearsMap" in unpickled_state
            assert len(unpickled_state["swearsMap"]) == 3
            assert scrubword("hell") in unpickled_state["swearsMap"]
        finally:
            if os.path.exists(temp_json_file):
                os.unlink(temp_json_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
