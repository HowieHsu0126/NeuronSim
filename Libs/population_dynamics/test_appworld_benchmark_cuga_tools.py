"""Run in the prepared CUGA environment; native API replies are synthetic.

These tests use the installed tracker, registry-disabled query helpers and
CallApiHelper. They do not construct a planning graph or run an AppWorld task.
"""
from __future__ import annotations

from copy import deepcopy
import tempfile
import threading
import unittest
from unittest.mock import patch

import aiohttp
from cuga.backend.activity_tracker.tracker import ActivityTracker
from cuga.backend.cuga_graph.nodes.cuga_lite.executors.common.call_api_helper import CallApiHelper
from cuga.backend.cuga_graph.nodes.cuga_lite.providers.combined import CombinedToolProvider
from cuga.backend.tools_env.registry.utils.api_utils import get_apps, get_apis
from cuga.backend.tools_env.registry.utils.types import AppDefinition
from cuga.config import settings

from population_dynamics.appworld_benchmark_cuga_tools import CugaNativeToolBindings


def native_documents():
    parameters = {"type": "object", "properties": {
        "access_token": {"type": "string", "description": "Token produced by a native login action."},
        "file_system_access_token": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
    }, "required": ["access_token", "file_system_access_token"], "additionalProperties": False}
    docs = {"synthetic_app": {"read": {
        "app_name": "synthetic_app", "api_name": "read", "description": "Synthetic API fixture.",
        "path": "/synthetic_app/read", "method": "GET", "parameters": [],
        "response_schemas": {"success": {"type": "array", "items": {"type": "integer"}},
                             "failure": {"type": "object", "properties": {"message": {"type": "string"}}}},
    }}}
    functions = [{"type": "function", "function": {"name": "synthetic_app__read",
                  "description": "Synthetic API fixture.", "parameters": parameters}}]
    return docs, functions, {"synthetic_app": "Source application description."}


class CugaNativeBindingsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native_tools_")
        self.addCleanup(self.temporary.cleanup)
        self.tracker = ActivityTracker()
        self.tracker.set_base_dir(self.temporary.name)
        self.tracker.set_tools([])
        settings.update({"ADVANCED_FEATURES": {"REGISTRY": False, "TOOL_CALL_TIMEOUT": 5}}, merge=True)
        self.invocations = []
        self.reply = None
        self.failure = None
        self.delay = False

        async def execute(app, api, args):
            self.invocations.append((app, api, deepcopy(args)))
            if self.delay:
                threading.Event().wait(.02)
            if self.failure is not None:
                raise self.failure
            return deepcopy(self.reply)

        self.execute = execute
        self.docs, self.functions, self.descriptions = native_documents()
        self.network = patch.object(aiohttp, "ClientSession", side_effect=AssertionError("unexpected registry HTTP"))
        self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(self.tracker.set_tools, [])

    def binding(self, **kwargs):
        arguments = dict(api_docs=self.docs, function_calling=self.functions,
                         app_descriptions=self.descriptions, executor=self.execute, app_definition=AppDefinition)
        arguments.update(kwargs)
        return CugaNativeToolBindings(self.tracker, **arguments)

    async def test_real_cuga_path_preserves_json_types_and_string_values(self):
        values = [{"items": [1, None]}, [1, {"nested": False}], 7, 2.5, True, False,
                  None, "plain text", "true", "null", "7", "[1, 2]", '{"x": 1}',
                  {"status_code": 400, "message": "native returned error data"}]
        with self.binding() as binding:
            call_api = CallApiHelper.create_local_call_api_function()
            for value in values:
                with self.subTest(value=value):
                    self.reply = value
                    before = len(self.invocations)
                    actual = await call_api("synthetic_app", "read", {})
                    self.assertEqual(actual, value)
                    self.assertIs(type(actual), type(value))
                    self.assertEqual(len(self.invocations), before + 1)
            self.assertEqual(len(binding.calls), len(values))
            self.assertTrue(all(row["status"] == "returned" for row in binding.calls))

    async def test_parameters_are_not_cast_filled_or_removed(self):
        args = {"access_token": "generated-token", "file_system_access_token": "generated-file-token",
                "limit": "4", "undeclared": [False, None]}
        with self.binding() as binding:
            await CallApiHelper.create_local_call_api_function()("synthetic_app", "read", args)
            self.assertEqual(self.invocations, [("synthetic_app", "read", args)])
            self.assertIsInstance(self.invocations[0][2]["limit"], str)
            args["undeclared"].append("later mutation")
            self.assertEqual(binding.calls[0]["arguments"]["undeclared"], [False, None])

    async def test_source_app_and_api_documents_survive_real_query_helpers(self):
        expected_schema = deepcopy(self.functions[0]["function"]["parameters"])
        expected_response = deepcopy(self.docs["synthetic_app"]["read"]["response_schemas"])
        with self.binding():
            self.functions[0]["function"]["parameters"]["properties"].clear()
            self.docs["synthetic_app"]["read"]["response_schemas"].clear()
            apps = await get_apps()
            self.assertEqual([(app.name, app.description) for app in apps], list(self.descriptions.items()))
            record = (await get_apis("synthetic_app"))["read"]
            self.assertEqual(record["parameters"], expected_schema)
            self.assertEqual(record["response_schemas"], expected_response)
            self.assertEqual((record["method"], record["path"]), ("GET", "/synthetic_app/read"))
            record["parameters"]["properties"].clear()
            self.assertEqual((await get_apis("synthetic_app"))["read"]["parameters"], expected_schema)
            self.assertEqual(await get_apis("absent_app"), {})
            self.assertEqual(self.invocations, [])

    async def test_default_provider_initializes_from_bound_tools_without_registry(self):
        # Full-graph construction also initializes its unused lite subgraph's
        # provider. Discover this task without HTTP or invoking any native API.
        # Actual full-graph API calls use the separately tested CallApiHelper
        # path, rather than these StructuredTool wrappers.
        with self.binding():
            provider = CombinedToolProvider()
            await provider.initialize()
            apps = await provider.get_apps()
            self.assertEqual([(app.name, app.description) for app in apps], list(self.descriptions.items()))
            tools = await provider.get_all_tools()
            self.assertEqual([tool.name for tool in tools], ["read"])
            self.assertEqual(self.invocations, [])
        self.assertEqual(self.tracker.tools, {})

    async def test_native_exception_propagates_once_without_a_success_payload(self):
        self.failure = ValueError("synthetic native failure")
        with self.binding() as binding:
            with self.assertRaisesRegex(ValueError, "synthetic native failure"):
                await CallApiHelper.create_local_call_api_function()("synthetic_app", "read", {})
            self.assertEqual(len(self.invocations), 1)
            self.assertEqual(binding.calls[0]["status"], "raised")
            self.assertNotIn("returned_type", binding.calls[0])

    async def test_old_tool_references_close_and_next_task_replaces_bindings(self):
        first = self.binding()
        with first:
            old_tool = self.tracker.tools["synthetic_app"][0]
            self.reply = "task-a"
            self.assertEqual(await CallApiHelper.create_local_call_api_function()("synthetic_app", "read", {}), "task-a")
        self.assertEqual(self.tracker.tools, {})
        self.assertEqual(self.tracker.apps, [])
        self.assertNotIn("get_tools_by_server", vars(self.tracker))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await old_tool.ainvoke({})
        with self.assertRaisesRegex(RuntimeError, "reused"):
            first.__enter__()
        self.reply = "task-b"
        with self.binding() as second:
            self.assertEqual(await CallApiHelper.create_local_call_api_function()("synthetic_app", "read", {}), "task-b")
            self.assertEqual(len(second.calls), 1)
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(self.invocations), 2)

    async def test_unknown_api_is_not_dispatched(self):
        with self.binding():
            with self.assertRaises(ValueError):
                await self.tracker.invoke_tool("synthetic_app", "absent", {})
            self.assertEqual(self.invocations, [])

    async def test_elapsed_clock_survives_frozen_python_timers(self):
        self.delay = True
        with self.binding() as binding, patch("time.time", return_value=0), patch("time.perf_counter", return_value=0):
            await CallApiHelper.create_local_call_api_function()("synthetic_app", "read", {})
        self.assertGreaterEqual(binding.calls[0]["wall_seconds"], .01)

    async def test_mismatched_native_catalogs_are_rejected_before_binding(self):
        for defect in ("missing_function", "duplicate", "wrong_identity", "missing_response", "invalid_description"):
            docs, functions, descriptions = native_documents()
            if defect == "missing_function": functions.clear()
            elif defect == "duplicate": functions.append(deepcopy(functions[0]))
            elif defect == "wrong_identity": docs["synthetic_app"]["read"]["api_name"] = "other"
            elif defect == "missing_response": docs["synthetic_app"]["read"].pop("response_schemas")
            else: descriptions["synthetic_app"] = 7
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.binding(api_docs=docs, function_calling=functions, app_descriptions=descriptions)
            self.assertEqual(self.tracker.tools, {})

    async def test_task_export_uses_public_views_without_ground_truth_access(self):
        class Docs(dict):
            def function_calling(inner):
                return self.functions

        class Task:
            api_docs = Docs(self.docs)
            app_descriptions = self.descriptions

            @property
            def ground_truth(inner):
                raise AssertionError("ground truth must not be read")

        with CugaNativeToolBindings.from_task(self.tracker, Task(), executor=self.execute, app_definition=AppDefinition):
            self.assertEqual(set(await get_apis("synthetic_app")), {"read"})
        self.assertEqual(self.invocations, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
