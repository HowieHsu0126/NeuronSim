"""Bind one task's public AppWorld APIs to the pinned CUGA external-tool path.

The caller owns the real native executor and its execution/scoring receipts.
This adapter preserves documents, argument values and returned values. It does
not authenticate, retry, score, or infer API success from a returned payload.
It is specific to CUGA's set_tools/invoke_tool/model_dump interfaces.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import time
from typing import Any


NativeExecutor = Callable[[str, str, dict[str, Any]], Awaitable[Any]]


class CugaNativeResult:
    """Keep tracker JSON parsing and helper string coercion away from a value.

    This is a CUGA protocol object, not a Pydantic model. The pinned tracker
    passes the object through. The pinned helper calls model_dump once and
    returns that value without another conversion.
    """

    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = deepcopy(value)

    def model_dump(self) -> Any:
        return deepcopy(self._value)


class _NativeParameterSchema:
    """Expose the native schema without coercing the executor's arguments."""

    def __init__(self, schema: Mapping[str, Any]) -> None:
        self._schema = deepcopy(dict(schema))

    def model_json_schema(self) -> dict[str, Any]:
        return deepcopy(self._schema)


@dataclass
class _NativeTool:
    name: str
    description: str
    metadata: dict[str, str]
    args_schema: _NativeParameterSchema
    binding: CugaNativeToolBindings

    async def ainvoke(self, args: dict[str, Any]) -> CugaNativeResult:
        return await self.binding._invoke(self.metadata["server_name"], self.name, args)


class CugaNativeToolBindings:
    """Replace task tools and documents, then release all owned bindings.

    Use with one fresh CUGA task/graph and registry=False. The caller configures
    CUGA before graph import and supplies its actual AppDefinition constructor.
    The executor must preserve native errors and settle uncertain operations;
    this adapter records propagation without interpreting or repeating them.
    """

    def __init__(
        self, tracker: Any, *, api_docs: Mapping[str, Mapping[str, Mapping[str, Any]]],
        function_calling: Sequence[Mapping[str, Any]], app_descriptions: Mapping[str, str],
        executor: NativeExecutor, app_definition: Callable[..., Any],
    ) -> None:
        if not callable(executor) or not callable(app_definition):
            raise TypeError("native executor and CUGA app-definition constructor are required")
        self.tracker = tracker
        self.executor = executor
        self.app_definition = app_definition
        self.calls: list[dict[str, Any]] = []
        self._active = False
        self._used = False
        self._catalog: dict[str, dict[str, dict[str, Any]]] = {}
        self._descriptions = deepcopy(dict(app_descriptions))
        schemas: dict[tuple[str, str], dict[str, Any]] = {}
        for row in function_calling:
            if row.get("type") != "function" or not isinstance(row.get("function"), Mapping):
                raise ValueError("native OpenAI function descriptors are required")
            function = row["function"]
            name = function.get("name")
            if not isinstance(name, str) or name.count("__") != 1:
                raise ValueError("native function names must identify app__api")
            app_name, api_name = name.split("__")
            identity = (app_name, api_name)
            parameters = function.get("parameters")
            if (identity in schemas or not isinstance(parameters, Mapping)
                    or parameters.get("type") != "object" or not isinstance(parameters.get("properties"), Mapping)):
                raise ValueError("native function schemas must have unique identities and object parameters")
            schemas[identity] = deepcopy(dict(parameters))
        identities = set()
        for app, apis in api_docs.items():
            if not isinstance(app, str) or not app or not isinstance(apis, Mapping):
                raise ValueError("native public API documents must be grouped by app")
            if self._descriptions.get(app) is not None and not isinstance(self._descriptions[app], str):
                raise ValueError("available source app descriptions must be text")
            self._catalog[app] = {}
            for api, document in apis.items():
                identity = (app, api)
                if (not isinstance(document, Mapping) or document.get("app_name") != app
                        or document.get("api_name") != api or identity not in schemas):
                    raise ValueError("standard and function-calling API identities differ")
                if any(key not in document for key in ("path", "method", "description", "response_schemas")):
                    raise ValueError("native path, method, description and response schemas are required")
                if not isinstance(document["description"], str):
                    raise ValueError("native API description must be text")
                identities.add(identity)
                self._catalog[app][api] = {
                    "app_name": app, "api_name": api,
                    "path": deepcopy(document["path"]), "method": deepcopy(document["method"]),
                    "description": document["description"], "parameters": schemas[identity],
                    "response_schemas": deepcopy(document["response_schemas"]),
                    # No registry-managed authentication. Native access_token
                    # requirements remain in the complete parameter schema.
                    "secure": False,
                }
        if not identities or identities != set(schemas):
            raise ValueError("public standard/function-calling API memberships must match exactly")
        self._query = self._get_tools_by_server

    @classmethod
    def from_task(cls, tracker: Any, task: Any, *, executor: NativeExecutor, app_definition: Callable[..., Any]):
        """Read only the public task fields used by the official agents."""
        docs = task.api_docs
        return cls(tracker, api_docs=docs, function_calling=docs.function_calling(),
                   app_descriptions=task.app_descriptions, executor=executor, app_definition=app_definition)

    def __enter__(self) -> CugaNativeToolBindings:
        if self._used:
            raise RuntimeError("a task binding cannot be reused")
        if "get_tools_by_server" in vars(self.tracker):
            raise RuntimeError("the tracker already has an instance document adapter")
        self._used = True
        tools = [
            _NativeTool(api, doc["description"], {"server_name": app},
                        _NativeParameterSchema(doc["parameters"]), self)
            for app, apis in self._catalog.items() for api, doc in apis.items()
        ]
        try:
            self.tracker.set_tools(tools)
            self.tracker.apps = [self.app_definition(name=app, description=self._descriptions.get(app), url=None)
                                 for app in self._catalog]
            # The source's default external-tool view discards response/path/
            # method information. Supply the source-native catalog at that
            # existing instance interface, without changing the planning graph.
            self.tracker.get_tools_by_server = self._query
            self._active = True
        except BaseException:
            self.tracker.set_tools([])
            raise
        return self

    def __exit__(self, error_type, error, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._active:
            return
        self._active = False
        if vars(self.tracker).get("get_tools_by_server") is not self._query:
            raise RuntimeError("native document binding was replaced before cleanup")
        del self.tracker.get_tools_by_server
        self.tracker.set_tools([])

    def _get_tools_by_server(self, app: str) -> dict[str, dict[str, Any]]:
        if not self._active:
            raise RuntimeError("task tool binding is closed")
        return deepcopy(self._catalog.get(app, {}))

    async def _invoke(self, app: str, api: str, args: dict[str, Any]) -> CugaNativeResult:
        if not self._active:
            raise RuntimeError("task tool binding is closed")
        if api not in self._catalog.get(app, {}):
            raise ValueError("API is absent from the current public task catalog")
        if not isinstance(args, dict):
            raise TypeError("native API arguments must be a dictionary")
        # Native API validation owns parameter validity. Do not add defaults,
        # remove token fields, cast types, or synthesize authentication here.
        arguments = deepcopy(args)
        record = {"index": len(self.calls), "app_name": app, "api_name": api,
                  "arguments": deepcopy(arguments), "status": "running"}
        self.calls.append(record)
        started = time.clock_gettime(time.CLOCK_MONOTONIC)
        try:
            result = await self.executor(app, api, arguments)
            wrapped = CugaNativeResult(result)
            record.update(status="returned", returned_type=type(result).__name__)
            return wrapped
        except BaseException as error:
            record.update(status="raised", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            record["wall_seconds"] = time.clock_gettime(time.CLOCK_MONOTONIC) - started
