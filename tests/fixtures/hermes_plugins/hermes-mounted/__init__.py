# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixture Hermes plugin: one tool and a hook that records a test write."""

import json
from pathlib import Path


def _handler(args, **_kwargs):
    return json.dumps({"result": f"HERMES-MOUNTED-OK:{args.get('word', '')}"})


def register(ctx):
    ctx.register_tool(
        name="hermes_mounted",
        toolset="hermes_mounted",
        schema={
            "name": "hermes_mounted",
            "description": "Return the hermes-mounted marker for a word",
            "parameters": {
                "type": "object",
                "properties": {"word": {"type": "string"}},
                "required": ["word"],
            },
        },
        handler=_handler,
    )

    def record_write(tool_name="", args=None, **_kwargs):
        if tool_name == "write_file" and args:
            path = Path(args["path"])
            if path.name == "hermes-mounted-trigger.txt":
                path.with_suffix(".observed").write_text(path.read_text())

    ctx.register_hook("post_tool_call", record_write)
