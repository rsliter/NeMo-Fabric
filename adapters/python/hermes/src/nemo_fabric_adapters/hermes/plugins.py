# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes plugin settings: the native section, mounted plugins, and the start checks.

``harness.settings.plugins`` names plugins by where they come from:

``bundled.entries.<id>``
    Plugins that ship with Hermes, turned on or off.
``external.paths`` and ``external.entries.<id>``
    Plugin folders that something else, such as a mounted volume, provides;
    each is linked into ``$HERMES_HOME/plugins`` before discovery.

Each entry takes ``enabled``, ``config`` and ``capabilities``. The adapter
translates them to Hermes's native ``plugins`` section: ``enabled`` true or
false joins ``plugins.enabled`` or ``plugins.disabled``, ``config`` becomes
``plugins.entries.<id>.settings``, and ``capabilities`` becomes
``plugins.entries.<id>.granted_capabilities``. Hermes's own plugin controls,
such as ``isolation`` and the timeouts, pass through. The older
``plugins_enabled`` joins ``plugins.enabled``. After Hermes discovers its
plugins, :func:`check_enabled` confirms that every plugin the configuration
turns on has loaded.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nemo_fabric_adapters.common.plugins as common_plugins
import nemo_fabric_adapters.common.utils as common_utils
from nemo_fabric_adapters.common.lifecycle import LifecycleError


LOGGER = logging.getLogger(__name__)
SETTINGS_FIELD = "harness.settings"
PLUGINS_FIELD = f"{SETTINGS_FIELD}.plugins"
HARNESS = "hermes"
LABEL = "Hermes"
ENABLED_DIRECTORY = f"{PLUGINS_FIELD}.external.enabled_directory"
MANIFEST_FILES = ("plugin.yaml", "plugin.yml")
SECTIONS = ("bundled", "external")
# Hermes's own plugin controls, written to ``plugins`` as they are.
CONTROLS = (
    "load_timeout_seconds",
    "hook_callback_timeout",
    "clone_timeout_seconds",
    "auto_update_check_hours",
    "auto_apply",
    "isolation",
    "host",
)
BUNDLED_SOURCE = "bundled"


@dataclass(frozen=True)
class PluginEntry:
    """One ``bundled`` or ``external`` entry from ``harness.settings.plugins``."""

    section: str
    plugin: str
    entry: Mapping[str, Any]

    @property
    def field(self) -> str:
        return f"{PLUGINS_FIELD}.{self.section}.entries.{self.plugin}"

    @property
    def enabled(self) -> bool | None:
        value = self.entry.get("enabled")
        return value if isinstance(value, bool) else None


def _authored(settings: Mapping[str, Any]) -> Mapping[str, Any]:
    value = settings.get("plugins")
    return value if isinstance(value, Mapping) else {}


def _section(settings: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = _authored(settings).get(section)
    return value if isinstance(value, Mapping) else {}


def plugin_entries(settings: Mapping[str, Any]) -> list[PluginEntry]:
    """Every ``bundled`` and ``external`` entry, in that order.

    A plugin id may appear in only one section; otherwise fails with
    ``hermes_invalid_config``.
    """

    found: list[PluginEntry] = []
    sections: dict[str, str] = {}
    for section in SECTIONS:
        entries = _section(settings, section).get("entries")
        for plugin_id, entry in (
            entries if isinstance(entries, Mapping) else {}
        ).items():
            plugin = PluginEntry(
                section, plugin_id, entry if isinstance(entry, Mapping) else {}
            )
            if plugin_id in sections:
                raise LifecycleError(
                    "hermes_invalid_config",
                    f"Hermes plugin {plugin_id!r} is listed as both "
                    f"{sections[plugin_id]} and {section}",
                    metadata={"field": plugin.field, "plugin": plugin_id},
                )
            sections[plugin_id] = section
            found.append(plugin)
    return found


def external_paths(settings: Mapping[str, Any]) -> list[str]:
    """``plugins.external.paths``."""

    return common_utils.normalize_list(_section(settings, "external").get("paths"))


def enabled_ids(settings: Mapping[str, Any]) -> list[tuple[str, str]]:
    """``(field, id)`` for each plugin the configuration turns on.

    Reads the ``bundled`` and ``external`` entries with ``enabled: true``,
    then the older ``plugins_enabled``, without repeats.
    """

    ids = [
        (plugin.field, plugin.plugin)
        for plugin in plugin_entries(settings)
        if plugin.enabled is True
    ]
    seen = {plugin_id for _, plugin_id in ids}
    for index, plugin_id in enumerate(
        common_utils.normalize_list(settings.get("plugins_enabled"))
    ):
        if plugin_id not in seen:
            seen.add(plugin_id)
            ids.append((f"{SETTINGS_FIELD}.plugins_enabled[{index}]", plugin_id))
    return ids


def disabled_ids(settings: Mapping[str, Any]) -> list[str]:
    """Plugin ids whose entry sets ``enabled: false``."""

    return [
        plugin.plugin for plugin in plugin_entries(settings) if plugin.enabled is False
    ]


def native_section(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The ``plugins`` section to write to ``config.yaml``.

    Empty lists and entries are left out, so a configuration without plugin
    settings writes no section.
    """

    authored = _authored(settings)
    section: dict[str, Any] = {
        name: copy.deepcopy(authored[name]) for name in CONTROLS if name in authored
    }
    enabled = [plugin_id for _, plugin_id in enabled_ids(settings)]
    disabled = disabled_ids(settings)
    if enabled:
        section["enabled"] = enabled
    if disabled:
        section["disabled"] = disabled
    entries: dict[str, Any] = {}
    for plugin in plugin_entries(settings):
        native: dict[str, Any] = {}
        if isinstance(plugin.entry.get("config"), Mapping):
            native["settings"] = copy.deepcopy(dict(plugin.entry["config"]))
        if plugin.entry.get("capabilities"):
            native["granted_capabilities"] = list(plugin.entry["capabilities"])
        if native:
            entries[plugin.plugin] = native
    if entries:
        section["entries"] = entries
    return section


def configured(settings: Mapping[str, Any]) -> bool:
    """Whether the configuration names any plugin the start checks cover."""

    return bool(enabled_ids(settings) or plugin_entries(settings))


def _matches(
    plugin_id: str, listed: Iterable[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Hermes's entries for ``plugin_id``.

    ``plugins.enabled`` selects a plugin by its registry key (``disk-cleanup``,
    ``image_gen/openai``) or its bare manifest name, so both are matched.
    """

    by_key = [entry for entry in listed if entry.get("key") == plugin_id]
    if by_key:
        return by_key
    return [entry for entry in listed if entry.get("name") == plugin_id]


def check_enabled(
    enabled: list[tuple[str, str]],
    listed: Iterable[Mapping[str, Any]],
    disabled: Iterable[str] = (),
) -> None:
    """Fail unless every ``(field, id)`` in ``enabled`` is present, enabled, and
    loaded, and not also turned off.

    ``listed`` is Hermes's ``get_plugin_manager().list_plugins()`` after
    discovery. Hermes silently ignores an unknown ``plugins.enabled`` id and
    records a load failure only as the entry's ``error``, so both would
    otherwise go unnoticed until the agent misses the capability.
    """

    listed = list(listed)
    disabled = set(disabled)

    def lookup(plugin_id: str) -> str | None:
        if plugin_id in disabled:
            return "its entry also sets enabled: false"
        matches = _matches(plugin_id, listed)
        if not matches:
            return None
        if any(entry.get("enabled") and not entry.get("error") for entry in matches):
            return ""
        return next(
            (str(entry["error"]) for entry in matches if entry.get("error")),
            "not enabled",
        )

    common_plugins.check_requested(enabled, lookup, harness=HARNESS, label=LABEL)


def check_bundled(
    entries: Iterable[PluginEntry], listed: Iterable[Mapping[str, Any]]
) -> None:
    """Fail when a ``bundled`` entry names a plugin Hermes doesn't ship."""

    listed = list(listed)
    for plugin in entries:
        if plugin.section != "bundled":
            continue
        matches = _matches(plugin.plugin, listed)
        if not matches:
            raise LifecycleError(
                common_plugins.not_found_code(HARNESS),
                f"{LABEL} plugin {plugin.plugin!r} was not found",
                metadata={"field": plugin.field, "plugin": plugin.plugin},
            )
        sources = {str(entry.get("source")) for entry in matches}
        if BUNDLED_SOURCE not in sources:
            raise LifecycleError(
                "hermes_invalid_config",
                f"Hermes plugin {plugin.plugin!r} doesn't ship with Hermes; "
                "list it under plugins.external",
                metadata={
                    "field": plugin.field,
                    "plugin": plugin.plugin,
                    "source": sorted(sources),
                },
            )


def check_loaded_plugins(settings: Mapping[str, Any]) -> None:
    """Run the start checks against Hermes's discovered plugins."""

    enabled = enabled_ids(settings)
    entries = plugin_entries(settings)
    if not enabled and not any(plugin.section == "bundled" for plugin in entries):
        return
    from hermes_cli.plugins import get_plugin_manager

    listed = get_plugin_manager().list_plugins()
    check_bundled(entries, listed)
    check_enabled(enabled, listed, disabled_ids(settings))


# --- Agent-installed plugins -----------------------------------------------

AGENT_INSTALLS = "installs"


def agent_environments(hermes_home: Path) -> list[Path]:
    """Plugin environments Hermes built in the home, ``installs/<key>/environments/<id>``.

    Hermes's own launcher also writes bootstrap records under ``installs``,
    so only a built environment counts.
    """

    installs = hermes_home / AGENT_INSTALLS
    if not installs.is_dir():
        return []
    return sorted(
        environment
        for environments in installs.glob("*/environments")
        if environments.is_dir()
        for environment in environments.iterdir()
    )


def check_no_agent_installs(hermes_home: Path) -> None:
    """Fail when the Hermes home holds a plugin environment an agent created.

    Importing Hermes switches the process to that environment, dropping NeMo
    Fabric's packages, so start fails before the import. The folder is left
    for the user to remove.
    """

    environments = agent_environments(hermes_home)
    if not environments:
        return
    installs = hermes_home / AGENT_INSTALLS
    raise LifecycleError(
        "hermes_plugin_agent_installed_unsupported",
        f"The Hermes home {str(hermes_home)!r} holds a plugin environment "
        f"({AGENT_INSTALLS}/) created by an agent plugin install. NeMo Fabric "
        "doesn't support agent-installed Hermes plugins. Delete "
        f"{str(installs)!r} and the plugin's folder under plugins/, and provide "
        "the plugin through plugins.external instead.",
        metadata={
            "path": str(installs),
            "hermes_home": str(hermes_home),
            "environments": [str(path) for path in environments],
        },
    )


# --- External plugins -------------------------------------------------------


@dataclass(frozen=True)
class MountedPlugin:
    """A plugin folder and the setting that selected it."""

    field: str
    name: str
    path: Path


def _manifest_path(folder: Path) -> Path | None:
    return next(
        (path for name in MANIFEST_FILES if (path := folder / name).is_file()), None
    )


def _manifest_name(folder: Path, field: str) -> str:
    """The plugin's name as Hermes reads it: ``name`` in the manifest, else the
    folder's name."""

    # Parse the manifest as Hermes does: its YAML helper, not PyYAML, which
    # a Hermes install doesn't ship.
    import hermes_yaml as yaml

    manifest = _manifest_path(folder)
    if manifest is None:
        raise LifecycleError(
            "hermes_plugin_invalid",
            f"Hermes plugin folder {str(folder)!r} has no plugin.yaml",
            metadata={
                "field": field,
                "path": str(folder),
                "reason": "missing_manifest",
            },
        )
    try:
        data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise LifecycleError(
            "hermes_plugin_invalid",
            f"Hermes plugin manifest {str(manifest)!r} can't be read: {error}",
            metadata={
                "field": field,
                "path": str(manifest),
                "reason": "invalid_manifest",
            },
        ) from error
    if not isinstance(data, dict):
        raise LifecycleError(
            "hermes_plugin_invalid",
            f"Hermes plugin manifest {str(manifest)!r} is not a mapping",
            metadata={
                "field": field,
                "path": str(manifest),
                "reason": "invalid_manifest",
            },
        )
    name = data.get("name", folder.name)
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or name.startswith((".", "__"))
    ):
        raise LifecycleError(
            "hermes_plugin_invalid",
            f"Hermes plugin manifest {str(manifest)!r} has an invalid name {name!r}",
            metadata={"field": field, "path": str(manifest), "reason": "invalid_name"},
        )
    return name


def _directory_children(
    settings: Mapping[str, Any], base_dir: str | Path
) -> list[Path]:
    directory = _section(settings, "external").get("enabled_directory")
    if directory is None:
        return []
    root = common_plugins.resolve_plugin_path(
        directory, base_dir, harness=HARNESS, label=LABEL, field=ENABLED_DIRECTORY
    )
    try:
        children = sorted(root.iterdir())
    except OSError as error:
        raise LifecycleError(
            "hermes_plugin_invalid",
            f"Cannot list Hermes plugin directory {str(root)!r}: {error}",
            metadata={"field": ENABLED_DIRECTORY, "path": str(root)},
        ) from error

    plugins = []
    for child in children:
        if (
            child.name.startswith((".", "__"))
            or child.is_symlink()
            or not child.is_dir()
        ):
            continue
        entrypoint = child / "__init__.py"
        manifest = _manifest_path(child)
        if (
            manifest
            and not manifest.is_symlink()
            and entrypoint.is_file()
            and not entrypoint.is_symlink()
        ):
            plugins.append(child)
    return plugins


def settings_for_mounted(
    settings: dict[str, Any], mounted: Iterable[MountedPlugin]
) -> dict[str, Any]:
    """Apply directory defaults for this start, preserving explicit per-plugin settings."""
    discovered = [
        plugin.name for plugin in mounted if plugin.field == ENABLED_DIRECTORY
    ]
    if not discovered:
        return settings
    authored = _authored(settings)
    external = _section(settings, "external")
    entries = dict(external.get("entries", {}))
    for name in discovered:
        entries[name] = {"enabled": True, **entries.get(name, {})}
    return {
        **settings,
        "plugins": {**authored, "external": {**external, "entries": entries}},
    }


def plan_mounted(
    settings: Mapping[str, Any], base_dir: str | Path
) -> list[MountedPlugin]:
    """Resolve individual paths and the immediate plugins in the opted-in directory.

    A relative entry resolves against ``base_dir``. Fails with
    ``hermes_plugin_not_found`` for a missing folder or an ``external`` entry
    no folder provides, ``hermes_plugin_invalid`` for a folder without a
    readable manifest, and ``hermes_plugin_duplicate`` when two folders share
    a name.
    """

    sources = [
        (f"{PLUGINS_FIELD}.external.paths[{index}]", Path(entry))
        for index, entry in enumerate(external_paths(settings))
    ]
    sources.extend(
        (ENABLED_DIRECTORY, child) for child in _directory_children(settings, base_dir)
    )
    names: dict[str, MountedPlugin] = {}
    for field, entry in sources:
        resolved = common_plugins.resolve_plugin_path(
            entry, base_dir, harness=HARNESS, label=LABEL, field=field
        )
        path = entry.expanduser()
        if not path.is_absolute():
            path = Path(base_dir) / path
        if not resolved.is_dir():
            raise LifecycleError(
                "hermes_plugin_invalid",
                f"Hermes plugin path {str(path)!r} is not a folder",
                metadata={"field": field, "path": str(path), "reason": "not_a_folder"},
            )
        plugin = MountedPlugin(field, _manifest_name(path, field), resolved)
        earlier = names.get(plugin.name)
        if earlier and not (field == ENABLED_DIRECTORY and earlier.path == resolved):
            raise LifecycleError(
                "hermes_plugin_duplicate",
                f"Hermes plugin {plugin.name!r} is in {earlier.field} and {field}",
                metadata={
                    "field": field,
                    "plugin": plugin.name,
                    "paths": [str(earlier.path), str(plugin.path)],
                },
            )
        names[plugin.name] = plugin
    for plugin in plugin_entries(settings):
        if plugin.section == "external" and plugin.plugin not in names:
            raise LifecycleError(
                common_plugins.not_found_code(HARNESS),
                f"Hermes plugin {plugin.plugin!r} is not in any "
                "configured external plugin folder",
                metadata={"field": plugin.field, "plugin": plugin.plugin},
            )
    return list(names.values())


LINKS_RECORD = ".nemo-fabric-plugin-links.json"


def _read_links_record(hermes_home: Path) -> dict[str, str]:
    """The links an earlier start made, by plugin name, with their targets."""

    try:
        record = json.loads((hermes_home / LINKS_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(record, dict):
        return {}
    return {str(name): str(target) for name, target in record.items()}


def _write_links_record(hermes_home: Path, record: Mapping[str, str]) -> None:
    path = hermes_home / LINKS_RECORD
    if record:
        # Replace the record whole, so a failed write keeps the previous one.
        partial = path.with_name(f"{path.name}.partial")
        partial.write_text(json.dumps(dict(record), sort_keys=True), encoding="utf-8")
        os.replace(partial, path)
    else:
        path.unlink(missing_ok=True)


def _is_recorded_link(link: Path, record: Mapping[str, str]) -> bool:
    """Whether ``link`` is a symlink NeMo Fabric made and nothing has replaced."""

    try:
        return link.is_symlink() and os.readlink(link) == record.get(link.name)
    except OSError:
        return False


def stage_mounted(
    mounted: Iterable[MountedPlugin],
    hermes_home: Path,
    links: list[Path] | None = None,
) -> list[Path]:
    """Link each mounted plugin into ``$HERMES_HOME/plugins/<name>``.

    Each link is recorded in ``$HERMES_HOME/.nemo-fabric-plugin-links.json``. A link of that
    name that an earlier start recorded is replaced; anything else of that
    name, whether a folder or a symlink, belongs to the user or the agent, so
    start fails with ``hermes_plugin_duplicate`` rather than shadow it.
    Returns the links made, for :func:`remove_mounted`, appending each to
    ``links`` as it's made when given; links made before a failure are
    removed.
    """

    plugins_dir = hermes_home / "plugins"
    links = [] if links is None else links
    record = _read_links_record(hermes_home)
    try:
        for plugin in mounted:
            plugins_dir.mkdir(parents=True, exist_ok=True)
            link = plugins_dir / plugin.name
            if _is_recorded_link(link, record):
                link.unlink()
            elif link.is_symlink() or link.exists():
                raise LifecycleError(
                    "hermes_plugin_duplicate",
                    f"Hermes plugin {plugin.name!r} from {plugin.field} has the "
                    "name of a plugin already in the Hermes home",
                    metadata={
                        "field": plugin.field,
                        "plugin": plugin.name,
                        "paths": [str(link), str(plugin.path)],
                    },
                )
            record[plugin.name] = str(plugin.path)
            _write_links_record(hermes_home, record)
            link.symlink_to(plugin.path, target_is_directory=True)
            links.append(link)
    except BaseException:
        remove_mounted(links)
        raise
    return links


def remove_mounted(links: Iterable[Path]) -> None:
    """Remove the links :func:`stage_mounted` made; the plugin folders stay.

    A link the user or the agent replaced since start is left alone.
    """

    records: dict[Path, dict[str, str]] = {}
    for link in links:
        hermes_home = link.parent.parent
        record = records.setdefault(hermes_home, _read_links_record(hermes_home))
        try:
            if _is_recorded_link(link, record):
                link.unlink()
        except OSError:
            # Keep the entry, so a later start can still replace the link.
            LOGGER.warning(
                "Hermes plugin link %s could not be removed", link, exc_info=True
            )
            continue
        record.pop(link.name, None)
    for hermes_home, record in records.items():
        try:
            _write_links_record(hermes_home, record)
        except OSError:
            LOGGER.warning(
                "Hermes plugin link record in %s could not be updated",
                hermes_home,
                exc_info=True,
            )
