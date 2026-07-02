# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""NVIDIA Sphinx theme customization."""

import datetime
from pathlib import Path
from string import Template

from pydata_sphinx_theme.utils import config_provided_by_user, get_theme_options_dict
from sphinx.addnodes import desc_name, desc_parameter, desc_signature_line
from sphinx.errors import ThemeError
from sphinx.util.fileutil import copy_asset_file

__version__ = "0.0.9.post1"
__all__ = []

_THEME_DIR = Path(__file__).parent / "theme" / "nvidia_sphinx_theme"
_DEFAULT_FAVICON_FILE = _THEME_DIR / "static" / "favicon.png"
_DEFAULT_FOOTER_LINKS = [
    {"name": "Privacy Policy", "url": "https://www.nvidia.com/en-us/about-nvidia/privacy-policy/"},
    {"name": "Your Privacy Choices", "url": "https://www.nvidia.com/en-us/about-nvidia/privacy-center/"},
    {"name": "Terms of Service", "url": "https://www.nvidia.com/en-us/about-nvidia/terms-of-service/"},
    {"name": "Accessibility", "url": "https://www.nvidia.com/en-us/about-nvidia/accessibility/"},
    {"name": "Corporate Policies", "url": "https://www.nvidia.com/en-us/about-nvidia/company-policies/"},
    {"name": "Product Security", "url": "https://www.nvidia.com/en-us/product-security/"},
    {"name": "Contact", "url": "https://www.nvidia.com/en-us/contact/"},
]


def _update_html_title(app):
    """Update the html_title if not set by the user."""
    if config_provided_by_user(app, "html_title"):
        return

    app.config.html_title = app.config.project


def _set_logo(app):
    """Ensure the NVIDIA logo is used for all docs."""
    theme_options = get_theme_options_dict(app)

    # always use the NVIDIA logo
    if "logo" in theme_options:
        raise ThemeError(
            "the logo is set by the theme and cannot be set by the user, do not set `html_theme_settings['logo']`"
        )

    logo = theme_options.setdefault("logo", {})
    logo["image_light"] = "_static/nvidia-logo-horiz-rgb-blk-for-screen.svg"
    logo["image_dark"] = "_static/nvidia-logo-horiz-rgb-wht-for-screen.svg"
    logo["text"] = f"{app.config.html_title}"
    logo["alt_text"] = f"{app.config.html_title} - Home"


def _update_author(app):
    """Update the author if not set by the user."""
    if config_provided_by_user(app, "author"):
        return

    app.config.author = "NVIDIA Corporation"


def _update_copyright(app):
    """Update the copyright based on the template and given parameters if not set by the user."""
    if config_provided_by_user(app, "copyright"):
        return

    theme_options = get_theme_options_dict(app)

    # set default base config
    parameters = theme_options.get("copyright_override") or {}

    # set defaults if not given by the user
    template = parameters.setdefault("template", "${start}${separator}${end} ${author}")
    parameters.setdefault("start", str(datetime.date.today().year))
    parameters.setdefault("end", str(datetime.date.today().year))
    parameters.setdefault("separator", "-")
    parameters.setdefault("author", app.config.author)

    # handle the case where only one year should be displayed
    if parameters["start"] == parameters["end"]:
        parameters["start"] = ""
        parameters["separator"] = ""

    app.config.copyright = Template(template).safe_substitute(parameters)


def _verify_announcement_is_not_remote(app):
    """Check that that the announcement is not remotely loading content."""
    theme_options = get_theme_options_dict(app)

    if theme_options.get("announcement", "").startswith("http"):
        raise ThemeError("announcements loading remote content are not permitted")


def _set_default_footer_links(app):
    """Set the default footer links if not set by the user."""
    theme_options = get_theme_options_dict(app)
    footer_links = theme_options.get("footer_links")

    # if set by the user nothing to do
    if footer_links is not None:
        return

    theme_options["footer_links"] = _DEFAULT_FOOTER_LINKS


def _update_toc_object_entries_show_parents(app):
    """Update toc_object_entries_show_parents if not set by the user."""
    if config_provided_by_user(app, "toc_object_entries_show_parents"):
        return

    app.config.toc_object_entries_show_parents = "hide"


def _update_maximum_signature_line_length(app):
    """Update maximum_signature_line_length if not set by the user."""
    if config_provided_by_user(app, "maximum_signature_line_length"):
        return

    # approximate maximum width of member signatures
    app.config.maximum_signature_line_length = 70


def _make_api_consistent(app, doctree):
    """Sphinx C and C++ domain do not include <em> tags which makes them inconsistent with Py API."""
    for param in doctree.findall(desc_parameter):
        param.delattr("noemph")


def _mark_template_sig_name(app, doctree, docname):
    """To simplify styling of signatures, mark the sig-name objects found if they are templates."""
    for sig in doctree.findall(desc_signature_line):
        if sig.sphinx_line_type != "templateParams":
            continue
        for name in sig.findall(desc_name):
            name["classes"].append("sig-name-template")


def _update_search_scripts(app, pagename, templatename, context, doctree):
    """Take the default string definition of search scripts and make it a list."""
    scripts = context.get("theme_search_scripts", [])

    if isinstance(scripts, str):
        scripts = [s.strip() for s in scripts.strip().split(",") if s.strip()]
    elif isinstance(scripts, list):
        pass
    else:
        raise ValueError("`html_theme_options['search_scripts']` must be a list of script paths")

    context["theme_search_scripts"] = scripts


def _update_favicon(app, pagename, templatename, context, doctree):
    """Update the favicon path if not given by the user."""
    if config_provided_by_user(app, "html_favicon"):
        return

    context["favicon_url"] = _DEFAULT_FAVICON_FILE.name


def _ensure_sidebar_toctree(app, pagename, templatename, context, doctree):
    """Prevent the base theme from hiding the sidebar."""
    context["suppress_sidebar_toctree"] = lambda *args, **kwargs: False


def _copy_favicon(app, exception):
    """Copy the favicon image if not given by the user."""
    if exception is not None:
        return

    if app.builder.format != "html":
        return

    if config_provided_by_user(app, "html_favicon"):
        return

    staticdir = Path(app.builder.outdir) / "_static"
    copy_asset_file(_DEFAULT_FAVICON_FILE, staticdir)


def setup(app):
    """Configure theme specific custom Sphinx functionality."""
    app.add_html_theme("nvidia_sphinx_theme", _THEME_DIR)
    app.connect("builder-inited", _update_html_title)
    app.connect("builder-inited", _set_logo)
    app.connect("builder-inited", _set_default_footer_links)
    app.connect("builder-inited", _update_author)
    app.connect("builder-inited", _update_copyright)
    app.connect("builder-inited", _verify_announcement_is_not_remote)
    app.connect("builder-inited", _update_toc_object_entries_show_parents)
    app.connect("builder-inited", _update_maximum_signature_line_length)
    app.connect("doctree-read", _make_api_consistent)
    app.connect("doctree-resolved", _mark_template_sig_name)
    app.connect("html-page-context", _update_search_scripts)
    app.connect("html-page-context", _update_favicon, priority=100)  # run before sphinx handler setup_resource_paths
    app.connect("html-page-context", _ensure_sidebar_toctree, priority=1000)  # run after base theme has set this
    app.connect("build-finished", _copy_favicon)

    app.config.templates_path.append(str(_THEME_DIR / "components"))

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
