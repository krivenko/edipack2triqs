# -*- coding: utf-8 -*-
#
# TRIQS documentation build configuration file

import pathlib
import sys
from sphinx.writers.html import HTMLTranslator
from docutils import nodes
from docutils.nodes import Element


sys.path.insert(0, "sphinxext")
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))


# exclude these folders from scanning by sphinx
exclude_patterns = ['_templates']

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'sphinx.ext.intersphinx',
              'sphinx.ext.doctest',
              'sphinx.ext.todo',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx.ext.githubpages',
              'nbsphinx',
              'myst_parser',
              'matplotlib.sphinxext.plot_directive',
              'sphinxfortran_ng.fortran_domain'
              ]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

project = 'edipack2triqs'
version = '0.6.0'
copyright = '2024-2025, Igor Krivenko, Lorenzo Crippa'

source_suffix = '.rst'
templates_path = ['_templates']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Turn on sphinx.ext.autosummary
autosummary_generate = True
autosummary_imported_members = False

# sphinx.ext.autodoc options
autodoc_member_order = 'bysource'

# this makes the current project version available as var in every rst file
rst_epilog = """
.. |PROJECT_VERSION| replace:: {version}
""".format(version=version)

# this requires the sphinx_rtd_theme to be installed via pip
html_theme = 'sphinx_rtd_theme'
# this loads the custom css file to change the page width
html_style = 'css/custom.css'

html_favicon = '_static/triqs_logo/triqs_favicon.ico'

# options for the the rtd theme
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#7E588A',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 5,
    'includehidden': True,
    'titles_only': False
}

html_show_sphinx = False

html_context = {'header_title': 'edipack2triqs'}

html_static_path = ['_static']
html_sidebars = {'index': ['sideb.html', 'searchbox.html']}

htmlhelp_basename = 'edipack2triqsdoc'

# Plot options
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3.12', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'triqslibs': ('https://triqs.github.io/triqs/latest', None),
    'edipack': ('https://edipack.github.io/EDIpack', None),
    'edipack2py': ('https://edipack.github.io/EDIpack2py', None)
}


# Open links in new tab instead of same window
class PatchedHTMLTranslator(HTMLTranslator):

    def visit_reference(self, node: Element) -> None:
        atts = {'class': 'reference'}
        if node.get('internal') or 'refuri' not in node:
            atts['class'] += ' internal'
        else:
            atts['class'] += ' external'
            # ---------------------------------------------------------
            # Customize behavior (open in new tab, secure linking site)
            atts['target'] = '_blank'
            atts['rel'] = 'noopener noreferrer'
            # ---------------------------------------------------------
        if 'refuri' in node:
            atts['href'] = node['refuri'] or '#'
            if self.settings.cloak_email_addresses and \
               atts['href'].startswith('mailto:'):
                atts['href'] = self.cloak_mailto(atts['href'])
                self.in_mailto = True
        else:
            assert 'refid' in node, \
                   'References must have "refuri" or "refid" attribute.'
            atts['href'] = '#' + node['refid']
        if not isinstance(node.parent, nodes.TextElement):
            assert len(node) == 1 and isinstance(node[0], nodes.image)
            atts['class'] += ' image-reference'
        if 'reftitle' in node:
            atts['title'] = node['reftitle']
        if 'target' in node:
            atts['target'] = node['target']
        self.body.append(self.starttag(node, 'a', '', **atts))

        if node.get('secnumber'):
            self.body.append(('%s' + self.secnumber_suffix) %
                             '.'.join(map(str, node['secnumber'])))


def setup(app):
    app.set_translator('html', PatchedHTMLTranslator)
