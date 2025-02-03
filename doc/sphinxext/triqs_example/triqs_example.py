# -*- coding: utf-8 -*-
# seems to be executed at the level of the conf.py
# so we need to link the lib at that place...
"""
"""
import os
import sys
import codecs
from os import path
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from sphinx.errors import SphinxError


class TriqsExampleError(SphinxError):
    category = 'triqs_example error'


class TriqsExampleRun:
    config = dict()

    @classmethod
    def builder_init(cls, app):
        pass


class TriqsExample(Directive):
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        'linenos': directives.flag,
    }

    def run(self):
        document = self.state.document
        filename = self.arguments[0]
        if not document.settings.file_insertion_enabled:
            return [document.reporter.warning('File insertion disabled',
                                              line=self.lineno)]
        env = document.settings.env
        if filename.startswith('/') or filename.startswith(os.sep):
            rel_fn = filename[1:]
        else:
            docdir = path.dirname(env.doc2path(env.docname, base=None))
            rel_fn = path.normpath(path.join(docdir, filename))
        try:
            fn = path.join(env.srcdir, rel_fn)
        except UnicodeDecodeError:
            # the source directory is a bytestring with non-ASCII characters;
            # let's try to encode the rel_fn in the file system encoding
            rel_fn = rel_fn.encode(sys.getfilesystemencoding())
            fn = path.join(env.srcdir, rel_fn)

        encoding = self.options.get('encoding', env.config.source_encoding)
        try:
            f = codecs.open(fn, 'rU', encoding)
            lines = f.readlines()
            f.close()
        except (IOError, OSError):
            return [document.reporter.warning(
                'Include file %r not found or reading it failed' % filename,
                line=self.lineno)]
        except UnicodeError:
            return [document.reporter.warning(
                'Encoding %r used for reading included file %r seems to '
                'be wrong, try giving an :encoding: option' %
                (encoding, filename))]

        output_encoding = 'utf8'

        # Build the code text
        code = ''.join(lines).strip()
        filename_clean = filename.rsplit('.', 1)[0]
        if filename_clean.startswith('./'):
            filename_clean = filename_clean[2:]

        import subprocess as S
        error = True
        try:
            stdout = ''
            resout = S.check_output("./%s/doc_%s" % (docdir, filename_clean),
                                    stderr=S.STDOUT, shell=True)
            if resout:
                stdout = '---------- Result is -------\n' + resout.strip()
            error = False
        except S.CalledProcessError as E:
            stdout = '---------- RunTime error -------\n'
            stdout += E.output

        # Process output
        if stdout:
            stdout = stdout.decode(output_encoding, 'ignore')
            out = ''.join(stdout).decode(output_encoding)
        else:
            out = ''

        # Get the original code with prefixes
        code_out = '\n'.join((code, out))

        if error:  # report on console
            print(" Error in processing ")
            print(code_out)

        literal = nodes.literal_block(code_out, code_out)
        literal['language'] = 'c'
        literal['linenos'] = 'linenos' in self.options
        return [literal]


def setup(app):
    app.add_directive('triqs_example', TriqsExample)
    app.connect('builder-inited', TriqsExampleRun.builder_init)
