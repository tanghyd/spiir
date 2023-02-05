# Developer Guide

## Documentation Style

All documentation should follow the [NumPy Documentation Style Guide].
The NumPy style guide is a derivation of the Google Developer documentation style guide.
All docstrings should follow the style and order of sections as defined in the links above.

Further information about writing Python docstrings can be found in the multi-line
docstring specifications from [PEP 257] detailing docstring conventions for scripts,
modules, functions, and classes. For an example of NumPy Style Python docstrings we
refer to an [example from Sphinx].

Note that this NumPy example starts with a UTF-8 encoding declaration that we have
chosen to omit in our package documentation because UTF-8 is already the default
encoding, as described in [lexical analysis] section of the Python documentation.

## Precommit Git Hooks

This repository uses [pre-commit] to ensure that code standards are met in regard to
formatting and style. For example, we ensure all Python code is formatted according to
the style of [Black], and that all markdown text is also formatted properly (with code
snippets inside docstrings also matching Black's style with mdformat-black). We also
add configuration files in pyproject.toml and setup.cfg that ensure [isort] and [flake8]
run matching Black's configuration (although flake8 does not run during CI yet).
Additionally, there are also a number of standard pre-commit hooks that check valid
XML, YAML, JSON file types, check for accidental merge conflict text, unnecessary
whitespace, and missing end of file new lines.

## License

SPIIR is licensed according to the GNU General Public License v3, which is defined in
further detail in the [project license](../about/license.md) page.

As a note for contributors, this license file is located at docs/about/license.md, and
is technically a symbolic link to the LICENSE file kept at the root of the repository
(which is there for easy access for visitors and for discovery by GitHub/GitLab
repositories). To create this symlink, route to the docs/about directory (or any
location where your desired output file will be kept, adjusting paths accordingly) and
run the following:

```bash
ln -s -T ../../LICENSE license.md
```

[black]: https://black.readthedocs.io/en/stable/1
[example from sphinx]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy
[flake8]: https://flake8.pycqa.org/en/latest/
[isort]: https://pycqa.github.io/isort/
[lexical analysis]: https://docs.python.org/3/reference/lexical_analysis.html#encoding-declarations
[numpy documentation style guide]: https://numpydoc.readthedocs.io/en/latest/format.html
[pep 257]: https://peps.python.org/pep-0257/#multi-line-docstrings
[pre-commit]: https://pre-commit.com/
