import os

import pytest
import _pytest.capture

from nengo.utils.stdlib import execfile

# Monkeypatch _pytest.capture.DontReadFromInput
#  If we don't do this, importing IPython will choke as it reads the current
#  sys.stdin to figure out the encoding it will use; pytest installs
#  DontReadFromInput as sys.stdin to capture output.
#  Running with -s option doesn't have this issue, but this monkeypatch
#  doesn't have any side effects, so it's fine.
_pytest.capture.DontReadFromInput.encoding = "utf-8"
_pytest.capture.DontReadFromInput.write = lambda: None
_pytest.capture.DontReadFromInput.flush = lambda: None


all_examples = []
examples_dir = os.path.realpath(".")

for subdir, _, files in os.walk(examples_dir):
    if (os.path.sep + '.') in subdir:
        continue
    files = [f for f in files if f.endswith('.ipynb')]
    examples = [os.path.join(subdir, os.path.splitext(f)[0]) for f in files]
    all_examples.extend(examples)

# os.walk goes in arbitrary order, so sort after the fact to keep pytest happy
all_examples.sort()


def assert_noexceptions(nb_file, tmpdir):
    plt = pytest.importorskip('matplotlib.pyplot')
    pytest.importorskip("IPython", minversion="1.0")
    pytest.importorskip("jinja2")
    from nengo.utils.ipython import export_py, load_notebook
    nb_path = os.path.join(examples_dir, "%s.ipynb" % nb_file)
    nb = load_notebook(nb_path)
    pyfile = "%s.py" % (
        tmpdir.join(os.path.splitext(os.path.basename(nb_path))[0]))
    export_py(nb, pyfile)
    execfile(pyfile, {})
    plt.close('all')


def iter_cells(nb_file, cell_type="code"):
    from nengo.utils.ipython import load_notebook
    nb = load_notebook(os.path.join(examples_dir, "%s.ipynb" % nb_file))

    if nb.nbformat <= 3:
        cells = []
        for ws in nb.worksheets:
            cells.extend(ws.cells)
    else:
        cells = nb.cells

    for cell in cells:
        if cell.cell_type == cell_type:
            yield cell


@pytest.mark.example
@pytest.mark.parametrize('nb_file', all_examples)
def test_no_signature(nb_file):
    from nengo.utils.ipython import load_notebook
    nb = load_notebook(os.path.join(examples_dir, "%s.ipynb" % nb_file))
    assert 'signature' not in nb.metadata, "Notebook has signature"


@pytest.mark.example
@pytest.mark.parametrize('nb_file', all_examples)
def test_no_outputs(nb_file):
    """Ensure that no cells have output."""
    pytest.importorskip("IPython", minversion="1.0")

    for cell in iter_cells(nb_file):
        assert cell.outputs == [], "Cell outputs not cleared"
