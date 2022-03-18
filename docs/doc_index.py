#!/usr/bin/python3
class ExamplePage:
    def __init__(self, entries, title, prefix_path, index_name):
        self.entries = entries
        self.title = title
        self.prefix_path = prefix_path
        self.index_name = index_name

    def get_index(self):
        s = """
.. title:: {}
.. toctree::
   :maxdepth: 2

""".format(self.title)
        for entry in self.entries:
            if isinstance(entry, ExamplePage):
                if entry.index_name is not None:
                    s += "  " + entry.index_name
                else:
                    s += "  {}/index\n".format(entry.prefix_path)
            else:
                s += "  {}".format(entry.jupyter_name)
        return s

    def write_to_location():
        pass


class ExampleEntry:
    def __init__(self, jupyter_name, operator = None):
        self.jupyter_name = jupyter_name
        self.operator = operator

class DocReference:
    def __init__(self, operator, docstring=None):
        self.operator = operator
        self.docstring = docstring

# page listing examples
def example_page(entries, *, title, prefix_path="", index_name=None):
    return ExamplePage(entries, title, prefix_path, index_name)

# example listed on the page
def example_entry(jupyter_name, operator: str or DocReference=None):
    if isinstance(operator, str):
        operator = doc_reference(operator)
    return ExampleEntry(jupyter_name, operator)

# Add a reference for this tutorial in given operator doc with optional docstring
def doc_reference(operator, docstring=None):
    return DocReference(operator, docstring)

# Downside of this approach is the need to define the doc tree bottom-up
# We can maybe split them into separate files corresponding to the actual index locations?

data_loading = example_page([
    example_entry('external_input.ipynb', [doc_reference('external_source')]),
    example_entry('parallel_external_source.ipynb',
              doc_reference('external_source', 'How to use parallel mode for external source')),
    example_entry('parallel_external_source_fork.ipynb',
              doc_reference('external_source', 'How to use parallel mode for external source in fork mode')),
    example_entry('dataloading_lmdb.ipynb'),
    example_entry('dataloading_recordio.ipynb'),
    example_entry('dataloading_tfrecord.ipynb'),
    example_entry('dataloading_webdataset.ipynb'),
    example_entry('coco_reader.ipynb'),
    example_entry('numpy_reader.ipynb', 'readers.numpy'),
], title='Data Loading', prefix_path='general/data_loading')


general_purpose = example_page([
    example_entry('reductions.ipynb'),
    example_entry('tensor_join.ipynb'),
    example_entry('reinterpret.ipynb'),
    example_entry('normalize.ipynb'),
], title="General purpose", prefix_path="general")

operations = example_page([
    general_purpose,
    # image_processing,
    # audio_processing,
    # video_processing,
], title="Operations", index_name="operations_index")

tot = example_page([
    data_loading,
    operations,
    # use_cases,
    # other,
], title="Examples and tutorials")



print(tot.get_index())