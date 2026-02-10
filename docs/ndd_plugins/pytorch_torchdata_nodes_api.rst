.. _ndd_torchdata:

TorchData Integration Reference
===============================

.. currentmodule:: nvidia.dali.experimental.dynamic.pytorch.nodes

DALI Dynamic provides integration with :mod:`torchdata.nodes` to build composable
data loading pipelines. The following node classes can be composed with standard
``torchdata.nodes`` building blocks such as :class:`~torchdata.nodes.Prefetcher`
and :class:`~torchdata.nodes.Loader`.

Reader
------

.. autoclass:: Reader
   :members:

DictMapper
----------

.. autoclass:: DictMapper
   :members:

ToTorch
-------

.. autoclass:: ToTorch
   :members:

Usage Pattern
-------------

A typical pipeline composes these nodes with ``torchdata.nodes`` utilities:

.. code-block:: python

    import nvidia.dali.experimental.dynamic as ndd
    import torchdata.nodes as tn

    reader_node = ndd.pytorch.nodes.Reader(
        ndd.readers.File,
        batch_size=batch_size,
        file_root=data_dir,
        random_shuffle=True,
    )
    mapper_node = ndd.pytorch.nodes.DictMapper(
        source=reader_node,
        map_fn=my_processing_function,
    )
    torch_node = ndd.pytorch.nodes.ToTorch(mapper_node)
    prefetch_node = tn.Prefetcher(torch_node, prefetch_factor=2)
    loader = tn.Loader(prefetch_node)

    for images, labels in loader:
        # images, labels are torch.Tensors on GPU
        ...


The above snippet defines the following simple graph:

.. raw:: html

    <style>
        .graphviz { background: transparent !important; }
    </style>

.. digraph:: simple_pipeline
    :align: center

    rankdir=LR;
    bgcolor="transparent";
    margin="0.71, 0";
    dpi=300;
    node [shape=box, style="filled,rounded", fontname="NVIDIA Sans, sans-serif",
            fontsize=14, penwidth=0];
    edge [color="#707070"];

    Reader      [fillcolor="#76B900", fontcolor="white", label="Reader"];
    DictMapper  [fillcolor="#76B900", fontcolor="white", label="DictMapper"];
    ToTorch     [fillcolor="#76B900", fontcolor="white", label="ToTorch"];
    Prefetcher  [fillcolor="#DE3412", fontcolor="white", label="Prefetcher"];

    Reader -> DictMapper -> ToTorch -> Prefetcher;

| In practice, :mod:`torchdata.nodes` allows composing nodes to define complex graphs. For instance,
| if we wanted to apply a transformation to the labels we could add another :class:`DictMapper` node
| that takes the output of the :class:`Reader`. :func:`torchdata.nodes.Mapper` can be used to
| combine the outputs:

.. digraph:: complex_pipeline
    :align: center

    rankdir=LR;
    bgcolor="transparent";
    dpi=300;
    node [shape=box, style="filled,rounded", fontname="NVIDIA Sans, sans-serif",
            fontsize=14, penwidth=0, fontcolor="white"];
    edge [color="#707070"];

    Reader       [fillcolor="#76B900", label="Reader"];
    DictMapper1  [fillcolor="#76B900", label="DictMapper"];
    DictMapper2  [fillcolor="#76B900", label="DictMapper"];
    Mapper       [fillcolor="#DE3412", label="Mapper"];
    ToTorch      [fillcolor="#76B900", label="ToTorch"];
    Prefetcher   [fillcolor="#DE3412", label="Prefetcher"];

    Reader -> DictMapper1 -> Mapper;
    Reader -> DictMapper2 -> Mapper;
    Mapper -> ToTorch -> Prefetcher;
