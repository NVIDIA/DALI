Checkpointing
=============

.. currentmodule:: nvidia.dali

Checkpointing is a feature in DALI which allows you to save the current state of the pipeline to a file.
Then, you can restore the pipeline from a saved checkpoint and the new pipeline will produce exactly the same outputs as the old one would.
It is particularly useful for long-running training jobs which are likely to be interrupted.

A checkpoint of DALI pipeline contains information about states of all random number generators used in the pipeline and about the progress of each reader.

Checkpointing API
-----------------

Enabling checkpointing
~~~~~~~~~~~~~~~~~~~~~~

To enable checkpointing, set ``enable_checkpointing=True`` when creating a pipeline.
With this option enabled, DALI will track the state of each operator, allowing you to save it on demand.
Enabling checkpointing shouldn't have any impact on the performance.

.. code-block:: python

  @pipeline_def(..., enable_checkpointing=True)
  def pipeline():
      ...

  p = pipeline()
  p.build()


.. note::
    Readers with ``shuffle_after_epoch=True`` might produce different (but still equally random) output if checkpointing is enabled.


Saving a checkpoint
~~~~~~~~~~~~~~~~~~~

To save a checkpoint, you need to call :meth:`Pipeline.checkpoint` method, which will return a serialized checkpoint as a string.
Optionally, you can pass filename as an argument and DALI will save the checkpoint there.

.. code-block:: python

  for _ in range(iters):
      output = p.run()

  # Write the checkpoint to file:
  checkpoint = p.checkpoint()
  open('checkpoint_file.cpt', 'wb')

  # Or simply:
  checkpoint = p.checkpoint('checkpoint_file.cpt')

.. note::
    Calling :meth:`Pipeline.checkpoint method` may introduce an observable overhead.
    We recommend you not to call it too often.

Restoring from checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~

You can later restore pipeline state from a saved checkpoint.
To do so, pass `checkpoint` argument to :class:`Pipeline` on construction.
Such a pipeline should then return exactly the same outputs as the original one.

.. code-block:: python

  checkpoint = open('checkpoint_file.cpt', 'rb').read()
  p_restored = pipeline(checkpoint=checkpoint)
  p_restored.build()

.. warning::
    Make sure that the pipeline that you're restoring is the same as the original one,
    i.e. contains the same operators with the same arguments.
    Restoring from a checkpoint created with a different pipeline will result in undefined behavior.

External source checkpointing
-----------------------------

:meth:`fn.external_source` operator only partially supports checkpointing.

Checkpointing is supported only if ``source`` is a single-argument callable accepting
batch index, ``BatchInfo`` or ``SampleInfo``.
For such ``sources``, the queries will continue from the point saved in the checkpoint.

Other kinds of ``source`` don't support checkpointing.
Their state won't be saved in a checkpoint and
after restoring from a checkpoint, they will start from the beginning.
If you want to use checkpointing, we recommend you rewrite your source
to be a supported callable.