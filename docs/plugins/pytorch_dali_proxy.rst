PyTorch DALI Proxy
==================

Overview
--------

**DALI Proxy** is a tool designed to integrate NVIDIA DALI pipelines with PyTorch data workers while maintaining the simplicity of PyTorch's dataset logic. The key features of DALI Proxy include:

- **Efficient GPU Utilization**: DALI Proxy ensures GPU data processing occurs on the same process running the main loop. This avoids performance degradation caused by multiple CUDA contexts for the same GPU.
- **Selective Offloading**: Users can offload parts of the data processing pipeline to DALI while retaining PyTorch Dataset logic, making it ideal for multi-modal applications.

This tutorial will explain the key components, workflow, and usage of DALI Proxy in PyTorch.

DALI Proxy Workflow
-------------------

**Key Components**

1. **DALI Pipeline**  
   A user-defined DALI pipeline processes input data.

2. **DALI Server**  
   The server runs a background thread to execute the DALI pipeline asynchronously.

3. **DALI Proxy**  
   A callable interface between PyTorch data workers and the DALI Server.

4. **PyTorch Dataset and DataLoader**  
   The Dataset remains agnostic of DALI internals and uses the Proxy for preprocessing.

**Workflow Summary**

- A DALI pipeline is defined and connected to a **DALI Server**, which executes the pipeline in a background thread.
- The **DALI Proxy** provides an interface for PyTorch data workers to request DALI processing asynchronously.
- Each data worker invokes the proxy, which returns a **reference to a future processed sample**.
- During batch collation, the proxy groups data into a batch and sends it to the server for execution.
- The server processes the batch asynchronously and outputs the actual data to an output queue.
- The PyTorch DataLoader retrieves either the processed data or references to pending pipeline runs. If it encounters pipeline run references, it queries the DALI server for the actual data, waiting if necessary until the data becomes available in the output queue.

Example Usage
-------------

DALI Proxy in a Nutshell
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from torchvision import datasets, transforms
   from nvidia.dali import pipeline_def, fn, types
   from nvidia.dali.plugin.pytorch.experimental import proxy as dali_proxy

   # Step 1: Define a DALI pipeline
   @pipeline_def
   def my_dali_pipeline():
       images = fn.external_source(name="images", no_copy=True)
       images = fn.resize(images, size=[224, 224])
       return fn.crop_mirror_normalize(
           images, dtype=types.FLOAT, output_layout="CHW",
           mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
           std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
       )

   # Step 2: Initialize DALI server
   with dali_proxy.DALIServer(my_dali_pipeline(batch_size=64, num_threads=3, device_id=0)) as dali_server:
       # Step 3: Define a PyTorch Dataset using the DALI proxy
       dataset = datasets.ImageFolder("/path/to/images", transform=dali_server.proxy)
       
       # Step 4: Use DALI proxy DataLoader
       loader = dali_proxy.DataLoader(dali_server, dataset, batch_size=64, num_workers=8, drop_last=True)
       
       # Step 5: Consume data
       for data, target in loader:
           print(data.shape)  # Processed data ready

How It Works
------------

**1. DALI Pipeline**

The DALI pipeline defines the data processing steps. Input data is fed using ``fn.external_source``.

.. code-block:: python

   from nvidia.dali import pipeline_def, fn, types

   @pipeline_def
   def example_pipeline():
       images = fn.external_source(name="images", no_copy=True)
       images = fn.io.file.read(images)
       images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
       return fn.resize(images, size=[224, 224])

   pipeline = example_pipeline(batch_size=32, num_threads=2, device_id=0)

**2. DALI Server and Proxy**

The DALI Server manages the execution of the pipeline. The Proxy acts as an interface for PyTorch data workers.

.. code-block:: python

   from nvidia.dali.plugin.pytorch.experimental import proxy as dali_proxy

   with dali_proxy.DALIServer(pipeline) as dali_server:
      # ... in the workers
      transform_fn = dali_server.proxy  # A callable interface for workers
      future_samples = [transform_fn(image) for image in images]

**3. Data Collation and Execution**

The ``default_collate`` function combines processed samples into a batch. DALI executes the pipeline asynchronously when a batch is collated.
This step is usually abstracted away inside the PyTorch DataLoader and the user doesn't need to take care of it explicitly.

.. code-block:: python

   from torch.utils.data.dataloader import default_collate

   # Collate samples into a single batch and send to DALI
   processed_batch = default_collate(future_samples)

**4. Integration with PyTorch Dataset**

The PyTorch Dataset can directly use the proxy as a transform function. Note that we can choose to offload only part of the
processing to DALI, while keeping some of the original data intact.

.. code-block:: python

   class CustomDataset(torch.utils.data.Dataset):
       def __init__(self, transform_fn, data):
           self.data = data
           self.transform_fn = transform_fn

       def __len__(self):
           return len(self.data)

       def __getitem__(self, idx):
           filename, label = self.data[idx]
           return self.transform_fn(filename), label  # Returns processed sample and the original label

**5. Integration with PyTorch DataLoader**

The ``DataLoader`` wrapper provided by DALI Proxy simplifies the integration process.

.. code-block:: python

   with dali_proxy.DALIServer(pipeline) as dali_server:
      dataset = CustomDataset(dali_server.proxy, data=images)
      loader = dali_proxy.DataLoader(dali_server, dataset, batch_size=32, num_workers=4)
      for data, _ in loader:
         print(data.shape)  # Ready-to-use processed batch

If using a custom ``DataLoader``, call the DALI server explicitly:

.. code-block:: python

   for data, _ in loader:
      # Replaces instances of ``DALIOutputBatchRef`` with actual data
      processed_data = dali_server.produce_data(data)
      print(processed_data.shape)  # data is now ready

Summary
-------

DALI Proxy provides a clean and efficient way to integrate NVIDIA DALI with PyTorch. By offloading computationally intensive tasks to DALI while keeping PyTorch's Dataset and DataLoader interface intact, it ensures flexibility and maximum performance.
This approach is particularly powerful in large-scale data pipelines and multi-modal workflows.