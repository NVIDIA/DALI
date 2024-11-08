#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::ptr;

    const BATCH_SIZE: i32 = 12;
    const NUM_THREADS: i32 = 1;
    const DEVICE_ID: i32 = 0;
    const SEED: i64 = 1234;
    const PIPELINED: bool = true;
    const PREFETCH_QUEUE_DEPTH: i32 = 2;
    const ASYNC: bool = false;

    #[test]
    fn test_get_output_name() {
        let output0_name = "compressed_images";
        let output1_name = "labels";

        unsafe {
            // Create pipeline handle
            let mut handle = ptr::null_mut();

            // Create pipeline and serialize it
            let pipeline = create_test_pipeline(output0_name, output1_name);
            let serialized = serialize_pipeline(&pipeline);

            // Create DALI pipeline from serialized data
            let ret = daliCreatePipeline(
                &mut handle,
                serialized.as_ptr() as *const i8,
                serialized.len() as i32,
                BATCH_SIZE,
                NUM_THREADS,
                DEVICE_ID,
                false as i32,
                PREFETCH_QUEUE_DEPTH,
                PREFETCH_QUEUE_DEPTH,
                PREFETCH_QUEUE_DEPTH,
                false as i32,
            );
            assert_eq!(ret, 0);

            // Test number of outputs
            assert_eq!(daliGetNumOutput(&handle), 2);

            // Test output names
            let name0 = CString::new(output0_name).unwrap();
            let name1 = CString::new(output1_name).unwrap();

            assert_eq!(
                std::ffi::CStr::from_ptr(daliGetOutputName(&handle, 0))
                    .to_str()
                    .unwrap(),
                output0_name
            );
            assert_eq!(
                std::ffi::CStr::from_ptr(daliGetOutputName(&handle, 1))
                    .to_str()
                    .unwrap(),
                output1_name
            );

            daliDeletePipeline(&mut handle);
        }
    }

    #[test]
    fn test_file_reader_pipe() {
        unsafe {
            let mut handle = ptr::null_mut();

            // Create and serialize pipeline
            let pipeline = create_test_pipeline("output", "labels");
            let serialized = serialize_pipeline(&pipeline);

            // Create DALI pipeline
            let ret = daliCreatePipeline(
                &mut handle,
                serialized.as_ptr() as *const i8,
                serialized.len() as i32,
                BATCH_SIZE,
                NUM_THREADS,
                DEVICE_ID,
                false as i32,
                PREFETCH_QUEUE_DEPTH,
                PREFETCH_QUEUE_DEPTH,
                PREFETCH_QUEUE_DEPTH,
                false as i32,
            );
            assert_eq!(ret, 0);

            // Prefetch data
            daliPrefetchUniform(&handle, PREFETCH_QUEUE_DEPTH);

            // Run pipeline multiple times
            for _ in 0..PREFETCH_QUEUE_DEPTH {
                daliRun(&handle);
                compare_pipeline_outputs(&handle, &pipeline);
            }

            daliDeletePipeline(&mut handle);
        }
    }

    // Helper functions
    fn create_test_pipeline(output0: &str, output1: &str) -> Pipeline {
        // This would create a Pipeline object with FileReader operator
        // Similar to the C++ implementation
        unimplemented!()
    }

    fn serialize_pipeline(pipeline: &Pipeline) -> Vec<u8> {
        // This would serialize the pipeline to protobuf
        // Similar to the C++ implementation
        unimplemented!()
    }

    fn compare_pipeline_outputs(handle: &DaliPipelineHandle, pipeline: &Pipeline) {
        // This would compare outputs between C API and Pipeline object
        // Similar to the C++ implementation
        unimplemented!()
    }
}
