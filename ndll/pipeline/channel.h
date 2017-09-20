#ifndef NDLL_PIPELINE_CHANNEL_H_
#define NDLL_PIPELINE_CHANNEL_H_

namespace ndll {

/**
 * Channels provide a mechanism for sharing data between operators external to the
 * main batch being processed by the pipeline. They can be used for...
 * - passing secondary outputs to ops later in the pipeline (e.g. bounding boxes)
 * - managing flags indicating when data is ready for use (useful for debugging)
 * 
 * No data should be allocated without the use of a 'Backend'. This is to avoid
 * issues that frameworks have built special memory allocators to handle. Copies
 * to device should not be handled by ops or channels, they should be managed
 * through the pipelines buffer packing mechanism. The resultant pointer can be
 * stored in a channel if a later op needs access to the same device-side data
 *
 * TODO(docs): Throroughly document the guarantees our executor makes about
 * when different functions will be executed
 */
struct Channel {
  // Channel is currently just a simple aggregate type. In the future
  // we could extend this with more functionality if need be.
};

} // namespace ndll

#endif // NDLL_PIPELINE_CHANNEL_H_
