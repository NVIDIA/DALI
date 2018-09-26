#include "bb_flip.h"

namespace dali {

DALI_REGISTER_OPERATOR(BbFlip, BbFlip, CPU);

DALI_SCHEMA(BbFlip)
                .DocStr("PTEEREFEREE")
                .NumInput(1)
                .NumOutput(1)
                .AddOptionalArg("myarg",
                        R"code(Size of the cropped image. If only a single value `c` is provided,
the resulting crop will be square with size `(c,c)`)code",
                        DALI_RGB);

} // namespace dali
