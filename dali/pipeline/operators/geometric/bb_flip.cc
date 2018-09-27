#include "bb_flip.h"

namespace dali {

DALI_REGISTER_OPERATOR(BbFlip, BbFlip, CPU);


DALI_SCHEMA(BbFlip)
                .DocStr(R"code(Operator for horizontal flip (mirror) of bounding box.
                        Input: Bounding box coordinates; in either (x1,y1,w,h) or (x1,y1,x2,y2) format)code")
                .NumInput(1)
                .NumOutput(1)
                .AddArg("coordinates_type",
                        R"code(True for width and height)code",
                        DALI_BOOL);

} // namespace dali
