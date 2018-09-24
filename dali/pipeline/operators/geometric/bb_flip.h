#ifndef DALI_BB_FLIP_H
#define DALI_BB_FLIP_H

#include <dali/pipeline/operators/operator.h>
#include <dali/pipeline/operators/common.h>

namespace dali {

//template<typename Backend>
//class BbFlip : public Operator<Backend> {
// public:
//    BbFlip(const OpSpec &spec) :
//            Operator<Backend>(spec) {
//
//    }
//    void RunImpl(Workspace<Backend> *ws, int idx = 0) {}
//    USE_OPERATOR_MEMBERS();
//};


//template <typename Backend>
class BbFlip : public Operator<CPUBackend> {
 public:
    explicit inline BbFlip(const OpSpec &spec) :
            Operator<CPUBackend>(spec),
            output_type_(spec.GetArgument<DALIImageType>("myarg")) {
    }


    virtual ~BbFlip() override = default;
    DISABLE_COPY_MOVE_ASSIGN(BbFlip);

 protected:

    inline void RunImpl(SampleWorkspace *ws, const int idx) override {
    }


 private:
    DALIImageType output_type_;
//    USE_OPERATOR_MEMBERS();
};

} // namespace dali

#endif //DALI_BB_FLIP_H
