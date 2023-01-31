


void EnforceConditionalInputKind(const TensorList<CPUBackend> &input, const std::string &name,
                                 const std::string side) {
  auto dim = input.shape().sample_dim();

  std::string preamble = make_string("Logical expression ``", name,
                                     "`` is restricted to scalar (0-d tensors) inputs of bool type.");
  std::string suggestion =
      "\n\nThis input restriction allows the logical expressions to always return scalar boolean "
      "outputs and to be used in unambiguous way in DALI conditionals. You may use bitwise "
      "arithmetic operators ``&``, ``|`` if you need to process inputs of higher dimensionality or "
      "different type - those operations performed on boolean inputs are equivalent to logical "
      "expressions.";

  DALI_ENFORCE(dim == 0,
               make_string(preamble, " Got a ", dim, "-d input on the ", side, ".", suggestion));
  auto type = input.type();
  DALI_ENFORCE(type == DALI_BOOL, make_string(preamble, " Got an input of type ", type, " on the ",
                                              side, ".", suggestion));
}