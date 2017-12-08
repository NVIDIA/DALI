// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_ARGUMENT_H_
#define NDLL_PIPELINE_ARGUMENT_H_

namespace ndll {

// NOTE: This class should really just be a protobuf message,
// but for now we don't wan't to add a protobuf dependency to
// ndll so we just baked our own.

/**
 * @brief Stores a single argument.
 *
 * TODO(tgale): Clearly document this API.
 */
class Argument {
public:
  Argument() :
    f_(0), i_(0), ui_(0), has_name_(false),
    has_f_(false), has_i_(false), has_ui_(false),
    has_s_(false) {}

  // Setters & getters for name
  inline bool has_name() const { return has_name_; }
  inline const string& get_name() const { return name_; }
  inline void set_name(string name) {
    has_name_ = true;
    name_ = name;
  }
  inline void clear_name() {
    has_name_ = false;
    name_ = "";
  }
  
  // Setters & getters for double
  inline bool has_f() const { return has_f_; }
  inline double get_f() const { return f_; }
  inline void set_f(double f) {
    has_f_ = true;
    f_ = f;
  }
  inline void clear_f() {
    has_f_ = false;
    f_ = 0;
  }

  // Setters & getters for int
  inline bool has_i() const { return has_i_; }
  inline int64 get_i() const { return i_; }
  inline void set_i(int64 i) {
    has_i_ = true;
    i_ = i;
  }
  inline void clear_i() {
    has_i_ = false;
    i_ = 0;
  }

  // Setters & getters for unsigned int
  inline bool has_ui() const { return has_ui_; }
  inline uint64 get_ui() const { return ui_; }
  inline void set_ui(uint64 ui) {
    has_ui_ = true;
    ui_ = ui;
  }
  inline void clear_ui() {
    has_ui_ = false;
    ui_ = 0;
  }
  
  // Setters & getters for string
  inline bool has_s() const { return has_s_; }
  inline const string& get_s() const { return s_; }
  inline void set_s(string s) {
    has_s_ = true;
    s_ = s;
  }
  inline void clear_s() {
    has_s_ = false;
    s_ = "";
  }

  // Setters & getters for repeated doubles
  inline int rf_size() const { return rf_.size(); }
  inline const vector<double>& rf() const {
    return rf_;
  }
  inline double rf(int index) const {
    NDLL_ENFORCE((index >= 0) && ((size_t)index < rf_.size()), "Index out of valid range.");
    return rf_[index];
  }
  inline void set_rf(int index, double f) {
    NDLL_ENFORCE((index >= 0) && ((size_t)index < rf_.size()), "Index out of valid range.");
    rf_[index] = f;
  }
  inline void add_rf(double f) {
    rf_.push_back(f);
  }
  inline void clear_rf() {
    rf_.clear();
  }

  // Setters & getters for repeated ints
  inline int ri_size() const { return ri_.size(); }
  inline const vector<int64>& ri() const {
    return ri_;
  }
  inline int64 ri(int index) const {
    NDLL_ENFORCE((index >= 0) && ((size_t)index < ri_.size()), "Index out of valid range.");
    return ri_[index];
  }
  inline void set_ri(int index, int64 i) {
    NDLL_ENFORCE((index >= 0) && ((size_t)index < ri_.size()), "Index out of valid range.");
    ri_[index] = i;
  }
  inline void add_ri(int64 i) {
    ri_.push_back(i);
  }
  inline void clear_ri() {
    ri_.clear();
  }

  // Setters & getters for repeated unsigned ints
  inline int rui_size() const { return rui_.size(); }
  inline const vector<uint64>& rui() const {
    return rui_;
  }
  inline uint64 rui(int index) const {
    NDLL_ENFORCE((index >= 0) && ((size_t)index < rui_.size()), "Index out of valid range.");
    return rui_[index];
  }
  inline void set_rui(int index, uint64 ui) {
    NDLL_ENFORCE((index >= 0) && ((size_t)index < rui_.size()), "Index out of valid range.");
    rui_[index] = ui;
  }
  inline void add_rui(uint64 ui) {
    rui_.push_back(ui);
  }
  inline void clear_rui() {
    rui_.clear();
  }
  

  // Setters & getters for repeated strings
  inline int rs_size() const { return rs_.size(); }
  inline const vector<string>& rs() const {
    return rs_;
  }
  inline const string& rs(int index) const {
    NDLL_ENFORCE((index >= 0) && ((size_t)index < rs_.size()), "Index out of valid range.");
    return rs_[index];
  }
  inline void set_rs(int index, const string& s) {
    NDLL_ENFORCE((index >= 0) && ((size_t)index < rs_.size()), "Index out of valid range.");
    rs_[index] = s;
  }
  inline void add_rs(const string& s) {
    rs_.push_back(s);
  }
  inline void clear_rs() {
    rs_.clear();
  }
  
private:
  string name_;
  
  double f_;
  int64 i_;
  uint64 ui_;
  string s_;
  
  vector<double> rf_;
  vector<int64> ri_;
  vector<uint64> rui_;
  vector<string> rs_;
  bool has_name_, has_f_, has_i_, has_ui_, has_s_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_ARGUMENT_H_
