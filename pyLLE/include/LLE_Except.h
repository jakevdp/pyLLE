#ifndef LLE_EXCEPT_H
#define LLE_EXCEPT_H

#include <string>
class LLE_Error : public std::exception {
 public:
  LLE_Error() throw() : msg_("LLE Error"){}
  LLE_Error(const std::string& msg) throw() : msg_("LLE Error: "){
    msg_ += msg;
  }
  ~LLE_Error() throw(){}
  const char * what() const throw() { return msg_.c_str(); }
 private:
  std::string msg_;
};

#endif //LLE_EXCEPT_H
