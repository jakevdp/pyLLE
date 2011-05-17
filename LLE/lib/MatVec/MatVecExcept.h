#ifndef MATVEC_EXCEPT_H
#define MATVEC_EXCEPT_H

#include <string>

class IndexError : public std::exception {
 public:
  IndexError() throw() : msg_("Index Error"){}
  IndexError(const std::string& msg) throw() : msg_("Index Error: "){
    msg_ += msg;
  }
  ~IndexError() throw(){}
  const char * what() const throw() { return msg_.c_str(); }
 private:
  std::string msg_;
};

class MatVecError : public std::exception {
 public:
  MatVecError() throw() : msg_("MatVec Error"){}
  MatVecError(const std::string& msg) throw() : msg_("MatVec Error: "){
    msg_ += msg;
  }
  ~MatVecError() throw(){}
  const char * what() const throw() { return msg_.c_str(); }
 private:
  std::string msg_;
};

class NotImplementedError : public std::exception {
 public:
  NotImplementedError() throw() : msg_("NotImplemented Error"){}
  NotImplementedError(const std::string& msg) throw() 
    : msg_("NotImplemented Error: "+msg){}
  ~NotImplementedError() throw(){}
  const char * what() const throw() { return msg_.c_str(); }
 private:
  std::string msg_;
};



#endif // MATVEC_EXCEPT_H
