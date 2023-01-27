#ifndef ard_test_h
#define ard_test_h
#include "Arduino.h"
#include <SerialCommand.h>
#include <SoftwareSerial.h>

class Controller {
public:
  void init();
private:
  // I feel like here certain pin parameters should be defined or a collable void like the setup void should be called
  // to initiate the necessary arduino logic to actually control everything.
};
#endif