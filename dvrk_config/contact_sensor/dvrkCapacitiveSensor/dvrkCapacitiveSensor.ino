#include <CapacitiveSensor.h>

#define SEND_PIN_1 2
#define SEND_PIN_2 7
#define SEND_PIN_3 10

#define RECEIVE_PIN_1 4
#define RECEIVE_PIN_2 8
#define RECEIVE_PIN_3 12

#define OUTPUT_PIN_1 5
#define OUTPUT_PIN_2 9
#define OUTPUT_PIN_3 13   // using pin 13 allows you to visualize you output with the arduino built-in LED


CapacitiveSensor psm1Sensor = CapacitiveSensor(SEND_PIN, RECEIVE_PIN_1);
CapacitiveSensor psm2Sensor = CapacitiveSensor(SEND_PIN, RECEIVE_PIN_2);
CapacitiveSensor psm3Sensor = CapacitiveSensor(SEND_PIN, RECEIVE_PIN_3);

void setup() {
  psm1Sensor.set_CS_AutocaL_Millis(0xFFFFFFFF);     // turn off autocalibrate on PSM1
  psm2Sensor.set_CS_AutocaL_Millis(0xFFFFFFFF);
  psm3Sensor.set_CS_AutocaL_Millis(0xFFFFFFFF);
  pinMode(OUTPUT_PIN_1, OUTPUT);
  pinMode(OUTPUT_PIN_2, OUTPUT);
  pinMode(OUTPUT_PIN_3, OUTPUT);
  Serial.begin(9600);
}

void loop() {

  // set all sending pin in Hi-Z  
  pinMode(SEND_PIN_1, INPUT);
  pinMode(SEND_PIN_2, INPUT);
  pinMode(SEND_PIN_3, INPUT);

  // read one sensor at a time while the others are in Hi-Z
  long value1 = readPSM1();
  long value2 = readPSM2();
  long value3 = readPSM3();

  // uncomment the following lines to see the output of you sensor and set the threshold
  // long start = millis();
  // Serial.print(millis() - start);        // check on performance in milliseconds
  
  // Serial.print("\t");                    
  // Serial.print(value1);                  // print sensor1 output 
  // Serial.print("\t");                    
  // Serial.print(value2);                  // print sensor2 output 
  // Serial.print("\t");                    
  // Serial.println(value3);                // print sensor3 output 

  int threshold = 205;
  
  if(value1>threshold) digitalWrite(OUTPUT_PIN_1, HIGH);
  else digitalWrite(OUTPUT_PIN_1, LOW);

  if(value2>threshold) digitalWrite(OUTPUT_PIN_2, HIGH);
  else digitalWrite(OUTPUT_PIN_2, LOW);

  if(value3>threshold) digitalWrite(OUTPUT_PIN_3, HIGH);
  else digitalWrite(OUTPUT_PIN_3, LOW);

  //uncomment the following lines to see how your binarization performs
  // Serial.print(digitalRead(OUTPUT_PIN_1));                  
  // Serial.print("\t");                    
  // Serial.print(digitalRead(OUTPUT_PIN_2));                  
  // Serial.print("\t");                    
  // Serial.println(digitalRead(OUTPUT_PIN_3));

  delay(10);
}


long readPSM1() {
  pinMode(SEND_PIN_1, OUTPUT);
  delay(1);
  long value1 = psm1Sensor.capacitiveSensor(10);
  pinMode(SEND_PIN_1, INPUT);
  return value1;
}

long readPSM2() {
  pinMode(SEND_PIN_2, OUTPUT);
  delay(1);
  long value2 = psm2Sensor.capacitiveSensor(10);
  pinMode(SEND_PIN_2, INPUT);
  return value2;
}

long readPSM3() {
  pinMode(SEND_PIN_3, OUTPUT);
  delay(1);
  long value3 = psm3Sensor.capacitiveSensor(10);
  pinMode(SEND_PIN_3, INPUT);
  return value3;
}

