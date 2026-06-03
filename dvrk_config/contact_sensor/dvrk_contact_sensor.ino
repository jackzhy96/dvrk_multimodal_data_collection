#include <CapacitiveSensor.h>

#define SEND_PIN 2
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
  psm1Sensor.set_CS_AutocaL_Millis(0xFFFFFFFF);     // turn off autocalibrate on sensor 1
  psm2Sensor.set_CS_AutocaL_Millis(0xFFFFFFFF);	    // turn off autocalibrate on sensor 2
  psm3Sensor.set_CS_AutocaL_Millis(0xFFFFFFFF);     // turn off autocalibrate on sensor 3
  pinMode(OUTPUT_PIN_1, OUTPUT);
  pinMode(OUTPUT_PIN_2, OUTPUT);
  pinMode(OUTPUT_PIN_3, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  
  long value1 = psm1Sensor.capacitiveSensor(10);
  long value2 = psm2Sensor.capacitiveSensor(10);
  long value3 = psm3Sensor.capacitiveSensor(10);

  // uncomment the following lines to see the output of you sensor and set the threshold
  // long start = millis();
  // Serial.print(millis() - start);        // check on performance in milliseconds
  
  // Serial.print("\t");                    
  // Serial.print(value1);                  // print sensor1 output 
  // Serial.print("\t");                    
  // Serial.print(value2);                  // print sensor2 output 
  // Serial.print("\t");                    
  // Serial.println(value3);                // print sensor3 output 

  int threshold = 105;
  
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
