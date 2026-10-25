// Potentiometer: middle pin -> A0, outer pins -> 5V and GND.
void setup() { Serial.begin(9600); }
void loop() {
  Serial.println(analogRead(A0));
  delay(50);
}
