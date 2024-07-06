/*
  Blink

  Turns an LED on for one second, then off for one second, repeatedly.

  Most Arduinos have an on-board LED you can control. On the UNO, MEGA and ZERO
  it is attached to digital pin 13, on MKR1000 on pin 6. LED_BUILTIN is set to
  the correct LED pin independent of which board is used.
  If you want to know what pin the on-board LED is connected to on your Arduino
  model, check the Technical Specs of your board at:
  https://www.arduino.cc/en/Main/Products

  modified 8 May 2014
  by Scott Fitzgerald
  modified 2 Sep 2016
  by Arturo Guadalupi
  modified 8 Sep 2016
  by Colby Newman

  This example code is in the public domain.

  https://www.arduino.cc/en/Tutorial/BuiltInExamples/Blink
*/

// the setup function runs once when you press reset or power the board
void setup() {

  Serial.begin(9600);
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(5, OUTPUT);
  pinMode(3, OUTPUT);

  digitalWrite(5, LOW);
  float messwert = 0;
  Serial.println("Al, L=25cm, d=2mm, m=g, 0.25mm Schritte, Delta=4mm");
  int e =16;
for(int j = 0;j<e;j++){
  for (int i=0; i<50; i++) {
    digitalWrite(3, HIGH);
    delay(1);
    digitalWrite(3, LOW);
    delay(1);
  }
  
  delay(200);
  messwert = analogRead(A0);
  messwert = messwert*5.0/1024.0;
  messwert = (messwert - 0.5)*2.5;
  Serial.println(messwert);
  
}
digitalWrite(5, HIGH);
for(int j=0;j<e+2;j++){
  for (int i=0; i<100; i++) {
    digitalWrite(3, HIGH);
    delay(1);
    digitalWrite(3, LOW);
    delay(1);
  }
 
  
}

}

// the loop function runs over and over again forever
void loop() {
 //Serial.println(",jgdf");
}
