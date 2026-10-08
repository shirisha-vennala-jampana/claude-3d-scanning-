// Minimal sensor logger for scan3d: carry it while you film.
// Prints CSV lines "seconds,key,value,unit" over USB serial; record them with
// examples/serial_to_csv.py, then:  scan3d sensor-log <scene> sensors.csv --offset <s>
//
// Wiring (Arduino Uno / Nano, 5 V, 10-bit ADC):
//   TMP36 temperature sensor: Vout -> A0
//   Light-dependent resistor + 10k divider -> A1
// On an ESP32 change VREF to 3.3 and ADC_MAX to 4095.

const float VREF = 5.0;
const float ADC_MAX = 1023.0;
const unsigned long PERIOD_MS = 500;

void setup() {
  Serial.begin(115200);
  Serial.println("timestamp,key,value,unit");
}

void loop() {
  static unsigned long next = 0;
  unsigned long now = millis();
  if (now < next) return;
  next = now + PERIOD_MS;
  float t = now / 1000.0;

  float volts = analogRead(A0) * VREF / ADC_MAX;
  float celsius = (volts - 0.5) * 100.0;  // TMP36: 10 mV per degree, 500 mV offset
  int light = analogRead(A1);

  Serial.print(t, 2); Serial.print(",temperature,"); Serial.print(celsius, 1); Serial.println(",C");
  Serial.print(t, 2); Serial.print(",light,"); Serial.print(light); Serial.println(",raw");
}
