#include <Arduino_LSM6DS3.h>

/* This "trick" is called inheritance. We make a new class that inherits everything from the old class.
   We can then add additional functionality to the new class. In this case, we add four functions to
   add the ability to change the sampling rate for the gyro and accelerometer.
*/
class IMUExtended : public LSM6DS3Class {
public:
  IMUExtended(TwoWire& wire, uint8_t slaveAddress)
    : LSM6DS3Class{ wire, slaveAddress } {
  }
  void SetAccGyroRate13Hz() {
    writeRegister(0x10, 0b00011000);  // See the LSM6DS3 documentation for what to write to these registers.
    writeRegister(0x11, 0b00011100);
  }
  void SetAccGyroRate26Hz() {
    writeRegister(0x10, 0b00101000);
    writeRegister(0x11, 0b00101100);
  }
  void SetAccGyroRate52Hz() {
    writeRegister(0x10, 0b00111000);
    writeRegister(0x11, 0b00111100);
  }
  void SetAccGyroRate104Hz() {
    writeRegister(0x10, 0b01001000);
    writeRegister(0x11, 0b01001100);
  }
};

// Instantiate the new class instead of the old class.
IMUExtended myIMU{ Wire, LSM6DS3_ADDRESS };

// Calibration variables
const int sampleRate = 13;     // Hz
const int duration = 20;       // seconds
const int calibSamples = sampleRate * duration;
float axData[calibSamples];
int calibIndex = 0; 

uint32_t Time{ 0 };

// Initialization
float x = 0;
float P = pow(0.0156, 2);  // P = sigmax**2
float R = pow(0.1, 2);     // R = sigmav**2
float Q = pow(0.0156, 2);  // Q = sigma2**2
float H = 1;               // H = c
float Phi = 0.95;          // Phi = a 
float I = 1;
float K;


void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!myIMU.begin()) {
    Serial.println("Failed to initialize IMU!");

    while (1);
  }

  Serial.println("Calibration start: keep board steady and horizontal.");

  myIMU.SetAccGyroRate13Hz();

  Serial.println("Acceleration in g's");
  Serial.println("X\tY\tZ");
}

void calibrate() {
  float sum = 0;
  float sumSq = 0;
  float minVal = axData[0];
  float maxVal = axData[0];

  for (int i = 0; i < calibSamples; i++) {
    float v = axData[i];
    sum += v;
    sumSq += v * v;
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }
  
  float mean = sum / calibSamples;
  float variance = sumSq / calibSamples - mean * mean;
  float stddev = sqrt(variance);

  Serial.println("----- Calibration Results -----");
  Serial.print("Bias (mean): "); Serial.println(mean);
  Serial.print("Std Dev: "); Serial.println(stddev);
  Serial.print("Min: "); Serial.println(minVal);
  Serial.print("Max: "); Serial.println(maxVal);
  Serial.println("-------------------------------");
}

void loop() {

  // Calibration
  if (calibIndex >= calibSamples){
    calibrate();
    while (1); // break
  } 
  
  float ax{ 0.f };
  float ay{ 0.f };
  float az{ 0.f };

  //myIMU.gyroscopeSampleRate();    // Note that this function does not return the rate, but just always 104 Hz
  //myIMU.accelerationSampleRate(); // Note that this function does not return the rate, but just always 104 Hz

  if (myIMU.accelerationAvailable()) {
    /*
    Serial.print("Time: ");
    Serial.println(millis() - Time);
    Time = millis();
    */
    myIMU.readAcceleration(ax, ay, az);
    axData[calibIndex] = ax;
    calibIndex++;
    // y(0,0) = x;
    float y_meas = ax;

    // Kalman Filter Measurement Update
    K = P * (H) / (H * P * H + R);
    x = x + K * (y_meas - H * x);
    P = (I - K * H) * P * (I - K * H) + K * R * K;

    // Store state estimate, then perform Kalman filter prediction
    //for (int i=0; i < n + 1; i++) {X[i][k] = x[i];}
    x = Phi * x;
    P = Phi * P * Phi + Q;

    /*
    Serial.print(ax);
    Serial.print(",");
    Serial.print(ay);
    Serial.print(",");
    Serial.println(az);

    
    Serial.print("K=");
    Serial.print(K);
    Serial.print('\t');
    Serial.print("x=");
    Serial.print(x);
    Serial.print('\t');
    Serial.print("P=");
    Serial.print(P);
    Serial.println('\t');
    */
  }
}

