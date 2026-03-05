/*
  Nano IMU 1D (x-axis) logger + Kalman filter using BasicLinearAlgebra

  KF model from exercise (with bias):
    x = [a v p b]^T
    a_{k+1} = phi*a_k + w_a
    v_{k+1} = v_k + Ts*a_k
    p_{k+1} = p_k + Ts*v_k
    b_{k+1} = b_k + w_b
    y_k     = a_k + b_k + v_k (measurement noise)

  Output CSV:
    t_s,phase,y,a_hat,v_hat,p_hat,b_hat,y_hat

  Commands in Serial Monitor:
    0 = idle, 1 = stationary, 2 = motion
    r = reset time
    p = pause/resume

  Requirements:
    - Install library: "BasicLinearAlgebra" (Arduino Library Manager)
    - Correct IMU include below (default: Nano 33 BLE Sense Rev2 BMI270)
*/

#include <Arduino.h>
#include <BasicLinearAlgebra.h>
using namespace BLA;

// ======= Choose ONE IMU library include (uncomment the one that compiles) =======
// Nano 33 BLE / BLE Sense (rev1)
// #include <Arduino_LSM9DS1.h>
// Nano 33 IoT (LSM6DS3)
#include <Arduino_LSM6DS3.h>
// Nano 33 BLE Sense Rev2 (BMI270)
//#include <Arduino_BMI270_BMM150.h>
// ==============================================================================

// ===== Units: set to 1 for m/s^2 output, 0 for g output =====
#define OUTPUT_IN_MPS2 1
static constexpr float G_TO_MPS2 = 9.80665f;

// ===== Sampling =====
static constexpr uint32_t FS_HZ = 13;
static constexpr uint32_t TS_US = 1000000UL / FS_HZ;
static constexpr float    Ts    = 1.0f / FS_HZ;

// ===== Calibration =====
// You can set this to 60 for the full minute the handout suggests; 6–10s is often enough in class.
static constexpr uint32_t CAL_SECONDS = 8;

// ===== Kalman tuning knobs (start values; tune in lab) =====
static constexpr float f_b_hz   = 0.5f;   // accel AR(1) bandwidth [Hz], try 0.2..2.0
static constexpr float sigma_a  = 0.6f;   // expected motion accel std [unit], try 0.2..2.0
static constexpr float sigma_qb = 1e-3f;  // bias random-walk std per sample [unit], try 1e-5..1e-2

// ===== Phase tagging =====
volatile int phase = 0;   // 0 idle, 1 stationary, 2 motion
bool paused = false;

// ===== Timing =====
uint32_t t0_us = 0;
uint32_t next_us = 0;

// ===== Running stats (Welford) =====
struct RunningStats {
  uint32_t n = 0;
  float mean = 0.0f;
  float M2 = 0.0f;

  void push(float x) {
    n++;
    float delta = x - mean;
    mean += delta / (float)n;
    float delta2 = x - mean;
    M2 += delta * delta2;
  }
  float variance() const { return (n > 1) ? (M2 / (float)(n - 1)) : 0.0f; }
};

// ===== IMU read helper =====
bool readAccel(float &ax, float &ay, float &az) {
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax, ay, az); // typically returns "g" in Arduino IMU libs
    return true;
  }
  return false;
}

// ===== Kalman filter using BasicLinearAlgebra =====
struct KF4_BLA {
  // state x = [a v p b]^T
  Matrix<4,1> x;

  // matrices
  Matrix<4,4> Phi, P, Q, I4;
  Matrix<1,4> H;     // 1x4
  float R = 1e-3f;   // measurement variance (scalar)

  void init(float Ts_, float b0, float R_meas) {
    const float omega_b = 2.0f * PI * f_b_hz;
    const float phi = expf(-omega_b * Ts_);

    // sigma_a^2 = sigma_qa^2 / (1-phi^2)  => sigma_qa = sigma_a*sqrt(1-phi^2)
    const float sigma_qa = sigma_a * sqrtf(fmaxf(1.0f - phi*phi, 1e-12f));
    R = fmaxf(R_meas, 1e-10f);

    // Init state
    x = {0.0f, 0.0f, 0.0f, b0};

    // Phi from handout
    Phi = { phi, 0,    0,    0,
            Ts_, 1,    0,    0,
            0,   Ts_,  1,    0,
            0,   0,    0,    1 };

    // Q = diag([sigma_qa^2, 0, 0, sigma_qb^2])
    Q = { sigma_qa*sigma_qa, 0, 0, 0,
          0,                0, 0, 0,
          0,                0, 0, 0,
          0,                0, 0, sigma_qb*sigma_qb };

    // H = [1 0 0 1]
    H = {1, 0, 0, 1};

    // Identity
    I4 = {1,0,0,0,
          0,1,0,0,
          0,0,1,0,
          0,0,0,1};

    // Initial covariance: let bias be uncertain enough to adapt
    const float Pb0 = fmaxf(25.0f * R, 1e-6f);

    P = { sigma_a*sigma_a, 0,     0,     0,
          0,              1e-2f,  0,     0,
          0,              0,     1e-2f,  0,
          0,              0,     0,     Pb0 };
  }

  void step(float y) {
    // Predict
    x = Phi * x;
    P = Phi * P * ~Phi + Q;

    // Update (scalar measurement)
    const float y_hat = (H * x)(0,0);
    const float S     = (H * P * ~H)(0,0) + R;     // scalar
    const Matrix<4,1> K = (P * ~H) * (1.0f / S);   // 4x1

    x = x + K * (y - y_hat);
    P = (I4 - K * H) * P;
  }

  float yhat() const { return (H * x)(0,0); }
};

KF4_BLA kf;

// ===== Serial helpers =====
void printHelp() {
  Serial.println(F("# Commands: 0=idle, 1=stationary, 2=motion, r=reset time, p=pause/resume"));
#if OUTPUT_IN_MPS2
  Serial.println(F("# Units: m/s^2"));
#else
  Serial.println(F("# Units: g"));
#endif
  Serial.println(F("t_s,phase,y,a_hat,v_hat,p_hat,b_hat,y_hat"));
}

void handleSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '0') phase = 0;
    else if (c == '1') phase = 1;
    else if (c == '2') phase = 2;
    else if (c == 'r' || c == 'R') {
      t0_us = micros();
      next_us = t0_us;
      Serial.println(F("# time reset"));
    }
    else if (c == 'p' || c == 'P') {
      paused = !paused;
      Serial.println(paused ? F("# paused") : F("# resumed"));
    }
  }
}

// ===== 13 Hz scheduler tick =====
inline bool tick13Hz() {
  uint32_t now = micros();
  if ((int32_t)(now - next_us) < 0) return false;
  next_us += TS_US;
  return true;
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  if (!IMU.begin()) {
    Serial.println(F("ERROR: IMU.begin() failed. Check board/library include."));
    while (1) { delay(1000); }
  }

  printHelp();

  // Initialize timing base
  t0_us = micros();
  next_us = t0_us;

  // ---- Calibration: keep still for CAL_SECONDS at 13 Hz ----
  Serial.println(F("# CAL: keep IMU still..."));
  RunningStats stats;
  const uint32_t cal_samples = CAL_SECONDS * FS_HZ;

  float ax_g = 0, ay_g = 0, az_g = 0;

  for (uint32_t i = 0; i < cal_samples; ) {
    if (!tick13Hz()) continue;

    if (!readAccel(ax_g, ay_g, az_g)) continue;

#if OUTPUT_IN_MPS2
    const float y = ax_g * G_TO_MPS2;
#else
    const float y = ax_g;
#endif

    stats.push(y);
    i++;
  }

  const float b0 = stats.mean;
  const float R0 = stats.variance();
  kf.init(Ts, b0, R0);

  Serial.print(F("# CAL done. b0=")); Serial.print(b0, 6);
  Serial.print(F("  R=")); Serial.println(R0, 10);

  // Reset time after calibration so logs start at ~0
  t0_us = micros();
  next_us = t0_us;
}

void loop() {
  handleSerial();
  if (!tick13Hz()) return;
  if (paused) return;

  float ax_g, ay_g, az_g;
  if (!readAccel(ax_g, ay_g, az_g)) return;

#if OUTPUT_IN_MPS2
  const float y = ax_g * G_TO_MPS2;
#else
  const float y = ax_g;
#endif

  kf.step(y);

  const float t_s = (micros() - t0_us) * 1e-6f;

  // CSV output
  Serial.print(t_s, 6); Serial.print(',');
  Serial.print(phase);  Serial.print(',');
  Serial.print(y, 6);   Serial.print(',');
  Serial.print(kf.x(0,0), 6); Serial.print(','); // a_hat
  Serial.print(kf.x(1,0), 6); Serial.print(','); // v_hat
  Serial.print(kf.x(2,0), 6); Serial.print(','); // p_hat
  Serial.print(kf.x(3,0), 6); Serial.print(','); // b_hat
  Serial.println(kf.yhat(), 6);                  // y_hat = a_hat + b_hat
}