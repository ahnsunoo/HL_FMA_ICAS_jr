// Front Motor Drive
#define MOTOR1_PWM 2
#define MOTOR1_ENA 3
#define MOTOR1_ENB 4

// Rear Motor Drive
#define MOTOR2_PWM 5
#define MOTOR2_ENA 6
#define MOTOR2_ENB 7

//Serial data
int cnt=-1;
unsigned char buf[30];

union
{
  float data ;
  char  bytedata[4];
    
} m_car_speed_float;

union
{
    short data ;
    char  bytedata[2];
    
} m_car_angle_int16;

///////////////////////////////////////  Steering PID 제어 /////////////////////////////////////////////
#define Steering_Sensor A15  // Analog input pin that the potentiometer is attached to
#define NEURAL_ANGLE 3
#define LEFT_STEER_ANGLE  -30
#define RIGHT_STEER_ANGLE  30
#define MOTOR3_PWM 8
#define MOTOR3_ENA 9
#define MOTOR3_ENB 10

float Kp = 15.5;
float Ki = 3.5;
float Kd = 6.5; //PID 상수 설정, 실험에 따라 정해야 함 중요!
double Setpoint, Input, Output; //PID 제어 변수
double error, error_old;
double error_s, error_d;
int pwm_output;

int sensorValue = 0;        // value read from the pot
int Steer_Angle_Measure = 0;        // value output to the PWM (analog out)
int Steering_Angle = NEURAL_ANGLE;

void steer_motor_control(int motor_pwm)
{
  if (motor_pwm > 0) // forward
  {
    digitalWrite(MOTOR3_ENA, LOW);
    digitalWrite(MOTOR3_ENB, HIGH);
    analogWrite(MOTOR3_PWM, motor_pwm);
  }
  else if (motor_pwm < 0) // backward
  {
    digitalWrite(MOTOR3_ENA, HIGH);
    digitalWrite(MOTOR3_ENB, LOW);
    analogWrite(MOTOR3_PWM, -motor_pwm);
  }
  else // stop
  {
    digitalWrite(MOTOR3_ENA, LOW);
    digitalWrite(MOTOR3_ENB, LOW);
    analogWrite(MOTOR3_PWM, 0);
  }
}

void PID_Control()
{
  error = Steering_Angle - Steer_Angle_Measure ;  // steering error 계산
  error_s += error;
  error_d = error - error_old;
  error_s = (error_s >=  100) ?  100 : error_s;
  error_s = (error_s <= -100) ? -100 : error_s;

  pwm_output = Kp * error + Kd * error_d + Ki * error_s;
  pwm_output = (pwm_output >=  255) ?  255 : pwm_output;
  pwm_output = (pwm_output <= -255) ? -255 : pwm_output;

  if (error == 0)
  {
    steer_motor_control(0);
    error_s = 0;
  }
  else          steer_motor_control(pwm_output);
  error_old = error;  
}

void steering_control()
{
  if (Steering_Angle <= LEFT_STEER_ANGLE + NEURAL_ANGLE)   Steering_Angle  = LEFT_STEER_ANGLE + NEURAL_ANGLE;
  if (Steering_Angle >= RIGHT_STEER_ANGLE + NEURAL_ANGLE)  Steering_Angle = RIGHT_STEER_ANGLE + NEURAL_ANGLE;
  PID_Control(); 
}

///////////////////////////////// Encoder  /////////////////////////////////////

#include <SPI.h>

#define ENC1_ADD 22
#define ENC2_ADD 23
signed long encoder1count = 0;
signed long encoder2count = 0;

void initEncoders() {    
  // Set slave selects as outputs
  pinMode(ENC1_ADD, OUTPUT);
  pinMode(ENC2_ADD, OUTPUT); 
  // Raise select pins
  // Communication begins when you drop the individual select signsl
  digitalWrite(ENC1_ADD,HIGH);
  digitalWrite(ENC2_ADD,HIGH);

  SPI.begin();
  
  // Initialize encoder 1
  // Clock division factor: 0
  // Negative index input
  // free-running count mode
  // x4 quatrature count mode (four counts per quadrature cycle)
  // NOTE: For more information on commands, see datasheet
  digitalWrite(ENC1_ADD,LOW);        // Begin SPI conversation
  SPI.transfer(0x88);                       // Write to MDR0
  SPI.transfer(0x03);                       // Configure to 4 byte mode
  digitalWrite(ENC1_ADD,HIGH);       // Terminate SPI conversation 

  // Initialize encoder 2
  // Clock division factor: 0
  // Negative index input
  // free-running count mode
  // x4 quatrature count mode (four counts per quadrature cycle)
  // NOTE: For more information on commands, see datasheet
  digitalWrite(ENC2_ADD,LOW);        // Begin SPI conversation
  SPI.transfer(0x88);                       // Write to MDR0
  SPI.transfer(0x03);                       // Configure to 4 byte mode
  digitalWrite(ENC2_ADD,HIGH);       // Terminate SPI conversation 
}

long readEncoder(int encoder_no) 
{
  
  // Initialize temporary variables for SPI read
  unsigned int count_1, count_2, count_3, count_4;
  long count_value;  
  
  
    digitalWrite(ENC1_ADD + encoder_no-1,LOW);      // Begin SPI conversation
   // digitalWrite(ENC4_ADD,LOW);      // Begin SPI conversation
    SPI.transfer(0x60);                     // Request count
    count_1 = SPI.transfer(0x00);           // Read highest order byte
    count_2 = SPI.transfer(0x00);           
    count_3 = SPI.transfer(0x00);           
    count_4 = SPI.transfer(0x00);           // Read lowest order byte
    digitalWrite(ENC1_ADD+encoder_no-1,HIGH);     // Terminate SPI conversation 
    //digitalWrite(ENC4_ADD,HIGH);      // Begin SPI conversation
// Calculate encoder count
  count_value= ((long)count_1<<24) + ((long)count_2<<16) + ((long)count_3<<8 ) + (long)count_4;
  
  return count_value;
}

void clearEncoderCount(int encoder_no) {
    
  // Set encoder1's data register to 0
  digitalWrite(ENC1_ADD+encoder_no-1,LOW);      // Begin SPI conversation  
  // Write to DTR
  SPI.transfer(0x98);    
  // Load data
  SPI.transfer(0x00);  // Highest order byte
  SPI.transfer(0x00);           
  SPI.transfer(0x00);           
  SPI.transfer(0x00);  // lowest order byte
  digitalWrite(ENC1_ADD+encoder_no-1,HIGH);     // Terminate SPI conversation 
  
  delayMicroseconds(100);  // provides some breathing room between SPI conversations
  
  // Set encoder1's current data register to center
  digitalWrite(ENC1_ADD+encoder_no-1,LOW);      // Begin SPI conversation  
  SPI.transfer(0xE0);    
  digitalWrite(ENC1_ADD+encoder_no-1,HIGH);     // Terminate SPI conversation 
}

///////////////////////////////// Motor Drive  /////////////////////////////////////

void front_motor_control(int motor1_pwm)
{
   if (motor1_pwm > 0) // forward
  {
    digitalWrite(MOTOR1_ENA, HIGH);
    digitalWrite(MOTOR1_ENB, LOW);
    analogWrite(MOTOR1_PWM, motor1_pwm);

  }
  else if (motor1_pwm < 0) // backward
  {

    digitalWrite(MOTOR1_ENA, LOW);
    digitalWrite(MOTOR1_ENB, HIGH);
    analogWrite(MOTOR1_PWM, -motor1_pwm);
  }
  else
  {
    digitalWrite(MOTOR1_ENA, LOW);
    digitalWrite(MOTOR1_ENB, LOW);
    digitalWrite(MOTOR1_PWM, 0);
  }
}

void rear_motor_control(int motor2_pwm)
{
   if (motor2_pwm > 0) // forward
  {
    digitalWrite(MOTOR2_ENA, HIGH);
    digitalWrite(MOTOR2_ENB, LOW);
    analogWrite(MOTOR2_PWM, motor2_pwm);

  }
  else if (motor2_pwm < 0) // backward
  {

    digitalWrite(MOTOR2_ENA, LOW);
    digitalWrite(MOTOR2_ENB, HIGH);
    analogWrite(MOTOR2_PWM, -motor2_pwm);
  }
  else
  {
    digitalWrite(MOTOR2_ENA, LOW);
    digitalWrite(MOTOR2_ENB, LOW);
    digitalWrite(MOTOR2_PWM, 0);
  }
}

void motor_control(int f_speed,int r_speed){
  front_motor_control(f_speed);
  rear_motor_control(r_speed);
  }
///////////////////////////////// MsTimer callback  /////////////////////////////////////

int f_speed=0;
int r_speed=0;
int velocity=0;
int steer_angle=0;

#include <MsTimer2.h>
void control_callback()
{
  static boolean output = HIGH;
  
  digitalWrite(13, output);
  output = !output;

  f_speed = r_speed = velocity;

  motor_control(f_speed, r_speed);

  // read the analog in value:
  sensorValue = analogRead(Steering_Sensor);
  // map it to the range of the analog out:
  Steer_Angle_Measure = map(sensorValue, 50, 1050, LEFT_STEER_ANGLE, RIGHT_STEER_ANGLE);
  Steering_Angle = NEURAL_ANGLE + steer_angle;
  steering_control();  
}

///////////////////////////////// Serial Event  /////////////////////////////////////

void Serial_read_data() {
  static int index = 0;
  static unsigned long last_time = 0;

  while (Serial.available()) {
    unsigned char c = Serial.read();
    if (index == 0 && c != '#') continue; // 시작 문자가 '#' 아니면 무시
    buf[index++] = c;
    if (index == 9) {
      if (buf[0] == '#' && buf[1] == 'C' && buf[8] == '*') {
        m_car_angle_int16.bytedata[0] = buf[2];
        m_car_angle_int16.bytedata[1] = buf[3];
        m_car_speed_float.bytedata[0] = buf[4];      
        m_car_speed_float.bytedata[1] = buf[5];      
        m_car_speed_float.bytedata[2] = buf[6];      
        m_car_speed_float.bytedata[3] = buf[7];

        Serial.print(m_car_angle_int16.data);  
        Serial.print(" ");  
        Serial.print(m_car_speed_float.data);  
        Serial.print(" ");  
        Serial.print(encoder1count);  
        Serial.print(" ");  
        Serial.println(encoder2count);
      }
      index = 0;
    }
  }
}


///////////////////////////////// Setup  /////////////////////////////////////

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.print("Serial Port Connected!");
  
  pinMode(13, OUTPUT);
  
  // Front Motor Drive Pin Setup
  pinMode(MOTOR1_PWM, OUTPUT);
  pinMode(MOTOR1_ENA, OUTPUT);  // L298 motor control direction
  pinMode(MOTOR1_ENB, OUTPUT);

  // Rear Motor Drive Pin Setup
  pinMode(MOTOR2_PWM, OUTPUT);
  pinMode(MOTOR2_ENA, OUTPUT);  // L298 motor control direction
  pinMode(MOTOR2_ENB, OUTPUT);

  // Steer
  pinMode(MOTOR3_PWM, OUTPUT);
  pinMode(MOTOR3_ENA, OUTPUT);  // L298 motor control direction
  pinMode(MOTOR3_ENB, OUTPUT);  // L298 motor control PWM

  error = error_s = error_d = error_old = 0.0;
  pwm_output = 0;

  // Encoder
  initEncoders();          // initialize encoder
  clearEncoderCount(1); 
  clearEncoderCount(2);

  MsTimer2::set(100, control_callback); // 500ms period
  MsTimer2::start();
}

///////////////////////////////// Loop  /////////////////////////////////////

void loop() {
  // put your main code here, to run repeatedly:
  encoder1count = readEncoder(1);
  encoder2count = readEncoder(2);
  Serial_read_data();
  velocity = m_car_speed_float.data;
  steer_angle = m_car_angle_int16.data;
}
