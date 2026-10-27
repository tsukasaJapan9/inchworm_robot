// シャクトリムシロボットに PPO の方策を載せる。
//
//   100Hz ループ: 観測を作る -> MLP 推論 -> 目標角を増分で積分 -> XL330 へ送信
//
// ボタン A: スタート / ボタン B: 停止 / ボタン C: ゼロ点キャリブレーション
//
// テレメトリ: USB シリアルに 1 行 1 サンプルで流す。待機中も流れるので、
// 機体を手で動かしながらシリアルモニタで姿勢を確認できる。
//   T,<millis>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<q1>,<q2>,<dq1>,<dq2>,<tgt1>,<tgt2>,<run>
//   加速度 [g] / 角速度 [rad/s] / 角度 [deg] / 角速度 [rad/s] / 目標角 [deg] /
//   走行中=1
//
// キャリブレーション: トルクを切った状態で節を手で水平にし、ボタン C を押す。
// そのときの Present Position をゼロ度として NVS
// に保存するので、電源を切っても残る。
//
// ★ が付いている定数は実機で必ず確認すること。シミュと符号や単位が違うと
//    方策はまったく別の動きをする。

#include <Dynamixel2Arduino.h>
#include <M5Unified.h>
#include <Preferences.h>

#include "policy_weights.h"

// ───────────────────────── 設定 ─────────────────────────

// identify_axes.py で確認済み: 前の節を振ると ID1 が 70.31 度動き、ID2 は 0.35
// 度
constexpr uint8_t ID_FRONT = 1;
constexpr uint8_t ID_REAR = 2;
constexpr float DXL_PROTOCOL = 2.0;
constexpr int32_t DXL_BAUD = 1000000; // ★ サーボ側の設定と合わせる

// works/goron（同じ配線で動いている先行機）から持ってきた実績値。Port A を使う
constexpr int PIN_DXL_RX = 33;
constexpr int PIN_DXL_TX = 32;
constexpr uint32_t DXL_SETTLE_MS =
    200; // ポートを開いた直後は最初のパケットが落ちる
constexpr uint8_t DXL_PING_RETRY =
    3; // 取りこぼしでサーボを見失わないよう数回試す

// 水平（方策の 0 度）に対応する Dynamixel の角度は、ボタン C
// のキャリブレーションで 決めて NVS に保存する。定数で持たない。

// 「持ち上げる向き」を正にするための符号。Drive Mode を反転させる代わりに
// ここで符号を掛けて合わせる。
// identify_axes.py --signs で実機に聞いて確定（2026-10-20）:
//   両方とも持ち上げで -90 度側に動いたので、前後とも -1。
//   鏡像ではなく同じ向きに付いている
constexpr float SIGN_FRONT = -1.0f;
constexpr float SIGN_REAR = -1.0f;

// 安全のための電流上限 [mA]。学習した方策のトルク実効値は 0.127 N·m
// （出力軸トルク定数 0.354 N·m/A から 359mA 相当）なので、その倍を上限にする。
// ★ 最初はもっと低い値から始めて、動きを見ながら上げるのが安全
// シミュの方策が使う瞬間最大トルクは 0.315 N·m = 890mA（失速の 61%）。
// 800mA だと肝心の持ち上げ局面で頭打ちになるので余裕を持たせる
constexpr uint16_t CURRENT_LIMIT_MA = 1200;

// サーボ内蔵 PD の比例ゲイン。
//   400  : 目標に 10 度遅れる（既定値）
//   1000 : 宙に持てば ±90 度振れるが、接地すると振れ幅が 26 度まで潰れた
//   2500 : 接地荷重に負けないように上げた値。振動するようなら下げる
constexpr uint16_t POSITION_P_GAIN_VALUE = 2500;
// P を上げたぶんの行き過ぎを抑える。0 のままだと振動しやすい
constexpr uint16_t POSITION_D_GAIN_VALUE = 500;

// サーボへ送る角度のハード上限。方策の目標角とは別に、指令の直前でもう一度潰す。
// ★ 機構的に当たる角度がこれより小さいなら下げること
//    （後ろの節が体にぶつかってストールした実績あり）
constexpr float Q_LIMIT_DEG = 90.0f;

// 角速度を割る値。学習時の分散で正規化すると過敏になるため、env.py 側で
// 固定値による無次元化に変えた。**ここを合わせないと方策に 10 倍の値が入る**
constexpr float GYRO_SCALE = 10.0f; // rad/s

// テレメトリを何制御ステップに 1 回送るか。
// Serial.printf は送信が終わるまで待つので、ここが制御周期を直接食う。
// 115200bps のとき 1 行 92 バイト = 8.0ms かかっており、2 ステップに 1 回でも
// 1 ステップあたり 4.0ms（予算 10ms の 40%）を消費して 87Hz に落ちていた。
// ボーレートを上げたうえで、走行中は 10 ステップに 1 回（10Hz）に間引く
constexpr int TELEMETRY_DECIM = 10;
constexpr uint32_t SERIAL_BAUD = 921600;

// IMU の軸を機体座標（x=前, y=左, z=上）に直す行列。body = M * imu。
// 実機で軸と符号を確かめて決めた値（2026-10-20）。
// M5Stack は画面を上にして寝かせてあり、z 軸まわりに 90 度回った取り付け:
//   前(+x) = -imu_y /  左(+y) = +imu_x /  上(+z) = +imu_z
static const float IMU_TO_BODY[3][3] = {
    {+0.0f, -1.0f, +0.0f},
    {+1.0f, +0.0f, +0.0f},
    {+0.0f, +0.0f, +1.0f},
};

struct Vec3 {
  float x, y, z;
};

static Vec3 MapImu(float ix, float iy, float iz) {
  const float v[3] = {ix, iy, iz};
  float o[3] = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      o[i] += IMU_TO_BODY[i][j] * v[j];
  return {o[0], o[1], o[2]};
}

// 観測そのものを 1 行で送る。PC 側はこれを MuJoCo に流し込んで姿勢を描く。
static void SendTelemetry(const Vec3 &acc, const Vec3 &gyr, const float *q_deg,
                          const float *dq_rad, const float *tgt_deg,
                          bool running) {
  Serial.printf(
      "T,%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.4f,%.4f,%.2f,%.2f,%d\n",
      (unsigned long)millis(), acc.x, acc.y, acc.z, gyr.x, gyr.y, gyr.z,
      q_deg[0], q_deg[1], dq_rad[0], dq_rad[1], tgt_deg[0], tgt_deg[1],
      running ? 1 : 0);
}

// ───────────────────────── 方策 ─────────────────────────

// obs -> 正規化 -> Tanh(64) -> Tanh(64) -> 行動。
// **クリップしない生の出力を返す。** 呼び出し側で [-1,1] に収めること。
// 学習した方策は常に飽和しているので、クリップ後の値で突き合わせても
// 実装の誤りを検出できない（どんな実装でも ±1 になる）
static void PolicyForward(const float *obs, float *action) {
  float z[POLICY_N_IN];
  for (int i = 0; i < POLICY_N_IN; ++i) {
    float v = (obs[i] - kObsMean[i]) / kObsStd[i];
    z[i] = v > POLICY_CLIP_OBS ? POLICY_CLIP_OBS
                               : (v < -POLICY_CLIP_OBS ? -POLICY_CLIP_OBS : v);
  }
  float h0[POLICY_N_HIDDEN];
  for (int j = 0; j < POLICY_N_HIDDEN; ++j) {
    float s = kB0[j];
    const float *w =
        &kW0[j * POLICY_N_IN]; // PyTorch の Linear は (out, in) 並び
    for (int i = 0; i < POLICY_N_IN; ++i)
      s += w[i] * z[i];
    h0[j] = tanhf(s);
  }
  float h1[POLICY_N_HIDDEN];
  for (int j = 0; j < POLICY_N_HIDDEN; ++j) {
    float s = kB1[j];
    const float *w = &kW1[j * POLICY_N_HIDDEN];
    for (int i = 0; i < POLICY_N_HIDDEN; ++i)
      s += w[i] * h0[i];
    h1[j] = tanhf(s);
  }
  for (int k = 0; k < POLICY_N_OUT; ++k) {
    float s = kB2[k];
    const float *w = &kW2[k * POLICY_N_HIDDEN];
    for (int i = 0; i < POLICY_N_HIDDEN; ++i)
      s += w[i] * h1[i];
    action[k] = s; // クリップしない
  }
}

static inline float Clip1(float v) {
  return v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
}

// export_policy.py が出したテストベクタ。C 実装が Python
// と一致するかを確かめる。
// クリップ前の生出力で比べる（この方策は常に飽和するので、クリップ後では検証にならない）
static void SelfTest() {
  const float obs[POLICY_N_IN] = {
      +0.834444f, +0.468774f, -0.289750f, +0.004572f, +0.241396f,
      -0.123605f, +0.157795f, -0.526733f, +9.772182f, -9.684257f,
      +1.000000f, -1.000000f, +0.193553f, -0.453871f};
  const float expect[POLICY_N_OUT] = {+0.340597f, -2.113979f};
  float act[POLICY_N_OUT];
  PolicyForward(obs, act);
  for (int i = 0; i < POLICY_N_OUT; ++i) {
    float err = fabsf(act[i] - expect[i]);
    Serial.printf("[selftest] a[%d]=%+.6f expect=%+.6f err=%.2e %s\n", i,
                  act[i], expect[i], err, err < 1e-4f ? "OK" : "NG");
  }
}

// ───────────────────────── 本体 ─────────────────────────

Dynamixel2Arduino dxl(Serial2); // 方向制御は基板側が面倒を見る
using namespace ControlTableItem;

static float g_home_deg[2] = {
    0.0f, 0.0f}; // 水平に対応する Dynamixel 角度（キャリブレーション結果）
static bool g_calibrated = false;
static Preferences g_prefs;
static float g_last_q[2] = {0.0f, 0.0f}; // Sync Read が落ちたときの保険
static float g_target_deg[2] = {0.0f,
                                0.0f}; // 積分している目標角（方策の座標系）
static float g_prev_action[2] = {0.0f, 0.0f}; // 前回の行動
static bool g_running = false;
static uint32_t g_overrun = 0;
static uint32_t g_read_fail = 0; // Sync Read が 2 台そろわなかった回数

static void StopRun();

static void SetupServo(uint8_t id) {
  dxl.torqueOff(id);
  // Extended Position（多回転）にする。0〜360 の Position モードだと、
  // ゼロ点が 0 付近にある関節で指令が 0 を下回った瞬間に 360 側へ折り返し、
  // サーボが反対回りに大回りして機体にぶつかる（実機で発生させた）
  dxl.setOperatingMode(id, OP_EXTENDED_POSITION);
  dxl.writeControlTableItem(CURRENT_LIMIT, id, CURRENT_LIMIT_MA);
  // 台形プロファイルを無効にする（0 = 制限なし）。
  // これが残っていると指令どおりに動かず、サーボ内蔵 PD の素性が見えない。
  // 加速度 429deg/s^2 に制限され、5 度動くのに 153ms かかっていた
  dxl.writeControlTableItem(PROFILE_ACCELERATION, id, 0);
  dxl.writeControlTableItem(PROFILE_VELOCITY, id, 0);
  dxl.writeControlTableItem(POSITION_P_GAIN, id, POSITION_P_GAIN_VALUE);
  dxl.writeControlTableItem(POSITION_D_GAIN, id, POSITION_D_GAIN_VALUE);
}

// ── 1 往復で 2 台ぶんを読み書きする（Sync Read / Sync Write）──
// 個別に読み書きすると 1 ループで 6 往復になり、1Mbps でも 11.5ms かかって
// 100Hz に間に合わなかった（実測 87Hz）。2 往復に減らす。
namespace bus {
constexpr uint16_t ADDR_PRESENT_VELOCITY =
    128; // ここから vel(4) + pos(4) が並んでいる
constexpr uint16_t ADDR_GOAL_POSITION = 116;

struct __attribute__((packed)) Feedback {
  int32_t velocity;
  int32_t position;
};
struct __attribute__((packed)) Goal {
  int32_t position;
};

static Feedback fb[2];
static Goal goal[2];
static uint8_t sr_buf[64];
static DYNAMIXEL::InfoSyncReadInst_t sr;
static DYNAMIXEL::XELInfoSyncRead_t sr_xel[2];
static DYNAMIXEL::InfoSyncWriteInst_t sw;
static DYNAMIXEL::XELInfoSyncWrite_t sw_xel[2];

static void Init(uint8_t id0, uint8_t id1) {
  const uint8_t ids[2] = {id0, id1};
  sr.packet.p_buf = sr_buf;
  sr.packet.buf_capacity = sizeof(sr_buf);
  sr.packet.is_completed = false;
  sr.addr = ADDR_PRESENT_VELOCITY;
  sr.addr_length = sizeof(Feedback);
  sr.p_xels = sr_xel;
  sr.xel_count = 2;
  sr.is_info_changed = true;
  sw.packet.p_buf = nullptr; // ライブラリ内部のバッファを使う
  sw.packet.is_completed = false;
  sw.addr = ADDR_GOAL_POSITION;
  sw.addr_length = sizeof(Goal);
  sw.p_xels = sw_xel;
  sw.xel_count = 2;
  sw.is_info_changed = true;
  for (int i = 0; i < 2; ++i) {
    sr_xel[i].id = ids[i];
    sr_xel[i].p_recv_buf = (uint8_t *)&fb[i];
    sw_xel[i].id = ids[i];
    sw_xel[i].p_data = (uint8_t *)&goal[i];
  }
}
} // namespace bus

// サーボの設定を読み出す。シミュの前提と食い違っていないかの確認用
static void DumpServo(uint8_t id) {
  const int herr = (int)dxl.readControlTableItem(HARDWARE_ERROR_STATUS, id);
  if (herr) {
    Serial.printf("[dxl] id=%u HARDWARE ERROR 0x%02X -> reboot\n", id, herr);
    dxl.reboot(id);
    delay(300);
  }
  Serial.printf(
      "[dxl] id=%u vlimit=%d pwmlimit=%d minpos=%d maxpos=%d volt=%d temp=%d\n",
      id, (int)dxl.readControlTableItem(VELOCITY_LIMIT, id),
      (int)dxl.readControlTableItem(PWM_LIMIT, id),
      (int)dxl.readControlTableItem(MIN_POSITION_LIMIT, id),
      (int)dxl.readControlTableItem(MAX_POSITION_LIMIT, id),
      (int)dxl.readControlTableItem(PRESENT_INPUT_VOLTAGE, id),
      (int)dxl.readControlTableItem(PRESENT_TEMPERATURE, id));
  Serial.printf(
      "[dxl] id=%u mode=%d P=%d I=%d D=%d prof_acc=%d prof_vel=%d ilimit=%d\n",
      id, (int)dxl.readControlTableItem(OPERATING_MODE, id),
      (int)dxl.readControlTableItem(POSITION_P_GAIN, id),
      (int)dxl.readControlTableItem(POSITION_I_GAIN, id),
      (int)dxl.readControlTableItem(POSITION_D_GAIN, id),
      (int)dxl.readControlTableItem(PROFILE_ACCELERATION, id),
      (int)dxl.readControlTableItem(PROFILE_VELOCITY, id),
      (int)dxl.readControlTableItem(CURRENT_LIMIT, id));
}

// 角度は循環量なので、差を取ったら必ず ±180 に畳む。
// これが無いと、Extended Position モードで記録したゼロ点（360 度超え）や
// 0/360 の境界をまたいだときに、まるごと 360 度ずれる
static float Wrap180(float deg) {
  while (deg > 180.0f)
    deg -= 360.0f;
  while (deg < -180.0f)
    deg += 360.0f;
  return deg;
}

// 方策の座標系 [deg] -> Dynamixel の角度 [deg]。
// Extended Position なので折り返さない。ゼロ点のまわりで連続した値になる
static float ToDxlDeg(int joint, float q_deg) {
  const float sign = joint == 0 ? SIGN_FRONT : SIGN_REAR;
  const float q = q_deg > Q_LIMIT_DEG
                      ? Q_LIMIT_DEG
                      : (q_deg < -Q_LIMIT_DEG ? -Q_LIMIT_DEG : q_deg);
  return g_home_deg[joint] + sign * q;
}

static float FromDxlDeg(int joint, float dxl_deg) {
  const float sign = joint == 0 ? SIGN_FRONT : SIGN_REAR;
  return sign * Wrap180(dxl_deg - g_home_deg[joint]);
}

// ──────────────── ゼロ点キャリブレーション ────────────────

static void LoadHome() {
  g_prefs.begin("inchworm", true);
  const bool has = g_prefs.isKey("home0") && g_prefs.isKey("home1");
  g_home_deg[0] = g_prefs.getFloat("home0", 180.0f);
  g_home_deg[1] = g_prefs.getFloat("home1", 180.0f);
  g_prefs.end();
  for (int j = 0; j < 2; ++j) {
    while (g_home_deg[j] < 0.0f)
      g_home_deg[j] += 360.0f;
    while (g_home_deg[j] >= 360.0f)
      g_home_deg[j] -= 360.0f;
  }
  g_calibrated = has;
  Serial.printf("[calib] %s home=(%.2f, %.2f) deg\n",
                has ? "loaded" : "NOT SET", g_home_deg[0], g_home_deg[1]);
}

// 節を手で水平にした状態で呼ぶ。いまの Present Position をゼロ度として覚える。
static void Calibrate() {
  for (uint8_t id : {ID_FRONT, ID_REAR})
    dxl.torqueOff(id); // 手で動かせるように
  delay(50);
  for (int j = 0; j < 2; ++j) {
    const uint8_t id = j == 0 ? ID_FRONT : ID_REAR;
    // 1 回の読みだとノイズが乗るので平均する
    float sum = 0.0f;
    constexpr int kN = 16;
    for (int i = 0; i < kN; ++i) {
      sum += dxl.getPresentPosition(id, UNIT_DEGREE);
      delay(5);
    }
    float h = sum / kN;
    while (h < 0.0f)
      h += 360.0f;
    while (h >= 360.0f)
      h -= 360.0f; // Extended Position の名残で 360 を超えることがある
    g_home_deg[j] = h;
  }
  g_prefs.begin("inchworm", false);
  g_prefs.putFloat("home0", g_home_deg[0]);
  g_prefs.putFloat("home1", g_home_deg[1]);
  g_prefs.end();
  g_calibrated = true;
  Serial.printf("[calib] saved home=(%.2f, %.2f) deg\n", g_home_deg[0],
                g_home_deg[1]);

  M5.Display.fillScreen(BLACK);
  M5.Display.setCursor(0, 0);
  M5.Display.println("calibrated");
  M5.Display.printf("home %.1f %.1f\n", g_home_deg[0], g_home_deg[1]);
  M5.Display.println("A:start B:stop C:zero");
}

static void StartRun() {
  if (!g_calibrated) {
    // ゼロ点が分からないまま動かすと、最初の 1 ステップで機体が飛ぶ
    Serial.println(
        "[run] refused: not calibrated. hold the body flat and press C");
    M5.Display.fillScreen(BLACK);
    M5.Display.setCursor(0, 0);
    M5.Display.println("NOT CALIBRATED");
    M5.Display.println("flat + press C");
    return;
  }
  for (uint8_t id : {ID_FRONT, ID_REAR}) {
    SetupServo(id);
    dxl.torqueOn(id);
  }
  // いまの姿勢から積分を始める（起動時に飛ばないように）
  for (int j = 0; j < 2; ++j) {
    const uint8_t id = j == 0 ? ID_FRONT : ID_REAR;
    g_target_deg[j] = FromDxlDeg(j, dxl.getPresentPosition(id, UNIT_DEGREE));
  }
  g_prev_action[0] = g_prev_action[1] = 0.0f;
  g_overrun = 0;
  g_read_fail = 0;
  g_running = true;
}

static void StopRun() {
  g_running = false;
  for (uint8_t id : {ID_FRONT, ID_REAR})
    dxl.torqueOff(id);
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  M5.Display.setTextSize(2);
  Serial.begin(SERIAL_BAUD);

  Serial2.begin(DXL_BAUD, SERIAL_8N1, PIN_DXL_RX, PIN_DXL_TX);
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(DXL_PROTOCOL);
  delay(DXL_SETTLE_MS);
  for (uint8_t id : {ID_FRONT, ID_REAR}) {
    bool ok = false;
    for (uint8_t r = 0; r < DXL_PING_RETRY && !ok; ++r)
      ok = dxl.ping(id);
    Serial.printf("[dxl] id=%u ping=%s model=%d\n", id, ok ? "OK" : "NG",
                  ok ? dxl.getModelNumber(id) : -1);
    if (ok) {
      SetupServo(id);
      DumpServo(id);
    }
  }

  bus::Init(ID_FRONT, ID_REAR);
  SelfTest();
  LoadHome();

  M5.Display.fillScreen(BLACK);
  M5.Display.setCursor(0, 0);
  M5.Display.println("inchworm policy");
  if (g_calibrated) {
    M5.Display.printf("home %.1f %.1f\n", g_home_deg[0], g_home_deg[1]);
  } else {
    M5.Display.println("NOT CALIBRATED");
  }
  M5.Display.println("A:start B:stop C:zero");
}

void loop() {
  M5.update();
  if (M5.BtnC.wasPressed() && !g_running)
    Calibrate(); // 走行中は触らせない
  if (M5.BtnA.wasPressed() && !g_running)
    StartRun();
  if (M5.BtnB.wasPressed() && g_running)
    StopRun();
  if (!g_running) {
    // 待機中も観測を表示し続ける。方策を走らせる前に、
    // 「水平で q=0」「持ち上げると q が正」「水平静置で
    // g=(0,0,-1)」を目で確かめられる。
    static uint32_t last = 0;
    if (millis() - last > 200) {
      last = millis();
      float ra[3];
      M5.Imu.update();
      M5.Imu.getAccel(&ra[0], &ra[1], &ra[2]);
      const Vec3 a = MapImu(ra[0], ra[1], ra[2]);
      const float n = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z) + 1e-6f;
      const float q0 =
          FromDxlDeg(0, dxl.getPresentPosition(ID_FRONT, UNIT_DEGREE));
      const float q1 =
          FromDxlDeg(1, dxl.getPresentPosition(ID_REAR, UNIT_DEGREE));
      M5.Display.setCursor(0, 60);
      M5.Display.printf("q %+6.1f %+6.1f deg\n", q0, q1);
      M5.Display.printf("g %+5.2f %+5.2f %+5.2f\n", -a.x / n, -a.y / n,
                        -a.z / n);
      M5.Display.printf("%s      \n",
                        g_calibrated ? "calibrated" : "NOT CALIB");
    }
    // 待機中もテレメトリを流す。手で動かして PC 側の描画と突き合わせるため
    static uint32_t last_tm = 0;
    if (millis() - last_tm >= 20) {
      last_tm = millis();
      float ra[3], rw[3];
      M5.Imu.update();
      M5.Imu.getAccel(&ra[0], &ra[1], &ra[2]);
      M5.Imu.getGyro(&rw[0], &rw[1], &rw[2]);
      const Vec3 a = MapImu(ra[0], ra[1], ra[2]);
      const Vec3 g = MapImu(rw[0], rw[1], rw[2]);
      const Vec3 g_rad = {g.x * DEG_TO_RAD, g.y * DEG_TO_RAD, g.z * DEG_TO_RAD};
      float q_deg[2], dq[2] = {0.0f, 0.0f};
      q_deg[0] = FromDxlDeg(0, dxl.getPresentPosition(ID_FRONT, UNIT_DEGREE));
      q_deg[1] = FromDxlDeg(1, dxl.getPresentPosition(ID_REAR, UNIT_DEGREE));
      SendTelemetry(a, g_rad, q_deg, dq, q_deg, false);
    }
    delay(10);
    return;
  }

  const uint32_t t0 = micros();

  // ── 観測を作る ──
  float a[3], w[3];
  M5.Imu.update();
  M5.Imu.getAccel(&a[0], &a[1], &a[2]); // [g] 静止時は上方向を指す
  M5.Imu.getGyro(&w[0], &w[1], &w[2]);  // [deg/s]
  const Vec3 acc = MapImu(a[0], a[1], a[2]);
  const Vec3 gyr = MapImu(w[0], w[1], w[2]);

  // 加速度計は静止時に「上向き」に +1g を読む。重力方向はその逆向き
  const float an = sqrtf(acc.x * acc.x + acc.y * acc.y + acc.z * acc.z) + 1e-6f;

  float q[2], dq[2];
  // 2 台そろって返ってきたときだけ信用する。1 台だけだと古い値を使い続けて
  // 目標角との差が開き続ける（実測で 189 度まで開いた）
  const uint8_t n_recv = dxl.syncRead(&bus::sr);
  const bool got = (n_recv == 2);
  if (!got)
    ++g_read_fail;
  // 続けて失敗するのは配線か通信設定の異常。目標角が暴走する前に止める
  if (g_read_fail > 50) {
    Serial.printf("[run] aborted: sync read failed %lu times\n",
                  (unsigned long)g_read_fail);
    StopRun();
    return;
  }
  for (int j = 0; j < 2; ++j) {
    const float sign = j == 0 ? SIGN_FRONT : SIGN_REAR;
    if (got) {
      q[j] = FromDxlDeg(j, bus::fb[j].position * 360.0f / 4096.0f) * DEG_TO_RAD;
      // Present Velocity の 1 = 0.229 rev/min
      dq[j] = sign * bus::fb[j].velocity * 0.229f * (2.0f * PI / 60.0f);
    } else {
      q[j] = g_last_q[j];
      dq[j] = 0.0f;
    }
    g_last_q[j] = q[j];
  }

  float obs[POLICY_N_IN] = {
      -acc.x / an,
      -acc.y / an,
      -acc.z / an,
      gyr.x * DEG_TO_RAD / GYRO_SCALE,
      gyr.y * DEG_TO_RAD / GYRO_SCALE,
      gyr.z * DEG_TO_RAD / GYRO_SCALE,
      q[0],
      q[1],
      dq[0],
      dq[1],
      g_prev_action[0],
      g_prev_action[1],
      g_target_deg[0] / POLICY_Q_MAX_DEG,
      g_target_deg[1] / POLICY_Q_MAX_DEG,
  };

  // ── 推論して目標角を積分 ──
  float act[POLICY_N_OUT];
  PolicyForward(obs, act);
  for (int j = 0; j < POLICY_N_OUT; ++j)
    act[j] = Clip1(act[j]);
  for (int j = 0; j < 2; ++j) {
    g_target_deg[j] += act[j] * POLICY_DELTA_MAX_DEG;
    if (g_target_deg[j] > POLICY_Q_MAX_DEG)
      g_target_deg[j] = POLICY_Q_MAX_DEG;
    if (g_target_deg[j] < -POLICY_Q_MAX_DEG)
      g_target_deg[j] = -POLICY_Q_MAX_DEG;
    g_prev_action[j] = act[j];
  }

  for (int j = 0; j < 2; ++j)
    bus::goal[j].position =
        (int32_t)lroundf(ToDxlDeg(j, g_target_deg[j]) * 4096.0f / 360.0f);
  bus::sw.is_info_changed = true;
  dxl.syncWrite(&bus::sw);

  // ── 周期を守る ──
  const uint32_t budget_us = 1000000 / POLICY_CTRL_HZ;
  const uint32_t used = micros() - t0;
  if (used < budget_us) {
    delayMicroseconds(budget_us - used);
  } else {
    ++g_overrun; // 間に合っていない。通信レートか処理を見直す
  }

  static uint32_t tick = 0;
  ++tick;
  if (tick % TELEMETRY_DECIM == 0) {
    const float q_deg[2] = {q[0] * RAD_TO_DEG, q[1] * RAD_TO_DEG};
    SendTelemetry(acc,
                  {gyr.x * DEG_TO_RAD, gyr.y * DEG_TO_RAD, gyr.z * DEG_TO_RAD},
                  q_deg, dq, g_target_deg, true);
  }
  if (tick % 25 == 0) {
    M5.Display.setCursor(0, 60);
    M5.Display.printf("q  %+6.1f %+6.1f deg\n", g_target_deg[0],
                      g_target_deg[1]);
    M5.Display.printf("a  %+5.2f %+5.2f   \n", act[0], act[1]);
    M5.Display.printf("us %4lu over %lu rf %lu  \n", (unsigned long)used,
                      (unsigned long)g_overrun, (unsigned long)g_read_fail);
  }
}
