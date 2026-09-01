# isaac_continuous_track

`gazebo_continuous_track` (Gazebo/ODE 플러그인)의 Isaac Sim / PhysX 5 이식.
설계 근거는 저장소 루트의 [`ISAAC_SIM_PORT_PLAN.md`](../ISAAC_SIM_PORT_PLAN.md).

궤도를 링크 체인으로 만들지 않는다. **차체에 매달린 세그먼트 링크 4개**를 한 그라우저
피치만큼만 미끄러뜨리고, 한 피치를 넘으면 0으로 되감는다. 그라우저 간격이 균일하므로
되감기 전후 형상이 동일하다.

## 파일 구성

| 파일 | 원본 대응 | pxr 필요 |
|---|---|---|
| `track_config.py` | `continuous_track_plugin.sdf` 스키마 + `make_track` xacro | 아니오 |
| `track_geometry.py` | `FillSegmentLength()`, `ComposeSegments()`의 배치 루프 | 아니오 |
| `track_builder.py` | `ComposeSegments()`의 프림 생성, `InitTrack()`의 충돌 필터 | **예** |
| `track_driver.py` | `UpdateTrack()` | 아니오 |
| `math_utils.py` | `ComputeChildPoseOffset()`, `Vector3d::DistToLine()` | 아니오 |
| `isaac_compat.py` | 버전별 Isaac 임포트 격리 (계획서 §8) | 아니오 |
| `example/spawn_track_robot.py` | 예제 로봇 + 슬립률 측정 (계획서 §6-2) | 예 |
| `tests/` | 이식 기하·드라이버 검증 — Isaac 없이 실행 | 아니오 |

USD를 만지는 건 `build_track()` 하나뿐이다. 이식된 알고리즘 본체와 드라이버는
pxr 의존이 없으므로 Isaac Sim 없이 그대로 돌려볼 수 있다:

```bash
python -m isaac_continuous_track.tests.test_track_geometry
python -m isaac_continuous_track.tests.test_track_driver
```

## 사용법

```python
from pxr import Usd
from isaac_continuous_track import DriveCfg, GrouserCfg, TrackDriver, build_track
from isaac_continuous_track import make_oval_track_cfg

cfg = make_oval_track_cfg(
    name="left_track",
    chassis_path="/World/Robot/chassis",
    sprocket_joint_path="/World/Robot/left_sprocket_joint",  # 이미 존재하는 회전 조인트
    pitch_diameter=0.2,
    length=0.5, radius=0.1, width=0.12,
    mass=2.0,
    elements_per_round=32,
    grouser=GrouserCfg(size=(0.012, 0.12, 0.012)),
    origin_pos=(0.0, 0.2, -0.2),          # 차체 프레임 기준 궤도 원점
    drive=DriveCfg(damping=5.0e4, max_force=500.0),
)
handle = build_track(stage, cfg, body_paths=["/World/Robot/chassis", "/World/Robot/left_sprocket"])

# 시뮬레이션 시작 후
driver = TrackDriver(view, [handle])   # view = Articulation("/World/Robot")
driver.reset()                          # 벨트 위상 동기화 + wrap 검출기 초기화
driver.register()                       # subscribe_physics_step_events
```

예제 실행 (윈도우, Isaac Sim 5.1). 두 방식 다 된다.

**1. 파일 경로로 직접** — 패키지를 Isaac Sim 설치 루트에 심볼릭 링크해 두는 방식.
`PYTHONPATH`가 필요 없다:

```bash
C:\isaac-sim-5.1.0\python.bat C:\isaac-sim-5.1.0\isaac_continuous_track\example\spawn_track_robot.py --headless --seconds 6
```

**2. 모듈로** — 저장소 루트에서:

```bash
set PYTHONPATH=%CD% && C:\isaac-sim-5.1.0\python.bat -m isaac_continuous_track.example.spawn_track_robot --headless --seconds 6
```

리눅스에서는 Isaac Sim 설치 디렉터리의 `./python.sh`가 같은 역할을 한다.

`print()` 출력이 안 보이면 `PYTHONUNBUFFERED=1`을 켠다 — Kit은 프로세스를 내릴 때
파이썬 stdout 버퍼를 플러시하지 않아서 버퍼에 남은 출력이 통째로 사라진다.
예제 스크립트 자체는 `flush=True`로 이미 대응해 두었다.

스크립트를 경로로 실행하면 파이썬은 `example\` 폴더만 `sys.path`에 넣으므로 두 단계 위의
패키지를 못 찾는다. 예제 상단에서 패키지의 부모 디렉터리를 직접 넣어 해결했다. 이때
심볼릭 링크를 `os.path.realpath()`로 풀어 **실제 저장소 경로**를 넣는다 — 링크 위치인
Isaac Sim 설치 루트를 넣으면 그 안의 `isaacsim\`(`__init__.py` 없음)이 네임스페이스
패키지로 잡혀 진짜 `isaacsim` 모듈을 가릴 수 있다.

## 원본과의 대응

**유지한 것**

- 세그먼트 4개의 부모는 전부 차체다. 스프라켓이 아니다. 따라서 스프라켓 조인트는
  궤도 부하를 느끼지 않는다 (계획서 §3.3).
- `joint_to_track`: prismatic이면 1.0, revolute면 자식 링크 원점에서 조인트 축까지의
  거리. `perimeter = Σ length`, `pitch = perimeter / elements_per_round`.
- 세그먼트 링크는 그라우저 외에 **자기 자신의 베이스 collider(box/cylinder)도 가진다**.
- 되감기 범위 `[-pitch/2, +pitch/2)`, 세그먼트 조인트 속도 지령
  `track_vel / joint_to_track`을 매 스텝 적용.

**의도적으로 바꾼 것**

| 원본 | 이식 | 근거 |
|---|---|---|
| 매 스텝 `SetPosition` | wrap을 넘은 스텝에만 `set_joint_positions` | PhysX는 contact patch와 friction anchor를 스텝 간 유지한다. 매 스텝 텔레포트하면 앵커가 계속 무효화된다 (계획서 §3.1) |
| ODE 조인트 모터 `fmax`/`vel` | `UsdPhysics.DriveAPI`, `stiffness=0` | 사실상 1:1 |
| variant + 런타임 collision/visibility 토글 | 제거 | 그라우저를 전부 동일 형상으로 두면 원리적으로 불필요 (계획서 §3.2) |
| `dGeomSetCollideBits` | `UsdPhysics.CollisionGroup` + `filteredGroups` | 정적 설정 |
| 런타임 SDF 생성 + private 멤버 접근 | USD 프림 직접 생성 | `gazebo_patch.hpp` 전체가 불필요해진다 |

### 단위 규약

`DriveCfg`의 게인은 **궤도 공간(track space)** 기준이다 — `damping`은 궤도속도 오차
1 m/s당 N, `max_force`는 궤도가 낼 수 있는 견인력 상한(N). `track_builder`가 세그먼트별
`joint_to_track` 값 `s`로 변환한다: 토크 한계 `F·s`, damping `D·s²`. prismatic은 `s=1`이라
그대로다. 여기에 더해 revolute 드라이브는 USD가 도(degree) 단위로 저작하므로
`π/180`을 한 번 더 곱한다. 런타임 텐서 API는 라디안이므로 양쪽이 맞아떨어진다.

## 검증

### `tests/test_track_geometry.py` — 기하

- `ComputeChildPoseOffset`의 1-파라미터 군 성질 — `offset(a)·offset(b) == offset(a+b)`.
  원본이 `base_pose = base_pose_step + base_pose`로 누적할 수 있었던 근거다.
- `perimeter == 2·length + 2·π·radius`, arc 세그먼트의 반지름 복원.
- 배치된 그라우저가 벨트 경로상 정확히 `k·pitch` 위치에 온다 (4개 세그먼트 전체에 걸쳐
  최대 오차 2.2e-16 m). 마지막↔처음 간격도 정확히 한 피치 — **패턴이 닫힌다**.
- 되감기 불변성: 모든 세그먼트를 한 피치 전진시키면 그라우저 집합이 자기 자신으로
  사상된다. 단, 세그먼트 경계마다 최대 1개는 예외다 — 세그먼트는 강체이므로 끝단의
  그라우저는 코너를 돌지 않고 직진한다. 편차는 한 피치로 유한하며, 되감기 범위가
  ±pitch/2이므로 실제로는 그 절반 이하다. **원본도 동일하게 동작한다** (이식 결함 아님).

### `tests/test_track_driver.py` — 런타임

가짜 articulation view와 이상적인 속도 드라이브를 물려 `UpdateTrack()`의 모든 분기를
검사한다:

- 세그먼트 조인트 지령이 `track_vel / joint_to_track`과 정확히 일치 (직선 0.5 m/s ↔
  원호 5.0 rad/s).
- 되감기 빈도가 `v_track / pitch`와 일치 — 1 m/s에서 4초간 78회(이론 79회). 되감기
  사이에 12.8스텝이 남는다: 계획서 §3.1이 노린 그대로, 마찰 앵커가 자리 잡을 시간이 있다.
- 되감기 목표값이 `wrapped / joint_to_track`과 오차 0.
- 되감기가 스프라켓·서스펜션의 위치와 **속도**를 건드리지 않는다
  (`preserveWorldVelocity=true` 재현).
- 역방향 주행과 다중 환경(N=4) 동시 구동. 정지한 환경은 `-pitch/2` 위상을 그대로
  유지한다 — 다른 환경이 되감기로 전체 텐서를 덮어써도 흐트러지지 않는다.

한 가지 짚어둘 동작: `track_pos == 0`일 때 원본 공식이 주는 `wrapped`는 0이 아니라
`-pitch/2`다. 정지 상태의 궤도는 반 피치 뒤에 서 있다. 원본과 동일하다.

### Isaac Sim 5.1 실측 (계획서 §6)

**§6-2 평지 직진 슬립률** — 예제 로봇, 지령 0.5 m/s

```
[t= 1.00s] v_body= 0.518 m/s  v_track= 0.500 m/s  slip=-0.036  z= 0.312 m
[t= 4.50s] v_body= 0.518 m/s  v_track= 0.500 m/s  slip=-0.036  z= 0.312 m
```

- `v_track`이 지령과 정확히 일치 — 드라이브 게인 변환이 맞다.
- `v_body`가 매끄럽고 되감기 주파수(9.8 Hz)의 리플이 없다. 계획서 §6-2가 경고한
  contact 재생성 아티팩트가 나타나지 않았다.
- slip이 −3.5%로 음수인 건 그라우저가 벨트면 밖으로 6 mm 나와 유효 구름반경이
  피치반경보다 크기 때문이다(상한 +6%). 정상이다.

**견인이 진짜인지 확인 — 마찰 0 반증 실험**

```bash
... spawn_track_robot.py --ground-friction 0.0 --track-friction 0.0
[t= 2.00s] v_body= 0.000 m/s  v_track= 0.500 m/s  slip= 1.000
```

마찰을 없애면 벨트는 계속 돌지만 차량은 **전혀 움직이지 않는다.** 되감기 텔레포트가
운동량을 주입하는 게 아니라 접촉 마찰이 실제로 일을 하고 있다는 뜻이다.

> **주의: 지면만 0으로 놓으면 이 테스트는 무의미하다.** PhysX는 접촉 마찰을 두 재질의
> 조합으로 계산하고 기본 모드가 **평균**이라, 벨트 재질이 0.8이면 실효 마찰이 0.4로
> 남는다. 실제로 지면만 0으로 놓았을 때는 μ=0.8일 때와 똑같이 주행해서, 한때 이것을
> 무반동 추진의 증거로 잘못 읽었다. **반드시 `--ground-friction 0 --track-friction 0`
> 둘 다 줘야 한다.**

**§6-5 제자리 선회** — 예제 로봇은 약 −36 deg/s로 정상 선회한다. 무슬립 이론값
`2v/W = 143 deg/s`보다 훨씬 작은 건 스키드 스티어에서 정상이다.

**Talon 4-플리퍼는 선회가 되지 않는다 (미해결).** 요각속도 0.00 deg/s. 벨트 표면속도는
좌우가 정확히 대립하는데(FL −0.399 / FR +0.404 / RL −0.408 / RR +0.405) 차체는 회전
대신 병진한다. 팔 각도(0/15/30°)와 마찰(0.9/0.3/0.15)을 바꿔도 불변. 같은 모듈로
예제 로봇은 선회하므로 모듈이 아니라 Talon 쪽 조건 문제다. 접지선이 앞뒤로 1.44 m
퍼져 있는데 트랙 간격은 0.48 m뿐이라 요 저항이 매우 크고(예전 컨베이어 방식도 선회
효율 0.135로 같은 벽에 부딪혔다), 여기에 더해 `vflip_link` 콜라이더가 아직 켜져 있어
견인 없이 요 저항만 더하고 있을 가능성이 있다.

## USD 물리에서 밟기 쉬운 함정 둘

실제로 돌려보며 걸린 것들이라 적어둔다. 둘 다 예외도 경고도 없이 조용히 틀린 결과를 낸다.

**1. 리지드바디 프림에 `xformOp:scale`을 주지 말 것.** USD 물리는 조인트의 `localPos`에
그 바디의 스케일을 곱한다. 스케일 (0.6, 0.4, 0.2)인 차체에 앵커 `(0, 0.2, -0.2)`를 주면
실제로는 `(0, 0.08, -0.04)`에 붙는다. 궤도와 스프라켓이 전부 차체 안으로 빨려들어가
차량이 배를 깔고 앉는다. 게다가 `Gf.Matrix4d.ExtractRotationQuat()`이 스케일 있는 행렬에서
정규화되지 않은 쿼터니언을 돌려주므로(실측 `(0.7416, 0, 0, 0)`) 프레임 계산도 같이 깨진다.

박스를 만들려면 **스케일 없는 Xform을 바디로 두고 그 아래 자식 프림에 스케일**을 준다.
`build_track()`이 차체 스케일을 검사해 명확한 에러를 내도록 해 두었고,
`_gf_to_pose()`는 행렬에서 스케일을 나눠낸 뒤 회전을 뽑는다.

**2. 무제한 조인트는 limit을 "안 쓰는" 것이지 `lower > upper`가 아니다.** `lower=1,
upper=-1`로 자유회전을 표현하려 하면 PhysX는 그대로 ±1도로 잠긴 조인트로 읽는다.
스프라켓이 전혀 돌지 않아 한참 헤맸다. limit 속성을 **아예 저작하지 않으면** 무제한이다.

## 알려진 제약

- 원호 세그먼트의 베이스 collider는 `UsdGeom.Cylinder`다. 씬에서 custom geometry
  cylinder가 꺼져 있으면 PhysX가 convex hull로 근사한다. 하중은 그라우저가 받으므로
  실용상 문제는 없지만 알아둘 것.
- `set_joint_positions`는 계획서 §4.3대로 **전체 DOF를 읽어 벨트 DOF만 고치고 통째로**
  쓴다. 속도는 백업/원복해 `preserveWorldVelocity=true`의 의미를 버전 무관하게 재현한다.
- 이종 그라우저 패턴, 스프라켓 부하 되먹임, 컨베이어 하이브리드, 서스펜션은
  1차 범위 밖이다 (계획서 §7).
