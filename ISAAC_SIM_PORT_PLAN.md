# gazebo_continuous_track → Isaac Sim 이식 계획

Gazebo/ODE 플러그인 `ContinuousTrack`을 Isaac Sim(PhysX 5)으로 옮기기 위한 구현 계획.
원본 동작의 근거는 [`include/gazebo_continuous_track/gazebo_continuous_track.hpp`](include/gazebo_continuous_track/gazebo_continuous_track.hpp),
[`urdf_xacro/macros_common_gazebo.urdf.xacro`](urdf_xacro/macros_common_gazebo.urdf.xacro).

---

## 0. 한 줄 요약

궤도를 링크 체인으로 만들지 않는다. **차체에 prismatic/revolute로 매달린 세그먼트 링크 4개**를
한 그라우저 피치만큼만 미끄러뜨리고, 한 피치를 넘으면 0으로 되감는다.
그라우저 간격이 균일하므로 되감기 전후 형상이 동일해 시각·물리적으로 연속으로 보인다.
이 메커니즘은 PhysX에서 그대로 성립한다.

---

## 1. 원본 메커니즘 분석 (이식 대상 명세)

### 1.1 궤도 형상 — 4세그먼트 타원

`make_track` 매크로가 만드는 표준 구성:

| 세그먼트 | 조인트 타입 | 축 | 링크 형상 | `end_position` |
|---|---|---|---|---|
| `straight_segment0` | prismatic | `1 0 0` | box (`length × width × radius`) | `length` |
| `arc_segment0` | revolute | `0 1 0`, 조인트 pose `0 0 -radius` | cylinder (반지름 `radius`) | `pi` |
| `straight_segment1` | prismatic | `1 0 0` (역방향 배치) | box | `length` |
| `arc_segment1` | revolute | `0 1 0` | cylinder | `pi` |

- **4개 조인트의 부모는 전부 차체(`parent` = base_link)**. 스프라켓이 아니다.
- 세그먼트 질량은 각각 `mass / 4`.
- 궤도 둘레 `perimeter = 2·length + 2·pi·radius`.

### 1.2 스칼라 변환 `joint_to_track`

`FillSegmentLength()` ([hpp:256](include/gazebo_continuous_track/gazebo_continuous_track.hpp:256)):

```
prismatic:  joint_to_track = 1.0            length = end_position
revolute :  joint_to_track = radius         length = radius · end_position
            (radius = 자식 링크 원점에서 조인트 축 직선까지의 거리)
```

조인트 위치 ↔ 궤도 진행거리 변환에 쓰인다. `joint_pos = track_len / joint_to_track`.

### 1.3 그라우저 배치

`ComposeSegments()` ([hpp:106](include/gazebo_continuous_track/gazebo_continuous_track.hpp:106)):

- 간격 `len_step = perimeter / elements_per_round`
- 둘레를 따라 누적 이동하며 각 세그먼트 링크의 로컬 프레임에 collision/visual을 자식으로 추가
- 위치는 `ComputeChildPoseOffset(joint, 0, len_traveled / joint_to_track)` — 즉 **조인트를 그만큼 움직였을 때 자식 링크가 가는 포즈**를 오프셋으로 사용. 직선부는 단순 평행이동, 원호부는 축 중심 회전.
- 세그먼트 링크는 그라우저 외에 **자기 자신의 베이스 collider(box/cylinder)도 가진다** — 벨트 몸체에 해당.

### 1.4 런타임 루프

`UpdateTrack()` ([hpp:392](include/gazebo_continuous_track/gazebo_continuous_track.hpp:392)):

```
track_pos = sprocket_joint_position · (pitch_diameter / 2)
track_vel = sprocket_joint_velocity · (pitch_diameter / 2)

len_per_element = perimeter / elements_per_round
wrapped = track_pos - len_per_element·floor(track_pos / len_per_element) - len_per_element/2
          # → [-pitch/2, +pitch/2)

for each segment:
    SetPosition(joint, wrapped / joint_to_track, preserveWorldVelocity=true)
    SetJointMotorVelocity(joint, track_vel / joint_to_track)   # ODE fmax + vel
```

- 위치는 **매 스텝** 직접 세팅 (드리프트 방지)
- 힘 전달은 ODE 조인트 모터(`fmax`/`vel`)가 담당 → 지면 마찰 반력이 조인트 구속을 타고 차체로 전달

### 1.5 variant (패턴 이종성 지원)

그라우저 종류가 여러 개일 때, 패턴을 한 칸씩 민 벨트 복사본을 `elements.size()`개 만들고
ODE 비트마스크(`dGeomSetCategoryBits/CollideBits`)로 하나만 활성화. 되감기 시점에 교체.

### 1.6 충돌 필터링

`InitTrack()` ([hpp:355](include/gazebo_continuous_track/gazebo_continuous_track.hpp:355)):
스프라켓 부모/자식 링크와 세그먼트 부모 링크를 `GZ_GHOST_COLLIDE` 카테고리로 묶어
**벨트가 차체·바퀴와 충돌하지 않게** 한다. 벨트는 외부 환경하고만 충돌.

---

## 2. Isaac Sim 매핑

| Gazebo / ODE | Isaac Sim / PhysX | 비고 |
|---|---|---|
| 세그먼트 prismatic/revolute 조인트 | `UsdPhysics.PrismaticJoint` / `RevoluteJoint`, **차체와 같은 articulation 내부** | 트리 구조라 문제없음 (차체 → 세그먼트 4개 병렬 브랜치 + 차체 → 스프라켓) |
| ODE 조인트 모터 `fmax`+`vel` | `UsdPhysics.DriveAPI`, `stiffness=0`, `damping=큰 값`, `maxForce=토크한계` | 사실상 1:1 대응 |
| `wrap::SetPosition(..., preserveWorldVelocity=true)` | `ArticulationView.set_joint_positions()` + 속도 원복 | §4.2 참고 |
| `dGeomSetCollideBits` 벨트↔차체 차단 | `UsdPhysics.CollisionGroup` + `filteredGroups`, 또는 `PhysxSchema` filtered pairs | 정적 설정이라 런타임 부담 없음 |
| variant collision 토글 | **제거** (§3.2) | |
| `msgs::Visual` 가시성 토글 | **제거** | |
| SDF 런타임 생성 + private 멤버 해킹 ([gazebo_patch.hpp](include/gazebo_continuous_track/gazebo_patch.hpp)) | Python으로 USD 프림 직접 생성 | 코드가 대폭 짧아지는 지점 |
| `ConnectWorldUpdateBegin` | `omni.physx.get_physx_interface().subscribe_physics_step_events()` | |
| ODE surface `mu`, `min_depth` | `UsdPhysics.MaterialAPI` (`staticFriction`/`dynamicFriction`), `PhysxCollisionAPI` (`contactOffset`/`restOffset`) | |

### 2.1 채택하지 않는 대안과 이유

| 대안 | 기각 사유 |
|---|---|
| 벨트를 kinematic 리지드바디로 두고 포즈로 이동 | PhysX kinematic은 반력을 받지 않고, 지면은 static → **상호작용 자체가 없음**. 차체가 안 밀림 |
| 링크 하나하나 이어붙인 실제 체인 트랙 | 원 논문이 피하려던 그 방식. 느리고 불안정 |
| 컨베이어(surface velocity)만 사용 | 싸고 안정적이지만 **그라우저가 지형에 파고드는 효과가 사라짐** — 이 논문의 핵심이 없어짐 |
| 컨베이어 + 이동 세그먼트 하이브리드 | 유효한 최적화이나 **1차 구현 범위 밖**. §7 참고 |

---

## 3. 원본 대비 의도적 변경점

### 3.1 되감기를 매 스텝이 아니라 wrap 시점에만

**원본**: 매 스텝 `SetPosition`.
**변경**: 평상시엔 속도 드라이브로만 굴리고, `wrapped`가 되감기는 스텝에만 `set_joint_positions`.

이유 — ODE는 매 스텝 contact를 새로 생성하므로 매 스텝 텔레포트에 관대하다.
PhysX는 contact patch와 friction anchor를 스텝 간 **유지**하므로 매 스텝 텔레포트하면
앵커가 계속 무효화되어 마찰이 제대로 쌓이지 않는다.

되감기 빈도 검산: 피치 5 cm, 궤도속도 1 m/s → **초당 20회**.
물리 250 Hz면 되감기 사이에 12스텝, 500 Hz면 25스텝. 앵커가 자리 잡는 데 2~3스텝이면 충분하므로
대부분의 스텝은 앵커가 유지된 상태로 돈다.

드리프트 보정은 wrap 시점의 위치 재설정이 겸한다.

### 3.2 variant 제거

**전제: 그라우저를 전부 동일 형상으로 만든다.**

그러면 `elements.size() == 1`이 되어 variant가 원리적으로 불필요해진다.
결과적으로 **런타임 collision 토글과 visibility 토글이 전부 사라진다** —
GPU 파이프라인/Fabric에서 비싸거나 지원이 애매한 두 연산을 처음부터 안 쓰게 된다.

이종 패턴이 필요해지면 §7의 후속 과제로 뺀다.

### 3.3 (선택) 스프라켓 부하 되먹임

원본은 세그먼트 조인트 부모가 차체라 **스프라켓 조인트가 궤도 부하를 전혀 느끼지 않는다**.
스프라켓 모터 토크/전류를 시뮬레이션하려면 세그먼트 조인트 반력의 접선 성분을 합산해
스프라켓 조인트에 외부 토크로 되먹여야 한다. 1차 구현에서는 원본 동작을 그대로 두고,
필요 시 §7에서 추가.

---

## 4. 구현 설계

### 4.1 파일 구성

```
isaac_continuous_track/
├── track_config.py      # 치수/물리 파라미터 dataclass (SDF <plugin> 파라미터 대응)
├── track_builder.py     # USD 프림 생성: 세그먼트 링크·조인트·그라우저 배치
├── track_driver.py      # 물리 스텝 콜백: track_pos 계산, 드라이브 타깃, wrap 되감기
└── example/
    └── spawn_track_robot.py
```

`track_config.py` 는 [`sdf/continuous_track_plugin.sdf`](sdf/continuous_track_plugin.sdf) 의 스키마를 그대로 옮긴다:

```python
@dataclass
class SprocketCfg:
    joint_path: str          # <sprocket><joint>
    pitch_diameter: float    # <sprocket><pitch_diameter>

@dataclass
class SegmentCfg:
    joint_path: str          # <trajectory><segment><joint>
    end_position: float      # <trajectory><segment><end_position>

@dataclass
class TrackCfg:
    name: str
    sprocket: SprocketCfg
    segments: list[SegmentCfg]
    elements_per_round: int  # <pattern><elements_per_round>
    grouser_size: tuple[float, float, float]
    chassis_path: str
```

### 4.2 `track_builder.py`

**Phase A — 형상 계산** (`FillSegmentLength()` 이식)

```python
for seg in cfg.segments:
    if is_revolute(seg):
        radius = dist_point_to_line(child_origin, joint_pos, joint_axis)
        seg.joint_to_track = radius
        seg.length = radius * seg.end_position
    else:  # prismatic
        seg.joint_to_track = 1.0
        seg.length = seg.end_position
perimeter = sum(s.length for s in segments)
pitch = perimeter / cfg.elements_per_round
```

**Phase B — 프림 생성**

세그먼트마다:
1. `Xform` + `UsdPhysics.RigidBodyAPI` + `MassAPI`(질량 = 궤도 총질량 / 세그먼트 수)
2. 베이스 collider (직선=box, 원호=cylinder) — §1.3 참고, 빠뜨리지 말 것
3. 차체와의 `PrismaticJoint`/`RevoluteJoint`, 축·pose는 xacro 템플릿과 동일
   - 직선: axis `X`
   - 원호: joint pose `(0, 0, -radius)`, axis `Y`
4. `DriveAPI` 적용 — `stiffness=0`, `damping=D`, `maxForce=F` (§5 튜닝표)
5. 조인트 limit: **`lower=-pitch, upper=+pitch`** 로 넉넉히 (되감기 범위 ±pitch/2를 감싸도록)

**Phase C — 그라우저 배치** (`ComposeSegments()` 이식)

```python
len_traveled, elem_count, len_left = 0.0, 0, 0.0
for seg in segments:
    len_left += seg.length
    while len_left >= 0.0 and elem_count < cfg.elements_per_round:
        offset = child_pose_offset(seg, 0.0, len_traveled / seg.joint_to_track)
        add_grouser_collider(seg.prim, offset, cfg.grouser_size, name=f"element{elem_count}")
        len_left     -= pitch
        len_traveled += pitch
        elem_count   += 1
    len_traveled -= seg.length
```

`child_pose_offset` = 조인트를 `q`만큼 움직였을 때 자식 링크 포즈의 변화량
(직선: 축 방향 `q` 평행이동 / 원호: 조인트 축 중심 `q` 회전).
원본 `ComputeChildPoseOffset()` 과 동일.

**Phase D — 충돌 필터**

`UsdPhysics.CollisionGroup` 두 개:
- `belt_group` : 세그먼트 링크 전부
- `body_group` : 차체, 스프라켓, 바퀴 등

서로를 `filteredGroups`에 등록 → 벨트↔차체 충돌 차단, 벨트↔환경은 유지.
`belt_group` 자기 자신도 필터링해 세그먼트끼리 충돌하지 않게 한다.

### 4.3 `track_driver.py`

```python
class TrackDriver:
    def __init__(self, view, cfg, geom):
        self.prev_wrap_index = None

    def on_physics_step(self, dt):
        r = self.cfg.sprocket.pitch_diameter / 2.0
        q  = view.get_joint_positions()
        qd = view.get_joint_velocities()

        track_pos = q[:, self.sprocket_dof]  * r
        track_vel = qd[:, self.sprocket_dof] * r

        wrap_index = torch.floor(track_pos / self.geom.pitch)
        wrapped = track_pos - self.geom.pitch * wrap_index - self.geom.pitch / 2.0

        # 1) 속도 타깃은 매 스텝 (ODE fmax/vel 대응)
        for seg in self.segs:
            vel_targets[:, seg.dof] = track_vel / seg.joint_to_track
        view.set_joint_velocity_targets(vel_targets)

        # 2) 위치 되감기는 wrap을 넘은 스텝에만
        crossed = (wrap_index != self.prev_wrap_index)
        if crossed.any():
            q_new  = q.clone()
            qd_bak = qd.clone()                    # ← preserveWorldVelocity 대응
            for seg in self.segs:
                q_new[crossed, seg.dof] = wrapped[crossed] / seg.joint_to_track
            view.set_joint_positions(q_new)        # 전체 DOF 써야 다른 조인트 안 깨짐
            view.set_joint_velocities(qd_bak)      # 속도 원복
        self.prev_wrap_index = wrap_index
```

**주의점**

- `set_joint_positions` 는 articulation 캐시 전체를 적용한다. 벨트 DOF만 부분 갱신하려면
  **현재 전체 위치를 읽어 벨트 DOF만 수정한 뒤 통째로 쓴다.** 스프라켓·서스펜션 상태를 클로버링하면 안 됨.
- 위치 적용이 속도를 건드리는지는 Isaac Sim 버전에 따라 다르다. **속도를 백업했다 원복**하면
  버전 무관하게 원본의 `preserveWorldVelocity=true` 의미를 재현한다.
- 초기화 시 `prev_wrap_index` 를 첫 스텝 값으로 세팅해 시작 프레임에서 헛되감기를 막는다.
- 텐서 연산으로 짜두면 Isaac Lab 다중 환경(N envs)에 그대로 확장된다.

---

## 5. 물리 파라미터 튜닝 기준선

| 항목 | 시작값 | 조정 지침 |
|---|---|---|
| 물리 스텝 | 1/250 s | §6-2 리플 보이면 1/500 |
| 씬 solver | TGS | PGS는 이 구성에서 불리 |
| position iteration | 8 | 벨트 흔들리면 16까지 |
| velocity iteration | 1~2 | 과하게 올리지 말 것 |
| 세그먼트 damping `D` | 궤도 목표속도에서 필요한 힘 / 속도오차 기준으로 역산, 크게 | 작으면 궤도가 지면에 끌려 뒤처짐 |
| 세그먼트 `maxForce` | 원본 `GetEffortLimit()` 대응. 미지정이면 사실상 무제한 | 이 값이 궤도 최대 견인력 상한 |
| 세그먼트 질량 | 궤도 총질량 / 4 (원본과 동일) | 차체 대비 질량비 1:100 이내 유지 |
| `contactOffset` | 0.02 m | 그라우저 크기 대비 과하지 않게 |
| `restOffset` | 0.0 | ODE `min_depth`(기본 0.01) 대응 검토 |
| static/dynamic friction | `mu`(기본 0.5)를 양쪽에 동일 적용 | 논문 재현 시 실측값 |
| 그라우저 피치 | 작을수록 되감기 아티팩트 감소, 대신 collider 수 증가 | §6-2로 결정 |

---

## 6. 검증 계획

원본과의 등가성을 확인하는 순서. 앞 단계가 통과해야 다음으로 간다.

**1) 무부하 궤도 흐름 (지면 접촉 없음)**
로봇을 공중에 고정하고 스프라켓을 일정 각속도로 회전.
- 세그먼트 링크의 월드 속도가 `ω·r` 로 **등속**이어야 함
- 되감기 시점에 속도 스파이크가 없어야 함 → 있으면 §4.3의 속도 원복이 안 되고 있는 것
- 그라우저 간격이 되감기 전후로 동일한지 시각 확인

**2) 평지 직진 슬립률 — 가장 중요한 지표**
스프라켓 각속도 `ω` 를 주고 차체 속도 `v_body` 측정.
```
slip = 1 - v_body / (ω · pitch_diameter/2)
```
- 정상: slip이 작고 **매끄러움**
- 이상: `v_body` 에 **되감기 주파수(= v_track/pitch, 예 20 Hz)와 같은 주기의 리플**
  → contact 재생성 아티팩트. 대응 순서: ① 물리 dt 낮추기 ② 피치 줄이기 ③ friction 올리기

**3) 경사 등판 한계각**
마찰계수로부터 예측되는 한계각과 시뮬레이션 값 비교.

**4) 단차 극복** — 그라우저 기하가 실제로 걸리는지
컨베이어(surface velocity) 방식과 결과가 갈리는 지점이고, **이 이식의 존재 이유를 검증하는 항목**이다.
그라우저 높이보다 낮은/비슷한/높은 단차 3종.

**5) 제자리 선회 (좌우 궤도 반대방향)**
선회 반경과 요레이트 안정성.

**6) 성능**
세그먼트 4개 + 그라우저 N개 구성의 스텝 시간. 실시간 배수 기록.
Isaac Lab 다환경이면 env 수 스케일링 곡선.

---

## 7. 후속 과제 (1차 범위 밖)

- **이종 그라우저 패턴** — 필요해지면 variant 대신, 그라우저별 개별 조인트나
  세그먼트 링크를 패턴 주기 단위로 분할하는 방식 검토. 런타임 collision 토글은 끝까지 피한다.
- **스프라켓 부하 되먹임** (§3.3) — 모터 토크/전류 모델링이 필요할 때.
- **컨베이어 하이브리드** — 그라우저는 이동 세그먼트로, 평평한 벨트 몸체는
  surface velocity로 처리해 collider 수를 줄이는 최적화. §6-4로 등가성 확인 후에만.
- **서스펜션/텐셔너** — 원본에도 없음. 궤도 경로가 고정 타원이라는 가정이 깨지면
  세그먼트 형상을 매 스텝 재계산해야 하므로 설계가 크게 달라진다.
- **URDF/xacro 임포트 경로** — 기존 로봇 정의를 재사용하려면
  [`macros_track_gazebo.urdf.xacro`](urdf_xacro/macros_track_gazebo.urdf.xacro) 파라미터를
  `TrackCfg` 로 변환하는 컨버터.

---

## 8. 주요 리스크

| 리스크 | 영향 | 완화 |
|---|---|---|
| 되감기 시 friction anchor 무효화 → 미끄러짐/힘 스파이크 | 중 | §3.1 wrap 시점 한정, dt·피치 튜닝. §6-2가 조기 경보 |
| `set_joint_positions` 가 articulation 상태를 예상 밖으로 리셋 | 중 | 전체 DOF 읽고-쓰기 + 속도 백업/원복 |
| 그라우저 collider 수에 따른 성능 저하 | 중 | 피치를 필요 이상 줄이지 않기, §7 하이브리드 |
| 차체 대비 세그먼트 질량비로 인한 solver 불안정 | 낮 | 질량비 1:100 이내, position iteration 상향 |
| Isaac Sim 버전별 Python API 경로 변경 (`omni.isaac.core` ↔ `isaacsim.core`) | 낮 | 임포트를 한 모듈로 격리 |
