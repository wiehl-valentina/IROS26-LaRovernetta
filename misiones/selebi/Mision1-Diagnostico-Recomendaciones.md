# Lista de Arreglos para LaRovernetta (fixVelocidad-Tom)

## Problemas identificados en la corrida

| Síntoma | Causa raíz | Consecuencia |
|---------|-----------|--------------|
| No sigue checkpoint #3 (a 150° atrás) | Abanico ±75° capado, mete siempre el borde | Angular saturado = sin info de error |
| Se recupera pero no avanza | Retroceso vetado por cobertura=0% (cámara no mira atrás) | Loop infinito: frena→recupera→apunta obstáculo |
| Zigzag izquierda-derecha | Wrap de 180° + ruido brújula oscila entre lados | `commit_override=30°` no detiene saltos de 60-90° |
| Pose congelada en mapa BEV | Odometría integra comando (`linear=0`), no medida | Mapa desalineado; fantasmas que no se limpian |
| Desatascos: 0 | No hay escape automático de geometrías | Queda atrapado indefinidamente |

---

## FASE A: CRÍTICA — Retroceso y Pivote GPS

### A.1. Habilitar retroceso a ciegas

**Archivo:** `genie/genie_rover/bridge.py`  
**Función:** Buscar `_closest_navigable_direction()` o bloque de `RECUPERACION`

**Cambio:**
```python
# Viejo (línea aproximada ~580-590):
if map_behind['cobertura'] < 50%:
    # detras no parece seguro (o sin datos todavia), salteo el retroceso
    skip_backward = True

# Nuevo:
# La cámara nunca ve atrás => cobertura SIEMPRE es 0%
# En cambio, retrocede si:
# - Estás bloqueado adelante (clearance < 0.5m)
# - Y hace tiempo (>4 frames)
if clearance_m < 0.5 and consecutive_blocked_frames >= 4:
    # Retroceso a ciegas: velocidad baja, timeout corto
    self.send(DriveCommand(linear=-0.25, angular=0.0, reason="retroceso_ciegos_recovery"))
    time.sleep(1.5)  # 1.5 s @ 0.25 m/s ≈ 0.4 m atrás
    # Luego el recovery normal elige dirección
else:
    skip_backward = False
```

**Por qué funciona:** Rompe el loop "apunto al obstáculo → freno → apunto al obstáculo". Los 0.4 m hacia atrás desengatillan la geometría lo suficiente.

---

### A.2. Pivote gobernado por bearing GPS (no por error del camino)

**Archivo:** `genie/genie_rover/bridge.py`  
**Función:** En el ciclo principal, ANTES de `plan_on_bev()`

**Buscar y reemplazar (línea ~350-370):**
```python
# Viejo:
def _on_frame(self, ...):
    ...
    goal = goal_from_gps(...)
    # Llama directo al planner con la meta
    planned_path = plan_on_bev(bev, goal_xy, ...)

# Nuevo:
def _on_frame(self, ...):
    ...
    goal = goal_from_gps(...)
    
    # Si la meta está fuera del abanico, pivota primero
    if abs(goal.relative_bearing_deg) > self.planner_cfg.fan_max_angle_deg:
        # El planner no la puede ver; gira hacia ella
        error_rad = math.radians(goal.relative_bearing_deg)
        ang = np.clip(-self.cfg['nav']['kp_angular'] * error_rad * 0.0,  # No proporcional aquí
                      -self.cfg['nav']['max_angular'], 
                      self.cfg['nav']['max_angular'])
        # O mejor: velocidad constante hasta que entre en el abanico
        if goal.relative_bearing_deg > 0:
            ang_cmd = self.cfg['nav']['max_angular']  # Gira izquierda
        else:
            ang_cmd = -self.cfg['nav']['max_angular']  # Gira derecha
        
        self.send(DriveCommand(0.0, ang_cmd, 
                  reason=f"pivote_previo({goal.relative_bearing_deg:+.0f} grados)"))
        return  # No planifica hasta que entre en rango
    
    # Normal: llama al planner
    planned_path = plan_on_bev(bev, goal_xy, ...)
```

**Por qué funciona:** Resuelve el giro 150° *en el lugar* antes de que el planner se vea forzado a aproximar. Una vez que la meta entra en el abanico (±75°), el planner toma el control.

---

## FASE B: ALTA — Config de controlador

### B.1. Bajar kp_angular para evitar saturación prematura

**Archivo:** `genie/configs/frodobot_rover.yaml`  
**Línea:** `kp_angular: 0.45`

**Cambio:**
```yaml
# Viejo:
kp_angular: 0.45  # Satura a 57° => casi toda error es información cero

# Nuevo:
kp_angular: 0.004  # Satura a 75° => proporcional en todo el rango útil
# Fórmula: si quiero saturar a 75°, 
#   0.004 rad/s * 75° (en rad ≈ 1.31) = 0.0052 rad/s < 0.45
# Ajusta según pruebas
```

**Verificación:**
- Error 30° → angular ≈ 0.12 ✓ (proporcional)
- Error 60° → angular ≈ 0.24 ✓ (proporcional)
- Error 75° → angular ≈ 0.30 ✓ (todavía <0.45)

---

### B.2. Bajar align_threshold y subir turn_speed

**Archivo:** `genie/configs/frodobot_rover.yaml`  
**Líneas:** `align_threshold_deg` y `turn_speed`

**Cambio:**
```yaml
# Viejo:
align_threshold_deg: 90.0  # Inalcanzable desde planner (capped a 75°)
turn_speed: 0.40           # Más lento que curvar

# Nuevo:
align_threshold_deg: 65    # Ahora sí es alcanzable
turn_speed: 0.45           # Igual a max_angular (giro rápido)
```

**Por qué:** El rango [65°, 75°] ahora dispara pivote real (no aproximación). Es la rama conservadora del paper: gira en el lugar, sin avance lateral.

---

## FASE C: ALTA — Histéresis para ambigüedad de 180°

### C.1. Agregar histéresis en NavigationBridge

**Archivo:** `genie/genie_rover/bridge.py`  
**En clase NavigationBridge, __init__:**

```python
class NavigationBridge:
    def __init__(self, ...):
        ...
        self.backing_side = None  # None | 'left' | 'right'
        self.backing_since = None
```

**En el ciclo principal (función `_on_frame`), DESPUÉS de `goal_from_gps`:**

```python
goal = goal_from_gps(...)

# Histéresis para evitar oscilación cuando la meta está atrás
if abs(goal.relative_bearing_deg) > 135:
    # Meta detrás: elige un lado y mantenlo
    if self.backing_side is None:
        # Primera vez que ves meta detrás: elige el lado con más espacio
        left_free = traverse_map[left_corridor]  # Pedir al mapa
        right_free = traverse_map[right_corridor]
        self.backing_side = 'left' if left_free > right_free else 'right'
        self.backing_since = time.time()
        print(f"[bridge] meta detrás, elijo lado {self.backing_side}")
    
    # Mantén el lado elegido
    desired_bearing = goal.relative_bearing_deg
    if self.backing_side == 'left':
        desired_bearing = max(desired_bearing, 100)  # Fuerza hacia la izquierda
    else:
        desired_bearing = min(desired_bearing, -100)  # Fuerza hacia la derecha
    goal.relative_bearing_deg = desired_bearing

else:
    # Meta adelante: olvida el lado elegido
    if self.backing_side is not None:
        print(f"[bridge] meta adelante, cancelo histéresis (backing_side={self.backing_side})")
        self.backing_side = None
        self.backing_since = None
```

**Por qué funciona:** Cuando `|rel| > 135°`, ruido de brújula ya no puede cambiar de lado. Reduces los saltos de 60-90° que disparan `commit_override`.

---

## FASE D: MEDIA — Odometría con velocidad medida

### D.1. Reemplazar integración de comando por medida

**Archivo:** `genie/genie_rover/odometry.py`  
**Función:** `integrate()` o `step()`

**Cambio (pseudo-código):**
```python
# Viejo:
def integrate_pose(pose, cmd_linear, cmd_angular, dt):
    pose.x += cmd_linear * math.cos(pose.theta) * dt
    pose.y += cmd_linear * math.sin(pose.theta) * dt
    pose.theta += cmd_angular * dt
    return pose

# Nuevo:
def integrate_pose(pose, cmd_linear, cmd_angular, dt, 
                   actual_rpm_left, actual_rpm_right, wheel_radius, track_width):
    # Convierte RPM medido a velocidades lineales reales
    v_left = rpm_left * 2*pi*wheel_radius / 60
    v_right = rpm_right * 2*pi*wheel_radius / 60
    actual_linear = (v_left + v_right) / 2
    actual_angular = (v_right - v_left) / track_width
    
    # Integra lo QUE SE MOVIÓ, no lo que pediste
    pose.x += actual_linear * math.cos(pose.theta) * dt
    pose.y += actual_linear * math.sin(pose.theta) * dt
    pose.theta += actual_angular * dt  # Usa gyro si está disponible
    return pose
```

**Por qué:** Cuando el rover frena (`linear=0`), la pose se actualiza a 0, no congelada. El mapa BEV se recentra automáticamente.

---

## FASE E: MEDIA — Desatasco y recentrado forzado

### E.1. Desatasco forzado con retroceso + giro

**Archivo:** `genie/genie_rover/bridge.py`  
**Función:** Nueva función `_forced_unstick()` o en `_unstick()`

```python
def _forced_unstick(self):
    """Escape forzado cuando clearance no mejora en N frames."""
    if self.stats.consecutive_blocked_frames < 30:
        return False  # No dispara todavía
    
    print(f"[bridge] DESATASCO FORZADO: {self.stats.consecutive_blocked_frames} frames bloqueado")
    
    # 1. Retrocede
    self.send(DriveCommand(-0.25, 0.0, "retroceso_desatasco"))
    time.sleep(1.5)
    
    # 2. Gira 90°
    self.send(DriveCommand(0.0, 0.45, "giro_90_desatasco"))
    time.sleep(1.0)
    
    # 3. Reset
    self.send(DriveCommand(0.0, 0.0, "fin_desatasco"))
    self.stats.consecutive_blocked_frames = 0
    self.stats.unstucks += 1
    return True
```

**Dónde llamar:** En `_on_frame()`, después de chequeo de `front_is_blocked`, si `consecutive_blocked_frames >= 30`.

---

### E.2. Recentrado forzado del mapa

**Archivo:** `genie/genie_rover/bridge.py`  
**En `_on_frame()`, después de actualizar BEV:**

```python
# Fuerza recentrado cada 5 segundos aunque no sea por margen
if time.time() - self.last_forced_recentroid > 5.0:
    if self.pmap is not None:
        self.pmap.force_recentroid(current_pose)
    self.last_forced_recentroid = time.time()
```

**En `__init__`:**
```python
self.last_forced_recentroid = time.time()
```

---

## Orden de implementación recomendado

1. **A.1** → Habilitar retroceso (30 min)
2. **A.2** → Pivote GPS (1 h)
3. **Test corrida** con estos dos → ¿Cruza cp#3 sin atascarse?
4. **B.1, B.2** → Config (30 min)
5. **C.1** → Histéresis (30 min)
6. **Test corrida** → ¿Sigue sin zigzag?
7. **D.1** → Odometría medida (1-2 h, toca más del código)
8. **E.1, E.2** → Red de seguridad (30 min)

---

## Checklist de testing

- [ ] Retroceso se ejecuta (log: `retroceso_ciegos` o `retroceso_recovery`)
- [ ] Pivote GPS cuando `|rel| > 75°` (log: `pivote_previo`)
- [ ] No hay más `desatascos forzados: 0` (debe ser > 0 si está atrapado)
- [ ] Angular no está siempre saturado en `siguiendo camino`
- [ ] Backing_side no cambia 10 veces en 30 frames
- [ ] Pose en el mapa se mueve incluso cuando `linear=0.0`

