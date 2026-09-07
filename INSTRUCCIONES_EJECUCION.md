# Guía de Ejecución: La Rovernetta (GeNIE + Earth Rovers SDK)

Para levantar el proyecto y poner el rover a navegar de forma autónoma, es necesario ejecutar dos procesos en paralelo. Como cada componente tiene dependencias muy distintas (el SDK usa librerías web y GeNIE usa PyTorch/SAM2), **cada proceso debe correr en su propio entorno virtual y en una terminal separada**.

---

## 1. Levantar el SDK del Earth Rover (Proceso A)

Este proceso se encarga de la comunicación directa con el rover (video, telemetría y comandos de motor) a través de los servidores de FrodoBots.

1. **Abrir la primera terminal.**
2. Moverse a la carpeta del SDK:
   ```bash
   cd earth-rovers-sdk
   ```
3. Activar el entorno virtual del SDK:
   ```bash
   source .venv_sdk/bin/activate 
   ```
4. Iniciar el servidor (API local en el puerto 8000):
   ```bash
   hypercorn main:app --reload
   ```

> **Importante:** Deja esta terminal abierta y corriendo. No la cierres ni presiones `Ctrl+C`.

---

## 2. Levantar el cerebro autónomo GeNIE (Proceso B)

Este proceso analiza las fotos del SDK, extrae el mapa (BEV) usando el modelo SAM, y le envía las órdenes de manejo al SDK.

1. **Abrir una SEGUNDA terminal.**
2. Moverse a la carpeta de GeNIE:
   ```bash
   cd genie
   ```
3. Activar el entorno virtual de GeNIE:
   ```bash
   source .venv_genie/bin/activate
   ```
4. Lanzar el puente de control autónomo:
   ```bash
   python -m genie_rover.bridge \
       --config configs/frodobot_rover.yaml \
       --go \
       --start-mission \
       --max-seconds 120
   ```

### Notas sobre el comando GeNIE:
- `--go`: Habilita el movimiento real. Sin este flag, el programa hace un "dry run" (calcula todo pero no mueve los motores).
- `--start-mission`: Se encarga de autorizar la misión automáticamente (requerido por el SDK).
- `--max-seconds 120`: El bot se detendrá automáticamente a los 2 minutos por seguridad. (Puedes modificar este número o quitarlo si quieres navegación continua).

---

## Detención de emergencia

- Para frenar el rover en cualquier momento, haz un click en la terminal de GeNIE (Proceso B) y presiona `Ctrl + C`. El bridge atrapará la señal y le mandará una orden inmediata de freno total a los motores antes de cerrarse.
- Si por alguna razón el programa se cuelga, el SDK que dejamos corriendo en la Terminal A tiene un *watchdog* de seguridad que cortará la corriente de las ruedas automáticamente.

