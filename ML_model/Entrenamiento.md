Con lo que tenés armado en el repo (ML_model/, genie/sam2/configs/), el camino es este:

1. Armar el dataset final

Una vez etiquetados los pares _rgb.jpg/_mask.png (máscara blanca=transitable, negra=no):

bash
python -m dataset_tools armar \
    --labeled data/candidatos \
    --out training/FOLD_custom \
    --val-frac 0.15

Esto te arma el layout img_folder/{train,val}/<id>/00000.jpg + gt_folder/{train,val}/<id>/00000.png (estilo MOSE — cada imagen es un "video" de 1 frame) que espera el training config de SAM2. El val_frac es la separación que después te sirve para controlar si funcionó.

2. Lanzar el entrenamiento

El código de entrenamiento no está vendorizado en genie/ — hay que clonar el repo de Meta aparte:

bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e ".[dev]"

Copiar genie/sam2/configs/sam2.1_training_tiny/sam2.1_custom2.yaml (o el fold que corresponda) a los configs de esa clone, y editar dos cosas:

dataset.img_folder / dataset.gt_folder → apuntar a training/FOLD_custom/train/...
el checkpoint_path de inicialización → tu checkpoint actual (checkpoint_finetuned_v2.pt), no arrancar desde cero

Lanzar:

bash
python training/train.py \
    -c configs/sam2.1_training_tiny/sam2.1_custom2.yaml \
    --use-cluster 0 --num-gpus 1

Hiperparámetros de referencia (los que ya usaron para el checkpoint que tenés): resolución 1024, batch 8, AdamW, base_lr 5e-6, vision_lr 3e-6, 5 épocas. Para un fine-tune chico (unos cientos de frames nuevos, no miles): mismo LR, bajá a 2-3 épocas — con menos datos, más épocas es la receta directa para el catastrophic forgetting que se menciona más abajo.

3. Qué controlar durante el entrenamiento

Con TensorBoard:

bash
tensorboard --logdir sam2_logs/

Mirá loss_mask, loss_dice, loss_iou (log_freq: 10, cada 10 steps):

Normal: bajan las primeras épocas y se aplanan. Con pocos datos y LR bajo, la caída va a ser chica — no esperes un dropoff dramático.
Alerta: loss en NaN/inf (LR muy alto o batch con máscara corrupta), o la loss sube en vez de bajar de entrada (algo mal en el checkpoint de init o en el formato de máscara).
El repo guarda checkpoints en sam2_logs/<config>/checkpoints/ según save_freq — quedate con el de la última época salvo que veas la loss divergir antes.
4. Qué controlar después (esto es lo que de verdad importa)

La loss de training no te dice si el modelo mejoró en la práctica — para eso:

Correr el checkpoint nuevo sobre los frames de val/ que separaste en el paso 1 (esos tienen máscara real, nunca los vio en training):
bash
   SAMTP_CHECKPOINT=/ruta/al/nuevo.pt \
     python -m rover_traversability.demo predict frame_val.jpg --out check.png
Comparar overlay viejo vs. nuevo, frame por frame, prestando especial atención a los frames que el modelo VIEJO ya acertaba — el fallo clásico de un fine-tune chico es catastrophic forgetting: mejora en tus frames nuevos pero empieza a fallar en escenas que antes resolvía bien.
Si querés un número y no solo mirar a ojo: como ya tenés máscara real para cada frame de val/, es trivial calcular IoU (intersección/unión entre mask > 0.5 predicha y la máscara etiquetada) por frame y promediar — esto no viene armado en el repo, pero es un script de 15 líneas si querés que te lo arme.
Recién ahí, subir de riesgo en orden: demo live (guarda overlays, no manda comandos) → demo drive en espacio abierto con el dedo en Ctrl-C → misión completa.
5. Shippearlo
bash
hf upload yourteam/samtp-yourteam nuevo.pt checkpoint_finetuned_v2.pt
export SAMTP_HF_REPO=yourteam/samtp-yourteam

Guardá el sha256 del checkpoint nuevo al lado de tus resultados — así después "qué modelo era este run" siempre se puede responder.

¿Ya tenés los videos grabados con mission_recorder.py, o todavía estás en esa etapa? Si querés, te armo el script de IoU sobre el val set o el de comparación de overlays lado a lado.