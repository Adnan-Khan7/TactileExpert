# TactileExpert

Human-in-the-loop pipeline for generating and evaluating tactile graphics
using GPT-image-1 and BANA-trained quality evaluators.

![Pipeline](pipeline_diagram.png)

## Pipeline

1. **Upload** a natural reference photograph
2. **Generate** — GPT-image-1 produces a tactile graphic candidate from a
   BANA-grounded prompt (editable)
3. **Evaluate** — CLIP Probe and VLM Fine-tuned score the candidate on four
   BANA quality dimensions: Too Thick, Broken Lines, Missing Parts, Missing Texture
4. **Edit** — select which evaluator verdict to trust, modify the suggested
   instruction if needed, and regenerate via GPT-image-1 edit mode
5. **Iterate** — repeat until satisfied or both models show no defects

API key is entered directly in the UI for every call and is never stored.

## Running on Compute Canada (SLURM)

```bash
cd ~/TactileExpert
mkdir -p logs
sbatch slurm/run_pipeline.sh
```

Check the log for the node name:
```bash
tail -f logs/pipeline_<jobid>.out
```

Then open an SSH tunnel from your laptop:
```bash
ssh -L 7860:<nodename>:7860 nibi
```

Open `http://localhost:7860` in your browser.

## Models

| Model | Architecture | Macro F1 (cleaned test) |
|---|---|---|
| CLIP Probe | CLIP ViT-L/14 + MLP heads | 0.676 |
| VLM Fine-tuned | Qwen2-VL-2B + LoRA (epoch 7) | 0.541 |

Checkpoints: `models/clip_probe/` and `models/vlm_checkpoint/`.
The VLM requires the Qwen2-VL-2B-Instruct base model from HuggingFace.

## Project

Part of the TactileEval research project evaluating AI-generated tactile
graphics for blind and visually impaired users.
