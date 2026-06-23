Autonomous Driving Multimodal Pipeline

Perception and Reasoning in Autonomous Driving: A Three-Stage Multimodal Approach Using OWL-ViT, YOLOv8, and Large Vision-Language Models

<br>
Overview
<br>
A three-stage multimodal pipeline for autonomous driving perception and decision-making:

Stage            Model                                        Role
Stage A          OWL-ViT(google/owlvit-base-patch32)          Zero-shot region proposals
Stage B          YOLOv8s (fine-tuned)                         Object detection & classification
Stage C          Llama 4 Scout 17B (Groq API)                 Scene reasoning & driving decision

Dataset: Udacity Self-Driving Car Dataset — 13,280 dashcam images, 8 classes

Best mAP50: 0.784 · Precision: 0.849 · Recall: 0.720
<br><br>


Pipeline Architecture

<img width="877" height="421" alt="image" src="https://github.com/user-attachments/assets/02d9c943-cf38-4358-a0bd-fbb3faf23cd4" />

<br>
Requirements

Google Colab with T4 GPU (recommended)<br>
Groq API key — for Llama 4 Scout inference<br>
Roboflow API key — for dataset download<br>

<br>
Installation & Setup

1. Clone the repository

git clone https://github.com/[your-username]/Autonomous-Driving-Multimodal-Pipeline.git

2. Open in Google Colab

Upload the .ipynb notebook to Google Colab and set the runtime to GPU (T4):

Runtime → Change runtime type → T4 GPU

3. Install dependencies

pip install -r requirements.txt

4. Set up API keys

<br>

Running the Project

Step 1 — Download the Dataset

Run the dataset cell in the notebook. It automatically downloads the Udacity Self-Driving Car Dataset (13,280 images, 8 classes) from Roboflow.

Classes: biker · car · pedestrian · trafficLight · trafficLight_green · trafficLight_red · trafficLight_yellow · truck

<br>
Step 2 — Train YOLOv8s

Run the training cell Or you can use mine best(1).pt in training folder. Key hyperparameters:

yamlmodel:    yolov8s.pt<br>
epochs:   100  # early stopping patience=15, converges ~epoch 37<br>
imgsz:    640<br>
batch:    16<br>
optimizer: MuSGD (auto)<br>
augment:<br>
  fliplr:      0.5<br>
  hsv_v:       0.1<br>
  hsv_s:       0.1<br>
  mosaic:      1.0<br>
  close_mosaic: 10<br>

Best weights are saved automatically as best.pt based on validation mAP50.<br>

<br>
Step 3 — Run the Pipeline

Run the pipeline cell with any dashcam image. The three stages execute sequentially:


Stage A — OWL-ViT generates region proposals (threshold=0.15)
Stage B — YOLOv8s runs on the full image + OWL-ViT crops, deduplicated via seen_boxes
Stage C — Llama 4 Scout receives image + detection summary via Groq API → driving decision

<br>


Step 4 — View Results

The notebook outputs:


Annotated image with bounding boxes and class labels
Detection summary table (class, confidence, bounding box)
Llama 4 Scout driving decision with natural language justification


<br>

Create a .env file in the root of the project:
<br>
envGROQ_API_KEY=your_groq_api_key_here<br>
ROBOFLOW_API_KEY=your_roboflow_api_key_here<br>
<br>
Project Structure

Autonomous-Driving-Multimodal-Pipeline/<br>
│<br>
├── pipeline.ipynb          # Main notebook — training + full pipeline<br>
├── requirements.txt        # Python dependencies<br>
├── .env.example            # API key template<br>
├── best.pt                 # Trained YOLOv8s weights (after training)<br>
├── data/                   # Dataset directory (auto-created by Roboflow)<br>
└── results/                # Training curves, confusion matrix, predictions<br>

Notes

OWL-ViT runs on CPU by default — no GPU required for Stage A<br>
Groq API has rate limits — add time.sleep(1) between requests if needed<br>
best.pt weights from Step 2 are reused across all pipeline stages<br>
For HPC/Singularity training, see the singularity/ folder for the container definition<br>


Academic Info

This project assignment is for the course Intelligent Information Systems at FINKI, under the mentorship of the course professors.<br>

Faculty: Faculty of Computer Science and Engineering (FINKI)
Course: Intelligent Information Systems
Date: 23.06.2026
