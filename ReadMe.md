# PDF Annotation with PyTorch Profiler

This Python project processes text (such as OCR outputs or NLP results), applies sentence-to-criteria matching, color–codes the results in a PDF, and measures performance using the PyTorch Profiler with TensorBoard integration.  
It uses the **DRESS dataset** for evaluation and development.  

---

## Features
- **PDF Annotation**: Highlights sentences based on similarity to rubric criteria.  
- **Citation Highlighting**: Marks citation sentences with a dedicated fixed color.  
- **Performance Profiling**: Integrates PyTorch Profiler for CPU/GPU activity tracking.  
- **TensorBoard Support**: View profiling results visually via TensorBoard.  

---

## Dataset Reference
This project uses the **DRESS dataset** as described by:  

> Haneul Yoo, *DRESS: A Dataset for Recognizing and Extracting Salient Sentences in Documents*,  
> [Dataset Website](https://haneul-yoo.github.io/dress)  

Please follow the dataset license and citation requirements if you use it for research or development.

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/pdf-annotation-profiler.git
cd pdf-annotation-profiler
```

### 2. Create Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

If CUDA support is needed for GPU profiling:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Usage

### Running the Main Script
The main script requires arguments for:
- **`--submission`**: Path to the student submission PDF.  
- **`--rubric`**: Path to the rubric PDF.  
- **`--comments`**: Path to the comments json file.
- **`--output`**: Path to save the annotated PDF.  
- **`--logdir`**: Directory to save profiler logs (default: `profiler_logs`).  

Example:
```bash
python annotate_pdf.py \
  --submission input/student_submission.pdf \
  --rubric input/rubric.pdf \
  --comments comments.json
  --output output/annotated.pdf \
  --logdir profiler_logs
```

### Viewing Profiler Logs
```bash
tensorboard --logdir=./profiler_logs
```
Then open the displayed URL (e.g., `http://localhost:6006`) to explore CPU/GPU activity and bottlenecks.

---

## Folder Structure
```
pdf-annotation-profiler/
│── annotate_pdf.py
│── requirements.txt
|── comments.json
│── profiler_logs/          # Generated logs for TensorBoard
│── input/                  # Input PDFs and rubric files
│── output/                 # Annotated PDFs
```

---

## License
```
Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to use, copy, modify, merge, publish,
and distribute the Software for **non-commercial purposes only**.

Commercial use, including but not limited to selling, sublicensing, or incorporating the
Software into a commercial product, is strictly prohibited without prior written permission
from the copyright holder.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```
