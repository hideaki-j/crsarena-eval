# CRSArena-Eval Interactive Interface

An interactive web-based evaluation tool for CRS Arena benchmarks. Upload your run file and instantly see correlation metrics (Pearson & Spearman) for turn-level and dialogue-level aspects.

## 🚀 Live Demo

Visit the GitHub Pages site to use the interactive evaluator:
**[Your GitHub Pages URL]**

## 📊 Features

- **Drag & Drop Interface**: Simply drag your run file or click to browse
- **Instant Evaluation**: Client-side processing for immediate results
- **Comprehensive Metrics**: 
  - Turn-level aspects: relevance, interestingness
  - Dialogue-level aspects: understanding, task_completion, interest_arousal, efficiency, dialogue_overall
- **Dataset Breakdown**: Separate metrics for ReDial and OpenDialKG datasets
- **Beautiful Tables**: Clear presentation of Pearson and Spearman correlations

## 🎯 How to Use

1. **Visit the web interface** (GitHub Pages URL)
2. **Upload your run file** (e.g., `face_run.json`)
   - Drag and drop onto the upload area, OR
   - Click "Choose File" to browse your files
3. **View results** automatically displayed in tables
   - Turn-level correlation metrics
   - Dialogue-level correlation metrics
   - Results broken down by dataset (ReDial/OpenDialKG)

## 📁 File Format

Your run file should be a JSON file with the following structure:

```json
[
  {
    "conv_id": "barcor_redial_03368a16-93bd-4b21-885d-b9a21e3498ba",
    "turns": [
      {
        "turn_ind": 1,
        "turn_level_pred": {
          "relevance": 1.0,
          "interestingness": 0.8375
        }
      }
    ],
    "dial_level_pred": {
      "understanding": 0.5125,
      "task_completion": 0.3417,
      "interest_arousal": 0.4083,
      "efficiency": 0.3333,
      "dialogue_overall": 0.3958
    }
  }
]
```

## 🛠️ Local Development

To run locally:

1. Clone this repository
2. Serve the files with a local web server:
   ```bash
   python -m http.server 8000
   ```
3. Open `http://localhost:8000` in your browser

## 📦 Project Structure

```
.
├── index.html              # Main web interface
├── evaluator.js            # Evaluation logic (ported from Python)
├── crs_arena_eval/
│   ├── crs_arena_eval.json # Gold standard annotations
│   └── run/
│       ├── eval.py         # Original Python evaluation script
│       └── face_run.json   # Example run file
└── README.md
```

## 🔬 Command-Line Evaluation (Alternative)

You can also use the Python script directly:

```bash
python crs_arena_eval/run/eval.py --run_file crs_arena_eval/run/face_run.json
```

## 📈 Metrics Explained

- **Pearson Correlation**: Measures linear correlation between predictions and gold labels
- **Spearman Correlation**: Measures rank correlation (monotonic relationship)
- Values range from -1 to 1, where:
  - 1 = perfect positive correlation
  - 0 = no correlation
  - -1 = perfect negative correlation

## 🤝 Contributing

Feel free to open issues or submit pull requests to improve the evaluation interface!

## 📄 License

[Your License Here]