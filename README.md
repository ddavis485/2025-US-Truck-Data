# 2025 US Truck Data

A dataset of 2025 US pickup trucks from brands like Chevrolet, Ford, GMC, Ram, and Toyota, covering models like Silverado, F-150, and Tundra. Includes JSON data with specs (towing, payload, fuel economy), a FAISS index for AI/LLM similarity search, and a FastAPI-based RAG pipeline for querying truck data. Ideal for automotive, automotive ecommerce, and AI research.

## Makes and Models
The dataset includes the following Makes and Models of 2025 US pickup trucks:

- **Chevrolet**
  - Colorado
  - Silverado 1500
  - Silverado 1500 ZR2
  - Silverado 2500HD
  - Silverado 3500HD
  - Silverado EV
- **Ford**
  - F-150
  - F-150 Lightning
  - F-150 Raptor
  - Super Duty F-250
  - Super Duty F-350
- **GMC**
  - Canyon
  - Hummer EV Pickup
  - Sierra 1500
  - Sierra 2500HD
  - Sierra 3500HD
  - Sierra EV
- **Ram**
  - 1500
  - 2500
  - 3500
  - RHO
- **Toyota**
  - Tacoma
  - Tacoma Hybrid
  - Tundra
  - Tundra Hybrid

Each model includes various trims, cab types, bed lengths, and drive types (2WD/4WD), with detailed specifications.

## Repository Structure
- `data/`
  - `processed_truck_data.json`: JSON file with truck specifications.
  - `truck_embeddings.faiss`: FAISS index for content-based similarity search.
  - `raw_samples/` (optional): Sample raw JSON files for testing preprocessing.
- `scripts/`
  - `preprocess.py`: Python script to process raw JSON files and generate outputs.
  - `examples/`
    - `query_faiss.py`: Example script for FAISS similarity search.
- `src/`
  - `main.py`: FastAPI application for querying truck data via a RAG pipeline.
  - `rag_pipeline.py` (optional): Alternative RAG pipeline implementation.
- `README.md`: Project overview and instructions.
- `requirements.txt`: Python dependencies.
- `LICENSE`: License for the data and code.

## Data Description
The dataset covers 2025 US pickup trucks with fields like:
- Brand, model, trim, year
- MSRP, towing capacity, payload capacity
- Engine displacement, fuel type (gas, diesel, hybrid, electric), drive type
- Cargo bed length, tire size

The FAISS index, built with the `all-mpnet-base-v2` model, enables fast similarity searches on truck descriptions for AI/LLM applications.

## Getting Started
### Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Install SpaCy's English model:
  ```bash
  python -m spacy download en_core_web_sm
  ```

### Using the Data Directly
- **JSON Data**: Load `data/processed_truck_data.json` with a JSON parser:
  ```python
  import pandas as pd
  df = pd.read_json('data/processed_truck_data.json')
  print(df[['brand', 'model', 'trim']].head())
  ```
- **FAISS Index**: Use `data/truck_embeddings.faiss` for similarity searches:
  ```python
  import faiss
  from sentence_transformers import SentenceTransformer
  index = faiss.read_index('data/truck_embeddings.faiss')
  embedder = SentenceTransformer('all-mpnet-base-v2')
  query = "Crew Cab with high towing capacity"
  query_embedding = embedder.encode([query])
  D, I = index.search(query_embedding, k=5)
  print(df.iloc[I[0]][['brand', 'model', 'trim']])
  ```

### Using the API
The project includes a FastAPI-based RAG pipeline (`src/main.py`) for querying vehicle data conversationally with the ability to offer a website to purchase related equipment for specific vehicle.

- **Run the API**:
  ```bash
  uvicorn src.main:app --host 0.0.0.0 --port 8000
  ```
- **Query the API**:
  - Use `curl` or a browser to query the API at `http://localhost:8000/api/pipeline?query=<your-query>`.
  - Example:
    ```bash
    curl "http://localhost:8000/api/pipeline?query=What%20is%20the%20towing%20capacity%20of%20a%202025%20Chevrolet%20Silverado%201500%20LT%20Crew%20Cab%204WD%20Short%20Box?"
    ```
  - Response example:
    ```json
    {
      "result": "The towing capacity for a 2025 Chevrolet Silverado 1500 2WD LT Crew Cab Short Box 2-Wheel Drive is:\nMaximum Towing Capacity (pounds): 7700\n\nWhat else would you like to know about this vehicle (e.g., fuel economy, payload capacity), or would you like to start a new search?",
      "clarification": null,
      "aftermarket_url": "https://website/store/exterior-accessories/towing?sort=quick%20delivery&year=2025&make=Chevrolet&model=Silverado%201500&trim=LT&drive=2WD"
    }
    ```

### Reproducing the FAISS Index
The `preprocess.py` script processes raw JSON files from a directory like:
```
E:\VehicleData\Data\2025\<brand>\<model>\<file>.json
```
Run:
```bash
python scripts/preprocess.py
```
Update the `json_files` path in `preprocess.py` to match your raw data directory. Raw files are not included due to size constraints.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ddavis485/2025-US-Truck-Data.git
   cd 2025-US-Truck-Data
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. Run the API (optional):
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000
   ```

## License
This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). This license allows non-commercial use with attribution but prohibits commercial use. See the [LICENSE](LICENSE) file for details.

## Large File Handling
Large files (`processed_truck_data.json`, `truck_embeddings.faiss`) use Git LFS. To download:
```bash
git lfs install
git lfs pull
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Contact
For questions, open an issue on GitHub.
