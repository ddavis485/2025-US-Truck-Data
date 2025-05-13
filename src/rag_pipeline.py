import json
from fastapi import FastAPI, HTTPException
import pandas as pd
import faiss
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from urllib.parse import urlencode
import logging
import requests
import os
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

class RAGPipeline:
    def __init__(self, data_path: str, index_path: str):
        """Initialize the RAG pipeline with data, FAISS index, and models."""
        self.data_path = data_path
        self.index_path = index_path
        model_path = os.getenv("MODEL_PATH", "all-mpnet-base-v2")
        try:
            self.embedder = SentenceTransformer(model_path)
            logging.info(f"Loaded SentenceTransformer from {model_path}")
        except ValueError as e:
            logging.warning(f"Failed to load model from {model_path}: {str(e)}. Falling back to downloading all-mpnet-base-v2.")
            self.embedder = SentenceTransformer("all-mpnet-base-v2")
            if os.path.exists(os.path.dirname(model_path)):
                self.embedder.save(model_path)
                logging.info(f"Saved model to {model_path}")
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-large')
        self.qa_tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-large')
        self.nlp = spacy.load('en_core_web_sm')
        self.classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        self.documents = self.load_data()
        self.index = self.load_index()
        self.chat_history = []  # Store conversation history
        self.category_map = {
            "suspension": ["suspension", "lift kit", "shock"],
            "tires": ["tire", "tyre"],
            "wheels": ["wheel", "rim"],
            "wheels-tires-package": ["wheel and tire", "tire package"],
            "bed-accessories": ["bed", "cargo", "bed liner", "tonneau", "bed cover"],
            "engine-performance": ["engine", "horsepower", "torque"],
            "intake": ["intake", "air intake"],
            "driveline": ["driveline", "driveshaft"],
            "brakes": ["brake", "braking"],
            "exhaust": ["exhaust", "muffler"],
            "tuners-and-gauges": ["tuner", "gauge", "performance chip"],
            "lighting-accessories": ["lighting accessory", "light accessory"],
            "exterior-lighting": ["exterior light", "headlamp", "taillamp"],
            "interior-lighting": ["interior light", "cabin light"],
            "off-road-lighting": ["off-road light", "offroad light"],
            "auxiliary-lighting": ["auxiliary light", "fog light"],
            "armor-and-protection": ["armor", "skid plate", "protection"],
            "body": ["body", "fender", "hood"],
            "bumpers-and-accessories": ["bumper", "bumper accessory"],
            "horns": ["horn"],
            "overlanding-and-camping": ["overlanding", "camping", "roof rack"],
            "safety-and-storage": ["safety storage", "cargo net"],
            "step-bars": ["step bar", "running board"],
            "towing": ["towing", "hitch", "trailer"],
            "winches-and-recovery": ["winch", "recovery"],
            "interior-protection-and-storage": ["interior protection", "floor mat"],
            "interior-safety": ["interior safety", "safety kit"],
            "seats": ["seat", "seat cover"]
        }

    def load_data(self) -> pd.DataFrame:
        """Load processed truck data from JSON."""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            logging.info(f"Loaded {len(df)} documents from {self.data_path}")
            # Add detailed logging for Chevrolet and Silverado counts
            chevrolet_count = len(df[df['brand'].str.lower() == 'chevrolet'])
            silverado_count = len(df[df['model'].str.lower().str.contains('silverado')])
            logging.info(f"Found {chevrolet_count} Chevrolet entries and {silverado_count} Silverado entries")
            # Log a sample of Silverado entries to confirm
            silverado_sample = df[df['model'].str.lower().str.contains('silverado')].head(2).to_dict('records')
            for doc in silverado_sample:
                logging.info(f"Sample Silverado entry: {doc['brand']} {doc['model']} {doc['trim']}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

    def load_index(self) -> faiss.IndexFlatL2:
        """Load FAISS index."""
        try:
            index = faiss.read_index(self.index_path)
            logging.info(f"Loaded FAISS index with dimension {index.d}, {index.ntotal} vectors")
            return index
        except Exception as e:
            logging.error(f"Error loading FAISS index: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading FAISS index: {str(e)}")

    def extract_entities(self, query: str) -> Dict[str, str]:
        """Extract entities (brand, model, trim, year, spec) from the query using spaCy."""
        logging.info(f"Processing query: {query}")
        doc = self.nlp(query)
        entities = {"brand": None, "model": None, "trim": None, "year": None, "spec": None}

        for ent in doc.ents:
            logging.info(f"Entity: {ent.text}, Label: {ent.label_}")
            if ent.label_ == "ORG" and ent.text.lower() in ["chevrolet", "toyota", "gmc", "ford", "ram"]:
                entities["brand"] = ent.text
            elif ent.label_ == "DATE" and ent.text.isdigit() and 2000 <= int(ent.text) <= 2030:
                entities["year"] = ent.text
            elif ent.label_ in ["PRODUCT", "NORP"]:
                entities["model"] = ent.text if not entities["model"] else entities["model"]

        # Keyword-based fallback for model
        model_keywords = ["colorado", "silverado", "tundra", "sierra", "2500", "1500", "2500hd", "3500hd"]
        for keyword in model_keywords:
            if keyword in query.lower() and not entities["model"]:
                entities["model"] = keyword.capitalize()
                break

        spec_keywords = ["tire size", "towing capacity", "curb weight", "fuel economy", "cold cranking amps", "payload capacity"]
        for keyword in spec_keywords:
            if keyword in query.lower():
                entities["spec"] = keyword
                break

        for trim in self.documents["trim"].unique():
            if trim.lower() in query.lower():
                entities["trim"] = trim
                break

        logging.info(f"Extracted entities: {entities}")
        return entities

    def detect_upsell_category(self, query: str) -> str:
        """Detect aftermarket category using transformer-based classification and chat history."""
        query_lower = query.lower()
        for category, keywords in self.category_map.items():
            if any(keyword in query_lower for keyword in keywords):
                return category

        candidate_labels = list(self.category_map.keys()) + ["accessories"]
        result = self.classifier(query, candidate_labels, multi_label=False)
        category = result["labels"][0]

        for prev_query, _ in self.chat_history[::-1]:
            for cat, keywords in self.category_map.items():
                if any(keyword in prev_query.lower() for keyword in keywords):
                    return cat

        return category if category != "accessories" else "accessories"

    def generate_aftermarket_url(self, brand: str, model: str, trim: str, year: str, drive: str, category: str, engine_displacement: str, cylinders: str, fuel_type: str, cab: str) -> str:
        """Generate and validate aftermarket parts URL."""
        base_url = "https://www.domain.com/store"
        category_path = {
            "wheels-tires-package": "wheels",
            "bed-accessories": "exterior-accessories/bed-accessories",
            "engine-performance": "performance/engine-performance",
            "intake": "performance/intake",
            "driveline": "performance/driveline",
            "brakes": "performance/brakes",
            "exhaust": "performance/exhaust",
            "tuners-and-gauges": "performance/tuners-and-gauges",
            "lighting-accessories": "lighting/lighting-accessories",
            "exterior-lighting": "lighting/exterior-lighting",
            "interior-lighting": "lighting/interior-lights",
            "off-road-lighting": "lighting/off-road-lighting",
            "auxiliary-lighting": "lighting/auxiliary-lighting",
            "armor-and-protection": "exterior-accessories/armor-and-protection",
            "body": "exterior-accessories/body",
            "bumpers-and-accessories": "exterior-accessories/bumpers-and-accessories",
            "horns": "exterior-accessories/horns",
            "overlanding-and-camping": "exterior-accessories/overlanding-and-camping",
            "safety-and-storage": "exterior-accessories/safety-and-storage",
            "step-bars": "exterior-accessories/step-bars",
            "towing": "exterior-accessories/towing",
            "winches-and-recovery": "exterior-accessories/winches-and-recovery",
            "interior-protection-and-storage": "interior-accessories/interior-protection-and-storage",
            "interior-safety": "interior-accessories/interior-safety",
            "seats": "interior-accessories/seats"
        }.get(category, category)

        # Clean the model for URL generation (remove configuration details)
        clean_model = model
        config_keywords = [
            "Crew Cab", "Double Cab", "Regular Cab", "Short Box", "Long Box", "Standard Box",
            "2-Wheel Drive", "4-Wheel Drive", "2WD", "4WD", "4x2", "4x4"
        ]
        for keyword in config_keywords:
            clean_model = clean_model.replace(keyword, "").strip()

        params = {
            "sort": "quick%20delivery" if category != "tires" else "popular",
            "year": year,
            "make": brand.capitalize(),
            "model": clean_model,
            "trim": trim if trim else "",
            "drive": drive,
            "DRChassisID": "undefined",
            "saleToggle": "0",
            "qdToggle": "0"
        }

        if category == "tires":
            params["tiresOnly"] = "true"
            params["diameter_from"] = "18"
        if category == "wheels" or category == "wheels-tires-package":
            params["vehicle_type"] = "Truck"
            params["DRChassisID"] = "92277"
        if category in ["engine-performance", "intake", "driveline", "brakes", "exhaust", "tuners-and-gauges"]:
            if engine_displacement != "N/A":
                params["liter"] = engine_displacement
            if cylinders != "N/A":
                params["cylinders"] = cylinders
            if fuel_type != "N/A":
                params["fuel_type_name"] = fuel_type
            params["cab"] = cab
        if category == "interior-lighting":
            params["liter"] = engine_displacement
            params["cylinders"] = cylinders
            params["fuel_type_name"] = fuel_type
            params["cab"] = cab.replace("7 foot", "6.5 foot")

        url = f"{base_url}/{category_path}?{urlencode(params)}"

        try:
            response = requests.head(url, timeout=5)
            if response.status_code == 200:
                return url
            else:
                logging.warning(f"Invalid URL: {url}, falling back to general search")
                return f"{base_url}/search?q={urlencode(brand)}+{urlencode(clean_model)}+{year}"
        except requests.RequestException:
            logging.warning(f"URL validation failed for {url}, returning general search")
            return f"{base_url}/search?q={urlencode(brand)}+{urlencode(clean_model)}+{year}"

    def retrieve_documents(self, query: str, entities: Dict[str, str], k: int = 3) -> List[Dict]:
        """Retrieve documents using hybrid search (semantic + DataFrame filtering)."""
        # Step 1: FAISS semantic search
        query_embedding = self.embedder.encode([query], show_progress_bar=False)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k * 2)
        retrieved_docs = [self.documents.iloc[idx].to_dict() for idx in indices[0]]
        logging.info(f"Retrieved {len(retrieved_docs)} documents via FAISS search")
        for doc in retrieved_docs:
            logging.info(f"FAISS retrieved doc: {doc['brand']} {doc['model']} {doc['trim']}")

        # Step 2: Initial filtering (brand and year must match, partial match for model)
        filtered_docs = retrieved_docs
        if entities["brand"]:
            filtered_docs = [doc for doc in filtered_docs if doc["brand"].lower() == entities["brand"].lower()]
            logging.info(f"After brand filter (brand={entities['brand']}), {len(filtered_docs)} documents remain")
            for doc in filtered_docs:
                logging.info(f"After brand filter doc: {doc['brand']} {doc['model']} {doc['trim']}")

        if entities["model"]:
            filtered_docs = [
                doc for doc in filtered_docs
                if entities["model"].lower() in doc["model"].lower()
            ]
            logging.info(f"After model filter (model={entities['model']}), {len(filtered_docs)} documents remain")
            for doc in filtered_docs:
                logging.info(f"After model filter doc: {doc['brand']} {doc['model']} {doc['trim']}")

        if entities["trim"]:
            filtered_docs = [doc for doc in filtered_docs if entities["trim"].lower() in doc["trim"].lower()]
            logging.info(f"After trim filter (trim={entities['trim']}), {len(filtered_docs)} documents remain")
            for doc in filtered_docs:
                logging.info(f"After trim filter doc: {doc['brand']} {doc['model']} {doc['trim']}")

        if entities["year"]:
            filtered_docs = [doc for doc in filtered_docs if doc["year"] == entities["year"]]
            logging.info(f"After year filter (year={entities['year']}), {len(filtered_docs)} documents remain")
            for doc in filtered_docs:
                logging.info(f"After year filter doc: {doc['brand']} {doc['model']} {doc['trim']}")

        # Step 3: Fallback - Retrieve all brand documents if no matches
        if not filtered_docs and entities["brand"]:
            logging.info(f"No matches after initial filtering. Falling back to all documents for brand {entities['brand']}.")
            filtered_docs = [
                doc for doc in self.documents.to_dict('records')
                if doc["brand"].lower() == entities["brand"].lower()
            ]
            logging.info(f"After brand fallback (brand={entities['brand']}), {len(filtered_docs)} documents remain")
            for doc in filtered_docs[:5]:  # Log first 5 for brevity
                logging.info(f"After brand fallback doc: {doc['brand']} {doc['model']} {doc['trim']}")

            if entities["year"]:
                filtered_docs = [doc for doc in filtered_docs if doc["year"] == entities["year"]]
                logging.info(f"After year filter in brand fallback (year={entities['year']}), {len(filtered_docs)} documents remain")
                for doc in filtered_docs[:5]:
                    logging.info(f"After year filter in brand fallback doc: {doc['brand']} {doc['model']} {doc['trim']}")

        # Step 4: If still no matches, try partial model match across all documents
        if not filtered_docs and entities["brand"] and entities["model"]:
            logging.info(f"No matches after brand fallback. Trying partial model match across all documents.")
            filtered_docs = [
                doc for doc in self.documents.to_dict('records')
                if doc["brand"].lower() == entities["brand"].lower() and
                   entities["model"].lower() in doc["model"].lower()
            ]
            logging.info(f"After partial model match across all documents (model={entities['model']}), {len(filtered_docs)} documents remain")
            for doc in filtered_docs[:5]:
                logging.info(f"After partial model match doc: {doc['brand']} {doc['model']} {doc['trim']}")

            if entities["year"]:
                filtered_docs = [doc for doc in filtered_docs if doc["year"] == entities["year"]]
                logging.info(f"After year filter in partial model match (year={entities['year']}), {len(filtered_docs)} documents remain")
                for doc in filtered_docs[:5]:
                    logging.info(f"After year filter in partial model match doc: {doc['brand']} {doc['model']} {doc['trim']}")

        return filtered_docs[:k]

    def is_vague_query(self, query: str, entities: Dict[str, str]) -> bool:
        """Determine if the query is vague."""
        if "best" in query.lower() or "top" in query.lower() or "good" in query.lower() or "any" in query.lower():
            return True
        specific_entities = sum(1 for k, v in entities.items() if v is not None and k != "spec")
        return specific_entities < 2

    def generate_clarification(self, query: str, retrieved_docs: List[Dict]) -> str:
        """Generate a smart clarification prompt to narrow down the query."""
        # Extract queried brand and model
        queried_brand = next((ent["brand"] for ent in [self.extract_entities(query)] if ent["brand"]), None)
        queried_model = next((ent["model"] for ent in [self.extract_entities(query)] if ent["model"]), None)

        if not retrieved_docs:
            # If no documents are found, suggest available brands and models
            available_brands = sorted(set(doc["brand"] for doc in self.documents.to_dict('records')))
            clarification = f"No relevant vehicles found for {queried_brand} {queried_model}."
            if available_brands:
                clarification += f" I have data for the following brands: {', '.join(available_brands)}."
                if queried_brand and queried_brand.lower() in [b.lower() for b in available_brands]:
                    available_models = sorted(set(doc["model"] for doc in self.documents.to_dict('records') if doc["brand"].lower() == queried_brand.lower()))
                    if available_models:
                        clarification += f" For {queried_brand}, available models are: {', '.join(available_models)}."
            clarification += " Please specify a different brand or model (e.g., Chevrolet Colorado, Toyota Tundra)."
            return clarification
        
        # Get all models for the brand
        brand_docs = [doc for doc in retrieved_docs if doc["brand"].lower() == queried_brand.lower()]
        models = sorted(set(doc["model"] for doc in brand_docs))
        
        # If multiple models exist, prompt for model selection
        if len(models) > 1:
            model_examples = ", ".join(models)
            return f"I found multiple {queried_brand} models in my database: {model_examples}. Which model are you looking for (e.g., {models[0]}, {models[1]})?"
        
        # If only one model but multiple trims, prompt for trim selection
        if len(models) == 1:
            trims = sorted(set(doc["trim"] for doc in brand_docs))
            if len(trims) > 1:
                trim_examples = ", ".join(trims[:2])  # Show only first two trims to keep prompt concise
                return f"I found multiple trims for the {queried_brand} {models[0]}: {trim_examples}, and more. Which trim would you like to explore (e.g., {trims[0]})?"
        
        # If the queried model doesn't match any retrieved models, suggest alternatives
        if queried_model and not any(queried_model.lower() in doc["model"].lower() for doc in retrieved_docs):
            available_models = sorted(set(doc["model"] for doc in retrieved_docs))
            if available_models:
                model_examples = ", ".join(available_models)
                return f"I couldn’t find a {queried_model} for {queried_brand}, but I found these models: {model_examples}. Did you mean one of these (e.g., {available_models[0]})?"
            return f"No {queried_model} found for {queried_brand}. Please check the model name or specify a different vehicle."

        # If the query is broad (e.g., multiple documents but no specific model or trim), provide suggestions
        suggestions = [
            f"{doc['brand']} {doc['model']} {doc['trim']}"
            for doc in retrieved_docs[:2]
        ]
        return f"Your query is broad. Are you interested in {', or '.join(suggestions)}? You can also specify a trim or feature for more details."

    def answer_query(self, query: str, context: str) -> str:
        """Answer the query using the QA model."""
        prompt = f"Answer the following question based on the provided context. If the context doesn't contain the answer, say so.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
        inputs = self.qa_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.qa_model.generate(
            inputs["input_ids"],
            max_length=150,
            num_beams=5,
            early_stopping=True
        )
        answer = self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def process_query(self, query: str) -> Dict[str, Optional[str]]:
        """Process a user query and return the answer, clarification, and aftermarket URL."""
        self.chat_history.append((query, None))
        if len(self.chat_history) > 10:
            self.chat_history.pop(0)

        entities = self.extract_entities(query)
        logging.info(f"Extracted entities: {entities}")

        # Retrieve documents with partial model match
        retrieved_docs = self.retrieve_documents(query, entities, k=3)

        # If no documents match, retrieve all brand documents for clarification
        if not retrieved_docs and entities["brand"]:
            logging.info(f"No documents matched the query criteria. Retrieving all {entities['brand']} documents for clarification.")
            brand_docs = [
                doc for doc in self.documents.to_dict('records')
                if doc["brand"].lower() == entities["brand"].lower()
            ]
            if entities["year"]:
                brand_docs = [doc for doc in brand_docs if doc["year"] == entities["year"]]
            retrieved_docs = brand_docs
            logging.info(f"Retrieved {len(retrieved_docs)} brand documents for clarification")
            for doc in retrieved_docs:
                logging.info(f"Brand clarification doc: {doc['brand']} {doc['model']} {doc['trim']}")

        # Generate context for answering
        context = "\n".join([doc["content"] for doc in retrieved_docs])

        # Check if the query is vague
        if self.is_vague_query(query, entities):
            clarification = self.generate_clarification(query, retrieved_docs)
            return {
                "result": "Please clarify your query for a more specific answer.",
                "clarification": clarification,
                "aftermarket_url": None
            }

        # If no documents are found, return a clarification prompt
        if not retrieved_docs:
            clarification = self.generate_clarification(query, retrieved_docs)
            return {
                "result": "No matching vehicle found. Please check the details and try again.",
                "clarification": clarification,
                "aftermarket_url": None
            }

        # Check for multiple models or trims and prompt for clarification
        queried_brand = entities["brand"]
        models = sorted(set(doc["model"] for doc in retrieved_docs if doc["brand"].lower() == queried_brand.lower()))
        if len(models) > 1:
            clarification = self.generate_clarification(query, retrieved_docs)
            return {
                "result": "Multiple models found.",
                "clarification": clarification,
                "aftermarket_url": None
            }

        trims = sorted(set(doc["trim"] for doc in retrieved_docs))
        if len(retrieved_docs) > 1 and entities["spec"] is None:
            clarification = self.generate_clarification(query, retrieved_docs)
            return {
                "result": "Multiple variants found.",
                "clarification": clarification,
                "aftermarket_url": None
            }

        # If a single document is found, answer the query
        answer = self.answer_query(query, context)
        doc = retrieved_docs[0]
        category = self.detect_upsell_category(query)
        aftermarket_url = self.generate_aftermarket_url(
            brand=doc["brand"],
            model=doc["model"],
            trim=doc["trim"],
            year=doc["year"],
            drive=doc["drive_type"],
            category=category,
            engine_displacement=doc["engine_displacement"],
            cylinders=doc["cylinders"],
            fuel_type=doc["fuel_type"],
            cab=doc["cab"]
        )
        answer += f"\n\nExplore aftermarket {category.replace('-', ' ')} options: {aftermarket_url}"

        self.chat_history[-1] = (query, answer)
        return {
            "result": answer,
            "clarification": None,
            "aftermarket_url": aftermarket_url
        }

# Initialize pipeline
pipeline = RAGPipeline(
    data_path="/data/processed_truck_data.json",
    index_path="/data/truck_embeddings.faiss"
)

@app.get("/")
async def root():
    return {"message": "Welcome to FusionAI!"}

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/pipeline")
async def run_pipeline(query: str):
    return pipeline.process_query(query)