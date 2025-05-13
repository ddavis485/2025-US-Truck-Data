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
        self.chat_history = []  # Store conversation history with user queries and responses
        self.last_vehicle_context = None  # Store the last vehicle details for context
        self.last_entities = None  # Store the last entities for refining searches
        self.category_map = {
            "suspension": ["suspension", "lift kit", "shock"],
            "tires": ["tire", "tyre"],
            "wheels": ["wheel", "rim", "wheels", "wheel size"],
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
        # Define supported specifications and their corresponding data fields
        self.spec_field_map = {
            "wheel size": ["Front Wheel Size (inches)", "Rear Wheel Size (inches)", "Spare Wheel Size (inches)"],
            "towing capacity": ["towing_capacity", "Maximum Towing Capacity (pounds)", "Towing Capacity (lbs)"],
            "fuel economy": ["Fuel Economy City (mpg)", "Fuel Economy Highway (mpg)", "EPA Fuel Economy, combined/city/highway (mpg)"],
            "payload capacity": ["Payload Capacity (lbs)", "Maximum Payload Capacity (pounds)", "As Spec'd Payload Capacity (pounds)"],
            "curb weight": ["Curb Weight (lbs)"],
            "cold cranking amps": ["Cold Cranking Amps", "Cold Cranking Amps @ 0 F"],
            "horsepower": ["Horsepower (hp)", "Maximum Horsepower @ RPM"],
            "torque": ["Torque (lb-ft)", "Maximum Torque @ RPM"]
        }
        # Define upsell links for categories with only required parameters
        self.upsell_links = {
            "suspension": "https://www.customwheeloffset.com/store/suspension?sort=instock&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "tires": "https://www.customwheeloffset.com/store/tires?tiresOnly=true&diameter_from=18&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}&sort=popular",
            "wheels": "https://www.customwheeloffset.com/store/wheels?sort=instock&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "wheels-tires-package": "https://www.customwheeloffset.com/store/wheels?sort=instock&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "bed-accessories": "https://www.customwheeloffset.com/store/exterior-accessories/bed-accessories?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "engine-performance": "https://www.customwheeloffset.com/store/performance/engine-performance?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "intake": "https://www.customwheeloffset.com/store/performance/intake?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "driveline": "https://www.customwheeloffset.com/store/performance/driveline?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "brakes": "https://www.customwheeloffset.com/store/performance/brakes?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "exhaust": "https://www.customwheeloffset.com/store/performance/exhaust?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "tuners-and-gauges": "https://www.customwheeloffset.com/store/performance/tuners-and-gauges?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "lighting-accessories": "https://www.customwheeloffset.com/store/lighting/lighting-accessories?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "exterior-lighting": "https://www.customwheeloffset.com/store/lighting/exterior-lighting?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "interior-lighting": "https://www.customwheeloffset.com/store/lighting/interior-lights?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "off-road-lighting": "https://www.customwheeloffset.com/store/lighting/off-road-lighting?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "auxiliary-lighting": "https://www.customwheeloffset.com/store/lighting/auxiliary-lighting?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "armor-and-protection": "https://www.customwheeloffset.com/store/exterior-accessories/armor-and-protection?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "body": "https://www.customwheeloffset.com/store/exterior-accessories/body?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "bumpers-and-accessories": "https://www.customwheeloffset.com/store/exterior-accessories/bumpers-and-accessories?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "horns": "https://www.customwheeloffset.com/store/exterior-accessories/horns?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "overlanding-and-camping": "https://www.customwheeloffset.com/store/exterior-accessories/overlanding-and-camping?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "safety-and-storage": "https://www.customwheeloffset.com/store/exterior-accessories/safety-and-storage?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "step-bars": "https://www.customwheeloffset.com/store/exterior-accessories/step-bars?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "towing": "https://www.customwheeloffset.com/store/exterior-accessories/towing?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "winches-and-recovery": "https://www.customwheeloffset.com/store/exterior-accessories/winches-and-recovery?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "interior-protection-and-storage": "https://www.customwheeloffset.com/store/interior-accessories/interior-protection-and-storage?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "interior-safety": "https://www.customwheeloffset.com/store/interior-accessories/interior-safety?sort=quick%20delivery&year={year}&make={brand}&model={model}&trim={trim}&drive={drive_type}",
            "seats": "https://www.customwheeloffset.com/store/interior-accessories/seats?subcategory=Seats"
        }
        # Define trim keywords for matching, prioritizing longer matches
        self.trim_keywords = [
            "rebel", "work truck", "trail boss", "z71", "lt", "1lt", "2fl", "wt", 
            "high country", "custom", "rst", "ltz", "at4", "at4x", "denali", 
            "denali ultimate", "pro", "sle", "slt", "sr5", "sr", "trd sport", 
            "trd off-road", "limited", "platinum", "trd pro"
        ]

    def load_data(self) -> pd.DataFrame:
        """Load processed truck data from JSON."""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            logging.info(f"Loaded {len(df)} documents from {self.data_path}")
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

    def _extract_entities_from_query(self, query: str) -> Dict[str, str]:
        """Helper method to extract entities from a single query without recursion."""
        doc = self.nlp(query)
        entities = {"brand": None, "model": None, "trim": None, "year": None, "spec": None, "cab": None, "drive_type": None, "box_length": None}

        # Extract entities from the query
        query_lower = query.lower()
        for ent in doc.ents:
            logging.info(f"Entity: {ent.text}, Label: {ent.label_}")
            if ent.label_ == "ORG" or ent.text.lower() in ["chevrolet", "toyota", "gmc", "ford", "ram"]:
                entities["brand"] = ent.text
            elif ent.label_ == "DATE" and ent.text.isdigit() and 2000 <= int(ent.text) <= 2030:
                entities["year"] = ent.text
            elif ent.label_ in ["PRODUCT", "NORP"]:
                entities["model"] = ent.text if not entities["model"] else entities["model"]

        # Fallback for brand extraction if NLP misses it
        if not entities["brand"]:
            brand_keywords = ["chevrolet", "toyota", "gmc", "ford", "ram"]
            for brand in brand_keywords:
                if brand in query_lower:
                    entities["brand"] = brand.capitalize()
                    break

        # Keyword-based fallback for model with composite names
        model_keywords = ["colorado", "silverado 1500", "silverado 2500hd", "silverado 3500hd", "tundra", "sierra", "2500", "1500", "2500hd", "3500hd"]
        for keyword in model_keywords:
            if keyword in query_lower and not entities["model"]:
                entities["model"] = keyword.capitalize()
                break

        # Spec keywords (expanded to include more vehicle details)
        spec_keywords = [
            "tire size", "towing capacity", "curb weight", "fuel economy", 
            "cold cranking amps", "payload capacity", "wheel size", "wheels",
            "horsepower", "torque"
        ]
        for keyword in spec_keywords:
            if keyword in query_lower:
                entities["spec"] = keyword
                break

        # Cab configuration keywords
        cab_keywords = ["crewmax", "crew cab", "double cab", "regular cab"]
        for keyword in cab_keywords:
            if keyword in query_lower:
                entities["cab"] = keyword.capitalize()
                break

        # Drive type keywords
        drive_keywords = ["2wd", "4wd", "2-wheel drive", "4-wheel drive", "4x2", "4x4"]
        for keyword in drive_keywords:
            if keyword in query_lower:
                entities["drive_type"] = "2WD" if "2wd" in keyword or "2-wheel" in keyword or "4x2" in keyword else "4WD"
                break

        # Box length keywords
        box_keywords = ["short box", "standard box", "long box"]
        for keyword in box_keywords:
            if keyword in query_lower:
                entities["box_length"] = keyword.capitalize()
                break

        # Trim matching
        for keyword in self.trim_keywords:
            if keyword in query_lower:
                entities["trim"] = keyword.upper() if keyword == "sr5" else keyword.capitalize()
                break

        # If a trim or cab keyword is found but no model, check if the keyword is part of a model or trim
        if (entities["trim"] or entities["cab"]) and not entities["model"]:
            search_keyword = (entities["trim"] or entities["cab"]).lower()
            for model in self.documents["model"].unique():
                if search_keyword in model.lower():
                    entities["model"] = model
                    if entities["trim"] and search_keyword == entities["trim"].lower():
                        entities["trim"] = None
                    if entities["cab"] and search_keyword == entities["cab"].lower():
                        entities["cab"] = None
                    break

        return entities

    def extract_entities(self, query: str) -> Dict[str, str]:
        """Extract entities from the query and use conversation history without recursion."""
        logging.info(f"Processing query: {query}")
        entities = {"brand": None, "model": None, "trim": None, "year": None, "spec": None, "cab": None, "drive_type": None, "box_length": None}

        # First, check conversation history to set baseline entities
        historical_entities = {"brand": None, "model": None, "trim": None, "year": None, "spec": None, "cab": None, "drive_type": None, "box_length": None}
        for prev_query, _ in self.chat_history[:-1]:  # Exclude the current query
            prev_entities = self._extract_entities_from_query(prev_query)
            # Update historical entities, preserving the most recent non-None values
            for key, value in prev_entities.items():
                if value is not None:
                    historical_entities[key] = value

        # Set baseline entities from history
        entities.update(historical_entities)

        # Now extract entities from the current query and update only if the field is explicitly set
        current_entities = self._extract_entities_from_query(query)
        for key, value in current_entities.items():
            if value is not None:  # Only update if the current query provides a value
                entities[key] = value

        # If the current query specifies only a brand or model, retain other historical context
        if (current_entities.get("brand") or current_entities.get("model")) and not current_entities.get("trim"):
            for key in ["trim", "cab", "drive_type", "year", "box_length"]:
                if historical_entities[key] and entities[key] is None:
                    entities[key] = historical_entities[key]

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

    def generate_aftermarket_url(self, doc: Dict, category: str) -> str:
        """Generate aftermarket parts URL using dynamic fields for the specified category."""
        if category not in self.upsell_links:
            return None

        base_url = self.upsell_links[category]
        # Ensure all parameters are strings and properly formatted
        params = {
            "year": str(doc.get("year", "")),
            "brand": doc.get("brand", "").capitalize(),
            "model": doc.get("model", "").replace(" ", "%20"),
            "trim": doc.get("trim_name", "").replace(" ", "%20"),
            "drive_type": doc.get("drive_type", "")
        }
        # Format the URL with the parameters
        url = base_url.format(**params)
        logging.info(f"Generated aftermarket URL for {category}: {url}")
        return url

    def retrieve_documents(self, query: str, entities: Dict[str, str], k: int = 3) -> List[Dict]:
        """Retrieve documents using hybrid search (semantic + DataFrame filtering)."""
        # Step 1: FAISS semantic search to get initial set and distances
        query_embedding = self.embedder.encode([query], show_progress_bar=False)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k * 10)
        retrieved_docs = [self.documents.iloc[idx].to_dict() for idx in indices[0]]
        logging.info(f"Retrieved {len(retrieved_docs)} documents via FAISS search")
        for doc in retrieved_docs:
            logging.info(f"FAISS retrieved doc: {doc['brand']} {doc['model']} {doc['trim']}")

        # Step 2: Get all documents for clarification (apply all known filters)
        clarification_docs = self.documents.to_dict('records')
        if entities["brand"]:
            clarification_docs = [doc for doc in clarification_docs if doc["brand"].lower() == entities["brand"].lower()]
            logging.info(f"Clarification docs after brand filter (brand={entities['brand']}), {len(clarification_docs)} documents remain")

        if entities["year"]:
            clarification_docs = [doc for doc in clarification_docs if doc["year"] == entities["year"]]
            logging.info(f"Clarification docs after year filter (year={entities['year']}), {len(clarification_docs)} documents remain")

        if entities["cab"]:
            clarification_docs = [
                doc for doc in clarification_docs
                if entities["cab"].lower() in (doc["cab"].lower() if doc["cab"] else doc["trim"].lower())
            ]
            logging.info(f"Clarification docs after cab filter (cab={entities['cab']}), {len(clarification_docs)} documents remain")

        if entities["drive_type"]:
            clarification_docs = [
                doc for doc in clarification_docs
                if entities["drive_type"].lower() == doc["drive_type"].lower()
            ]
            logging.info(f"Clarification docs after drive_type filter (drive_type={entities['drive_type']}), {len(clarification_docs)} documents remain")

        if entities["model"]:
            clarification_docs = [
                doc for doc in clarification_docs
                if entities["model"].lower() in doc["model"].lower()  # Partial match for model
            ]
            logging.info(f"Clarification docs after model filter (model={entities['model']}), {len(clarification_docs)} documents remain")

        if entities["trim"]:
            clarification_docs = [
                doc for doc in clarification_docs
                if entities["trim"].lower() in doc["trim"].lower() and any(keyword in doc["trim"].lower() for keyword in [entities["trim"].lower()])
            ]
            logging.info(f"Clarification docs after trim filter (trim={entities['trim']}), {len(clarification_docs)} documents remain")

        if entities["box_length"]:
            clarification_docs = [
                doc for doc in clarification_docs
                if entities["box_length"].lower() in doc["trim"].lower()
            ]
            logging.info(f"Clarification docs after box_length filter (box_length={entities['box_length']}), {len(clarification_docs)} documents remain")

        # Step 3: Get filtered documents for answering (apply all filters)
        filtered_docs = self.documents.to_dict('records')
        if entities["brand"]:
            filtered_docs = [doc for doc in filtered_docs if doc["brand"].lower() == entities["brand"].lower()]
            logging.info(f"Answer docs after brand filter (brand={entities['brand']}), {len(filtered_docs)} documents remain")

        if entities["model"]:
            filtered_docs = [
                doc for doc in filtered_docs
                if entities["model"].lower() in doc["model"].lower()  # Partial match for model
            ]
            logging.info(f"Answer docs after model filter (model={entities['model']}), {len(filtered_docs)} documents remain")

        if entities["trim"]:
            filtered_docs = [
                doc for doc in filtered_docs
                if entities["trim"].lower() in doc["trim"].lower() and any(keyword in doc["trim"].lower() for keyword in [entities["trim"].lower()])
            ]
            logging.info(f"Answer docs after trim filter (trim={entities['trim']}), {len(filtered_docs)} documents remain")

        if entities["cab"]:
            filtered_docs = [
                doc for doc in filtered_docs
                if entities["cab"].lower() in (doc["cab"].lower() if doc["cab"] else doc["trim"].lower())
            ]
            logging.info(f"Answer docs after cab filter (cab={entities['cab']}), {len(filtered_docs)} documents remain")

        if entities["drive_type"]:
            filtered_docs = [
                doc for doc in filtered_docs
                if entities["drive_type"].lower() == doc["drive_type"].lower()
            ]
            logging.info(f"Answer docs after drive_type filter (drive_type={entities['drive_type']}), {len(filtered_docs)} documents remain")

        if entities["year"]:
            filtered_docs = [doc for doc in filtered_docs if doc["year"] == entities["year"]]
            logging.info(f"Answer docs after year filter (year={entities['year']}), {len(filtered_docs)} documents remain")

        if entities["box_length"]:
            filtered_docs = [
                doc for doc in filtered_docs
                if entities["box_length"].lower() in doc["trim"].lower()
            ]
            logging.info(f"Answer docs after box_length filter (box_length={entities['box_length']}), {len(filtered_docs)} documents remain")

        # If no documents match for answering, try a broader search without FAISS
        if not filtered_docs:
            logging.info("No documents matched for answering after filtering. Performing a broader search without FAISS.")
            filtered_docs = self.documents.to_dict('records')
            if entities["brand"]:
                filtered_docs = [doc for doc in filtered_docs if doc["brand"].lower() == entities["brand"].lower()]
            if entities["model"]:
                filtered_docs = [
                    doc for doc in filtered_docs
                    if entities["model"].lower() in doc["model"].lower()  # Partial match for model
                ]
            if entities["trim"]:
                filtered_docs = [
                    doc for doc in filtered_docs
                    if entities["trim"].lower() in doc["trim"].lower() and any(keyword in doc["trim"].lower() for keyword in [entities["trim"].lower()])
                ]
            if entities["cab"]:
                filtered_docs = [
                    doc for doc in filtered_docs
                    if entities["cab"].lower() in (doc["cab"].lower() if doc["cab"] else doc["trim"].lower())
                ]
            if entities["drive_type"]:
                filtered_docs = [
                    doc for doc in filtered_docs
                    if entities["drive_type"].lower() == doc["drive_type"].lower()
                ]
            if entities["year"]:
                filtered_docs = [doc for doc in filtered_docs if doc["year"] == entities["year"]]
            if entities["box_length"]:
                filtered_docs = [
                    doc for doc in filtered_docs
                    if entities["box_length"].lower() in doc["trim"].lower()
                ]
            logging.info(f"Answer docs after broader search, {len(filtered_docs)} documents remain")

        # If still no documents for answering, return empty list
        if not filtered_docs:
            logging.info("No documents found for answering even after broader search.")
            return clarification_docs, [], distances, indices

        # Sort by FAISS distance if available, but do not truncate yet
        filtered_docs_with_distance = []
        faiss_ids = {doc["id"]: doc for doc in retrieved_docs}
        for doc in filtered_docs:
            if doc["id"] in faiss_ids and distances.size > 0 and indices.size > 0:
                idx = indices[0][list(faiss_ids.keys()).index(doc["id"])]
                distance = distances[0][list(indices[0]).index(idx)]
                filtered_docs_with_distance.append((doc, distance))
            else:
                filtered_docs_with_distance.append((doc, float('inf')))

        filtered_docs_with_distance.sort(key=lambda x: x[1])
        filtered_docs = [doc for doc, _ in filtered_docs_with_distance]

        logging.info(f"Returning {len(clarification_docs)} clarification docs and {len(filtered_docs)} answer docs")
        return clarification_docs, filtered_docs, distances, indices

    def is_vague_query(self, query: str, entities: Dict[str, str]) -> bool:
        """Determine if the query is vague."""
        if "best" in query.lower() or "top" in query.lower() or "good" in query.lower() or "any" in query.lower():
            return True
        specific_entities = sum(1 for k, v in entities.items() if v is not None and k != "spec")
        return specific_entities < 2

    def generate_clarification(self, query: str, retrieved_docs: List[Dict], queried_brand: str, entities: Dict[str, str], level: str = "model") -> str:
        """Generate a smart clarification prompt to narrow down the query."""
        queried_model = entities.get("model")
        queried_trim = entities.get("trim")
        queried_cab = entities.get("cab")

        if not retrieved_docs:
            available_brands = sorted(set(doc["brand"] for doc in self.documents.to_dict('records')))
            clarification = f"I couldn't find any vehicles matching {queried_brand} {queried_model} {queried_trim if queried_trim else queried_cab if queried_cab else ''}."
            if available_brands:
                clarification += f" I have data for brands like {', '.join(available_brands[:3])}. Try specifying a different brand or model, like a Chevrolet Colorado or Toyota Tundra."
            else:
                clarification += " Try specifying a different brand or model to start your search."
            return clarification

        # Get all models for the brand
        brand_docs = [doc for doc in retrieved_docs if doc["brand"].lower() == queried_brand.lower()]
        
        if level == "model":
            models = sorted(set(doc["model"] for doc in brand_docs))
            if len(models) > 1:
                model_examples = ", ".join(models[:2])
                clarification = f"I found multiple {queried_brand} models for {entities.get('year')}: {model_examples}, among others. Which model would you like to explore (e.g., {models[0]} or {models[1]})?"
                return clarification

        # If only one model but multiple trims, prompt for trim selection
        if level == "trim":
            trims = sorted(set(doc["trim"] for doc in brand_docs))
            if len(trims) > 1:
                trim_examples = ", ".join(trims[:2])
                clarification = f"I found a few trims for the {queried_brand} {queried_model} in {entities.get('year')}, like {trim_examples}. Which trim are you interested in?"
                return clarification

        # If the queried model doesn't match any retrieved models, suggest alternatives
        if queried_model and not any(queried_model.lower() in doc["model"].lower() for doc in retrieved_docs):
            available_models = sorted(set(doc["model"] for doc in retrieved_docs))
            if available_models:
                model_examples = ", ".join(available_models[:2])
                return f"I couldn't find a {queried_model} for {queried_brand}, but I did find models like {model_examples}. Did you mean one of these?"
            return f"No {queried_model} found for {queried_brand}. Please check the model name or specify a different vehicle."

        suggestions = [
            f"{doc['brand']} {doc['model']} {doc['trim']}"
            for doc in retrieved_docs[:2]
        ]
        return f"Your query is a bit broad. Are you looking for something like {', '.join(suggestions)}? You can also specify a trim or feature for more details."

    def answer_query(self, query: str, doc: Dict, spec: str) -> str:
        """Answer the query based on the specified vehicle detail (spec)."""
        if not spec or spec not in self.spec_field_map:
            # Default to wheel sizes if no spec is specified or recognized
            spec = "wheel size"

        fields = self.spec_field_map[spec]
        answer = f"The {spec} for a {doc['year']} {doc['brand']} {doc['model']} {doc['trim']} is:\n"
        found_value = False
        for field in fields:
            value = doc.get(field, "N/A")
            if value and value != "null" and value != "N/A":
                # For fuel economy, handle special formatting if needed
                if spec == "fuel economy" and field == "EPA Fuel Economy, combined/city/highway (mpg)":
                    # Example format: "20 (2024) / 18 (2024) / 22 (2024)"
                    try:
                        parts = value.split('/')
                        if len(parts) >= 3:
                            city = parts[1].strip().split()[0]  # "18"
                            highway = parts[2].strip().split()[0]  # "22"
                            answer += f"Fuel Economy City (mpg): {city}\n"
                            answer += f"Fuel Economy Highway (mpg): {highway}\n"
                            found_value = True
                            break
                    except:
                        continue
                else:
                    answer += f"{field}: {value}\n"
                    found_value = True
                    break

        if not found_value:
            answer += f"{spec}: Not available\n"

        return answer.strip()

    def process_query(self, query: str) -> Dict[str, Optional[str]]:
        """Process a user query with a modern, intuitive conversational flow."""
        # Check if the user wants to start a new search
        query_lower = query.lower()
        if "start a new search" in query_lower or "reset" in query_lower:
            self.chat_history = []
            self.last_vehicle_context = None
            self.last_entities = None
            return {
                "result": "I've reset our conversation. Let's start fresh! What vehicle would you like to explore?",
                "clarification": None,
                "aftermarket_url": None
            }

        # Update chat history
        self.chat_history.append((query, None))
        if len(self.chat_history) > 10:
            self.chat_history.pop(0)

        # Log the current state of the chat history for debugging
        logging.info(f"Chat history: {[item[0] for item in self.chat_history]}")

        # Extract entities from the current query, including history
        entities = self.extract_entities(query)
        logging.info(f"Final entities after extraction: {entities}")

        # Store the entities for future reference
        self.last_entities = entities

        # Check if the query is a refinement or a new search
        if not entities["brand"] and not entities["model"] and self.last_vehicle_context:
            # Assume the user is refining the last vehicle
            entities["brand"] = self.last_vehicle_context["brand"]
            entities["model"] = self.last_vehicle_context["model"]
            entities["year"] = self.last_vehicle_context["year"]
            entities["trim"] = self.last_vehicle_context.get("trim")
            entities["drive_type"] = self.last_vehicle_context.get("drive_type")
            entities["cab"] = self.last_vehicle_context.get("cab")
            entities["box_length"] = self.last_vehicle_context.get("box_length")
            logging.info(f"Assumed context from last vehicle: {entities}")

        # Retrieve documents with the full set of filters
        clarification_docs, retrieved_docs, distances, indices = self.retrieve_documents(query, entities, k=3)

        # Check if we need clarification on the brand
        queried_brand = entities["brand"]
        if not queried_brand:
            brands = sorted(set(doc["brand"] for doc in clarification_docs))
            if len(brands) > 1:
                clarification = f"I found vehicles from multiple brands: {', '.join(brands[:3])}. Which brand are you interested in (e.g., {brands[0]} or {brands[1]})?"
                return {
                    "result": "Let's narrow it down a bit.",
                    "clarification": clarification,
                    "aftermarket_url": None
                }
            elif brands:
                queried_brand = brands[0]
                entities["brand"] = queried_brand
                logging.info(f"Assumed brand {queried_brand} since only one was found.")

        # Filter clarification_docs by brand
        clarification_docs = [doc for doc in clarification_docs if doc["brand"].lower() == queried_brand.lower()]

        # Filter by model
        models = sorted(set(doc["model"] for doc in clarification_docs))
        if entities["model"]:
            clarification_docs = [
                doc for doc in clarification_docs
                if entities["model"].lower() in doc["model"].lower()
            ]
            logging.info(f"Clarification docs after model filter (model={entities['model']}), {len(clarification_docs)} documents remain")

        # Apply trim and drive type filters before model clarification
        filtered_docs = clarification_docs
        if entities["trim"]:
            filtered_docs = [
                doc for doc in filtered_docs
                if entities["trim"].lower() in doc["trim"].lower() and any(keyword in doc["trim"].lower() for keyword in [entities["trim"].lower()])
            ]
            logging.info(f"Clarification docs after trim filter (trim={entities['trim']}), {len(filtered_docs)} documents remain")

        if entities["drive_type"]:
            filtered_docs = [
                doc for doc in filtered_docs
                if entities["drive_type"].lower() == doc["drive_type"].lower()
            ]
            logging.info(f"Clarification docs after drive_type filter (drive_type={entities['drive_type']}), {len(filtered_docs)} documents remain")

        # Check if we can narrow down to a single document
        if len(filtered_docs) == 1:
            retrieved_docs = filtered_docs
        elif len(filtered_docs) == 0:
            # If filters are too strict, revert to clarification_docs for clarification
            filtered_docs = clarification_docs
        else:
            # Instead of prompting for model clarification, select the first matching document
            # Prioritize documents with the most common configuration (e.g., shortest bed length)
            filtered_docs.sort(key=lambda x: float(x.get("bed_length", "999")))  # Sort by bed length, smallest first
            retrieved_docs = [filtered_docs[0]]  # Select the first document
            logging.info(f"Selected document with model {retrieved_docs[0]['model']} and trim {retrieved_docs[0]['trim']}")

        # Filter retrieved_docs by updated entities (brand and model)
        retrieved_docs = [
            doc for doc in retrieved_docs
            if entities["brand"].lower() == doc["brand"].lower() and entities["model"].lower() in doc["model"].lower()
        ]

        # If there are multiple trims, narrow down intelligently
        trims = sorted(set(doc["trim"] for doc in retrieved_docs))
        if len(retrieved_docs) > 1:
            # Apply additional filters (cab, drive_type, box_length) to reduce the number of matches
            filtered_docs = retrieved_docs
            if entities["cab"]:
                filtered_docs = [
                    doc for doc in filtered_docs
                    if entities["cab"].lower() in (doc["cab"].lower() if doc["cab"] else doc["trim"].lower())
                ]
            if entities["drive_type"]:
                filtered_docs = [
                    doc for doc in filtered_docs
                    if entities["drive_type"].lower() == doc["drive_type"].lower()
                ]
            if entities["box_length"]:
                filtered_docs = [
                    doc for doc in filtered_docs
                    if entities["box_length"].lower() in doc["trim"].lower()
                ]

            if len(filtered_docs) == 1:
                # If we can narrow down to one document, use it
                retrieved_docs = filtered_docs
            elif len(filtered_docs) == 0:
                # If filters are too strict, revert to retrieved_docs and prompt for clarification
                pass
            else:
                # Still multiple trims, prompt for clarification
                trims = sorted(set(doc["trim"] for doc in filtered_docs))
                clarification = f"I found a few trims for the {queried_brand} {entities['model']} in {entities.get('year')}: {', '.join(trims[:2])}. Which trim would you like to know more about?"
                return {
                    "result": "Let's narrow it down a bit.",
                    "clarification": clarification,
                    "aftermarket_url": None
                }

        # If we have exactly one document, provide the answer
        if len(retrieved_docs) == 1:
            doc = retrieved_docs[0]
            answer = self.answer_query(query, doc, entities["spec"])

            # Update the last vehicle context with all relevant details
            self.last_vehicle_context = {
                "year": doc["year"],
                "brand": doc["brand"],
                "model": doc["model"],
                "trim": doc["trim"],
                "drive_type": doc["drive_type"],
                "cab": doc.get("cab"),
                "box_length": "Short Box" if "short box" in doc["trim"].lower() else "Standard Box" if "standard box" in doc["trim"].lower() else "Long Box" if "long box" in doc["trim"].lower() else None
            }

            # Detect the category for upsell link
            category = self.detect_upsell_category(query)
            aftermarket_url = self.generate_aftermarket_url(doc, category) if category in self.upsell_links else None

            # Add a conversational follow-up question
            if aftermarket_url:
                answer += f"\n\nLooking to enhance your vehicle? Check out some great {category.replace('-', ' ')} options here:\n{aftermarket_url}"
            answer += f"\n\nWhat else would you like to know about this vehicle (e.g., fuel economy, payload capacity), or would you like to start a new search?"

            self.chat_history[-1] = (query, answer)
            return {
                "result": answer,
                "clarification": None,
                "aftermarket_url": aftermarket_url
            }

        # If no documents remain after filtering, provide context-aware feedback
        if self.last_vehicle_context:
            last_vehicle = f"{self.last_vehicle_context['year']} {self.last_vehicle_context['brand']} {self.last_vehicle_context['model']} {self.last_vehicle_context['trim']}"
            clarification = (f"Sorry, I couldn't find a match for your query. The last vehicle we discussed was a {last_vehicle}. "
                            f"Would you like to refine your search for this vehicle, or start a new search altogether?")
        else:
            suggestions = [
                f"{doc['brand']} {doc['model']} {doc['trim']}"
                for doc in clarification_docs[:2]
            ]
            clarification = f"Sorry, I couldn't find a match for your query. Here are a couple of options I can help with: {', '.join(suggestions)}. What would you like to explore?"

        return {
            "result": "I couldn't find an exact match for your query.",
            "clarification": clarification,
            "aftermarket_url": None
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